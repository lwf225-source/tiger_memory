"""86 synthetic P3b acceptance; real isolated stdlib worker + local HTTP only."""
import base64
from dataclasses import replace
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from tiger_memory import embedding_supervisor as ipc
from tiger_memory.embedding_budget import PersistentEmbeddingBudget
from tiger_memory.providers import ProviderUnavailable
from tiger_memory.remote_embeddings import DashScopeEmbeddingProvider
from tiger_memory.request_budget import RequestBudget, request_scope
from tests.test_embedding_budget import policy, raw_vector
from tests.child_env import artifact_binding, child_popen

WORKER = Path(ipc.__file__).with_name('embedding_worker.py')
WORKER_HASH = hashlib.sha256(WORKER.read_bytes()).hexdigest()
CANARY = 'SYNTHETIC_WORKER86_CANARY'


def config(**kwargs):
    return replace(ipc.P3bConfig(enabled=True, slots=2, quarantine_cap=1,
        ready_timeout=2.,g_net=.2,g_term=.1,g_reap=.2,g_shutdown=.5,
        expected_worker_sha256=WORKER_HASH), **kwargs)


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_POST(self):
        payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.received.append({'count':len(payload['input']),'path':self.path})
        result={'model':'text-embedding-v4','usage':{'total_tokens':len(payload['input'])},
                'data':[{'index':i,'embedding':raw_vector(t,payload['dimensions'])}
                        for i,t in enumerate(payload['input'])]}
        mode=self.server.mode
        if mode=='no-usage':result.pop('usage')
        if mode=='bad-vector':result['data'][0]['embedding']=[1.]
        if mode=='over-usage':result['usage']['total_tokens']=8193*len(payload['input'])
        raw=json.dumps(result).encode()
        try:
            if mode=='slow-header':time.sleep(.7)
            if mode=='redirect':
                self.send_response(302);self.send_header('Location',self.server.endpoint+'/redirect');self.end_headers();return
            if mode=='oversize':raw=b' '* (ipc.RESPONSE_LIMIT+1)
            self.send_response(200);self.send_header('Content-Length',str(len(raw)));self.end_headers()
            if mode in ('drip','grace'):
                for start in range(0,len(raw),20):
                    self.wfile.write(raw[start:start+20]);self.wfile.flush();time.sleep(.015 if mode=='grace' else .05)
            elif mode=='delayed':time.sleep(.15);self.wfile.write(raw)
            else:self.wfile.write(raw)
        except (OSError,BrokenPipeError):pass


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve()
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.server.received=[];self.server.mode='normal'
        self.server.endpoint='http://127.0.0.1:%d/embeddings'%self.server.server_port
        self.thread=threading.Thread(target=self.server.serve_forever,kwargs={'poll_interval':.02},daemon=True);self.thread.start()
        self.addCleanup(self.server.server_close);self.addCleanup(self.server.shutdown)
        self.budget=PersistentEmbeddingBudget.initialize(self.root/'ledger',policy(endpoint=self.server.endpoint))
        self.supervisors=[]
        self.addCleanup(self.finish)

    def finish(self):
        reports=[]
        for supervisor in self.supervisors:
            reports.append(supervisor.close())
        # Public event records must never carry credential or input canaries.
        self.assertNotIn(CANARY,json.dumps(reports))
        target=os.environ.get('TIGER_WORKER_EVIDENCE')
        if target:
            Path(target,self.id()+'.json').write_text(json.dumps({'snapshots':reports,
                'loopback_requests':self.server.received,'ledger':self.budget.snapshot()},indent=2))

    def supervisor(self,cls=ipc.NetworkSupervisor,**kwargs):
        s=cls(config(**kwargs));self.supervisors.append(s);return s

    def provider(self,s=None,**kwargs):
        return DashScopeEmbeddingProvider(self.server.endpoint,CANARY,self.root/'cache',64,
            allow_test_loopback=True,budget=self.budget,checkpoint_namespace='worker86',network_supervisor=s,**kwargs)

    def events(self,s,name):return [e for e in s.snapshot()['events'] if e['event']==name]

    def context(self,body=b'{}',seconds=3.,budgets=()):
        return ipc.RequestContext(73,'b'*64,hashlib.sha256(body).hexdigest(),time.monotonic()+seconds,budgets)

    def test_real_multibatch_cache_and_fd(self):
        s=self.supervisor();p=self.provider(s);values=p.embed([CANARY+str(i) for i in range(11)])
        self.assertEqual(len(values),11);self.assertEqual([r['count'] for r in self.server.received],[10,1])
        before=self.budget.snapshot();self.provider(s).embed([CANARY+str(i) for i in range(11)])
        self.assertEqual(before,self.budget.snapshot());self.assertEqual(len(self.events(s,'permit-complete')),2)
        self.assertEqual(len(self.events(s,'result-accepted')),2);self.assertEqual(len(self.events(s,'reaped')),2)
        for e in self.events(s,'ready'):self.assertEqual(set(e['open_fds']),{0,1,2,e['lifeline_fd']})
        self.assertEqual(s.snapshot()['active_slots'],0)

    def test_generation_kept_integer(self):
        s=self.supervisor()
        with request_scope(RequestBudget(4,generation=912)):self.provider(s).embed(['a'])
        self.assertTrue(all(type(e['generation']) is int and e['generation']==912 for e in s.snapshot()['events']))

    def test_slot_full_no_reserve_and_cache_available(self):
        s=self.supervisor(slots=1);p=self.provider(s);p.embed(['cached']);before=self.budget.snapshot()
        h=s.prepare(self.context(),allow_loopback=True)
        with self.assertRaisesRegex(ProviderUnavailable,'capacity'):self.provider(s).embed(['miss'])
        self.provider(s).embed(['cached']);self.assertEqual(before,self.budget.snapshot());s.cleanup(h)

    def test_spawn_failure_no_reserve_no_fallback(self):
        s=self.supervisor();before=self.budget.snapshot()
        with patch.object(s,'_spawn',side_effect=OSError('synthetic')):
            with self.assertRaises(ProviderUnavailable):self.provider(s).embed(['a'])
        self.assertEqual(before,self.budget.snapshot());self.assertEqual(self.server.received,[]);self.assertEqual(s.snapshot()['active_slots'],0)

    def test_bad_ready_no_reserve(self):
        s=self.supervisor();original=s._read
        def read(*args):
            obj=original(*args);obj['generation']+=1;return obj
        with patch.object(s,'_read',side_effect=read):
            with self.assertRaisesRegex(ProviderUnavailable,'ready'):self.provider(s).embed(['a'])
        self.assertIsNone(self.provider(s).last_attempt);self.assertEqual(self.budget.snapshot()['claims'],0)

    def test_ready_timeout_no_reserve(self):
        s=self.supervisor(ready_timeout=.05)
        def blocked_spawn(fd):
            return subprocess.Popen([sys.executable,'-I','-c','import time;time.sleep(2)'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,close_fds=True,pass_fds=(fd,))
        with patch.object(s,'_spawn',side_effect=blocked_spawn):
            with self.assertRaisesRegex(ProviderUnavailable,'deadline'):self.provider(s).embed(['a'])
        self.assertEqual(self.budget.snapshot()['claims'],0);self.assertEqual(s.snapshot()['active_slots'],0)

    def test_reserve_failure_cleans_ready(self):
        s=self.supervisor()
        with patch.object(self.budget,'reserve',side_effect=ProviderUnavailable('synthetic-reserve')):
            with self.assertRaisesRegex(ProviderUnavailable,'synthetic-reserve'):self.provider(s).embed(['a'])
        self.assertEqual(len(self.events(s,'ready')),1);self.assertEqual(s.snapshot()['active_slots'],0);self.assertEqual(self.server.received,[])

    def test_cancel_before_reserve(self):
        s=self.supervisor();b=RequestBudget(4);b.cancel('disconnect')
        with request_scope(b):
            with self.assertRaises(ProviderUnavailable):self.provider(s).embed(['a'])
        self.assertEqual(s.snapshot()['events'],[]);self.assertEqual(self.budget.snapshot()['claims'],0)

    def test_cancel_after_reserve_unknown_blocks_all_misses(self):
        s=self.supervisor();b=RequestBudget(4);original=self.budget.reserve
        def reserve(*args):
            result=original(*args);b.cancel('disconnect');return result
        with request_scope(b),patch.object(self.budget,'reserve',side_effect=reserve):
            with self.assertRaises(ProviderUnavailable):self.provider(s).embed(['a'])
        before=self.budget.snapshot();self.assertEqual(before['claims'],1)
        with self.assertRaisesRegex(ProviderUnavailable,'work-blocked'):self.provider(s).embed(['new']*10+['a'])
        self.assertEqual(before,self.budget.snapshot());self.assertEqual(self.server.received,[]);self.assertEqual(len(self.events(s,'started')),1)

    def test_partial_permit_no_wire_keeps_unknown(self):
        s=self.supervisor();original=s._write
        def write(h,value,limit,until,admission=True):
            if value['type']=='PERMIT':
                os.write(h.process.stdin.fileno(),ipc.frame(value,limit)[:57]);raise ProviderUnavailable('synthetic-partial')
            return original(h,value,limit,until,admission)
        with patch.object(s,'_write',side_effect=write):
            with self.assertRaisesRegex(ProviderUnavailable,'synthetic-partial'):self.provider(s).embed(['a'])
        self.assertEqual(self.server.received,[]);self.assertEqual(self.budget.snapshot()['claims'],1)
        self.assertEqual(list((self.root/'cache').rglob('raw-*')),[])

    def test_slow_headers_cutoff(self):self.cutoff_case('slow-header')
    def test_slow_drip_cutoff(self):self.cutoff_case('drip')
    def cutoff_case(self,mode):
        s=self.supervisor(g_net=.02);self.server.mode=mode
        with request_scope(RequestBudget(.25)):
            p=self.provider(s)
            with self.assertRaises(ProviderUnavailable):p.embed(['a'])
        self.assertEqual(p.last_attempt['state'],'UNKNOWN');self.assertFalse([e for e in self.events(s,'result-accepted') if e['status']=='ok'])
        self.assertEqual(list((self.root/'cache').rglob('raw-*')),[]);self.assertEqual(s.snapshot()['active_slots'],0)

    def test_accepted_finish_can_exceed_deadline(self):
        s=self.supervisor();b=RequestBudget(2)
        with request_scope(b):
            p=self.provider(s);original=p._checkpoint.put
            def put(*args):
                original(*args)
                if args[0]=='raw':b.cancel('disconnect')
            with patch.object(p._checkpoint,'put',side_effect=put):
                with self.assertRaises(ProviderUnavailable):p.embed(['a'])
        self.assertEqual(p.last_attempt['state'],'SETTLED');self.assertTrue(self.events(s,'result-accepted'))
        self.provider(s).embed(['a']);self.assertEqual(len(self.server.received),1)

    def test_missing_usage_raw_reuse_unknown(self):
        self.server.mode='no-usage';s=self.supervisor();p=self.provider(s)
        with self.assertRaisesRegex(ProviderUnavailable,'usage-untrusted'):p.embed(['a'])
        self.assertEqual(p.last_attempt['state'],'UNKNOWN');self.provider(s).embed(['a']);self.assertEqual(len(self.server.received),1)

    def test_bad_vector_no_raw_unknown(self):
        self.server.mode='bad-vector';s=self.supervisor();p=self.provider(s)
        with self.assertRaisesRegex(ProviderUnavailable,'dimension-mismatch'):p.embed(['a'])
        self.assertEqual(p.last_attempt['state'],'UNKNOWN');self.assertFalse(list((self.root/'cache').rglob('raw-*')))

    def test_usage_over_limit_stop_retains_raw(self):
        self.server.mode='over-usage';s=self.supervisor();p=self.provider(s)
        with self.assertRaisesRegex(ProviderUnavailable,'usage-untrusted'):p.embed(['a'])
        self.assertTrue(self.budget.snapshot()['stopped']);self.assertTrue(list((self.root/'cache').rglob('raw-*')))

    def test_poll_fault_preserves_accepted_result_and_quarantine(self):
        s=self.supervisor();original=s._poll
        def poll(h):
            if h.accepted:raise OSError('synthetic-poll')
            return original(h)
        with patch.object(s,'_poll',side_effect=poll):
            self.provider(s).embed(['a'])
            self.assertEqual(s.snapshot()['quarantined'],1)
            with self.assertRaisesRegex(ProviderUnavailable,'capacity'):self.provider(s).embed(['new'])
            self.provider(s).embed(['a'])
        time.sleep(.05);s.reap_pending();self.assertEqual(s.snapshot()['active_slots'],0)
        self.assertTrue(self.events(s,'cleanup-unconfirmed'))

    def test_unreap_adapter_quarantines_until_confirmation(self):
        s=self.supervisor();h=s.prepare(self.context(),allow_loopback=True)
        with patch.object(s,'_poll',return_value=None),patch.object(s,'_wait',return_value=None):s.cleanup(h)
        self.assertEqual(s.snapshot()['quarantined'],1)
        time.sleep(.05);s.reap_pending();self.assertEqual(s.snapshot()['active_slots'],0)

    def test_wrong_binding_result_not_accepted(self):
        for key,value in [('generation',99),('nonce','c'*32),('attempt','c'*32),('fingerprint','c'*64),('batch_digest','c'*64)]:
            with self.subTest(key=key):
                s=self.supervisor();original=s._read
                def read(*args):
                    result=original(*args)
                    if result['type']=='RESULT':result[key]=value
                    return result
                with patch.object(s,'_read',side_effect=read):
                    with self.assertRaisesRegex(ProviderUnavailable,'binding'):self.provider(s).embed([key])
                self.assertFalse(self.events(s,'result-accepted'))
        self.assertFalse(list((self.root/'cache').rglob('raw-*')))

    def test_result_late_after_validation_abandoned(self):
        s=self.supervisor(g_net=0);original=s._read;b=RequestBudget(2)
        def read(*args):
            value=original(*args)
            if value['type']=='RESULT':b.cancel('disconnect')
            return value
        with request_scope(b),patch.object(s,'_read',side_effect=read):
            with self.assertRaises(ProviderUnavailable):self.provider(s).embed(['a'])
        self.assertFalse(self.events(s,'result-accepted'));self.assertFalse(list((self.root/'cache').rglob('raw-*')))

    def test_redirect_no_retry(self):
        self.server.mode='redirect';s=self.supervisor()
        with self.assertRaisesRegex(ProviderUnavailable,'worker-http'):self.provider(s).embed(['a'])
        self.assertEqual(len(self.server.received),1)

    def test_response_limit(self):
        self.server.mode='oversize';s=self.supervisor()
        with self.assertRaisesRegex(ProviderUnavailable,'response-too-large'):self.provider(s).embed(['a'])
        self.assertEqual(len(self.server.received),1)

    def test_proxy_environment_ignored(self):
        s=self.supervisor()
        with patch.dict(os.environ,{'http_proxy':'http://127.0.0.1:1','HTTP_PROXY':'http://127.0.0.1:1','HTTPS_PROXY':'http://127.0.0.1:1','NO_PROXY':''}):self.provider(s).embed(['a'])
        self.assertEqual(len(self.server.received),1)

    def test_encoded_size_preflight_no_reserve(self):
        s=self.supervisor()
        with self.assertRaisesRegex(ProviderUnavailable,'request-size'):s.validate_request(self.server.endpoint,CANARY,b'x'*ipc.PERMIT_LIMIT)
        self.assertEqual(s.snapshot()['events'],[]);self.assertEqual(self.budget.snapshot()['claims'],0)
        self.provider(s).embed(['\x01'*8192+str(i) for i in range(10)])
        self.assertEqual(len(self.server.received),2)

    def test_worker_hash_negative_does_not_execute(self):
        marker=self.root/'never';fake=self.root/'embedding_worker.py';fake.write_text('open('+repr(str(marker))+',"w").write("bad")')
        with patch.object(ipc,'__file__',str(self.root/'embedding_supervisor.py')):
            with self.assertRaisesRegex(ProviderUnavailable,'code-integrity'):ipc.NetworkSupervisor(config())
        self.assertFalse(marker.exists());self.assertNotIn('tiger_memory.embedding_worker',sys.modules)
        fake.unlink()
        with patch.object(ipc,'__file__',str(self.root/'embedding_supervisor.py')):
            with self.assertRaisesRegex(ProviderUnavailable,'code-integrity'):ipc.NetworkSupervisor(config())

    def test_defaults_disabled_and_platform_reject(self):
        s=ipc.NetworkSupervisor(ipc.P3bConfig());self.assertFalse(s.config.enabled)
        self.assertIsNone(self.provider(s)._network_supervisor)
        with patch.object(ipc.sys,'platform','win32'):
            with self.assertRaisesRegex(ProviderUnavailable,'platform-unvalidated'):ipc.NetworkSupervisor(config())
        with self.assertRaisesRegex(ProviderUnavailable,'configuration'):ipc.NetworkSupervisor(config(expected_worker_sha256=''))

    def test_duplicate_send_cannot_repeat_exchange(self):
        s=self.supervisor();p=self.provider(s);original=s.send;saved=[]
        def send(*args):saved.append(args);return original(*args)
        with patch.object(s,'send',side_effect=send):p.embed(['a'])
        with self.assertRaisesRegex(ProviderUnavailable,'abandoned'):original(*saved[0])
        self.assertEqual(len(self.server.received),1)

    def test_close_blocks_new_misses_cache_still_works(self):
        s=self.supervisor();self.provider(s).embed(['a']);s.close()
        self.provider(s).embed(['a'])
        with self.assertRaisesRegex(ProviderUnavailable,'capacity'):self.provider(s).embed(['b'])
        self.assertEqual(len(self.server.received),1)

    def test_framing_helpers_bound_without_worker_import(self):
        namespace={};source=WORKER.read_text();exec('import json,hashlib,struct\n'+source[source.index('MAGIC ='):source.index('\ndef context_ok')],namespace)
        value={'type':'TEST','controls':'\x01'*100}
        self.assertEqual(ipc.frame(value,4000),namespace['frame'](value,4000))
        with self.assertRaises(ValueError):ipc.decode_json(b'{"x":1,"x":2}')
        with self.assertRaises(ValueError):ipc.decode_json(b'{"x":NaN}')
        with self.assertRaises(ValueError):ipc.frame(value,5)

    def test_bad_wire_frames_bounded(self):
        from types import SimpleNamespace
        cases=[ipc.HEADER.pack(ipc.MAGIC,ipc.RESULT_LIMIT+1,b'x'*32),
               ipc.HEADER.pack(ipc.MAGIC,2,b'x'*32)+b'{}',
               ipc.HEADER.pack(b'BAD!',2,hashlib.sha256(b'{}').digest())+b'{}',
               ipc.HEADER.pack(ipc.MAGIC,10,b'x'*32)+b'{']
        duplicate=b'{"type":1,"type":2}'
        cases.append(ipc.HEADER.pack(ipc.MAGIC,len(duplicate),hashlib.sha256(duplicate).digest())+duplicate)
        for payload in cases:
            with self.subTest(size=len(payload)):
                s=self.supervisor();r,w=os.pipe();os.set_blocking(r,False)
                stream=os.fdopen(r,'rb',buffering=0);h=ipc.Handle(self.context());h.process=SimpleNamespace(stdout=stream)
                os.write(w,payload);os.close(w)
                try:
                    with self.assertRaises(ProviderUnavailable):s._read(h,ipc.RESULT_LIMIT,lambda:time.monotonic()+1)
                finally:stream.close()

    def test_partial_frame_at_cutoff(self):
        from types import SimpleNamespace
        s=self.supervisor();r,w=os.pipe();os.set_blocking(r,False)
        stream=os.fdopen(r,'rb',buffering=0);h=ipc.Handle(self.context());h.process=SimpleNamespace(stdout=stream)
        os.write(w,ipc.frame({'type':'RESULT'},1024)[:45]);end=time.monotonic()+.04
        try:
            with self.assertRaisesRegex(ProviderUnavailable,'deadline'):s._read(h,1024,lambda:end)
        finally:os.close(w);stream.close()

    def test_parent_write_backpressure_deadline(self):
        from types import SimpleNamespace
        s=self.supervisor();r,w=os.pipe();os.set_blocking(w,False)
        stream=os.fdopen(w,'wb',buffering=0);h=ipc.Handle(self.context(seconds=.05));h.process=SimpleNamespace(stdin=stream)
        try:
            with self.assertRaisesRegex(ProviderUnavailable,'deadline'):
                s._write(h,{'padding':'a'*500000},ipc.PERMIT_LIMIT,h.context.admission_deadline)
        finally:stream.close();os.close(r)

    def test_duplicate_sending_rejected(self):
        s=self.supervisor();original=s._read;previous=[]
        def read(*args):
            value=original(*args)
            if value['type']=='SENDING':previous.append(value)
            if value['type']=='RESULT':return previous[0]
            return value
        with patch.object(s,'_read',side_effect=read):
            with self.assertRaisesRegex(ProviderUnavailable,'order'):self.provider(s).embed(['a'])
        self.assertFalse(self.events(s,'result-accepted'));self.assertEqual(len(self.server.received),1)

    def test_observer_failure_is_recorded(self):
        s=self.supervisor();s._observer=lambda e: (_ for _ in ()).throw(ValueError('synthetic-observer'))
        self.provider(s).embed(['a']);self.assertGreater(s.snapshot()['observer_errors'],0)

    def test_existing_elapsed_not_reset_at_spawn(self):
        s=self.supervisor();p=self.provider(s,max_elapsed_seconds=.03);time.sleep(.04)
        with self.assertRaisesRegex(ProviderUnavailable,'elapsed'):p.embed(['a'])
        self.assertFalse(s.snapshot()['events']);self.assertEqual(self.budget.snapshot()['claims'],0)

    def test_parent_kill_lifeline_and_stop_watchdog(self):
        import select
        binding=artifact_binding()
        facts=[]
        for stage in ('before-init','ready','reserved','partial-permit','sending','stop'):
            with self.subTest(stage=stage):
                root=self.root/stage;root.mkdir()
                actual='reserved' if stage=='stop' else stage
                proc=child_popen([sys.executable,'-B','-m','tests.embedding_worker_parent',
                    str(root),self.server.endpoint,actual],env={'PATH':os.defpath},cwd='/',stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,close_fds=True)
                try:
                    self.assertTrue(select.select([proc.stdout],[],[],4)[0])
                    line=proc.stdout.readline();self.assertTrue(line,proc.stderr.read() if proc.poll() is not None else 'no-ready')
                    ready=json.loads(line)
                    provenance=json.loads(proc.provenance_path.read_text())
                    self.assertTrue(provenance['validated'])
                    self.assertEqual(provenance['pid'],ready['pid'])
                    self.assertEqual(provenance['runtime_parent'],binding['runtime_parent'])
                    self.assertEqual(provenance['test_root'],binding['test_root'])
                    self.assertEqual(provenance['runtime_hashes'],binding['runtime_hashes'])
                    self.assertEqual(provenance['test_hashes'],binding['test_hashes'])
                    self.assertEqual(provenance['worker'],{'kind':'-m','module':'tests.embedding_worker_parent'})
                    for name in ('tests/embedding_worker_parent.py','tiger_memory/remote_embeddings.py',
                                 'tiger_memory/embedding_supervisor.py'):
                        package,relative=name.split('/',1)
                        parent=binding['runtime_parent'] if package=='tiger_memory' else binding['test_root']
                        hashes=binding['runtime_hashes'] if package=='tiger_memory' else binding['test_hashes']
                        self.assertEqual(provenance['loaded'][name],
                            {'origin':str(Path(parent)/name),'sha256':hashes[relative]})
                    watcher=select.kqueue()
                    watcher.control([select.kevent(ready['worker_pid'],filter=select.KQ_FILTER_PROC,
                        flags=select.KQ_EV_ADD|select.KQ_EV_ONESHOT,fflags=select.KQ_NOTE_EXIT)],0,0)
                    try:
                        if stage=='stop':os.kill(proc.pid,signal.SIGSTOP)
                        else:proc.kill();proc.wait(timeout=2)
                        events=watcher.control(None,1,2)
                        self.assertTrue(events,'worker exit not observed')
                        self.assertTrue(events[0].fflags & select.KQ_NOTE_EXIT)
                        facts.append({'stage':stage,'worker_exit_kqueue':True,'reaped_by_test_parent':False,
                                      'fixture_child_provenance':provenance})
                    finally:watcher.close()
                    if stage=='stop':proc.kill();proc.wait(timeout=2)
                    ledger=PersistentEmbeddingBudget(root/'ledger',policy(endpoint=self.server.endpoint),ledger_id=ready['ledger_id'])
                    expected=0 if stage in ('before-init','ready') else 1
                    self.assertEqual(ledger.snapshot()['claims'],expected)
                    if expected:
                        retry=DashScopeEmbeddingProvider(self.server.endpoint,'SYNTHETIC_PARENT86',root/'cache',64,
                            allow_test_loopback=True,budget=ledger,checkpoint_namespace='parent86',network_supervisor=self.supervisor())
                        with self.assertRaisesRegex(ProviderUnavailable,'work-blocked'):retry.embed(['synthetic parent death'])
                finally:
                    if proc.poll() is None:proc.kill();proc.wait(timeout=2)
                    proc.stdout.close();proc.stderr.close()
        target=os.environ.get('TIGER_WORKER_EVIDENCE')
        if target:Path(target,'parent-death-facts.json').write_text(json.dumps(facts,indent=2))

    def test_actual_response_after_d_within_c_finishes_raw(self):
        s=self.supervisor(g_net=.5);self.server.mode='grace'
        b=RequestBudget(.20,generation=555)
        with request_scope(b):
            p=self.provider(s)
            with self.assertRaises(ProviderUnavailable):p.embed(['a'])
        accepted=self.events(s,'result-accepted');self.assertEqual(len(accepted),1)
        self.assertEqual(accepted[0]['status'],'ok');self.assertGreater(accepted[0]['monotonic'],b.deadline)
        self.assertLess(accepted[0]['monotonic'],b.deadline+.5);self.assertEqual(p.last_attempt['state'],'SETTLED')
        self.provider(s).embed(['a']);self.assertEqual(len(self.server.received),1)

    def test_real_duplicate_permit_never_second_http(self):
        s=self.supervisor();original=s._write
        def write(h,value,limit,until,admission=True):
            original(h,value,limit,until,admission)
            if value['type']=='PERMIT':original(h,value,limit,until,admission)
        with patch.object(s,'_write',side_effect=write):self.provider(s).embed(['a'])
        self.assertEqual(len(self.server.received),1);self.assertEqual(len(self.events(s,'result-accepted')),1)

    def test_boot_idle_exits_before_init(self):
        s=self.supervisor();read,write=os.pipe();os.set_inheritable(write,False)
        h=ipc.Handle(self.context(seconds=10));h.life=write;s._handles[h.nonce]=h
        try:
            h.process=s._spawn(read);os.close(read)
            started=time.monotonic()
            with h.lock:code=s._wait(h,7)
            self.assertIn(code,(72,73))  # receive timeout and watchdog race, both exit without permit.
            self.assertGreater(time.monotonic()-started,4.7)
        finally:s.cleanup(h)
        self.assertEqual(s.snapshot()['active_slots'],0);self.assertEqual(self.budget.snapshot()['claims'],0)

    def test_other_exec_does_not_inherit_lifeline_write(self):
        s=self.supervisor();h=s.prepare(self.context(seconds=2),allow_loopback=True)
        other=subprocess.Popen([sys.executable,'-I','-c','import time;time.sleep(1)'],close_fds=True,
            stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        try:
            # An unrelated live child must not delay lifeline EOF after parent closes it.
            os.close(h.life);h.life=None
            with h.lock:code=s._wait(h,.5)
            self.assertEqual(code,74);self.assertIsNone(other.poll())
        finally:
            s.cleanup(h);other.terminate();other.wait(timeout=2)

    def test_launch_and_persistence_canary_redaction(self):
        s=self.supervisor();original=s._spawn;seen=[]
        def spawn(fd):
            proc=original(fd);seen.append(proc.args);return proc
        with patch.object(s,'_spawn',side_effect=spawn):self.provider(s).embed([CANARY+' body'])
        self.assertNotIn(CANARY,json.dumps(seen))
        for path in self.root.rglob('*'):
            if path.is_file():self.assertNotIn(CANARY.encode(),path.read_bytes(),str(path))
        self.assertTrue(all(args[1:4]==['-I','-B','-c'] for args in seen))

    def test_close_race_cannot_spawn_after_abandon(self):
        s=self.supervisor();entered=threading.Event();release=threading.Event();errors=[]
        original=ipc.time.monotonic;calls=[0]
        # Pause prepare just after registering its slot, before spawn acquires h.lock.
        def clock():
            if threading.current_thread().name=='prepare86' and s._handles and not entered.is_set():
                entered.set();release.wait(2)
            return original()
        def run():
            try:s.prepare(self.context(),allow_loopback=True)
            except ProviderUnavailable:errors.append('abandoned')
        with patch.object(ipc.time,'monotonic',side_effect=clock):
            thread=threading.Thread(target=run,name='prepare86');thread.start()
            self.assertTrue(entered.wait(2));s.close();release.set();thread.join(2)
        self.assertFalse(thread.is_alive());self.assertEqual(errors,['abandoned'])
        self.assertFalse(self.events(s,'started'));self.assertEqual(s.snapshot()['active_slots'],0)

    def test_ready_field_types_rejected(self):
        for field,value in [('open_fds',[0,True,2,9]),('lifeline_fd',[]),('pid',False)]:
            with self.subTest(field=field):
                s=self.supervisor();original=s._read
                def read(*args):
                    obj=original(*args);obj[field]=value;return obj
                with patch.object(s,'_read',side_effect=read):
                    with self.assertRaisesRegex(ProviderUnavailable,'ready'):self.provider(s).embed([field])
        self.assertEqual(self.budget.snapshot()['claims'],0)

    def test_error_code_wrong_type_fixed_failure(self):
        s=self.supervisor();original=s._read
        def read(*args):
            result=original(*args)
            if result['type']=='RESULT':
                result.pop('body_b64',None);result.update(status='error',code=[])
            return result
        with patch.object(s,'_read',side_effect=read):
            with self.assertRaisesRegex(ProviderUnavailable,'worker-frame'):self.provider(s).embed(['a'])
        self.assertFalse(self.events(s,'result-accepted'))

    def test_synthetic_resolver_stall_no_real_dns(self):
        # Explicit verified fixture variant injects a sleeping resolver; no DNS call.
        code=WORKER.read_text().replace("if __name__ == '__main__':", "import socket\ndef stalled_resolver(*a,**k):\n    time.sleep(60)\nsocket.getaddrinfo=stalled_resolver\n\nif __name__ == '__main__':")
        worker=self.root/'embedding_worker.py';worker.write_text(code)
        with patch.object(ipc,'__file__',str(self.root/'embedding_supervisor.py')):
            s=self.supervisor(expected_worker_sha256=hashlib.sha256(code.encode()).hexdigest(),g_net=.01)
        with request_scope(RequestBudget(.25)):
            p=self.provider(s)
            with self.assertRaises(ProviderUnavailable):p.embed(['a'])
        self.assertEqual(self.server.received,[]);self.assertEqual(p.last_attempt['state'],'UNKNOWN')
        self.assertFalse(self.events(s,'result-accepted'));self.assertEqual(s.snapshot()['active_slots'],0)

    def test_signal_fault_keeps_slot_until_confirmed_exit(self):
        s=self.supervisor();h=s.prepare(self.context(),allow_loopback=True)
        with patch.object(s,'_signal',side_effect=OSError('synthetic-signal')):
            s.cleanup(h)
        self.assertEqual(s.snapshot()['quarantined'],1);self.assertFalse(self.events(s,'reaped'))
        # Lifeline EOF remains independent of the failed signal adapter.
        time.sleep(.06);s.reap_pending();self.assertEqual(s.snapshot()['active_slots'],0)


if __name__=='__main__':unittest.main()
