"""88 bounded offline reference acceptance, separate from quality/deployment."""
from contextlib import closing
import http.client
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from tiger_memory import aml_reference as ref
from tests.child_env import child_popen


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve()
    def tearDown(self):self.temp.cleanup()

    def cli(self,*args):
        env=dict(os.environ)
        env.update(DASHSCOPE_API_KEY='SYNTHETIC_ENV_SECRET88',TIGER_MEMORY_EMBEDDING_BUDGET_RECEIPT='/not-read/receipt',
                   TIGER_MEMORY_DIR='/not-read/store',TIGER_MEMORY_ALLOW_REMOTE='1')
        p=child_popen([sys.executable,'-B','-m','tiger_memory.aml_reference',*args],env=env,
                      cwd=self.root,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        stdout,stderr=p.communicate(timeout=30)
        self.assertEqual(p.returncode,0,(stdout,stderr));self.assertEqual(stderr,'')
        self.assertNotIn('SYNTHETIC_ENV_SECRET88',stdout)
        self.assertNotIn('/not-read',stdout)
        record=json.loads(p.provenance_path.read_text());self.assertTrue(record['validated'])
        output=os.environ.get('TIGER_REFERENCE_EVIDENCE')
        if output:
            destination=Path(output);destination.mkdir(parents=True,exist_ok=True)
            (destination/(self._testMethodName+'.json')).write_text(stdout)
        return json.loads(stdout)

    def test_default_static_does_not_consume_real_environment(self):
        r=self.cli();self.assertTrue(r['passed']);self.assertEqual(r['mode'],'static')
        self.assertFalse(r['production_ready']);self.assertFalse(r['secrets_read'])
        self.assertFalse(r['network_probe']);self.assertEqual(r['profile']['old75_adoption_gate'],'FAIL')
        self.assertGreater(r['runtime_files_verified'],42)

    def test_rehearsal_real_http_p3a_restart_cache_and_unknown(self):
        r=self.cli('--rehearse')
        self.assertEqual(r['mode'],'synthetic-loopback');self.assertEqual(r['service_restarts'],2)
        self.assertEqual(r['retry_new_requests'],0);self.assertEqual(r['unknown_retained'],2)
        self.assertTrue(r['valid_raw_unknown_reused']);self.assertTrue(r['missing_raw_unknown_blocked'])
        self.assertTrue(r['budget_unchanged_on_retry']);self.assertTrue(r['test_receipt_production_rejected'])
        self.assertEqual(r['sensitive_public_canary_hits'],0)
        self.assertTrue(all(x['healthy'] and x['writer_stopped'] for x in r['log']))
        self.assertEqual(len({x['boot_id'] for x in r['log']}),3)
        end=[e for e in r['events'] if e['phase']=='end']
        self.assertTrue(any(e['recovery']=='unknown' and e['attempts'] for e in end))
        self.assertTrue(any(e['operation']=='search' and e['status']==200 and e['server_write_completed'] for e in end))
        unknown={a for e in end if e['recovery']=='unknown' for a in e['attempts']}
        recovered={a for e in r['events'] if e['phase']=='boot' for a in e['attempts']}
        self.assertEqual(unknown,recovered);self.assertEqual(len(unknown),2)
        for e in r['events']:
            self.assertEqual(len(e['artifact_sha256']),64)
            self.assertNotIn('user_id',e);self.assertNotIn('request_id',e)

    def test_manifest_missing_changed_code_and_support_rejected(self):
        expected=json.loads(ref.support_file('aml_reference_manifest.json').read_text())
        bad=self.root/'manifest.json';bad.write_text('{}')
        real=ref.support_file
        with patch.object(ref,'support_file',side_effect=lambda name:bad if name=='aml_reference_manifest.json' else real(name)):
            self.assertFalse(ref.self_check()['passed'])
            expected['runtime']['aml_adapter.py']='0'*64;bad.write_text(json.dumps(expected))
            self.assertFalse(ref.self_check()['passed'])
            expected['runtime']['aml_adapter.py']=json.loads(real('aml_reference_manifest.json').read_text())['runtime']['aml_adapter.py']
            expected['support']['aml_reference_README.md']='0'*64;bad.write_text(json.dumps(expected))
            self.assertFalse(ref.self_check()['passed'])

    def test_network_fence_denies_nonloopback_dns_socket_and_exec(self):
        f=ref.NetworkFence(loopback=True)
        for event,args in [('socket.connect',(None,('192.0.2.1',443))),('socket.getaddrinfo',('example.invalid',443)),
                           ('socket.bind',(None,('0.0.0.0',0))),('socket.gethostbyname',('localhost',)),
                           ('os.system',('anything',)),('subprocess.Popen',('curl',['curl']))]:
            with self.assertRaises(PermissionError):f(event,args)
        f('socket.connect',(None,('127.0.0.1',1)))
        self.assertEqual(f.allowed,1);self.assertEqual(f.denied,6)
        with self.assertRaises(PermissionError):ref.NetworkFence()('socket.connect',(None,('127.0.0.1',1)))

    def test_log_schema_modes_rotation_and_close(self):
        log=ref.PrivateLog(self.root/'log','1'*64,artifact_sha256='2'*64,max_bytes=1024)
        try:
            for _ in range(8):self.assertTrue(log.record('boot'))
            self.assertEqual(len(list(log.directory.iterdir())),2)
            for p in log.directory.iterdir():
                self.assertEqual(p.stat().st_mode&0o777,0o600);self.assertLessEqual(p.stat().st_size,1024)
            with self.assertRaises(ValueError):log.record('SYNTHETIC88_BODY_CANARY')
            with self.assertRaises(ValueError):log.record('end',attempts=['not-an-attempt'])
        finally:self.assertTrue(log.close()['writer_stopped'])
        self.assertFalse(log.record('boot'))

    def test_disk_failure_latches_and_no_new_writes(self):
        log=ref.PrivateLog(self.root/'log','1'*64,artifact_sha256='2'*64)
        with patch.object(log,'_append',side_effect=OSError('SYNTHETIC_PRIVATE_PATH')) as append:
            self.assertFalse(log.record('boot'));self.assertFalse(log.record('begin'))
            self.assertEqual(append.call_count,1)
        self.assertFalse(log.close()['healthy'])

    def test_stuck_writer_bounds_request_wait_and_never_clears_failure(self):
        entered=threading.Event();release=threading.Event()
        log=ref.PrivateLog(self.root/'log','1'*64,artifact_sha256='2'*64,wait=.03)
        def stalled(_):entered.set();release.wait(2)
        try:
            with patch.object(log,'_append',side_effect=stalled):
                start=time.monotonic();self.assertFalse(log.record('boot'))
                self.assertLess(time.monotonic()-start,.5);self.assertTrue(entered.is_set())
                self.assertFalse(log.record('begin'));release.set()
        finally:release.set();result=log.close()
        self.assertFalse(result['healthy']);self.assertTrue(result['writer_stopped'])

    def test_rotation_failure_latches(self):
        log=ref.PrivateLog(self.root/'log','1'*64,artifact_sha256='2'*64,max_bytes=1024)
        try:
            self.assertTrue(log.record('boot'))
            with patch.object(ref.os,'replace',side_effect=OSError('disk full')):
                results=[log.record('boot') for _ in range(5)]
            self.assertIn(False,results);self.assertTrue(log.failed.is_set())
        finally:log.close()

    def test_failed_log_refuses_real_http_before_business(self):
        from tiger_memory import aml_adapter as aml
        log=ref.PrivateLog(self.root/'log','1'*64,artifact_sha256='2'*64);log.failed.set()
        server=aml.make_server('127.0.0.1',0);server.RequestHandlerClass=ref.reference_handler(aml,threading.local(),log)
        thread=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.02},daemon=True);thread.start()
        try:
            with patch.object(aml,'add',side_effect=AssertionError('must not run')) as business:
                with closing(http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=3)) as c:
                    c.request('POST','/add',body='{}');r=c.getresponse();payload=json.loads(r.read())
                    self.assertEqual(r.status,503);self.assertEqual(payload,{'detail':{'reason':'reference-evidence-unavailable'}})
                business.assert_not_called()
        finally:server.shutdown();server.server_close();thread.join(3);log.close()

    def test_end_log_failure_does_not_undo_business_and_blocks_next(self):
        from tiger_memory import aml_adapter as aml
        log=ref.PrivateLog(self.root/'log','1'*64,artifact_sha256='2'*64)
        append=log._append
        def failing(raw):
            if json.loads(raw)['phase']=='end':raise OSError('disk full')
            append(raw)
        log._append=failing
        server=aml.make_server('127.0.0.1',0);server.RequestHandlerClass=ref.reference_handler(aml,threading.local(),log)
        thread=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.02},daemon=True);thread.start()
        try:
            with patch.object(aml,'add',return_value={'success':True}) as business:
                with closing(http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=3)) as c:
                    c.request('POST','/add',body='{}');r=c.getresponse();r.read();self.assertEqual(r.status,200)
                    self.assertTrue(log.failed.wait(1))
                    c.request('POST','/add',body='{}');r=c.getresponse();r.read();self.assertEqual(r.status,503)
                self.assertEqual(business.call_count,1)
        finally:server.shutdown();server.server_close();thread.join(3)
        self.assertFalse(log.close()['healthy'])

    def test_queue_full_latches_without_waiting_for_disk(self):
        log=ref.PrivateLog(self.root/'log','1'*64,artifact_sha256='2'*64)
        try:
            with patch.object(log.q,'put_nowait',side_effect=__import__('queue').Full):
                start=time.monotonic();self.assertFalse(log.record('boot'))
                self.assertLess(time.monotonic()-start,.5)
            self.assertTrue(log.failed.is_set())
        finally:log.close()

    def test_reference_close_joins_handler_before_log_close(self):
        from tiger_memory import aml_adapter as aml
        log=ref.PrivateLog(self.root/'log','1'*64,artifact_sha256='2'*64,wait=.5)
        append=log._append;entered=threading.Event();release=threading.Event();closed=threading.Event()
        def delayed(raw):
            if json.loads(raw)['phase']=='end':entered.set();release.wait(2)
            append(raw)
        log._append=delayed
        server=ref.make_reference_server(aml,threading.local(),log)
        thread=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.01},daemon=True);thread.start()
        closer=None
        try:
            with closing(http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=3)) as c:
                c.request('GET','/health');r=c.getresponse();r.read();self.assertEqual(r.status,200)
            self.assertTrue(entered.wait(1))
            server.shutdown();thread.join(1)
            def close():server.server_close();closed.set()
            closer=threading.Thread(target=close);closer.start()
            self.assertFalse(closed.wait(.03))
            release.set();self.assertTrue(closed.wait(1));closer.join(1)
            self.assertTrue(log.close()['healthy'])
            rows=[json.loads(line) for line in log.path.read_text().splitlines()]
            self.assertEqual([r['phase'] for r in rows],['begin','end'])
        finally:
            release.set();server.shutdown();server.server_close();thread.join(1)
            if closer:closer.join(1)
            log.close()

    def test_new_policy_remains_draft_and_separate_from_old_gate(self):
        d=json.loads(ref.support_file('aml_candidate_decision.json').read_text())
        self.assertEqual(d['status'],'DRAFT_NOT_FROZEN_POLICY')
        self.assertFalse(d['future_policy']['can_freeze_now']);self.assertIsNone(d['future_policy']['margin'])
        self.assertEqual(d['identity']['frozen75']['old_adoption_gate'],'FAIL')
        self.assertFalse(d['identity']['deployment_candidate']['adopted'])
        self.assertEqual(d['identity']['deployment_candidate']['equivalent_to75'],'UNPROVEN')


if __name__=='__main__':unittest.main()
