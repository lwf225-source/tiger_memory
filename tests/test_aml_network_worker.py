"""86 entry integration: existing factory + one shared real worker supervisor."""
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import sqlite3
import struct
import threading
import time
from unittest.mock import patch, Mock

from tiger_memory import aml_adapter as aml, request_budget as rb, service
from tiger_memory.embedding_budget import PersistentEmbeddingBudget
from tiger_memory.embedding_supervisor import NetworkSupervisor, P3bConfig
from tiger_memory.remote_embeddings import DashScopeEmbeddingProvider
from tests.test_aml_adapter import AMLServerCase
from tests.common import MemoryTestCase
from tests.test_embedding_budget import policy, raw_vector

# External frozen component manifest v2, never derived from installed bytes.
EXPECTED_WORKER = '5a47086e92f143858e4bb1837aff6b12fa18bb9d05b91f7141155d893b3e3f4d'
CANARY = 'SYNTHETIC_ENTRY86_CREDENTIAL'

class LocalProvider(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_POST(self):
        payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.received.append({'rows':len(payload['input'])})
        self.server.entered.set()
        mode=self.server.mode
        if mode=='pause': self.server.release.wait(4)
        raw=json.dumps({'model':'text-embedding-v4','usage':{'total_tokens':len(payload['input'])},
            'data':[{'index':i,'embedding':raw_vector(t,payload['dimensions'])}
                    for i,t in enumerate(payload['input'])]}).encode()
        try:
            self.send_response(200);self.send_header('Content-Length',str(len(raw)));self.end_headers()
            if mode in {'after-D','after-C'}:
                # Keep individual socket reads alive; only complete the frame late.
                # Delaying all headers would instead hit the existing per-operation timeout.
                budget=self.server.current_budget()
                self.wfile.write(raw[:1]);self.wfile.flush()
                time.sleep(max(0,budget.deadline-.3-time.monotonic()))
                self.wfile.write(raw[1:-1]);self.wfile.flush()
                offset=.025 if mode=='after-D' else .55
                time.sleep(max(0,budget.deadline+offset-time.monotonic()))
                self.wfile.write(raw[-1:])
            else:self.wfile.write(raw)
        except (BrokenPipeError,ConnectionResetError,OSError): pass

class WorkerHTTPTests(AMLServerCase):
    def setUp(self):
        self.records=[];self.providers=[];self.fallback=[]
        make=aml.make_server
        with patch.object(aml,'make_server',side_effect=lambda host,port:make(host,port,
                request_timeout=1.4,max_connections=3,allow_test_deadline=True,
                request_observer=self.records.append)):
            super().setUp()
        self.remote=ThreadingHTTPServer(('127.0.0.1',0),LocalProvider)
        self.remote.received=[];self.remote.mode='normal'
        self.remote.entered=threading.Event();self.remote.release=threading.Event()
        self.remote.current_budget=lambda:self.providers[-1]._request_budget
        self.remote_thread=threading.Thread(target=self.remote.serve_forever,kwargs={'poll_interval':.02},daemon=True)
        self.remote_thread.start()
        self.endpoint='http://127.0.0.1:%d/embeddings'%self.remote.server_port
        self.ledger=PersistentEmbeddingBudget.initialize(Path(self.temp.name).resolve()/'ledger',policy(endpoint=self.endpoint))
        self.shared=NetworkSupervisor(P3bConfig(enabled=True,slots=1,quarantine_cap=1,
            ready_timeout=1.,g_net=.35,g_term=.1,g_reap=.2,g_shutdown=.5,
            expected_worker_sha256=EXPECTED_WORKER))
        def factory(user):
            p=DashScopeEmbeddingProvider(self.endpoint,CANARY,
                cache_dir=aml.user_storage_dir(user)/'embedding-cache',dimensions=64,
                allow_test_loopback=True,budget=self.ledger,checkpoint_namespace=aml.user_storage_dir(user).name,
                network_supervisor=self.shared)
            p._opener.open=Mock(side_effect=AssertionError('P3a fallback forbidden'))
            self.fallback.append(p._opener.open);self.providers.append(p);return p
        self.flags=patch.multiple(aml,AML_EMBEDDING_FACTORY=factory,AML_EMBEDDING='remote-v4',
            AML_STRICT_RETRIEVAL=True,AML_PROFILE='lexical',AML_QUERY_REWRITE='off',AML_FACT_PROJECTION='off',
            AML_RERANKER='off',AML_RECENCY_BOOST=False,AML_NARRATIVE_TIME=False,
            AML_SESSION_RRF=False,AML_SESSION_VIEWS=False)
        self.flags.start()

    def tearDown(self):
        self.remote.release.set()
        self.server.shutdown();self.server.server_close();self.thread.join(5)
        snapshot=self.shared.close()
        self.remote.shutdown();self.remote.server_close();self.remote_thread.join(2)
        try:
            self.assertEqual(snapshot['active_slots'],0)
            self.assertTrue(all(not f.called for f in self.fallback))
            self.assertNotIn(CANARY,json.dumps(snapshot))
            target=os.environ.get('ENTRY86_EVIDENCE')
            if target:
                root=Path(target);root.mkdir(parents=True,exist_ok=True)
                (root/(self.id()+'.json')).write_text(json.dumps({'http':self.records,'transport':snapshot,
                    'received':self.remote.received,'ledger':self.ledger.snapshot(),'fallback_calls':0},indent=2))
        finally:
            self.flags.stop();MemoryTestCase.tearDown(self)

    def sources(self):return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in self.base.rglob('*.md')}
    def events(self,name):return [e for e in self.shared.snapshot()['events'] if e['event']==name]
    def long_text(self):
        prefix=aml._embedding_document('user · 2024-01-01',['aml'],'x')[:-1]
        return ''.join(('block-%04d|'%i).ljust(8192,'x') for i in range(10))+'tail'.ljust(8192-len(prefix.encode()),'z')
    def wait_records(self,n):
        until=time.monotonic()+3
        while len(self.records)<n and time.monotonic()<until:time.sleep(.01)
        self.assertGreaterEqual(len(self.records),n)

    def test_multibatch_replay_search_cache_and_user_isolation(self):
        self.assertEqual(self.add('r','A','s',[self.long_text()])[0],200)
        self.assertEqual([x['rows'] for x in self.remote.received],[10,1])
        before=self.sources();ledger=self.ledger.snapshot()
        self.assertEqual(self.add('r','A','s',['changed'])[0],200)
        self.assertEqual(self.sources(),before);self.assertEqual(self.ledger.snapshot(),ledger)
        first=self.search('block','A',1);self.assertEqual(first[0],200)
        count=len(self.remote.received);ledger=self.ledger.snapshot()
        self.assertEqual(self.search('block','A',1),first)
        self.assertEqual(self.search('block','unknown',1),(200,{'data':[]}))
        self.assertEqual(self.search('block','A',0),(200,{'data':[]}))
        self.assertEqual(len(self.remote.received),count);self.assertEqual(self.ledger.snapshot(),ledger)
        self.assertEqual(self.add('r','B','s',['other user'])[0],200)
        self.assertEqual(len(self.remote.received),count+1)
        for e in self.events('ready'):
            self.assertEqual(e['generation'],1)
            self.assertEqual(set(e['open_fds']),{0,1,2,e['lifeline_fd']})
        self.assertTrue(all(e['code_sha256']==EXPECTED_WORKER for e in self.events('started')))

    def test_http_requests_share_slot_cache_works_while_miss_rejected(self):
        self.assertEqual(self.add('cached','C','s',['cached'])[0],200)
        self.remote.entered.clear();self.remote.mode='pause';result=[]
        t=threading.Thread(target=lambda:result.append(self.add('r','A','s',['inflight'])));t.start()
        self.assertTrue(self.remote.entered.wait(2));ledger=self.ledger.snapshot()
        self.assertEqual(self.shared.snapshot()['active_slots'],1)
        self.assertEqual(self.add('r','B','s',['new miss'])[0],503)
        self.assertEqual(self.add('cached','C','s',['changed'])[0],200)
        self.assertEqual(self.ledger.snapshot(),ledger)
        self.remote.release.set();t.join(3);self.assertFalse(t.is_alive());self.assertEqual(result[0][0],200)

    def test_cancel_after_reserve_unknown_blocks_same_and_changed_request_id(self):
        reserve=self.ledger.reserve
        def cancel(*args,**kwargs):
            result=reserve(*args,**kwargs);rb.current_request_budget().cancel('deadline');return result
        with patch.object(self.ledger,'reserve',side_effect=cancel):
            self.assertEqual(self.add('r','A','s',['original'])[0],503)
        before=self.sources();ledger=self.ledger.snapshot()
        self.assertEqual((ledger['requests'],ledger['unknown'],ledger['claims']),(1,1,1))
        self.assertEqual(self.add('r','A','s',['changed'])[0],503);self.assertEqual(self.sources(),before)
        self.assertEqual(self.add('new','A','s',['original'])[0],503)
        self.assertEqual(self.ledger.snapshot(),ledger);self.assertEqual(self.remote.received,[])
        self.assertEqual(len(self.events('ready')),1);self.assertEqual(self.events('permit-complete'),[])

    def test_late_accepted_batch_finishes_but_next_batch_waits_for_retry(self):
        self.remote.mode='after-D'
        self.assertEqual(self.add('r','A','s',[self.long_text()])[0],503)
        before=self.sources();state=self.ledger.snapshot()
        self.assertEqual((state['settled'],state['claims']),(1,10))
        self.assertEqual(len(self.events('result-accepted')),1)
        self.assertEqual(len(list(self.providers[0]._checkpoint.root.glob('raw-*'))),10)
        self.assertFalse(list(self.providers[0]._checkpoint.root.glob('pooled-*')))
        self.remote.mode='normal'
        self.assertEqual(self.add('r','A','s',['changed'])[0],200)
        self.assertEqual(self.sources(),before)
        self.assertEqual([r['rows'] for r in self.remote.received],[10,1])

    def test_after_cutoff_unknown_retry_never_sends_again(self):
        self.remote.mode='after-C'
        self.assertEqual(self.add('r','A','s',['original'])[0],503)
        state=self.ledger.snapshot();source=self.sources()
        self.assertEqual((state['unknown'],state['claims']),(1,1))
        self.assertFalse(self.events('result-accepted'));self.assertEqual(len(self.remote.received),1)
        self.assertEqual(self.add('r','A','s',['changed'])[0],503);self.assertEqual(self.sources(),source)
        self.assertEqual(self.add('new','A','s',['original'])[0],503)
        self.assertEqual(state,self.ledger.snapshot());self.assertEqual(len(self.remote.received),1)
        self.assertFalse(list(self.providers[0]._checkpoint.root.glob('raw-*')))

    def test_rst_preserves_accepted_business_and_retry_cache(self):
        self.remote.mode='pause'
        payload={'request_id':'r','user_id':'A','session_id':'s',
                 'messages':[{'role':'user','content':'original','timestamp':1704067200000}]}
        body=json.dumps(payload).encode()
        with socket.create_connection(('127.0.0.1',self.port),timeout=3) as sock:
            sock.sendall(b'POST /add HTTP/1.1\r\nContent-Length: '+str(len(body)).encode()+b'\r\n\r\n'+body)
            self.assertTrue(self.remote.entered.wait(2))
            sock.setsockopt(socket.SOL_SOCKET,socket.SO_LINGER,struct.pack('ii',1,0))
        self.remote.release.set();self.wait_records(1)
        self.assertTrue(self.records[0]['business_completed']);self.assertFalse(self.records[0]['response_delivered'])
        self.assertEqual(self.records[0]['reason'],'disconnect')
        self.assertEqual(self.ledger.snapshot()['settled'],1);self.assertEqual(len(self.events('reaped')),1)
        self.remote.mode='normal';source=self.sources()
        self.assertEqual(self.request('POST','/add',payload)[0],200)
        self.assertEqual(self.sources(),source);self.assertEqual(len(self.remote.received),1)

    def test_keepalive_generation_and_worker_nonce_are_separate(self):
        conn=http.client.HTTPConnection('127.0.0.1',self.port,timeout=4)
        try:
            for i in range(2):
                payload={'request_id':'r'+str(i),'user_id':'A','session_id':'s',
                    'messages':[{'role':'user','content':'value'+str(i),'timestamp':1704067200000}]}
                conn.request('POST','/add',json.dumps(payload));response=conn.getresponse()
                self.assertEqual(response.status,200);response.read()
        finally:conn.close()
        self.wait_records(2)
        events=self.events('ready');self.assertEqual([e['generation'] for e in events],[1,2])
        self.assertEqual(len({e['nonce'] for e in events}),2)
        self.assertTrue(all(not r['watchdog_alive'] for r in self.records[:2]))

    def test_search_snapshot_survives_late_transport_then_closes(self):
        self.assertEqual(self.add('r','A','s',['violet evidence'])[0],200)
        self.remote.mode='after-D';connections=[];observed=[];opened=service.connect_readonly
        def connect():
            conn=opened();connections.append(conn);return conn
        def observe(event):
            if event['event']=='result-accepted':
                observed.append({'snapshot_open':connections[-1].in_transaction,
                    'past_D':time.monotonic()>rb.current_request_budget().deadline})
        self.shared._observer=observe
        with patch.object(service,'connect_readonly',side_effect=connect):
            self.assertEqual(self.search('violet','A',1)[0],503)
        self.assertEqual(observed,[{'snapshot_open':True,'past_D':True}])
        for conn in connections:
            with self.assertRaisesRegex(sqlite3.ProgrammingError,'closed'): _=conn.in_transaction
        count=len(self.remote.received);self.remote.mode='normal';self.shared._observer=None
        self.assertEqual(self.search('violet','A',1)[0],200)
        self.assertEqual(len(self.remote.received),count)
