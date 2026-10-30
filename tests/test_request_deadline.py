"""Synthetic short deadlines. No production parameters or external transports."""
import fcntl
import hashlib
import http.client
import json
import os
from pathlib import Path
import socket
import sqlite3
import struct
import subprocess
import sys
import threading
import time
import unittest
from unittest.mock import patch

from tiger_memory import aml_adapter as aml, database, events, service
from tiger_memory import request_budget as rb
from tiger_memory.aml_add_lock import add_admission_lock
from tiger_memory.aml_http_lifecycle import SocketWatchdog
from tests.common import MemoryTestCase


def evidence(name, value):
    out = os.environ.get('DEADLINE_EVIDENCE')
    if out:
        p = Path(out); p.mkdir(parents=True, exist_ok=True)
        (p / (name + '.json')).write_text(json.dumps(value, indent=2))


class HTTPDeadlineTests(unittest.TestCase):
    def setUp(self):
        self.records = []
        self.server = aml.make_server('127.0.0.1', 0, request_timeout=.35,
            max_connections=2, allow_test_deadline=True, request_observer=self.records.append)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
        self.thread.start()
        self.flags = patch.multiple(aml, AML_EMBEDDING='off', AML_RERANKER='off',
            AML_QUERY_REWRITE='off', AML_FACT_PROJECTION='off')
        self.flags.start()

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(2)
        self.flags.stop()

    def connect(self):
        sock = socket.create_connection(('127.0.0.1', self.port), timeout=2)
        self.addCleanup(sock.close)
        return sock

    def wait_records(self, count=1):
        end = time.monotonic()+2
        while len(self.records)<count and time.monotonic()<end:
            time.sleep(.01)
        self.assertGreaterEqual(len(self.records),count)
        self.assertFalse(any(r['watchdog_alive'] for r in self.records))

    def test_slow_line_headers_body_close_without_business(self):
        cases = {'line':b'P', 'headers':b'POST /add HTTP/1.1\r\nX-Slow: ',
                 'body':b'POST /add HTTP/1.1\r\nHost: local\r\nContent-Length: 1000\r\n\r\n'}
        with patch.object(aml,'add') as add, patch.object(aml,'search') as search:
            for name,prefix in cases.items():
                sock=self.connect(); start=time.monotonic(); sock.sendall(prefix)
                end=start+.65
                while time.monotonic()<end:
                    try:sock.sendall(b' ')
                    except OSError:break
                    time.sleep(.025)
                try:self.assertEqual(sock.recv(1024),b'')
                except ConnectionResetError:pass
                elapsed=time.monotonic()-start
                self.assertLess(elapsed,1.5)
                sock.close()
                evidence('slow-'+name,{'elapsed':elapsed,'deadline':.35,'business':0})
            add.assert_not_called(); search.assert_not_called()
        self.wait_records(3)
        self.assertTrue(all(r['reason']=='deadline' for r in self.records))

    def test_half_close_is_legal_and_pipelining_is_not_consumed(self):
        with patch.object(aml,'add',return_value={'success':True}) as add:
            sock=self.connect()
            sock.sendall(b'POST /add HTTP/1.1\r\nHost: local\r\nContent-Length: 2\r\n\r\n{}'
                         b'GET /health HTTP/1.1\r\nHost: local\r\nConnection: close\r\n\r\n')
            sock.shutdown(socket.SHUT_WR)
            data=b''
            while True:
                part=sock.recv(4096)
                if not part:break
                data+=part
            self.assertEqual(data.count(b'HTTP/1.1 200'),2)
            self.assertEqual(add.call_count,1)
        self.wait_records(2)
        self.assertEqual([r['generation'] for r in self.records],[1,2])
        self.assertGreater(self.records[1]['deadline'],self.records[0]['deadline'])
        self.assertTrue(all(r['response_delivered'] for r in self.records))

    def test_old_watchdog_retired_before_new_keepalive(self):
        entered=threading.Event()
        def business(_):
            entered.set(); time.sleep(.18); return {'success':True}
        with patch.object(aml,'add',side_effect=business):
            sock=self.connect()
            sock.sendall(b'POST /add HTTP/1.1\r\nHost: local\r\nContent-Length: 2\r\n\r\n{}')
            response=http.client.HTTPResponse(sock);response.begin();response.read()
            self.assertEqual(response.status,200)
            time.sleep(.20) # Past generation 1's D, inside generation 2's D.
            sock.sendall(b'GET /health HTTP/1.1\r\nHost: local\r\nConnection: close\r\n\r\n')
            response=http.client.HTTPResponse(sock);response.begin();response.read()
            self.assertEqual(response.status,200)
        self.wait_records(2)
        evidence('keepalive',self.records)

    def test_slots_reject_before_creating_another_handler_thread(self):
        entered=[]
        original=self.server.process_request_thread
        def count(*args):entered.append(1);return original(*args)
        with patch.object(self.server,'process_request_thread',side_effect=count):
            first=self.connect();second=self.connect()
            end=time.monotonic()+1
            while len(entered)<2 and time.monotonic()<end:time.sleep(.005)
            third=self.connect()
            self.assertEqual(third.recv(1),b'')
            self.assertEqual(len(entered),2)
            first.close();second.close()
        evidence('connection-limit',{'admitted_threads':len(entered),'limit':2,'third_closed':True})

    def test_business_cancel_503_no_background_success(self):
        def business(_):
            rb.wait(2,'fake-work');raise AssertionError('must not finish')
        with patch.object(aml,'add',side_effect=business):
            sock=self.connect();sock.sendall(b'POST /add HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}')
            response=http.client.HTTPResponse(sock);response.begin();response.read()
            self.assertEqual(response.status,503)
        self.wait_records()
        self.assertFalse(self.records[0]['business_completed'])
        evidence('business-expiry',self.records)

    def test_rst_during_body_cancels_without_business(self):
        with patch.object(aml,'add') as add:
            sock=self.connect()
            sock.sendall(b'POST /add HTTP/1.1\r\nContent-Length: 1000\r\n\r\n{')
            time.sleep(.04)
            sock.setsockopt(socket.SOL_SOCKET,socket.SO_LINGER,struct.pack('ii',1,0))
            sock.close()
            self.wait_records()
            self.assertEqual(self.records[0]['reason'],'disconnect')
            add.assert_not_called()
        evidence('rst-body',self.records)

    def test_slow_response_preserves_completed_but_not_delivered(self):
        with patch.object(aml,'add',return_value={'success':True,'data':'x'*(12*1024*1024)}):
            sock=self.connect();sock.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,4096)
            sock.sendall(b'POST /add HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}')
            self.wait_records()
            record=self.records[0]
            self.assertTrue(record['business_completed'])
            self.assertFalse(record['response_delivered'])
            data=sock.recv(4096)
            self.assertIn(b'HTTP/1.1 200',data)
            self.assertNotIn(b'HTTP/1.1 503',data)
            evidence('slow-response',record)

    def test_default_disabled_and_deployment_optin_rejected(self):
        with self.assertRaises(ValueError):aml.make_server('127.0.0.1',0,request_timeout=.1)
        for value in (0,-1,float('nan'),float('inf'),True):
            with self.assertRaises(ValueError):
                aml.make_server('127.0.0.1',0,request_timeout=value,allow_test_deadline=True)


class LocalDeadlineTests(MemoryTestCase):
    def lock_child(self,path):
        path.parent.mkdir(parents=True,exist_ok=True)
        child=subprocess.Popen([sys.executable,'-c',
            'import os,fcntl,sys; f=os.open(sys.argv[1],os.O_CREAT|os.O_RDWR,0o600);'
            'fcntl.flock(f,fcntl.LOCK_EX);print("locked",flush=True);sys.stdin.read()',str(path)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        self.assertEqual(child.stdout.readline().strip(),'locked')
        def cleanup():
            child.communicate('',timeout=3)
        self.addCleanup(cleanup)
        return child

    def test_independent_process_locks_share_request_deadline(self):
        root=self.project.resolve()
        lockpaths=[root/'.aml-add.lock',self.base/'v2/locks/maintenance.lock',self.base/'v2/locks/synthetic.lock']
        actions=[lambda:add_admission_lock(root),lambda:events.maintenance_lock(),lambda:events.project_lock('synthetic')]
        # Use actual configured v2 root rather than assuming its spelling.
        from tiger_memory.model import v2_root
        lockpaths[1:]=[v2_root()/'locks/maintenance.lock',v2_root()/'locks/synthetic.lock']
        for path,action in zip(lockpaths,actions):
            child=self.lock_child(path);start=time.monotonic()
            with rb.request_scope(rb.RequestBudget(.18)),self.assertRaises(rb.RequestCancelled):
                with action():self.fail('expired waiter admitted')
            elapsed=time.monotonic()-start
            self.assertLess(elapsed,1.0)
            child.communicate('',timeout=3)
            evidence('lock-'+path.stem,{'elapsed':elapsed,'budget':.18,'admitted':False})

    def test_git_launch_failure_is_not_identity_fallback(self):
        with rb.request_scope(rb.RequestBudget(1)),patch('tiger_memory.request_waits.subprocess.Popen',side_effect=OSError('synthetic launch failure')):
            with self.assertRaises(OSError):service.find_project_root(self.project)

    def test_expired_add_creates_no_user_store(self):
        budget=rb.RequestBudget(1);budget.cancel('deadline')
        with rb.request_scope(budget),self.assertRaises(rb.RequestCancelled):
            aml.add({'request_id':'r','user_id':'u','session_id':'s','messages':[{'role':'user','content':'synthetic'}]})
        self.assertFalse(self.base.exists())

    def test_late_file_lock_acquisition_does_not_enter_business(self):
        from tiger_memory import request_waits
        budget=rb.RequestBudget(1)
        def acquire(*args):budget.cancel('deadline')
        with rb.request_scope(budget),patch.object(request_waits.fcntl,'flock',side_effect=acquire):
            with self.assertRaises(rb.RequestCancelled):request_waits.flock(123,fcntl.LOCK_EX)

    def test_catalog_statements_consume_one_deadline(self):
        conn=database.connect();conn.close()
        budget=rb.RequestBudget(.3)
        with rb.request_scope(budget):
            conn=database.connect_readonly()
            first=conn.execute('PRAGMA busy_timeout').fetchone()[0]
            time.sleep(.1)
            second=conn.execute('PRAGMA busy_timeout').fetchone()[0]
            self.assertLess(second,first-70)
            self.assertLessEqual(first,300)
            budget.cancel('deadline')
            with self.assertRaises(rb.RequestCancelled):conn.execute('SELECT 1')
            conn.close()
        evidence('sql-shared-deadline',{'first_busy_ms':first,'second_busy_ms':second})

    def test_safe_boundary_records_cancel_finishes_and_next_unit_stops(self):
        budget=rb.RequestBudget(.05);steps=[]
        with rb.request_scope(budget):
            with rb.safe_boundary('source'):
                budget.cancel('deadline');rb.check();steps.append('published')
            with self.assertRaises(rb.RequestCancelled):rb.check('next-source')
        self.assertEqual(steps,['published'])
        self.assertIsNone(rb.current_request_budget())

    def test_long_sql_progress_interrupt_and_close(self):
        conn=database.connect();conn.close()
        start=time.monotonic()
        with rb.request_scope(rb.RequestBudget(.08)):
            conn=database.connect_readonly()
            try:
                with self.assertRaises((rb.RequestCancelled,sqlite3.OperationalError)):
                    conn.execute('WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n WHERE x<100000000) SELECT sum(x) FROM n').fetchone()
                with self.assertRaises(rb.RequestCancelled):rb.check()
            finally:conn.close()
        self.assertLess(time.monotonic()-start,1)
        evidence('sql-progress',{'elapsed':time.monotonic()-start,'closed':True})

    def test_catalog_busy_does_not_get_new_thirty_seconds(self):
        conn=database.connect();conn.execute('BEGIN IMMEDIATE')
        start=time.monotonic()
        try:
            with rb.request_scope(rb.RequestBudget(.15)),self.assertRaises((rb.RequestCancelled,sqlite3.OperationalError)):
                other=database.connect()
                other.close()
        finally:conn.rollback();conn.close()
        self.assertLess(time.monotonic()-start,1)
        evidence('catalog-busy',{'elapsed':time.monotonic()-start,'budget':.15})

    def test_git_timeout_reaps_controlled_group_without_identity_fallback(self):
        script=Path(self.temp.name)/'git';pidfile=Path(self.temp.name)/'pids'
        script.write_text('#!'+sys.executable+'\nimport os,sys,time\np=os.fork()\nif p==0:\n while True: time.sleep(1)\nopen('+repr(str(pidfile))+',"w").write(str(os.getpid())+" "+str(p))\nwhile True: time.sleep(1)\n')
        script.chmod(0o700)
        start=time.monotonic()
        with patch.dict(os.environ,{'PATH':self.temp.name+os.pathsep+os.environ.get('PATH','')}):
            with rb.request_scope(rb.RequestBudget(.7)),self.assertRaises(rb.RequestCancelled):
                service.find_project_root(self.project)
        pids=[int(x) for x in pidfile.read_text().split()]
        # Direct child reaped; grandchild must not remain running (a zombie may
        # await its OS parent reaper). Do not claim arbitrary descendant fencing.
        with self.assertRaises(ChildProcessError):os.waitpid(pids[0],os.WNOHANG)
        # communicate returned EOF on stdout inherited by this controlled grandchild.
        # The fixture never closes stdout while alive: EOF proves its exit even
        # if the OS has not reaped the orphan zombie yet.
        status='controlled descendant stdout EOF; orphan reap not claimed'
        self.assertLess(time.monotonic()-start,1.5)
        evidence('git-group',{'direct_reaped':True,'grandchild_status':status,'elapsed':time.monotonic()-start})

    def test_source_cancel_after_publication_commits_prefix_and_retry_recovers(self):
        payload={'request_id':'deadline-synthetic','user_id':'deadline-u','session_id':'deadline-s',
                 'messages':[{'role':'user','content':'first source'},{'role':'user','content':'second source'}]}
        budget=rb.RequestBudget(5);original=service.atomic_write
        def publish(*args):
            original(*args)
            if Path(args[0]).suffix == '.md':budget.cancel('deadline')
        with patch.multiple(aml,AML_EMBEDDING='off',AML_FACT_PROJECTION='off',AML_ENTITY_EXTRACTION=False):
            with rb.request_scope(budget),patch.object(service,'atomic_write',side_effect=publish):
                with self.assertRaises(rb.RequestCancelled):aml.add(payload)
            files=list(self.base.rglob('*.md'));self.assertEqual(len(files),1)
            first={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
            with rb.request_scope(rb.RequestBudget(5)):self.assertTrue(aml.add(payload)['success'])
            self.assertEqual(len(list(self.base.rglob('*.md'))),2)
            self.assertTrue(all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in first.items()))
        evidence('source-cancel-prefix',{'prefix_count':1,'retry_count':2,'prefix_hash_unchanged':True})

    def test_search_cancel_not_legacy_empty_success(self):
        payload={'query':'test','user_id':'deadline-u','top_k':1}
        root=self.project.resolve()
        with rb.request_scope(rb.RequestBudget(1)),patch.object(aml,'_storage_identity_preflight',return_value=root),patch.object(aml.service,'context_data',side_effect=rb.RequestCancelled()):
            with self.assertRaises(rb.RequestCancelled):aml.search(payload)

class SearchDeadlineTests(MemoryTestCase):
    def setUp(self):
        super().setUp()
        self.flags=patch.multiple(aml,AML_EMBEDDING='off',AML_RERANKER='off',
            AML_QUERY_REWRITE='off',AML_FACT_PROJECTION='off',AML_ENTITY_EXTRACTION=False,
            AML_PROFILE='lexical',AML_STRICT_RETRIEVAL=True,AML_RECENCY_BOOST=False)
        self.flags.start();self.addCleanup(self.flags.stop)
        aml.add({'request_id':'q-r','user_id':'q-u','session_id':'q-s',
                 'messages':[{'role':'user','content':'violet one'},{'role':'user','content':'violet two'}]})
        self.query={'query':'violet','user_id':'q-u','top_k':1}

    def test_same_normal_result_full_eligible_input_and_snapshot(self):
        captured=[];inputs=[];connections=[]
        original=service.connect_readonly
        def opened():
            conn=original();connections.append(conn);return conn
        class Fake:
            def describe(self):return {'configured':True}
            def embed(self,texts):
                captured.append(connections[-1].in_transaction);inputs.append(len(texts))
                return [[1.,0.] for _ in texts]
        with patch.object(service,'connect_readonly',side_effect=opened),patch.object(aml,'_model_provider',side_effect=lambda role,*args:Fake() if role=='embedding' else None):
            baseline=aml.search(self.query)
            with rb.request_scope(rb.RequestBudget(5)):actual=aml.search(self.query)
        self.assertEqual(actual,baseline)
        self.assertEqual(inputs,[3,3]);self.assertEqual(captured,[True,True])
        self.assertEqual(len(actual['data']),1)
        evidence('search-baseline',{'equal':True,'eligible_inputs':inputs,'snapshot_in_embed':captured})

    def test_blocked_provider_keeps_snapshot_past_D_then_closes_and_fails(self):
        captured=[];connections=[];original=service.connect_readonly
        def opened():
            conn=original();connections.append(conn);return conn
        class Fake:
            def describe(self):return {'configured':True}
            def embed(self,texts):
                budget=rb.current_request_budget()
                time.sleep(budget.remaining()+.04)
                captured.append({'snapshot_open':connections[-1].in_transaction,
                                 'past_deadline':time.monotonic()>budget.deadline})
                return [[1.,0.] for _ in texts]
        with patch.object(service,'connect_readonly',side_effect=opened),patch.object(aml,'_model_provider',side_effect=lambda role,*args:Fake() if role=='embedding' else None):
            with rb.request_scope(rb.RequestBudget(.2)),self.assertRaises(rb.RequestCancelled):aml.search(self.query)
        self.assertEqual(captured,[{'snapshot_open':True,'past_deadline':True}])
        for conn in connections:
            with self.assertRaises(sqlite3.ProgrammingError):conn.execute('SELECT 1')
        evidence('search-late-snapshot',{'during':captured,'finally_closed':True,'hard_D_bound':False})

    def test_empty_paths_consume_D_and_never_construct_provider(self):
        with patch.object(aml,'_model_provider') as provider:
            with rb.request_scope(rb.RequestBudget(2)):
                self.assertEqual(aml.search(dict(self.query,top_k=0)),{'data':[]})
                self.assertEqual(aml.search(dict(self.query,user_id='unknown-synthetic')),{'data':[]})
            budget=rb.RequestBudget(1);budget.cancel('deadline')
            with rb.request_scope(budget),self.assertRaises(rb.RequestCancelled):aml.search(dict(self.query,top_k=0))
            provider.assert_not_called()
