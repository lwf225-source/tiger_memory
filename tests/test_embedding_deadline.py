"""84 P2/P3a cooperative deadlines: fake transport, real clocks/local locks."""
from contextlib import closing
import fcntl
import multiprocessing
import os
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest
from unittest.mock import patch

from tiger_memory.embedding_budget import PersistentEmbeddingBudget
from tiger_memory.providers import ProviderUnavailable
from tiger_memory.request_budget import RequestBudget, RequestCancelled, request_scope
from tests.test_embedding_budget import policy, provider, FakeTransport, evidence


def hold_resource(path, kind, ready, release):
    with patch('socket.socket', side_effect=AssertionError('NETWORK_DENIED')):
        if kind == 'checkpoint':
            fd=os.open(path,os.O_RDWR)
            try:
                fcntl.flock(fd,fcntl.LOCK_EX);ready.set();release.wait(10)
            finally:os.close(fd)
        else:
            with closing(sqlite3.connect(path)) as conn:
                conn.execute('BEGIN IMMEDIATE');ready.set();release.wait(10);conn.rollback()


class EmbeddingDeadlineTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name).resolve()
        self.policy=policy();self.budget=PersistentEmbeddingBudget.initialize(self.root/'budget',self.policy)
        guard=patch('socket.socket',side_effect=AssertionError('NETWORK_DENIED'));guard.start();self.addCleanup(guard.stop)

    def test_cancelled_plan_and_warm_cache_stop_but_no_budget_cli_unchanged(self):
        p=provider(self.root,self.budget);expected=p.embed(['warm'])
        before=self.budget.snapshot();d=RequestBudget(5);d.cancel('disconnect')
        with request_scope(d):
            for call in (lambda:p.plan(['warm']),lambda:p.embed(['warm']),lambda:p.embed([])):
                with self.assertRaises(RequestCancelled):call()
        self.assertEqual(self.budget.snapshot(),before)
        self.assertEqual(p.embed(['warm']),expected)
        self.assertEqual(len(p._opener.requests),1)
        evidence('deadline-cache-plan',{'cancelled_plan_and_cache':'rejected','new_sends':0,'cli_warm_unchanged':True})

    def test_request_budget_survives_new_provider_and_original_scope_exit(self):
        d=RequestBudget(.02)
        with request_scope(d):p=provider(self.root,self.budget)
        time.sleep(.03)
        with self.assertRaises(RequestCancelled):p.embed(['cold'])
        with request_scope(d):
            q=provider(self.root,self.budget)
            with self.assertRaises(RequestCancelled):q.embed(['cold'])
        self.assertEqual(self.budget.snapshot()['requests'],0)
        self.assertEqual(p._opener.requests+q._opener.requests,[])

    def test_timeout_parameter_consumes_same_request_remaining(self):
        d=RequestBudget(1.0);observed=[];fake=FakeTransport();opening=fake.open
        def open_(*args,**kw):observed.append(kw['timeout']);time.sleep(.025);return opening(*args,**kw)
        fake.open=open_
        with request_scope(d):
            p=provider(self.root,self.budget,transport=fake);p.embed([str(i) for i in range(11)])
        self.assertEqual(len(observed),2);self.assertGreater(observed[0],observed[1]);self.assertLess(observed[0],1.0)
        evidence('deadline-batch-timeouts',{'timeouts':observed,'request_seconds':1.0,'not_total_network_timeout':True})

    def test_cancel_during_reserved_commit_holds_claim_and_never_sends(self):
        d=RequestBudget(10);checksum=self.budget._work_checksum
        def cancel_then_hash(work):d.cancel('deadline');return checksum(work)
        with request_scope(d), patch.object(self.budget,'_work_checksum',side_effect=cancel_then_hash):
            p=provider(self.root,self.budget)
            with self.assertRaises(RequestCancelled):p.embed(['cold'])
        state=self.budget.snapshot();self.assertEqual((state['requests'],state['unknown'],state['claims']),(1,1,1))
        self.assertEqual(p._opener.requests,[])
        q=provider(self.root,self.budget)
        with self.assertRaisesRegex(ProviderUnavailable,'work-blocked'):q.embed(['cold'])
        self.assertEqual(q._opener.requests,[])
        evidence('deadline-reserve-commit',{'send0':True,'retry_send0':True,'ledger':state})

    def test_late_valid_response_publishes_settles_but_no_next_batch_or_pooling(self):
        d=RequestBudget(.035);fake=FakeTransport();opening=fake.open
        def late(*a,**k):time.sleep(.06);return opening(*a,**k)
        fake.open=late
        with request_scope(d):
            p=provider(self.root,self.budget,transport=fake)
            with self.assertRaises(RequestCancelled):p.embed([str(i) for i in range(11)])
        self.assertEqual(len(fake.requests),1)
        self.assertEqual(len(list(p._checkpoint.root.glob('raw-*'))),10)
        self.assertEqual(list(p._checkpoint.root.glob('pooled-*')),[])
        state=self.budget.snapshot();self.assertEqual((state['settled'],state['claims']),(1,10))
        q=provider(self.root,self.budget);q.embed([str(i) for i in range(10)])
        self.assertEqual(q._opener.requests,[])
        evidence('deadline-late-response',{'received_after_D':True,'sent_batches':1,'raw':10,'pooled_at_cancel':0,'ledger':state})

    def test_cancel_during_raw_publication_finishes_admitted_batch_then_stops(self):
        d=RequestBudget(10)
        with request_scope(d):
            p=provider(self.root,self.budget);put=p._checkpoint.put;count=[0]
            def publish(kind,key,values):
                put(kind,key,values)
                if kind=='raw':count[0]+=1;d.cancel('disconnect')
            with patch.object(p._checkpoint,'put',side_effect=publish):
                with self.assertRaises(RequestCancelled):p.embed(['one','two'])
        self.assertEqual(count,[2]);self.assertEqual(self.budget.snapshot()['settled'],1)
        q=provider(self.root,self.budget);q.embed(['one','two']);self.assertEqual(q._opener.requests,[])

    def test_late_invalid_response_keeps_unknown_and_blocks_retry(self):
        d=RequestBudget(10)
        def malformed(obj):d.cancel('deadline');obj['data']=[]
        with request_scope(d):
            p=provider(self.root,self.budget,transport=FakeTransport(mutate=malformed))
            with self.assertRaisesRegex(ProviderUnavailable,'count-mismatch'):p.embed(['one'])
        q=provider(self.root,self.budget)
        with self.assertRaisesRegex(ProviderUnavailable,'work-blocked'):q.embed(['one'])
        self.assertEqual(q._opener.requests,[]);self.assertEqual(self.budget.snapshot()['unknown'],1)

    def test_pooling_and_final_output_recheck_cancellation(self):
        # Cancellation while reconstructing cached raw must not create pooled
        # output. All data had previously been settled; no send is required.
        p=provider(self.root,self.budget);p.embed(['one'])
        for f in p._checkpoint.root.glob('pooled-*'):f.unlink()
        d=RequestBudget(10);q=provider(self.root,self.budget);get=q._checkpoint.get
        def read(kind,key):
            value=get(kind,key)
            if kind=='raw':d.cancel('shutdown')
            return value
        with request_scope(d),patch.object(q._checkpoint,'get',side_effect=read):
            with self.assertRaises(RequestCancelled):q.embed(['one'])
        self.assertEqual(list(q._checkpoint.root.glob('pooled-*')),[]);self.assertEqual(q._opener.requests,[])

    def test_real_process_checkpoint_and_budget_wait_use_remaining(self):
        p=provider(self.root,self.budget)
        with p._checkpoint.locked():pass
        ctx=multiprocessing.get_context('spawn')
        for kind,path in [('checkpoint',p._checkpoint.root/'.lock'),('budget',self.budget.root/'ledger.sqlite3')]:
            ready,release=ctx.Event(),ctx.Event();proc=ctx.Process(target=hold_resource,args=(str(path),kind,ready,release));proc.start()
            try:
                self.assertTrue(ready.wait(10));d=RequestBudget(.10);started=time.monotonic()
                with request_scope(d):
                    with self.assertRaises(RequestCancelled):
                        if kind=='checkpoint':p.embed(['cold'])
                        else:self.budget.snapshot()
                elapsed=time.monotonic()-started
                self.assertLess(elapsed,1.0)
                evidence('deadline-wait-'+kind,{'elapsed_seconds':elapsed,'D_seconds':.10,'new_sends':0,'schedule_tolerance_seconds':.90})
            finally:release.set();proc.join(10);self.assertEqual(proc.exitcode,0)
        self.assertEqual(self.budget.snapshot()['requests'],0)

    def test_checkpoint_late_success_does_not_enter_business(self):
        p=provider(self.root,self.budget);d=RequestBudget(10)
        original=fcntl.flock
        def acquired(fd,flags):original(fd,flags);d.cancel('deadline')
        with request_scope(d),patch('tiger_memory.embedding_checkpoint.fcntl.flock',side_effect=acquired):
            with self.assertRaises(RequestCancelled):p.embed(['cold'])
        self.assertEqual(p._opener.requests,[]);self.assertEqual(self.budget.snapshot()['requests'],0)
        # Released fd is reusable by a later legitimate request.
        provider(self.root,self.budget).embed(['cold'])

    def test_sql_progress_and_multiple_statements_share_one_deadline(self):
        d=RequestBudget(.04);started=time.monotonic()
        with request_scope(d):
            with self.assertRaises(RequestCancelled):
                with self.budget._transaction() as (conn,*_):
                    conn.execute('WITH RECURSIVE t(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM t WHERE x<100000000) SELECT SUM(x) FROM t').fetchone()
        elapsed=time.monotonic()-started;self.assertLess(elapsed,1.0)
        d=RequestBudget(.07);busy=[]
        with request_scope(d):
            with self.assertRaises(RequestCancelled):
                with self.budget._transaction() as (conn,*_):
                    for _ in range(10):
                        busy.append(conn.execute('PRAGMA busy_timeout').fetchone()[0]);time.sleep(.015)
        self.assertGreater(len(busy),1);self.assertGreater(busy[0],busy[-1])
        self.assertEqual(self.budget.snapshot()['requests'],0)
        evidence('deadline-sql',{'progress_seconds':elapsed,'busy_timeouts_ms':busy,'new_reservations':0})

    def test_cancel_wire_and_observation_are_distinct_admission_events(self):
        d=RequestBudget(5);trace={'D':d.deadline,'started':d.started};fake=FakeTransport()
        reserve=self.budget.reserve;opening=fake.open
        def reserved(*args):
            result=reserve(*args);trace['reserve_committed']=time.monotonic();return result
        def wire(*args,**kw):
            # A scheduler/transport window after the provider's last check.
            # This is one already admitted batch, not a new permission.
            d.cancel('disconnect');trace['cancel_event_set']=time.monotonic()
            trace['fake_receive']=time.monotonic();return opening(*args,**kw)
        fake.open=wire
        with request_scope(d),patch.object(self.budget,'reserve',side_effect=reserved):
            p=provider(self.root,self.budget,transport=fake)
            with self.assertRaises(RequestCancelled):p.embed([str(i) for i in range(11)])
            trace['provider_cancel_observed']=time.monotonic()
        self.assertLessEqual(trace['reserve_committed'],trace['cancel_event_set'])
        self.assertLessEqual(trace['cancel_event_set'],trace['fake_receive'])
        self.assertLessEqual(trace['fake_receive'],trace['provider_cancel_observed'])
        self.assertEqual(len(fake.requests),1);self.assertEqual(self.budget.snapshot()['requests'],1)
        evidence('deadline-admission-wire',{'timeline':trace,'admitted_batches':1,'fake_received_batches':1,
            'next_batch_permits':0,'ledger':self.budget.snapshot(),
            'network_worker':'none; P3a synchronous fake, no P3b reap claim'})
