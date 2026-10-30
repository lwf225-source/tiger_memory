"""84 durable per-key send claims: all inputs/transport/ledgers are synthetic."""
from contextlib import closing
from dataclasses import asdict
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import signal
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import patch

from tiger_memory.embedding_budget import PersistentEmbeddingBudget, _canonical, _hash
from tiger_memory.embedding_checkpoint import digest
from tiger_memory.providers import ProviderUnavailable, _digest
from tests.test_embedding_budget import policy, provider, FakeTransport, evidence, synthetic_claim


def make_legacy(root, configured, states=()):
    """Exact pre84 schema fixture, not a product downgrade or reset path."""
    root.mkdir(mode=0o700)
    ledger_id = 'a' * 32
    with closing(sqlite3.connect(root / 'ledger.sqlite3')) as conn:
        conn.executescript('''CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
          CREATE TABLE attempts(seq INTEGER PRIMARY KEY AUTOINCREMENT,attempt TEXT UNIQUE NOT NULL,
          rows INTEGER NOT NULL,reserved_tokens INTEGER NOT NULL,state TEXT NOT NULL,
          usage_tokens INTEGER,request_hash TEXT NOT NULL,reason TEXT NOT NULL,checksum TEXT NOT NULL);''')
        conn.executemany('INSERT INTO metadata VALUES (?,?)', [('schema','1'),('ledger_id',ledger_id),
            ('policy',_canonical(asdict(configured)).decode()),('policy_hash',configured.fingerprint),('stopped','0')])
        for i, state in enumerate(states, 1):
            row = dict(seq=i,attempt=('%032x' % i),rows=1,reserved_tokens=configured.token_limit_per_row,
                       state=state,usage_tokens=1 if state=='SETTLED' else None,
                       request_hash=digest(i),reason='validated-response' if state=='SETTLED' else 'reserved')
            conn.execute('INSERT INTO attempts VALUES (?,?,?,?,?,?,?,?,?)', (*row.values(),_hash(row)))
        conn.commit()
    (root/'ledger.sqlite3').chmod(0o600)
    (root/'identity.json').write_bytes(_canonical(dict(schema=1,ledger_id=ledger_id,policy_hash=configured.fingerprint)))
    (root/'identity.json').chmod(0o600)
    return PersistentEmbeddingBudget(root, configured, ledger_id=ledger_id)


def claim_worker(root, configured, ledger_id, ready, pipe, mode, cache='cache'):
    with patch('socket.socket', side_effect=AssertionError('NETWORK_DENIED')):
        budget = PersistentEmbeddingBudget(Path(root)/'budget',configured,ledger_id=ledger_id)
        fake = FakeTransport(fail_batch=1 if mode=='compete' else None, wire_file=Path(root)/'wire')
        p = provider(Path(root)/cache,budget,user='same',transport=fake)
        if mode == 'partial':
            original = p._checkpoint.put
            def put(kind, key, values):
                original(kind, key, values)
                if kind == 'raw':
                    pipe.send('one-raw-published')
                    signal.pause()
            p._checkpoint.put = put
        elif mode == 'settled':
            original = budget.settle
            def settle(*args):
                original(*args)
                pipe.send('settled-before-pooling')
                signal.pause()
            budget.settle = settle
        if ready: ready.wait(10)
        try:
            p.embed(['one','two'] if mode=='partial' else ['one'])
            pipe.send({'state':'ok','sends':len(fake.requests)})
        except ProviderUnavailable as e:
            pipe.send({'state':str(e),'sends':len(fake.requests)})
        finally:
            pipe.close()


class DurableClaimTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve();self.policy=policy()
        self.budget=PersistentEmbeddingBudget.initialize(self.root/'budget',self.policy)
        guard=patch('socket.socket',side_effect=AssertionError('NETWORK_DENIED'));guard.start();self.addCleanup(guard.stop)

    def reopen(self):
        return PersistentEmbeddingBudget(self.budget.root,self.policy,ledger_id=self.budget.ledger_id)

    def test_unknown_changed_order_batch_and_cache_path_do_not_send(self):
        p=provider(self.root,self.budget,transport=FakeTransport(fail_batch=1))
        with self.assertRaisesRegex(ProviderUnavailable,'network-failed'):p.embed(['one','two'])
        before=self.budget.snapshot()
        outcomes=[]
        for texts in (['one'],['two','one'],['new']*10+['two']):
            q=provider(self.root,self.reopen())
            with self.assertRaisesRegex(ProviderUnavailable,'work-blocked'):q.embed(texts)
            self.assertEqual(q._opener.requests,[]);outcomes.append({'input_count':len(texts),'sends':0})
        q=provider(self.root/'new-cache',self.reopen())
        with self.assertRaisesRegex(ProviderUnavailable,'work-blocked'):q.embed(['one'])
        self.assertEqual(q._opener.requests,[])
        self.assertEqual(before,self.budget.snapshot())
        evidence('claims-regroup',{'replays':outcomes,'ledger':before,'changed_cache_sends':0})

    def test_different_full_input_and_namespace_remain_new_work(self):
        p=provider(self.root,self.budget,transport=FakeTransport(fail_batch=1));text='x'*8192
        with self.assertRaises(ProviderUnavailable):p.embed([text+'a'])
        q=provider(self.root,self.reopen());q.embed([text+'b'])
        self.assertEqual(len(q._opener.requests),1)
        # Identical first chunk string has a different full-input semantic key.
        self.assertEqual(q._opener.requests[0]['input'][0],text)
        other=provider(self.root,self.reopen(),user='other');other.embed([text+'a'])
        self.assertEqual(len(other._opener.requests),1)
        self.assertEqual(self.budget.snapshot()['claims'],6)

    def test_settled_missing_and_corrupt_checkpoints_never_resend(self):
        p=provider(self.root,self.budget);p.embed(['one'])
        files={f.name:f.read_bytes() for f in p._checkpoint.root.glob('*.json')}
        for f in p._checkpoint.root.glob('*.json'):f.unlink()
        before=self.budget.snapshot()
        q=provider(self.root,self.reopen())
        with self.assertRaisesRegex(ProviderUnavailable,'work-blocked'):q.embed(['one'])
        self.assertEqual(q._opener.requests,[])
        # Restoring already validated checkpoint bytes enables local recovery,
        # not a new sending authorization or deletion of a financial claim.
        for name,data in files.items():(p._checkpoint.root/name).write_bytes(data)
        q.embed(['one']);self.assertEqual(q._opener.requests,[])
        self.assertEqual(before,self.budget.snapshot())
        next(p._checkpoint.root.glob('pooled-*')).write_text('{corrupt')
        with self.assertRaisesRegex(ProviderUnavailable,'checkpoint-corrupt'):q.embed(['one'])
        self.assertEqual(q._opener.requests,[])
        evidence('claims-settled-lost-data',{'ledger':before,'missing_sends':0,'restored_sends':0,'corrupt_sends':0})

    def test_claim_and_attempt_transaction_failure_rolls_back_all(self):
        db=self.budget.root/'ledger.sqlite3';before=self.budget.snapshot()
        with closing(sqlite3.connect(db)) as conn:
            conn.execute("CREATE TRIGGER fail_claim BEFORE INSERT ON work_claims BEGIN SELECT RAISE(ABORT,'synthetic'); END")
            conn.commit()
        with self.assertRaises(ProviderUnavailable):self.budget.reserve(2,digest('batch'),synthetic_claim(2))
        self.assertEqual(before,self.budget.snapshot())
        with closing(sqlite3.connect(db)) as conn:
            conn.execute('DROP TRIGGER fail_claim');conn.commit()
        self.budget.reserve(2,digest('batch'),synthetic_claim(2))
        self.assertEqual(self.budget.snapshot()['claims'],2)

    def test_threads_atomic_intersection_with_different_batch_hash(self):
        barrier=threading.Barrier(2);results=[]
        def run(i):
            b=self.reopen();barrier.wait(5)
            try:b.reserve(2,digest(['batch',i]),synthetic_claim(2));results.append('reserved')
            except ProviderUnavailable as e:results.append(str(e))
        threads=[threading.Thread(target=run,args=(i,)) for i in range(2)]
        for t in threads:t.start()
        for t in threads:t.join(10);self.assertFalse(t.is_alive())
        self.assertEqual(sorted(results),['embedding-budget-work-blocked','reserved'])
        self.assertEqual(self.budget.snapshot()['requests'],1)
        evidence('claims-threads',{'results':results,'ledger':self.budget.snapshot()})

    def test_independent_processes_different_cache_locks_share_atomic_claim(self):
        ctx=multiprocessing.get_context('spawn');ready=ctx.Event();workers=[]
        for cache in ('left','right'):
            r,w=ctx.Pipe(duplex=False)
            proc=ctx.Process(target=claim_worker,args=(self.root,self.policy,self.budget.ledger_id,ready,w,'compete',cache))
            proc.start();w.close();workers.append((proc,r))
        ready.set();results=[]
        try:
            for proc,r in workers:
                self.assertTrue(r.poll(15));results.append(r.recv());proc.join(15);self.assertEqual(proc.exitcode,0)
        finally:
            for proc,r in workers:
                if proc.is_alive():proc.kill();proc.join(5)
                r.close()
        self.assertEqual(sum(r['sends'] for r in results),1)
        self.assertEqual(sorted(r['state'] for r in results),['embedding-budget-work-blocked','embedding-network-failed'])
        self.assertEqual(self.budget.snapshot()['requests'],1)
        evidence('claims-processes',{'results':results,'ledger':self.budget.snapshot()})

    def test_sigkill_partial_raw_and_settled_before_pooling(self):
        ctx=multiprocessing.get_context('spawn')
        for phase in ('partial','settled'):
            root=self.root/phase;root.mkdir()
            b=PersistentEmbeddingBudget.initialize(root/'budget',self.policy)
            r,w=ctx.Pipe(duplex=False)
            proc=ctx.Process(target=claim_worker,args=(root,self.policy,b.ledger_id,None,w,phase))
            proc.start();w.close()
            try:
                self.assertTrue(r.poll(15));observed=r.recv()
            finally:
                proc.kill();proc.join(10);r.close()
            self.assertEqual(proc.exitcode,-signal.SIGKILL)
            reopened=PersistentEmbeddingBudget(b.root,self.policy,ledger_id=b.ledger_id)
            q=provider(root/'cache',reopened,user='same');before=reopened.snapshot()
            if phase=='partial':
                self.assertEqual(observed,'one-raw-published')
                with self.assertRaisesRegex(ProviderUnavailable,'work-blocked'):q.embed(['one','two'])
                q.embed(['one']);self.assertEqual(before['unknown'],1)
            else:
                self.assertEqual(observed,'settled-before-pooling');q.embed(['one']);self.assertEqual(before['settled'],1)
            self.assertEqual(q._opener.requests,[]);self.assertEqual(before,reopened.snapshot())
            evidence('claims-kill-'+phase,{'exit':proc.exitcode,'phase':observed,'replay_sends':0,'ledger':before})

    def test_legacy_upgrade_explicit_preserves_unknown_and_still_blocks(self):
        b=make_legacy(self.root/'legacy',self.policy,['UNKNOWN','SETTLED'])
        p=provider(self.root/'legacy-cache',b)
        # Seed a synthetic verified raw checkpoint, simulating existing data.
        text='legacy cache';key=digest({'input':_digest(text),'position':0,'chunk':_digest(text)})
        with p._checkpoint.locked():p._checkpoint.put('raw',key,[1.0]*64)
        before=b.snapshot();marker=(b.root/'identity.json').read_bytes();receipt=b.receipt()
        with self.assertRaisesRegex(ProviderUnavailable,'claims-upgrade-required'):p.embed(['cold'])
        p.embed([text]);self.assertEqual(p._opener.requests,[])
        with self.assertRaisesRegex(ProviderUnavailable,'migration-binding'):b.upgrade_claims('0'*64)
        self.assertEqual(before,b.snapshot())
        after=b.upgrade_claims(before['migration_state_sha256'])
        self.assertEqual(after['occupied_nano_cny'],before['occupied_nano_cny'])
        self.assertEqual(after['legacy_attempts'],2);self.assertEqual(after['unknown'],1)
        self.assertEqual(marker,(b.root/'identity.json').read_bytes());self.assertEqual(receipt,b.receipt())
        for user in ('new-a','new-b'):
            q=provider(self.root,b,user=user)
            with self.assertRaisesRegex(ProviderUnavailable,'legacy-work-blocked'):q.embed(['cold'])
            self.assertEqual(q._opener.requests,[])
        p.embed([text]);self.assertEqual(p._opener.requests,[])
        with self.assertRaisesRegex(ProviderUnavailable,'already-upgraded'):b.upgrade_claims(after['migration_state_sha256'])
        evidence('claims-migration',{'before':before,'after':after,'cold_sends':0,'warm_sends':0,'identity_unchanged':True})

    def test_empty_legacy_upgrade_and_failed_upgrade_are_transactional(self):
        b=make_legacy(self.root/'legacy',self.policy);before=b.snapshot()
        import tiger_memory.embedding_budget as mod
        original=mod._create_claim_tables
        def failing(conn):original(conn);raise OSError('synthetic migration failure')
        with patch.object(mod,'_create_claim_tables',side_effect=failing):
            with self.assertRaises(ProviderUnavailable):b.upgrade_claims(before['migration_state_sha256'])
        self.assertEqual(before,b.snapshot())
        b.upgrade_claims(before['migration_state_sha256']);p=provider(self.root,b);p.embed(['cold'])
        self.assertEqual(len(p._opener.requests),1)

    def test_claim_corruption_and_deleted_last_attempt_fail_closed(self):
        mutations=['DELETE FROM work_claims','DELETE FROM work_sets',
                   "UPDATE work_sets SET checksum='bad'", "UPDATE metadata SET value='2' WHERE key='claims_version'",
                   'DELETE FROM work_claims; DELETE FROM work_sets; DELETE FROM attempts']
        for i,sql in enumerate(mutations):
            root=self.root/str(i);root.mkdir();b=PersistentEmbeddingBudget.initialize(root/'budget',self.policy)
            p=provider(root,b);p.embed(['one'])
            with closing(sqlite3.connect(b.root/'ledger.sqlite3')) as conn:conn.executescript(sql);conn.commit()
            with self.assertRaises(ProviderUnavailable):PersistentEmbeddingBudget(b.root,self.policy,ledger_id=b.ledger_id)
            # Already-open provider also validates ledger before any new miss.
            with self.assertRaises(ProviderUnavailable):p.embed(['new'])
            self.assertEqual(len(p._opener.requests),1)

    def test_claim_checksums_bind_identity_and_no_text_or_request_identifier(self):
        p=provider(self.root,self.budget);p.embed(['SYNTHETIC_PRIVATE_CANARY'])
        with closing(sqlite3.connect(self.budget.root/'ledger.sqlite3')) as conn:
            dump='\n'.join(conn.iterdump())
        self.assertNotIn('SYNTHETIC_PRIVATE_CANARY',dump);self.assertNotIn('SYNTHETIC_NOT_A_REAL_KEY',dump)
        self.assertEqual(self.budget.snapshot()['claims'],1)
        with self.assertRaisesRegex(ProviderUnavailable,'claim-required'):self.budget.reserve(1,digest('missing context'))
