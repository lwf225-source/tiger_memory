"""Offline real flock/cache tests: no network or production storage."""
import concurrent.futures as cf
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from tiger_memory.remote_embeddings import DashScopeEmbeddingProvider as Provider
from tiger_memory.providers import ProviderUnavailable, _digest


def vector(text):
    n=hashlib.sha256(text.encode()).digest()
    return [float(n[i % len(n)]+1) for i in range(64)]


def provider(root):
    return Provider('https://example.invalid/v1/embeddings','OFFLINE-ONLY',root.resolve(),dimensions=64,checkpoint_namespace='same-user')


def fake(calls, entered=None, release=None):
    def request(texts,publish=None,raw_keys=None):
        calls.append(list(texts))
        if entered: entered.set()
        if release and not release.wait(3): raise AssertionError('blocked provider fixture')
        values=[vector(t) for t in texts]
        if publish:publish(values)
        return values
    return request


class EmbeddingParallelTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)/'cache'
        from tiger_memory import remote_embeddings as remote
        guard=patch.object(remote,'_CHECKPOINT_PARALLEL_ENABLED',True);guard.start();self.addCleanup(guard.stop)
        requested=patch.object(remote,'_CHECKPOINT_PARALLEL_REQUESTED',False);requested.start();self.addCleanup(requested.stop)

    def test_different_cold_inputs_overlap(self):
        texts=['a','b']
        self.assertNotEqual(_digest(texts[0])[:2],_digest(texts[1])[:2])
        gate=threading.Barrier(2);calls=[]
        def request(texts,publish=None,raw_keys=None):
            calls.append(list(texts));gate.wait(3)
            values=[vector(t) for t in texts]
            if publish:publish(values)
            return values
        ps=[provider(self.root) for _ in texts]
        for p in ps:p._request=request
        with cf.ThreadPoolExecutor(2) as pool:
            out=list(pool.map(lambda i:ps[i].embed([texts[i]]),range(2)))
        self.assertEqual(len(calls),2);self.assertEqual(len(out),2)

    def test_identical_work_single_request_and_same_outputs(self):
        started=threading.Event();release=threading.Event();calls=[]
        ps=[provider(self.root) for _ in range(8)]
        for p in ps:p._request=fake(calls,started,release)
        with cf.ThreadPoolExecutor(8) as pool:
            futures=[pool.submit(p.embed,['same','same']) for p in ps]
            self.assertTrue(started.wait(2));release.set();out=[f.result(5) for f in futures]
        self.assertEqual(len(calls),1)
        self.assertTrue(all(v==out[0] for v in out))

    def test_warm_input_bypasses_blocked_unrelated_cold_work(self):
        p=provider(self.root);p._request=fake([]);expected=p.embed(['warm'])
        cold=provider(self.root);ready=threading.Event();release=threading.Event();cold._request=fake([],ready,release)
        warm=provider(self.root);warm._request=lambda *a: self.fail('warm invoked provider')
        with cf.ThreadPoolExecutor(2) as pool:
            f=pool.submit(cold.embed,['cold']);self.assertTrue(ready.wait(2))
            try:self.assertEqual(pool.submit(warm.embed,['warm']).result(1),expected)
            finally:release.set()
            f.result(3)

    def test_legacy_exclusive_lock_blocks_new_requests(self):
        old=provider(self.root);p=provider(self.root);called=threading.Event();p._request=fake([],called)
        with cf.ThreadPoolExecutor(1) as pool:
            with old._checkpoint.locked():
                f=pool.submit(p.embed,['new']);self.assertFalse(called.wait(.1));self.assertFalse(f.done())
            f.result(3);self.assertTrue(called.is_set())

    def test_new_shared_lock_blocks_legacy_writer(self):
        p=provider(self.root);other=provider(self.root);entered=threading.Event()
        def old():
            with other._checkpoint.locked():entered.set()
        with cf.ThreadPoolExecutor(1) as pool:
            with p._checkpoint.locked(shared=True):
                f=pool.submit(old);self.assertFalse(entered.wait(.1))
            f.result(3);self.assertTrue(entered.is_set())

    def test_opposite_order_overlapping_inputs_no_deadlock(self):
        ps=[provider(self.root) for _ in range(2)];calls=[]
        for p in ps:p._request=fake(calls)
        with cf.ThreadPoolExecutor(2) as pool:
            a=pool.submit(ps[0].embed,['one','two']);b=pool.submit(ps[1].embed,['two','one'])
            self.assertEqual(a.result(5),list(reversed(b.result(5))))
        self.assertEqual(sum(map(len,calls)),2)

    def test_corrupt_cache_repaired_once_concurrently(self):
        p=provider(self.root);p._request=fake([]);expected=p.embed(['repair'])
        p._checkpoint._path('pooled',_digest('repair')).write_bytes(b'{bad json')
        # pooled repair reuses existing raw cache, with no external request.
        ps=[provider(self.root) for _ in range(4)]
        for p in ps:p._request=lambda *a:self.fail('raw cache should recover pooled')
        with cf.ThreadPoolExecutor(4) as pool:out=list(pool.map(lambda p:p.embed(['repair']),ps))
        self.assertEqual(out,[expected]*4)
        self.assertEqual(len(list(p._checkpoint.root.glob('.quarantine-pooled-*'))),1)

    def test_failure_releases_locks_and_preserves_published_chunks(self):
        p=provider(self.root);p._BATCH_SIZE=1;calls=[]
        text='a'*8192+'b'*8192+'c'*100
        def fail_second(texts,publish=None,raw_keys=None):
            calls.extend(texts)
            if len(calls)==2:raise ProviderUnavailable('offline-failure')
            values=[vector(t) for t in texts];publish(values);return values
        p._request=fail_second
        with self.assertRaises(ProviderUnavailable):p.embed([text])
        p._request=fake([]);actual=p.embed([text])
        clean=provider(Path(self.tmp.name)/'other');clean._BATCH_SIZE=1;clean._request=fake([])
        self.assertEqual(actual,clean.embed([text]))

    def test_shared_namespace_without_input_lock_cannot_quarantine(self):
        p=provider(self.root)
        with p._checkpoint.locked(shared=True):
            with self.assertRaisesRegex(ProviderUnavailable,'recovery-lock-required'):
                p._checkpoint.quarantine('pooled',_digest('x'))

    def test_stripe_symlink_is_rejected(self):
        p=provider(self.root)
        with p._checkpoint.locked(shared=True):
            target=Path(self.tmp.name)/'victim';target.write_text('untouched')
            (p._checkpoint.root/('.input-v1-'+_digest('x')[:2]+'.lock')).symlink_to(target)
        p._request=lambda *a:self.fail('unsafe lock must precede request')
        with self.assertRaises(ProviderUnavailable):p.embed(['x'])
        self.assertEqual(target.read_text(),'untouched')

    def test_vectors_cache_format_and_order_match_legacy_algorithm(self):
        # Legacy _embed_locked stays intact; compare the old exclusive execution.
        new=provider(self.root);old=provider(Path(self.tmp.name)/'old')
        new._request=fake([]);old._request=fake([])
        texts=['short','长内容'*3000,'short']
        with old._checkpoint.locked():expected=old._embed_locked(texts)
        self.assertEqual(new.embed(texts),expected)
        def data(p):return {x.name:x.read_bytes() for x in p._checkpoint.root.glob('*.json')}
        self.assertEqual(data(new),data(old))

    def test_fixed_operation_timings_do_not_mix_add_and_search(self):
        from tiger_memory import stage_timings as st
        before=st.timing_status()['stages']
        @st.timed('add_total')
        def add():
            with st.phase('embedding_input_wait'):pass
        @st.timed('search_total')
        def search():
            with st.phase('embedding_input_wait'):pass
        with cf.ThreadPoolExecutor(2) as pool:list(pool.map(lambda fn:fn(),[add,search]))
        after=st.timing_status()['stages']
        for op in ('add','search'):
            name=op+'_embedding_input_wait'
            self.assertEqual(after[name]['count']-before[name]['count'],1)
        self.assertIsNone(st._OPERATION.get())

    def test_same_provider_instance_supports_parallel_inputs(self):
        p=provider(self.root);gate=threading.Barrier(2)
        def request(texts,publish=None,raw_keys=None):
            gate.wait(3);values=[vector(t) for t in texts]
            if publish:publish(values)
            return values
        p._request=request
        with cf.ThreadPoolExecutor(2) as pool:
            a=pool.submit(p.embed,['a']);b=pool.submit(p.embed,['b'])
            self.assertEqual(len(a.result(5)),1);self.assertEqual(len(b.result(5)),1)

    def test_http400_is_not_retried_and_metadata_survives(self):
        import io
        from urllib.error import HTTPError
        from tiger_memory.safe_diagnostics import exception_record
        p=provider(self.root);calls=[]
        class Fail:
            def open(self,*a,**k):
                calls.append(1)
                raise HTTPError('https://example.invalid',400,'SECRET',{},io.BytesIO(
                    b'{"error":{"code":"context_length_exceeded","message":"SECRET"}}'))
        p._opener=Fail()
        with self.assertRaises(ProviderUnavailable) as caught:p.embed(['private input'])
        self.assertEqual(len(calls),1)
        record=exception_record(caught.exception,'add')
        self.assertEqual(record['exceptions'][0]['transport']['error_code'],'context_length_exceeded')
        self.assertNotIn('SECRET',json.dumps(record));self.assertNotIn('private input',json.dumps(record))

    def test_drain_mode_never_takes_shared_lock_then_activation_parallelizes(self):
        from tiger_memory import remote_embeddings as remote
        p=provider(self.root);p._request=fake([]);seen=[]
        original=p._checkpoint._file_lock
        from contextlib import contextmanager
        @contextmanager
        def recorded(name,shared=False):
            seen.append((name,shared))
            with original(name,shared=shared):yield
        with patch.object(remote,'_CHECKPOINT_PARALLEL_ENABLED',False), patch.object(p._checkpoint,'_file_lock',recorded):
            p.embed(['a'])
            self.assertFalse(remote.checkpoint_concurrency_status()['parallel_enabled'])
            self.assertEqual(seen,[('.lock',False)])
            remote.enable_checkpoint_parallel();seen.clear();p.embed(['a'])
            self.assertTrue(remote.checkpoint_concurrency_status()['parallel_enabled'])
            self.assertEqual(seen,[('.lock',True)])

    def test_activation_waits_for_existing_exclusive_work_and_blocks_new_entrants(self):
        from tiger_memory import remote_embeddings as remote
        entered=threading.Event();release=threading.Event();next_entered=threading.Event()
        a=provider(self.root);b=provider(self.root)
        a._request=fake([],entered,release);b._request=fake([],next_entered)
        with patch.object(remote,'_CHECKPOINT_PARALLEL_ENABLED',False):
            with cf.ThreadPoolExecutor(2) as pool:
                first=pool.submit(a.embed,['legacy']);self.assertTrue(entered.wait(2))
                remote.enable_checkpoint_parallel()
                state=remote.checkpoint_concurrency_status()
                self.assertFalse(state['parallel_enabled']);self.assertEqual(state['legacy_active'],1)
                later=pool.submit(b.embed,['new'])
                try:self.assertFalse(next_entered.wait(.1))
                finally:release.set()
                first.result(3);later.result(3)
            self.assertTrue(remote.checkpoint_concurrency_status()['parallel_enabled'])
            self.assertEqual(remote.checkpoint_concurrency_status()['legacy_active'],0)

if __name__=='__main__':unittest.main()
