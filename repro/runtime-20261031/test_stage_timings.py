"""Offline diagnostics acceptance: boundedness, privacy and unchanged outcomes."""
import concurrent.futures
import json
import threading
import unittest
from contextlib import contextmanager
from unittest.mock import patch
from urllib.error import URLError

from tests.common import MemoryTestCase
from tiger_memory import stage_timings as st
from tiger_memory import aml_adapter as aml, service, database, remote_embeddings


class TimingTests(unittest.TestCase):
    def setUp(self):
        # Test isolation only; runtime exposes no reset endpoint.
        with st._LOCK:
            for row in st._DATA.values():
                row.update(count=0, errors=0, active=0, total_ns=0, max_ns=0,
                           buckets=[0] * (len(st.BOUNDS_MS) + 1))

    def test_nested_histogram_and_live_active(self):
        with patch.object(st.time, 'perf_counter_ns', side_effect=[0, 1000000, 3000000, 10000000]):
            with st.phase('add_total'):
                with st.phase('index_write'):
                    self.assertEqual(st.timing_status()['stages']['add_total']['active'], 1)
        s = st.timing_status()
        self.assertEqual(s['semantics'], 'inclusive-nested-not-additive')
        self.assertEqual(s['stages']['add_total']['total_ms'], 10)
        self.assertEqual(s['stages']['index_write']['total_ms'], 2)
        self.assertEqual(s['stages']['index_write']['histogram'][1], 1)
        self.assertEqual(s['stages']['add_total']['active'], 0)

    def test_concurrency_and_bounded_private_labels(self):
        def work(i):
            for _ in range(100):
                with st.phase('add_total'):
                    with st.phase('private-user-secret-' + str(i)):
                        pass
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(work, range(16)))
        s = st.timing_status()
        self.assertEqual(s['stages']['add_total']['count'], 1600)
        self.assertEqual(sum(s['stages']['add_total']['histogram']), 1600)
        self.assertEqual(set(s['stages']), set(st.STAGES))
        self.assertNotIn('private-user-secret', json.dumps(s))

    def test_business_exception_and_failed_instrumentation(self):
        error = ValueError('private exception body')
        with self.assertRaises(ValueError) as caught:
            with st.phase('add_total'):
                raise error
        self.assertIs(caught.exception, error)
        self.assertEqual(st.timing_status()['stages']['add_total']['errors'], 1)
        with patch.object(st.time, 'perf_counter_ns', side_effect=RuntimeError('clock')):
            with self.assertRaises(ValueError) as caught:
                with st.phase('add_total'):
                    raise error
            self.assertIs(caught.exception, error)
            with st.phase('add_total'):
                value = 'business success'
            self.assertEqual(value, 'business success')
        with patch.object(st.time, 'perf_counter_ns', side_effect=[1, RuntimeError('exit clock')]):
            with st.phase('add_total'):
                pass
        self.assertEqual(st.timing_status()['stages']['add_total']['active'], 0)
        self.assertNotIn('private exception body', json.dumps(st.timing_status()))

    def test_timed_acquire_preserves_suppression_and_release(self):
        seen = []
        @contextmanager
        def lock():
            seen.append('enter')
            try:
                yield 42
            except ValueError:
                seen.append('suppressed')
            finally:
                seen.append('exit')
        with st.timed_acquire(lock(), 'project_lock_wait') as value:
            self.assertEqual(value, 42)
            raise ValueError('fixture')
        self.assertEqual(seen, ['enter', 'suppressed', 'exit'])
        self.assertEqual(st.timing_status()['stages']['project_lock_wait']['errors'], 0)

    def test_authenticated_status_only(self):
        handler = aml.AMLRequestHandler.__new__(aml.AMLRequestHandler)
        handler.path = '/status'
        sent = []
        handler._send_json = lambda code, payload: sent.append((code, payload))
        handler._authorized = lambda: False
        with patch.object(aml, '_auth_token', return_value='fixture-token'):
            handler.do_GET()
            self.assertEqual(sent[-1][0], 401)
            self.assertNotIn('stage_timings', sent[-1][1])
            handler._authorized = lambda: True
            handler.do_GET()
            self.assertEqual(sent[-1][0], 200)
            self.assertIn('stage_timings', sent[-1][1])
            handler.path = '/health'
            handler.do_GET()
            self.assertNotIn('stage_timings', sent[-1][1])


class TimingStorageTests(MemoryTestCase):
    def test_fake_embedding_retry_cache_and_first_vectors(self):
        provider = remote_embeddings.DashScopeEmbeddingProvider(
            'https://example.invalid/embeddings', 'offline-fixture',
            cache_dir=self.base.resolve() / 'embedding-fixture', dimensions=64)
        raw = json.dumps({'model': 'text-embedding-v4', 'data': [
            {'index': 0, 'embedding': [1.0] + [0.0] * 63}]}).encode()
        class Response:
            status = 200
            def __enter__(self): return self
            def __exit__(self, *exc): return False
            def read(self, size): return raw
        before = st.timing_status()['stages']
        with patch.object(provider._opener, 'open', side_effect=[URLError('offline fixture'), Response()]) as opened:
            with patch.object(remote_embeddings, 'sleep'):
                first = provider.embed(['private original fixture'])
                again = provider.embed(['private original fixture'])
            self.assertEqual(opened.call_count, 2)
        self.assertEqual(first, again)
        after = st.timing_status()['stages']
        self.assertEqual(after['embedding_http']['count'] - before['embedding_http']['count'], 2)
        self.assertEqual(after['embedding_backoff']['count'] - before['embedding_backoff']['count'], 1)
        self.assertEqual(after['embedding_total']['count'] - before['embedding_total']['count'], 2)
        self.assertGreater(after['embedding_cache_publish']['count'] - before['embedding_cache_publish']['count'], 0)
        self.assertNotIn('private original fixture', json.dumps(st.timing_status()))

    def test_real_offline_write_replay_and_failure_spans(self):
        root = aml.ensure_user_storage('timing-fixture')
        identity = service.ExpectedIdentity(root, 'session', 'timing-fixture')
        def write(body):
            return service.write_event(root, 'fixture', body,
                scope='session', scope_id='timing-fixture',
                idempotency_key='timing-idempotency', expected_identity=identity)
        event = write('first source bytes')
        replay = write('changed replay')
        self.assertEqual(event, replay)
        status = st.timing_status()['stages']
        for name in ['write_event', 'index_write', 'write_transaction', 'db_begin_wait',
                     'db_commit', 'source_publish', 'source_file_fsync', 'source_directory_fsync']:
            self.assertGreater(status[name]['count'], 0, name)
            self.assertEqual(status[name]['active'], 0, name)
        self.assertNotIn('first source bytes', json.dumps(st.timing_status()))
        c = database.connect_readonly()
        self.assertEqual(c.execute('SELECT count(*) FROM events').fetchone()[0], 1)
        c.close()
