"""Offline lock shortening with first-write, byte, and recovery parity."""
import json
import sqlite3
import threading
import time
from pathlib import Path
from unittest.mock import patch

from tests.common import MemoryTestCase
from tiger_memory import aml_adapter as aml, database, events, service, stage_timings as timing


class LockOptimizationTests(MemoryTestCase):
    def setUp(self):
        super().setUp()
        self.root = aml.ensure_user_storage('lock-fixture')
        self.identity = service.ExpectedIdentity(self.root, 'session', 'lock-fixture')
        # Warm the catalog without the path under test (no live/model work).
        c = database.connect()
        c.close()

    def write(self, body='original cobalt 中文', key='lock:1', **extra):
        return service.write_event(self.root, 'original title', body,
            tags=['public', 'aml-session:private-routing'], scope='session', scope_id='lock-fixture',
            idempotency_key=key, expected_identity=self.identity,
            recorded_at='2026-10-31T00:00:00Z', **extra)

    def rows(self, table):
        c = database.connect_readonly()
        try:
            return [dict(r) for r in c.execute('SELECT * FROM ' + table)]
        finally:
            c.close()

    def vector(self, event):
        c = database.connect_readonly()
        try:
            return c.execute('SELECT vector FROM vectors WHERE ref_id=?', (event.event_id,)).fetchone()[0]
        finally:
            c.close()

    def test_encoder_runs_before_writer_lock_and_exact_output(self):
        original = database.encode_vector
        observed = []
        def checked(text):
            # Real competing SQLite writer, timeout zero: this would fail inside
            # the application writer transaction, rather than blocking a test.
            other = sqlite3.connect(str(database.db_path()), timeout=0)
            try:
                other.execute('BEGIN IMMEDIATE')
                other.rollback()
            finally:
                other.close()
            self.assertEqual(timing.timing_status()['stages']['write_lock_held']['active'], 0)
            observed.append(text)
            return original(text)
        with patch.object(database, 'encode_vector', checked):
            event = self.write(('long cobalt 中文 ' * 1200).strip())
        text = ' '.join((event.metadata['title'], 'public', event.body))
        self.assertEqual(observed, [text])
        self.assertEqual(self.vector(event), original(text))
        self.assertNotIn('aml-session:', observed[0])
        self.assertEqual(events.parse_event(event.path), event)

    def test_indexed_replay_does_not_prepare_changed_payload(self):
        first = self.write()
        raw = first.path.read_bytes()
        vector = self.vector(first)
        with patch.object(database, 'prepare_lexical_vector', side_effect=AssertionError('must not prepare replay')) as prep:
            again = self.write('different replay text')
        prep.assert_not_called()
        self.assertEqual(first, again)
        self.assertEqual(raw, first.path.read_bytes())
        self.assertEqual(vector, self.vector(first))
        self.assertEqual(len(self.rows('events')), 1)

    def test_late_existing_key_discards_preparation_and_preserves_first(self):
        first = self.write()
        raw = first.path.read_bytes()
        vector = self.vector(first)
        original = service.event_for_idempotency
        calls = []
        def initially_missing(*args):
            calls.append(1)
            return None if len(calls) == 1 else original(*args)
        with patch.object(service, 'event_for_idempotency', initially_missing), patch.object(service, 'find_orphan', return_value=None):
            with patch.object(database, 'prepare_lexical_vector', side_effect=RuntimeError('offline precompute failure')) as prep:
                replay = self.write('changed late replay')
        self.assertEqual(prep.call_count, 1)
        self.assertEqual(first, replay)
        self.assertEqual(raw, first.path.read_bytes())
        self.assertEqual(vector, self.vector(first))

    def make_orphan(self):
        atomic = service.atomic_write
        def fail_after_source(path, text):
            atomic(path, text)
            if path.suffix == '.md':
                raise OSError('offline crash after publication')
        with patch.object(service, 'atomic_write', fail_after_source), self.assertRaises(OSError):
            self.write('immutable orphan original')
        paths = list(self.base.glob('v2/projects/*/events/*/*/*.md'))
        self.assertEqual(len(paths), 1)
        self.assertEqual(self.rows('events'), [])
        return events.parse_event(paths[0].resolve())

    def test_orphan_preparation_uses_original_bytes_and_recovery(self):
        orphan = self.make_orphan()
        raw = orphan.path.read_bytes()
        original = database.prepare_lexical_vector
        inputs = []
        def check(title, tags, body):
            inputs.append((title, tags, body))
            return original(title, tags, body)
        with patch.object(database, 'prepare_lexical_vector', check):
            recovered = self.write('changed retry must not become original')
        self.assertEqual(inputs, [(orphan.metadata['title'], 'public', orphan.body)])
        self.assertEqual(recovered, orphan)
        self.assertEqual(raw, orphan.path.read_bytes())
        self.assertEqual(self.vector(recovered), database.encode_vector(' '.join(inputs[0])))

    def test_changed_orphan_after_preparation_still_rejected(self):
        orphan = self.make_orphan()
        original = database.prepare_lexical_vector
        def mutate(title, tags, body):
            result = original(title, tags, body)
            events.atomic_write(orphan.path, events.serialize_event(orphan.metadata, 'external changed body'))
            return result
        with patch.object(database, 'prepare_lexical_vector', mutate), self.assertRaisesRegex(service.PLMError, 'source changed'):
            self.write('replay')
        self.assertEqual(self.rows('events'), [])

    def test_exact_binding_mismatch_recomputes_original_encoder(self):
        wrong = database.prepare_lexical_vector('foreign', '', 'wrong body')
        with patch.object(database, 'prepare_lexical_vector', return_value=wrong):
            event = self.write('proper body')
        expected = database.encode_vector(' '.join(('original title', 'public', 'proper body')))
        self.assertEqual(self.vector(event), expected)
        self.assertNotEqual(self.vector(event), wrong.vector)

    def test_preparation_failure_keeps_original_fallback(self):
        with patch.object(database, 'prepare_lexical_vector', side_effect=RuntimeError('preparation unavailable')):
            event = self.write()
        self.assertEqual(self.vector(event), database.encode_vector(' '.join(('original title', 'public', event.body))))

    def test_validation_failure_and_post_index_identity_change_rollback(self):
        before = timing.timing_status()['stages']['write_lock_held']['errors']
        with self.assertRaises(service.PLMError):
            self.write(extra={'unknown-protected-field': 'invalid'})
        self.assertEqual(self.rows('events'), [])
        original = service._index_validated_write
        def changed_alias(conn, *args, **kwargs):
            original(conn, *args, **kwargs)
            conn.execute('INSERT INTO projects(project_id,canonical_root,display_name,folder_name,created_at) VALUES(?,?,?,?,?)',
                         ('foreign', '/offline-foreign', 'foreign', 'foreign', '2026-10-31T00:00:00Z'))
            # Valid SQL, but the final fixed-target guard must still reject it.
            conn.execute('UPDATE project_aliases SET project_id=? WHERE alias_path=?', ('foreign', str(self.root)))
        with patch.object(service, '_index_validated_write', changed_alias), self.assertRaises(service.StorageIdentityConflict):
            self.write()
        self.assertEqual(self.rows('events'), [])
        self.assertEqual(list(self.base.glob('v2/projects/*/events/*/*/*.md')), [])
        row = timing.timing_status()['stages']['write_lock_held']
        self.assertEqual(row['active'], 0)
        self.assertEqual(row['errors'] - before, 2)

    def test_lock_held_span_excludes_actual_sqlite_admission_wait(self):
        holder = database.connect()
        holder.execute('BEGIN IMMEDIATE')
        failures = []
        completed = []
        def writer():
            try:
                completed.append(self.write())
            except BaseException as exc:
                failures.append(exc)
        t = threading.Thread(target=writer)
        t.start()
        try:
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                stages = timing.timing_status()['stages']
                if stages['db_begin_wait']['active']:
                    break
                time.sleep(0.005)
            self.assertGreater(stages['db_begin_wait']['active'], 0)
            self.assertGreater(stages['write_transaction']['active'], 0)
            self.assertEqual(stages['write_lock_held']['active'], 0)
        finally:
            holder.rollback()
            holder.close()
            t.join(5)
        self.assertFalse(t.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(len(completed), 1)
        self.assertEqual(timing.timing_status()['stages']['write_lock_held']['active'], 0)

    def test_unchanged_alias_register_has_zero_changes(self):
        project = service._expected_project(self.identity)
        c = database.connect()
        args = (project['project_id'], project['canonical_root'], project['display_name'], project['folder_name'])
        database.register_project(c, *args)
        c.commit()
        first = dict(c.execute('SELECT * FROM project_aliases').fetchone())
        before = c.total_changes
        database.register_project(c, *args)
        self.assertEqual(c.total_changes, before)
        self.assertEqual(dict(c.execute('SELECT * FROM project_aliases').fetchone()), first)
        c.execute('UPDATE project_aliases SET active=0')
        before = c.total_changes
        database.register_project(c, *args)
        self.assertEqual(c.total_changes - before, 1)
        self.assertEqual(c.execute('SELECT active FROM project_aliases').fetchone()[0], 1)
        c.rollback()
        c.close()
