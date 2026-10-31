"""Offline notification filtering and fail-safe regressions; no remote calls."""
import json
from pathlib import Path
import struct
import sys
import time
import unittest
from unittest.mock import patch

from tiger_memory import reliability, service
from tiger_memory.model import v2_root
from tiger_memory.recovery_watch import SourceWatch, WatchInvalidated, WatchUnavailable
import test_incremental_recovery as recovery_fixtures


class FilterDecoderTests(unittest.TestCase):
    def fixture(self):
        index = reliability._SourceIndex({'project_id': 'offline', 'folder_name': 'offline', 'canonical_root': '/offline'})
        watch = SourceWatch.__new__(SourceWatch)
        watch.fd = -1
        watch.tree_root = index.folder
        watch.mount_fingerprint = b'unchanged'
        watch.paths = {v2_root(): 1}
        watch.watches = {1: {(v2_root(), True)}}
        watch.directory_identity = {}
        index.watch = watch
        index.last_full_verify = time.monotonic()
        return index, watch

    def event(self, name, wd=1, mask=0x2):
        raw = name.encode() + b'\0'
        return struct.pack('iIII', wd, mask, 0, len(raw)) + raw

    def test_unrelated_wal_at_final_guard_does_not_invalidate(self):
        index, watch = self.fixture()
        with patch.object(SourceWatch, '_mount_fingerprint', return_value=b'unchanged'), patch('os.read', side_effect=[BlockingIOError(), self.event('catalog.sqlite3-wal'), BlockingIOError()]):
            self.assertIsNone(index.find('absent', lambda event: True, set()))

    def test_unrelated_bursts_do_not_consume_sync_attempts(self):
        index, watch = self.fixture()
        events = [self.event('catalog.sqlite3-wal')] * 100
        with patch.object(SourceWatch, '_mount_fingerprint', return_value=b'unchanged'), patch('os.read', side_effect=events + [BlockingIOError()]):
            index.sync()

    def test_temporary_and_outside_tree_changes_ignored_by_both_guards(self):
        index, watch = self.fixture()
        paths = {v2_root()/'catalog.sqlite3-wal', index.folder/'2026'/'10'/'source.tmp'}
        with patch.object(watch, 'changes', return_value=paths) as changes:
            self.assertIsNone(index.find('absent', lambda event: True, set()))
            self.assertEqual(changes.call_count, 2)

    def test_formal_new_source_at_final_guard_still_invalidates(self):
        index, watch = self.fixture()
        source = index.folder/'2026'/'10'/'20261031T000000Z_00000000-0000-0000-0000-000000000001.md'
        with patch.object(watch, 'changes', side_effect=[set(), {source}]):
            with self.assertRaisesRegex(WatchInvalidated, 'changed_during_lookup'):
                index.find('absent', lambda event: True, set())

    def test_inode_notification_survives_filter_even_outside_tree(self):
        index, watch = self.fixture()
        alias = v2_root()/'explicit-inode-alias'
        watch.watches[2] = {(alias, False)}
        with patch.object(SourceWatch, '_mount_fingerprint', return_value=b'unchanged'), patch('os.read', side_effect=[self.event('', wd=2), BlockingIOError()]):
            self.assertEqual(watch.changes(), {alias})

    def test_structural_ancestor_event_without_directory_bit_fails_safe(self):
        index, watch = self.fixture()
        ancestor = v2_root()/'projects'
        watch.paths[ancestor] = 2
        watch.directory_identity[ancestor] = ('fixture',)
        with patch.object(SourceWatch, '_mount_fingerprint', return_value=b'unchanged'), patch('os.read', side_effect=[self.event('projects', mask=0x200), BlockingIOError()]):
            with self.assertRaisesRegex(WatchInvalidated, 'directory_change'):
                watch.changes()

    def test_notification_overflow_and_mount_change_still_fail_safe(self):
        index, watch = self.fixture()
        with patch.object(SourceWatch, '_mount_fingerprint', return_value=b'unchanged'), patch('os.read', return_value=self.event('', wd=-1, mask=0x4000)):
            with self.assertRaises(WatchInvalidated):
                watch.changes()
        with patch.object(SourceWatch, '_mount_fingerprint', return_value=b'new'):
            with self.assertRaisesRegex(WatchInvalidated, 'mount_table_changed'):
                watch.changes()

    def test_reason_counters_have_fixed_labels_no_exception_text(self):
        before = reliability.recovery_status()
        reliability._count_fallback(WatchUnavailable('watch_limit'))
        reliability._count_fallback(WatchInvalidated('/private/path/token=SECRET'))
        reliability._count_fallback(RuntimeError('SECRET OTHER'))
        after = reliability.recovery_status()
        self.assertEqual(after['fallback_reasons'].get('watch_limit', 0)-before['fallback_reasons'].get('watch_limit', 0), 1)
        self.assertEqual(after['fallback_reasons'].get('unexpected', 0)-before['fallback_reasons'].get('unexpected', 0), 2)
        self.assertEqual(after['counters'].get('full_scan_fallbacks', 0)-before['counters'].get('full_scan_fallbacks', 0), 3)
        self.assertNotIn('SECRET', json.dumps(after))
        self.assertTrue(set(after['fallback_reasons']).issubset(reliability._KNOWN_FALLBACK_REASONS))


@unittest.skipUnless(sys.platform == 'linux', 'real inotify requires Linux')
class LinuxFilteredRecoveryTests(recovery_fixtures.FullFallbackTests):
    def warm(self):
        self.scan()
        index = next(iter(reliability._INCREMENTAL.values()))
        self.assertIsNotNone(index.watch, 'must exercise real kernel inotify')
        return index

    def test_shared_wal_changes_before_and_during_lookup_do_not_full_scan(self):
        index = self.warm()
        wal = v2_root()/'catalog.sqlite3-wal-fixture'
        before = reliability.recovery_status()['counters'].get('full_scan_fallbacks', 0)
        def committed(event):
            wal.write_bytes(b'during final source lookup')
            return service.batch_committed(event)
        with patch.object(index, 'build', side_effect=AssertionError('unexpected cold build')), patch.object(reliability, '_find_orphan_full', side_effect=AssertionError('unexpected full fallback')):
            for n in range(20):
                wal.write_bytes(str(n).encode())
                result = reliability.find_orphan(self.p, 'seed', committed, set())
                self.assertEqual(result.body, 'ORIGINAL BODY')
        self.assertEqual(reliability.recovery_status()['counters'].get('full_scan_fallbacks', 0), before)

    def test_temporary_source_write_during_lookup_does_not_full_scan(self):
        index = self.warm()
        temp = self.seed.path.with_suffix('.tmp')
        def committed(event):
            temp.write_text('not a published source')
            return True
        with patch.object(reliability, '_find_orphan_full', side_effect=AssertionError('unexpected full fallback')):
            self.assertEqual(reliability.find_orphan(self.p, 'seed', committed, set()).body, 'ORIGINAL BODY')
        temp.unlink()

    def test_published_source_during_lookup_still_falls_back(self):
        self.warm()
        published = []
        def committed(event):
            if not published:
                published.append(self.make('during-lookup', 'FIRST BYTES'))
            return True
        before = reliability.recovery_status()['fallback_reasons'].get('changed_during_lookup', 0)
        self.assertEqual(reliability.find_orphan(self.p, 'seed', committed, set()).body, 'ORIGINAL BODY')
        self.assertEqual(reliability.recovery_status()['fallback_reasons'].get('changed_during_lookup', 0), before+1)
        self.assertEqual(self.scan('during-lookup').body, 'FIRST BYTES')

    def test_resource_failure_and_cooldown_reasons_preserve_source_recovery(self):
        index = self.warm()
        index.close()
        before = reliability.recovery_status()['fallback_reasons']
        with patch.object(reliability, 'SourceWatch', side_effect=WatchUnavailable('watch_limit')):
            self.assertEqual(self.scan('seed').body, 'ORIGINAL BODY')
            self.make('during-cooldown', 'NEW FIRST BYTES')
            self.assertEqual(self.scan('during-cooldown').body, 'NEW FIRST BYTES')
        after = reliability.recovery_status()['fallback_reasons']
        self.assertEqual(after.get('watch_limit',0)-before.get('watch_limit',0), 1)
        self.assertEqual(after.get('cooldown',0)-before.get('cooldown',0), 1)

    def test_sixteen_projects_actual_sqlite_writers_keep_warm_indexes(self):
        from concurrent.futures import ThreadPoolExecutor
        import threading
        from tiger_memory import aml_adapter, database
        from tiger_memory.events import serialize_event
        writers = []
        # Establish each source tree and warm it before concurrency so the test
        # measures shared catalog WAL traffic, not initial missing directories.
        for n in range(16):
            user = "filter-sqlite-writer-" + str(n)
            root = aml_adapter.ensure_user_storage(user)
            identity = service.ExpectedIdentity(root, 'session', user)
            for seq in range(2):
                service.write_event(root, 'warm', 'stable warm bytes', scope='session', scope_id=user,
                                    idempotency_key='warm-' + str(seq), expected_identity=identity)
            writers.append((user, root, identity))
        indexes = [entry for entry in reliability._INCREMENTAL.values()
                   if entry.project['canonical_root'] in {str(x[1].resolve()) for x in writers}]
        self.assertEqual(len(indexes), 16)
        self.assertTrue(all(entry.watch is not None for entry in indexes))
        before = reliability.recovery_status()['counters'].get('full_scan_fallbacks', 0)
        barrier = threading.Barrier(16)
        def write(row):
            user, root, identity = row
            barrier.wait(timeout=10)
            events = []
            for seq in range(2):
                body = user + ' preserved source bytes ' + str(seq)
                event = service.write_event(root, 'concurrent', body, scope='session', scope_id=user,
                                            idempotency_key='concurrent-' + str(seq), expected_identity=identity)
                self.assertEqual(event.body, body)
                self.assertEqual(event.path.read_text(), serialize_event(event.metadata, body))
                events.append(event)
            return events
        with patch.object(reliability, '_find_orphan_full', side_effect=AssertionError('warm writer unexpectedly fell back')):
            with ThreadPoolExecutor(max_workers=16) as pool:
                results = list(pool.map(write, writers))
        self.assertEqual(reliability.recovery_status()['counters'].get('full_scan_fallbacks', 0), before)
        self.assertTrue(all(entry.watch is not None for entry in indexes))
        conn = database.connect_readonly()
        try:
            for events in results:
                for event in events:
                    row = conn.execute('SELECT body FROM events WHERE event_id=?', (event.event_id,)).fetchone()
                    self.assertEqual(row['body'], event.body)
        finally:
            conn.close()

# Reuse fixture setup only; run the existing full suite separately.
for name in dir(LinuxFilteredRecoveryTests):
    if name.startswith('test_') and name not in LinuxFilteredRecoveryTests.__dict__:
        setattr(LinuxFilteredRecoveryTests, name, None)
