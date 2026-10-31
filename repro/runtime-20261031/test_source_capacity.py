"""Isolated bounded-capacity tests; real Linux watch delivery is tested separately."""
import json
import sys
from pathlib import Path
from unittest.mock import patch
import unittest

import test_performance as fixtures
from test_incremental_recovery import clear_indexes
from tiger_memory import reliability
from tiger_memory.recovery_watch import SourceWatch, WatchUnavailable


class FakeWatch:
    def __init__(self, root, tree_root=None):
        self.pending = set()
        self.closed = False
    def add(self, path, directory=False):
        pass
    def changes(self):
        result, self.pending = self.pending, set()
        return result
    def close(self):
        self.closed = True


class SourceCapacityTests(fixtures.MetadataReuseTests):
    def setUp(self):
        clear_indexes()
        super().setUp()
        clear_indexes()
        self.watch_patch = patch.object(reliability, 'SourceWatch', FakeWatch)
        self.watch_patch.start()
    def tearDown(self):
        self.watch_patch.stop()
        clear_indexes()
        self.assertEqual(reliability._index_capacity_status()['cached_sources'], 0)
        self.assertEqual(reliability._index_capacity_status()['cached_metadata_bytes'], 0)
        super().tearDown()

    def test_project_above_old_source_limit_stays_incremental(self):
        for i in range(2050):
            self.make('history-' + str(i))
        before = reliability.recovery_status()['counters'].get('full_scan_fallbacks', 0)
        self.scan()
        entry = next(iter(reliability._INCREMENTAL.values()))
        self.assertIsNotNone(entry.watch)
        self.assertEqual(len(entry.entries), 2051)
        self.assertGreaterEqual(reliability.recovery_status()['counters']['snapshot_sources_observed_max'], 2051)
        with patch.object(reliability, '_fingerprint', side_effect=AssertionError('full scan')):
            with patch.object(Path, 'glob', side_effect=AssertionError('enumeration')):
                self.assertIsNone(self.scan())
        self.assertEqual(reliability.recovery_status()['counters'].get('full_scan_fallbacks', 0), before)
        path, meta = self.make('new-orphan', 'FIRST BODY')
        entry.watch.pending.add(path)
        self.assertEqual(self.scan('new-orphan').body, 'FIRST BODY')
        self.assertEqual(reliability._index_capacity_status()['cached_sources'], 2052)

    def test_new_per_project_source_bound_still_falls_back(self):
        self.make('seed', 'SECOND BODY')
        with patch.object(reliability._SourceIndex, 'MAX_SOURCES', 1):
            with self.assertRaisesRegex(reliability.SourceRecoveryError, 'AMBIGUOUS'):
                self.scan('seed')
        self.assertEqual(reliability._index_capacity_status()['cached_sources'], 0)
        self.assertGreaterEqual(reliability.recovery_status()['counters']['snapshot_sources_at_capacity_failure_max'], 1)

    def test_aggregate_count_limit_covers_distinct_indexes(self):
        path, meta = self.make('another')
        first, second = reliability._SourceIndex(self.p), reliability._SourceIndex(self.p)
        first.watch, second.watch = FakeWatch(None), FakeWatch(None)
        try:
            with patch.object(reliability, '_INDEX_MAX_TOTAL_SOURCES', 3):
                first.put(self.seed.path)
                first.put(path)
                second.put(self.seed.path)
                with self.assertRaisesRegex(WatchUnavailable, 'index_resource_limit'):
                    second.put(path)
                self.assertEqual(reliability._index_capacity_status()['cached_sources'], 3)
                first.close()
                second.put(path)
                self.assertEqual(reliability._index_capacity_status()['cached_sources'], 2)
        finally:
            first.close(); second.close()

    def test_aggregate_metadata_bound_and_close_release(self):
        first, second = reliability._SourceIndex(self.p), reliability._SourceIndex(self.p)
        first.watch, second.watch = FakeWatch(None), FakeWatch(None)
        try:
            first.put(self.seed.path)
            size = first.bytes
            with patch.object(reliability, '_INDEX_MAX_TOTAL_METADATA_BYTES', 2 * size - 1):
                with self.assertRaisesRegex(WatchUnavailable, 'index_resource_limit'):
                    second.put(self.seed.path)
                first.close()
                second.put(self.seed.path)
                self.assertEqual(reliability._index_capacity_status()['cached_metadata_bytes'], size)
        finally:
            first.close(); second.close()

    def test_partial_entry_allocation_failure_releases_exact_reservation(self):
        class FailedDictionary(dict):
            def setdefault(self, *args):
                raise MemoryError('fixture allocation failure')
        entry = reliability._SourceIndex(self.p)
        entry.watch = FakeWatch(None)
        entry.by_key = FailedDictionary()
        try:
            with self.assertRaises(MemoryError):
                entry.put(self.seed.path)
        finally:
            entry.close()
        self.assertEqual(reliability._index_capacity_status()['cached_sources'], 0)
        self.assertEqual(reliability._index_capacity_status()['cached_metadata_bytes'], 0)

    def test_concurrent_indexes_cannot_overbook_process_capacity(self):
        from concurrent.futures import ThreadPoolExecutor
        import threading
        entries = [reliability._SourceIndex(self.p) for _ in range(16)]
        for entry in entries:
            entry.watch = FakeWatch(None)
        barrier = threading.Barrier(16)
        def insert(entry):
            barrier.wait(timeout=10)
            try:
                entry.put(self.seed.path)
                return True
            except WatchUnavailable as exc:
                self.assertEqual(str(exc), 'index_resource_limit')
                return False
        try:
            with patch.object(reliability, '_INDEX_MAX_TOTAL_SOURCES', 4):
                with ThreadPoolExecutor(max_workers=16) as workers:
                    results = list(workers.map(insert, entries))
                self.assertEqual(sum(results), 4)
                self.assertEqual(reliability._index_capacity_status()['cached_sources'], 4)
                self.assertEqual(reliability._index_capacity_status()['cached_metadata_bytes'],
                                 sum(entry.bytes for entry in entries))
        finally:
            for entry in entries:
                entry.close()

    def test_entry_update_replaces_size_without_double_counting(self):
        self.scan()
        entry = next(iter(reliability._INCREMENTAL.values()))
        before = reliability._index_capacity_status()
        entry.put(self.seed.path)
        after = reliability._index_capacity_status()
        self.assertEqual(before, after)
        from tiger_memory.events import serialize_event
        self.seed.metadata['idempotency_key'] = 'a-longer-key'
        self.seed.path.write_text(serialize_event(self.seed.metadata, 'CHANGED BODY'))
        entry.watch.pending.add(self.seed.path)
        self.assertIsNone(self.scan('seed'))
        self.assertEqual(self.scan('a-longer-key').body, 'CHANGED BODY')
        self.assertEqual(reliability._index_capacity_status()['cached_sources'], 1)
        self.assertEqual(reliability._index_capacity_status()['cached_metadata_bytes'], entry.bytes)

    def test_global_capacity_fallback_cooldown_reads_new_actual_sources(self):
        self.make('second')
        before = reliability.recovery_status()['fallback_reasons'].get('index_resource_limit', 0)
        with patch.object(reliability, '_INDEX_MAX_TOTAL_SOURCES', 1):
            self.scan()
            entry = next(iter(reliability._INCREMENTAL.values()))
            self.assertIsNone(entry.watch)
            self.assertEqual(reliability._index_capacity_status()['cached_sources'], 0)
            self.make('new-orphan', 'NEW SOURCE BYTES')
            self.assertEqual(self.scan('new-orphan').body, 'NEW SOURCE BYTES')
            self.make('new-orphan', 'DUPLICATE')
            with self.assertRaisesRegex(reliability.SourceRecoveryError, 'AMBIGUOUS'):
                self.scan('new-orphan')
        self.assertEqual(reliability.recovery_status()['fallback_reasons'].get('index_resource_limit', 0), before + 1)

    def test_periodic_recheck_still_discovers_unreported_source(self):
        self.scan()
        entry = next(iter(reliability._INCREMENTAL.values()))
        self.make('unreported', 'NEW BODY')
        entry.last_full_verify -= 61
        self.assertEqual(self.scan('unreported').body, 'NEW BODY')
        self.assertEqual(reliability._index_capacity_status()['cached_sources'], 2)


for _name in list(dir(SourceCapacityTests)):
    if _name.startswith('test_') and _name not in SourceCapacityTests.__dict__:
        setattr(SourceCapacityTests, _name, None)


@unittest.skipUnless(sys.platform == 'linux', 'real inotify requires Linux')
class LinuxSourceCapacityTests(SourceCapacityTests):
    def setUp(self):
        super().setUp()
        self.watch_patch.stop()
    def tearDown(self):
        # stop() is idempotent; parent still clears indexes before storage removal.
        super().tearDown()
    def test_project_above_old_source_limit_stays_incremental(self):
        for i in range(2050):
            self.make('history-' + str(i))
        self.scan()
        entry = next(iter(reliability._INCREMENTAL.values()))
        self.assertIsNotNone(entry.watch)
        self.assertEqual(len(entry.entries), 2051)
        self.assertGreaterEqual(reliability.recovery_status()['counters']['snapshot_sources_observed_max'], 2051)
        with patch.object(reliability, '_fingerprint', side_effect=AssertionError('full scan')):
            with patch.object(Path, 'glob', side_effect=AssertionError('enumeration')):
                self.assertIsNone(self.scan())
        path, meta = self.make('new-orphan', 'REAL NOTIFICATION BODY')
        self.assertEqual(self.scan('new-orphan').body, 'REAL NOTIFICATION BODY')
        self.assertIsNotNone(entry.watch)
        self.assertEqual(len(entry.entries), 2052)

    def test_large_project_watch_exhaustion_falls_back_without_partial_index(self):
        for i in range(2050):
            self.make('history-' + str(i))
        self.make('recover-me', 'SOURCE STILL AUTHORITATIVE')
        before = reliability.recovery_status()['fallback_reasons'].get('watch_limit', 0)
        with patch.object(SourceWatch, 'MAX_WATCHES', 2050):
            self.assertEqual(self.scan('recover-me').body, 'SOURCE STILL AUTHORITATIVE')
        entry = next(iter(reliability._INCREMENTAL.values()))
        self.assertIsNone(entry.watch)
        self.assertEqual(reliability._index_capacity_status()['cached_sources'], 0)
        self.assertEqual(reliability.recovery_status()['fallback_reasons'].get('watch_limit', 0), before + 1)


# Real Linux test exercises kernel events; explicit FakeWatch queue tests above
# belong only to the simulation class, not to the Linux subclass.
for _name in list(dir(LinuxSourceCapacityTests)):
    if _name.startswith('test_') and _name not in LinuxSourceCapacityTests.__dict__:
        setattr(LinuxSourceCapacityTests, _name, None)
