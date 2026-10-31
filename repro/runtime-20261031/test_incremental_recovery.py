"""Filesystem-only fixtures; no production paths, APIs or model calls."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
import uuid

from test_performance import MetadataReuseTests
from tiger_memory import reliability
from tiger_memory.events import event_relative_path, serialize_event
from tiger_memory.model import v2_root
from tiger_memory.recovery_watch import SourceWatch, WatchUnavailable, WatchInvalidated


def clear_indexes():
    for entry in reliability._INCREMENTAL.values():
        entry.close()
    reliability._INCREMENTAL.clear()


class FullFallbackTests(MetadataReuseTests):
    def setUp(self):
        clear_indexes()
        super().setUp()
        clear_indexes()

    def tearDown(self):
        clear_indexes()
        super().tearDown()

    def test_watch_failure_is_full_scan_not_negative(self):
        p, m = self.make('crash', 'FIRST BYTES')
        with patch.object(reliability, 'SourceWatch', side_effect=WatchUnavailable()):
            self.assertEqual(self.scan('crash').body, 'FIRST BYTES')
            p.write_text('corrupt')
            with self.assertRaisesRegex(reliability.SourceRecoveryError, 'CORRUPT'):
                self.scan('absent')


@unittest.skipUnless(sys.platform == 'linux', 'real inotify requires Linux')
class LinuxIncrementalTests(FullFallbackTests):
    def warm(self):
        self.scan()
        entry = next(iter(reliability._INCREMENTAL.values()))
        self.assertIsNotNone(entry.watch, 'fixture must exercise actual inotify')
        return entry

    def test_unchanged_miss_does_not_lstat_or_enumerate_history(self):
        for i in range(100):
            self.make('history-' + str(i))
        self.warm()
        with patch.object(reliability, '_fingerprint', side_effect=AssertionError('history lstat')):
            with patch.object(Path, 'glob', side_effect=AssertionError('history walk')):
                # Fallback would call the patched glob and fail, not hide it.
                self.assertIsNone(self.scan())

    def test_one_new_source_is_only_metadata_revalidated(self):
        for i in range(100):
            self.make('history-' + str(i))
        self.warm()
        path, m = self.make('crash', 'FIRST')
        with patch.object(reliability, '_source_metadata', wraps=reliability._source_metadata) as read:
            self.assertEqual(self.scan('crash').body, 'FIRST')
            self.assertEqual([c.args[0] for c in read.call_args_list], [path])

    def test_hardlink_alias_write_invalidates_inode(self):
        path, m = self.make('before', 'FIRST')
        alias = path.parents[3] / 'external-alias'
        os.link(path, alias)
        self.warm()
        m['idempotency_key'] = 'after!'
        alias.write_text(serialize_event(m, 'OTHER'))
        self.assertIsNone(self.scan('before'))
        self.assertEqual(self.scan('after!').body, 'OTHER')

    def test_atomic_replace_invalidates_watched_inode(self):
        path, m = self.make('before')
        self.warm()
        temp = path.with_suffix('.tmp')
        m['idempotency_key'] = 'after!'
        temp.write_text(serialize_event(m, 'REPLACED'))
        os.replace(temp, path)
        self.assertIsNone(self.scan('before'))
        self.assertEqual(self.scan('after!').body, 'REPLACED')

    def test_replace_with_surviving_old_and_new_hardlink(self):
        path, m = self.make('before')
        old_alias = path.parents[3] / 'old-alias'
        new_alias = path.parents[3] / 'new-alias'
        os.link(path, old_alias)
        self.warm()
        temp = path.with_suffix('.tmp')
        m['idempotency_key'] = 'middle'
        temp.write_text(serialize_event(m, 'REPLACED'))
        os.link(temp, new_alias)
        os.replace(temp, path)
        self.assertEqual(self.scan('middle').body, 'REPLACED')
        # A second lookup warms the replacement's inode after safe fallback.
        self.scan()
        m['idempotency_key'] = 'after!'
        new_alias.write_text(serialize_event(m, 'LATEST'))
        self.assertIsNone(self.scan('middle'))
        self.assertEqual(self.scan('after!').body, 'LATEST')

    def test_repeated_real_writes_keep_warm_snapshot(self):
        from tiger_memory import service, aml_adapter
        root = aml_adapter.ensure_user_storage('warm-fixture')
        identity = service.ExpectedIdentity(root, 'session', 'warm-fixture')
        clear_indexes()
        for i in range(12):
            service.write_event(root, 'title', 'body', scope='session', scope_id='warm-fixture',
                                idempotency_key='warm-'+str(i), expected_identity=identity)
        entry = next(iter(reliability._INCREMENTAL.values()))
        self.assertIsNotNone(entry.watch)
        with patch.object(entry, 'build', side_effect=AssertionError('unexpected cold build')):
            service.write_event(root, 'title', 'body', scope='session', scope_id='warm-fixture',
                                idempotency_key='warm-final', expected_identity=identity)
        self.assertIsNotNone(entry.watch, 'must not silently fallback on every write')

    def test_same_mode_directory_chmod_does_not_invalidate(self):
        entry = self.warm()
        for directory in entry.watch.directory_identity:
            directory.chmod(directory.stat().st_mode & 0o7777)
        with patch.object(entry, 'build', side_effect=AssertionError('cold build')):
            self.assertIsNone(self.scan())
        self.assertIsNotNone(entry.watch)

    def test_other_process_orphan_without_new_writer_protocol(self):
        self.warm()
        m = copy.deepcopy(self.seed.metadata)
        m['id'] = str(uuid.uuid4()); m['idempotency_key'] = 'old-writer'
        path = v2_root() / event_relative_path(self.p['folder_name'], m['id'], m['recorded_at'])
        subprocess.run([sys.executable, '-B', '-c',
                        'import pathlib,sys;pathlib.Path(sys.argv[1]).write_text(sys.stdin.read())', str(path)],
                       input=serialize_event(m, 'OLD WRITER FIRST'), text=True, check=True)
        self.assertEqual(self.scan('old-writer').body, 'OLD WRITER FIRST')

    def test_new_month_backdated_source_forces_full_safe_rebuild(self):
        self.warm()
        m = copy.deepcopy(self.seed.metadata);m['id'] = str(uuid.uuid4())
        m['idempotency_key'] = 'backdated'; m['recorded_at'] = '2020-01-01T00:00:00Z'
        path = v2_root() / event_relative_path(self.p['folder_name'], m['id'], m['recorded_at'])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(serialize_event(m, 'BACKDATED'))
        self.assertEqual(self.scan('backdated').body, 'BACKDATED')

    def test_overflow_or_loss_forces_full_scan(self):
        entry = self.warm()
        path, m = self.make('lost', 'RECOVER')
        with patch.object(entry.watch, 'changes', side_effect=WatchInvalidated('overflow')):
            self.assertEqual(self.scan('lost').body, 'RECOVER')
        self.assertIsNone(entry.watch)
        self.assertEqual(self.scan('lost').body, 'RECOVER')
        self.assertIsNotNone(entry.watch)

    def test_resource_limit_fallback_preserves_duplicates(self):
        self.make('seed', 'DUPLICATE')
        with patch.object(SourceWatch, 'MAX_WATCHES', 1):
            with self.assertRaisesRegex(reliability.SourceRecoveryError, 'AMBIGUOUS'):
                self.scan('seed')

    def test_metadata_cap_fallback(self):
        with patch.object(reliability._SourceIndex, 'MAX_METADATA_BYTES', 1):
            self.assertEqual(self.scan('seed').body, 'ORIGINAL BODY')
        self.assertIsNone(next(iter(reliability._INCREMENTAL.values())).watch)

    def test_corruption_during_warmup_never_cached(self):
        path, m = self.make('other')
        path.write_text('unclassifiable damage')
        with self.assertRaisesRegex(reliability.SourceRecoveryError, 'CORRUPT'):
            self.scan('seed')
        path.write_text(serialize_event(m, 'REPAIRED'))
        self.assertEqual(self.scan('other').body, 'REPAIRED')

    def test_lru_eviction_closes_watch_and_cold_start_recovers(self):
        entry = self.warm(); descriptor = entry.watch.fd
        with patch.object(reliability, '_INCREMENTAL_LIMIT', 1):
            p = dict(self.p);p['project_id'] = 'another';p['folder_name'] = 'another'
            self.assertIsNone(reliability.find_orphan(p, 'absent', lambda e: True, set()))
        self.assertIsNone(entry.watch)
        self.assertEqual(self.scan('seed').body, 'ORIGINAL BODY')


@unittest.skipUnless(sys.platform == 'linux', 'real inotify requires Linux')
class NotificationDecoderTests(unittest.TestCase):
    def test_overflow_event_is_not_silently_dropped(self):
        import tempfile, struct
        with tempfile.TemporaryDirectory() as temp:
            watch = SourceWatch(temp)
            try:
                with patch('os.read', return_value=struct.pack('iIII', -1, 0x4000, 0, 0)):
                    with self.assertRaises(WatchInvalidated):
                        watch.changes()
            finally:
                watch.close()

class IncrementalIndexUnitTests(FullFallbackTests):
    """Index logic using explicit notifications, not proof of kernel delivery."""
    def setUp(self):
        super().setUp()
        class FakeWatch:
            def __init__(self, root, tree_root=None):
                self.pending = set();self.closed = False
            def add(self, path, directory=False):
                pass
            def changes(self):
                changes = self.pending;self.pending = set();return changes
            def close(self):
                self.closed = True
        self.watch_patch = patch.object(reliability, 'SourceWatch', FakeWatch)
        self.watch_patch.start()
    def tearDown(self):
        self.watch_patch.stop()
        super().tearDown()
    # Inherited filesystem notification tests apply only to a real watcher.
    # Only methods explicitly defined below use this fake backend.
    def test_index_delta_and_no_repeated_parse(self):
        self.scan(); entry = next(iter(reliability._INCREMENTAL.values()))
        with patch.object(reliability, '_source_metadata', wraps=reliability._source_metadata) as read:
            self.assertIsNone(self.scan())
            self.assertEqual(read.call_count, 0)
            path,m=self.make('delta','FIRST');entry.watch.pending.add(path)
            self.assertEqual(self.scan('delta').body,'FIRST')
            self.assertEqual(read.call_count,1)
    def test_index_overflow_fallback_and_rewarm(self):
        self.scan();entry=next(iter(reliability._INCREMENTAL.values()))
        path,m=self.make('lost','FIRST')
        with patch.object(entry.watch,'changes',side_effect=WatchInvalidated('overflow')):
            self.assertEqual(self.scan('lost').body,'FIRST')
        self.assertIsNone(entry.watch)
        self.assertEqual(self.scan('lost').body,'FIRST')
        self.assertIsNotNone(entry.watch)
    def test_resource_cooldown_always_scans_actual_source(self):
        self.scan();entry=next(iter(reliability._INCREMENTAL.values()));entry.close()
        with patch.object(reliability, 'SourceWatch', side_effect=WatchUnavailable('capacity')) as factory:
            self.scan()
            path,m=self.make('during-cooldown','NEW')
            self.assertEqual(self.scan('during-cooldown').body,'NEW')
            self.assertEqual(factory.call_count,1)
    def test_unreported_change_seen_at_periodic_verification(self):
        self.scan();entry=next(iter(reliability._INCREMENTAL.values()))
        path,m=self.make('out-of-band','BODY')
        entry.last_full_verify -= 61
        self.assertEqual(self.scan('out-of-band').body,'BODY')
    def test_duplicate_and_commit_delete_not_cached(self):
        self.scan();entry=next(iter(reliability._INCREMENTAL.values()))
        path,m=self.make('delta');entry.watch.pending.add(path)
        self.assertIsNone(reliability.find_orphan(self.p,'delta',lambda e:False,set()))
        self.assertEqual(self.scan('delta').event_id,m['id'])
        self.assertIsNone(self.scan('delta',{m['id']}))
        path2,m2=self.make('delta');entry.watch.pending.add(path2)
        with self.assertRaisesRegex(reliability.SourceRecoveryError,'AMBIGUOUS'):self.scan('delta')

# Do not inherit notification-dependent methods into the simulated backend.
for _name in list(dir(IncrementalIndexUnitTests)):
    if _name.startswith('test_') and _name not in IncrementalIndexUnitTests.__dict__:
        setattr(IncrementalIndexUnitTests, _name, None)

class NotificationDecoderUnitTests(unittest.TestCase):
    """Exercise parsing/loss/replacement using synthetic kernel-format records."""
    def watch(self, temp):
        w=SourceWatch.__new__(SourceWatch)
        w.fd=123;w.paths={};w.watches={};w.directory_identity={}
        w.tree_root=Path(temp)/'events';w.mount_fingerprint=b'mounts'
        return w
    def decode(self,w,wd,mask,name=''):
        import struct
        raw=name.encode()+b'\0' if name else b''
        record=struct.pack('iIII',wd,mask,0,len(raw))+raw
        with patch.object(SourceWatch,'_mount_fingerprint',return_value=b'mounts'):
            with patch('os.read',side_effect=[record,BlockingIOError()]):return w.changes()
    def test_overflow_and_mount_change(self):
        w=self.watch('/tmp')
        with self.assertRaises(WatchInvalidated):self.decode(w,-1,0x4000)
        with patch.object(SourceWatch,'_mount_fingerprint',return_value=b'changed'):
            with self.assertRaises(WatchInvalidated):w.changes()
    def test_known_path_replace_invalidates_even_if_inode_survives(self):
        w=self.watch('/tmp');p=w.tree_root/'2026'/'01';f=p/'source.md'
        w.watches={1:{(p,True)}};w.paths={p:1,f:2}
        with self.assertRaises(WatchInvalidated):self.decode(w,1,0x80,'source.md')
    def test_sibling_tree_change_ignored(self):
        w=self.watch('/tmp');w.watches={1:{(Path('/tmp'),True)}};w.paths={Path('/tmp'):1}
        self.assertEqual(self.decode(w,1,0x100|w.ISDIR,'unrelated'),set())
    def test_same_mode_directory_attrib_ignored(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            w=self.watch(temp);p=Path(temp);s=p.stat()
            w.watches={1:{(p,True)}};w.paths={p:1}
            w.directory_identity={p:(s.st_dev,s.st_ino,s.st_mode,s.st_uid,s.st_gid)}
            self.assertEqual(self.decode(w,1,0x4|w.ISDIR),set())
