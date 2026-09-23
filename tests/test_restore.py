from __future__ import annotations

import io
import json
import os
import tarfile
from pathlib import Path

from tiger_memory.database import connect
from tiger_memory.model import PLMError
from tiger_memory.ops import backup_memory, restore_backup
from tiger_memory.paths import restored_legacy_path
from tiger_memory.privacy import ledger_path, read_ledger
from tiger_memory.service import doctor, forget, migrate_legacy, rebuild_index, write_event, write_fact
from tests.common import MemoryTestCase, write_legacy


class RestoreTests(MemoryTestCase):
    def test_backup_then_purge_restore_with_latest_ledger_does_not_resurrect(self):
        original = write_event(self.project, "删除来源", "需要清除的原始知识")
        derived = write_fact(self.project, "build.command", "npm run build", source_event_id=original.event_id)
        kept = write_event(self.project, "保留来源", "应当保留的恢复知识")
        backup = backup_memory()
        self.assertTrue(Path(backup["deletion_ledger"]).is_file())
        self.assertEqual(read_ledger(ledger_path())["deletions"], [])
        forget(self.project, original.event_id, purge=True)
        source_ledger = ledger_path()
        destination = Path(self.temp.name).resolve() / "restored"
        previous = os.environ["TIGER_MEMORY_DIR"]
        report = restore_backup(Path(backup["archive"]), destination, source_ledger)
        self.assertTrue(report["healthy"])
        self.assertFalse(report["active_store_changed"])
        self.assertEqual(os.environ["TIGER_MEMORY_DIR"], previous)
        self.assertTrue(kept.path.is_file())
        os.environ["TIGER_MEMORY_DIR"] = str(destination)
        try:
            conn = connect()
            self.assertIsNone(conn.execute("SELECT 1 FROM events WHERE event_id=?", (original.event_id,)).fetchone())
            self.assertIsNone(conn.execute("SELECT 1 FROM events WHERE event_id=?", (derived.event_id,)).fetchone())
            self.assertIsNone(conn.execute("SELECT 1 FROM facts WHERE assertion_event_id=?", (derived.event_id,)).fetchone())
            self.assertIsNotNone(conn.execute("SELECT 1 FROM events WHERE event_id=?", (kept.event_id,)).fetchone())
            for row in conn.execute("SELECT source_path FROM events"):
                self.assertTrue(Path(row[0]).is_file())
                self.assertTrue(str(row[0]).startswith(str(destination)))
            conn.close()
            self.assertTrue(doctor()["healthy"])
            self.assertTrue(rebuild_index()["swapped"])
            self.assertTrue(doctor()["healthy"])
            raw = "\n".join(path.read_text(encoding="utf-8") for path in (destination / "v2" / "projects").glob("*/events/*/*/*.md"))
            self.assertNotIn("需要清除的原始知识", raw)
        finally:
            os.environ["TIGER_MEMORY_DIR"] = previous

    def test_missing_ledger_is_rejected_without_destination_or_environment_change(self):
        write_event(self.project, "备份", "恢复前必须提供账本")
        backup = backup_memory()
        destination = Path(self.temp.name) / "restored"
        previous = os.environ["TIGER_MEMORY_DIR"]
        for missing in (None, Path(self.temp.name) / "missing-ledger.json"):
            with self.assertRaises(PLMError):
                restore_backup(Path(backup["archive"]), destination, missing)
            self.assertFalse(destination.exists())
            self.assertEqual(os.environ["TIGER_MEMORY_DIR"], previous)

    def test_archive_traversal_links_and_special_files_are_rejected(self):
        backup = backup_memory()
        destination = Path(self.temp.name) / "restored"
        archive = Path(self.temp.name) / "malicious.tar"
        for kind in ("traversal", "absolute", "symlink", "hardlink", "fifo"):
            with tarfile.open(str(archive), "w") as bundle:
                good = tarfile.TarInfo("v2/safe.txt")
                good.size = 4
                bundle.addfile(good, io.BytesIO(b"safe"))
                name = "../escaped.txt" if kind == "traversal" else "/absolute.txt" if kind == "absolute" else "v2/unsafe"
                bad = tarfile.TarInfo(name)
                if kind in {"symlink", "hardlink"}:
                    bad.type = tarfile.SYMTYPE if kind == "symlink" else tarfile.LNKTYPE
                    bad.linkname = "../../escaped.txt"
                elif kind == "fifo":
                    bad.type = tarfile.FIFOTYPE
                bundle.addfile(bad)
            with self.assertRaises(PLMError):
                restore_backup(archive, destination, Path(backup["deletion_ledger"]))
            self.assertFalse(destination.exists())
            self.assertFalse((Path(self.temp.name) / "escaped.txt").exists())
            self.assertFalse(list(Path(self.temp.name).glob(".plm-restore-*")))

    def test_restore_never_overwrites_active_or_nonempty_store(self):
        write_event(self.project, "现有数据", "必须保持原样")
        backup = backup_memory()
        other = Path(self.temp.name) / "occupied"
        other.mkdir()
        sentinel = other / "user-data.txt"
        sentinel.write_text("preserve", encoding="utf-8")
        for destination in (self.base, self.base / "nested", self.base.parent, other):
            with self.assertRaises(PLMError):
                restore_backup(Path(backup["archive"]), destination, Path(backup["deletion_ledger"]))
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")
        self.assertTrue(doctor()["healthy"])

    def test_ledger_traversal_cannot_delete_an_external_file(self):
        write_event(self.project, "备份", "安全恢复")
        backup = backup_memory()
        outside = Path(self.temp.name) / "preserve.txt"
        outside.write_text("preserve", encoding="utf-8")
        malicious = Path(self.temp.name) / "malicious-ledger.json"
        malicious.write_text(json.dumps({"schema_version": 1, "deletions": [{
            "project_id": "fixture", "event_ids": [], "legacy_relative_paths": ["../../preserve.txt"],
        }]}), encoding="utf-8")
        with self.assertRaises(PLMError):
            restore_backup(Path(backup["archive"]), Path(self.temp.name) / "restored", malicious)
        self.assertEqual(outside.read_text(encoding="utf-8"), "preserve")

    def test_restored_legacy_purge_affects_only_restored_store(self):
        original_path = write_legacy(self.base, self.project, "旧格式知识", "遗留笔记恢复和清除").resolve()
        migrated = migrate_legacy()
        self.assertEqual(migrated["migrated"], 1)
        backup = backup_memory()
        destination = Path(self.temp.name).resolve() / "restored"
        restore_backup(Path(backup["archive"]), destination, Path(backup["deletion_ledger"]))
        previous = os.environ["TIGER_MEMORY_DIR"]
        os.environ["TIGER_MEMORY_DIR"] = str(destination)
        try:
            expected = destination / original_path.relative_to(self.base.resolve())
            self.assertEqual(restored_legacy_path(str(original_path)), str(expected))
            self.assertTrue(rebuild_index()["swapped"])
            conn = connect()
            row = conn.execute("SELECT event_id,source_legacy_path FROM events WHERE source='legacy_import'").fetchone()
            conn.close()
            self.assertEqual(row["source_legacy_path"], str(expected))
            forget(self.project, row["event_id"], purge=True)
            self.assertFalse(expected.exists())
            self.assertTrue(original_path.exists())
            self.assertTrue(doctor()["healthy"])
        finally:
            os.environ["TIGER_MEMORY_DIR"] = previous

    def test_historical_archive_without_origin_metadata_can_be_restored(self):
        original = write_legacy(self.base, self.project, "历史归档", "旧归档不含根目录元数据").resolve()
        migrate_legacy()
        backup = backup_memory()
        old_archive = Path(self.temp.name) / "historical.tar"
        with tarfile.open(backup["archive"], "r:*") as source, tarfile.open(str(old_archive), "w") as target:
            for member in source:
                if member.name in {"v2/backup-metadata.json", "v2/restore-origin.json", "v2/privacy/deletions.json"}:
                    continue
                target.addfile(member, source.extractfile(member) if member.isfile() else None)
        destination = Path(self.temp.name).resolve() / "restored"
        self.assertTrue(restore_backup(old_archive, destination, Path(backup["deletion_ledger"]))["healthy"])
        previous = os.environ["TIGER_MEMORY_DIR"]
        os.environ["TIGER_MEMORY_DIR"] = str(destination)
        try:
            self.assertEqual(restored_legacy_path(str(original)), str(destination / original.relative_to(self.base.resolve())))
            self.assertTrue(rebuild_index()["swapped"])
        finally:
            os.environ["TIGER_MEMORY_DIR"] = previous
