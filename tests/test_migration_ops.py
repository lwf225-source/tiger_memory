from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from tiger_memory.database import connect, db_path
from tiger_memory.ops import cutover, install_skill, rollback_skill
from tiger_memory.model import PLMError, v2_root
from tiger_memory.service import doctor, migrate_legacy, migration_plan, rebuild_index, write_config, write_event
from tests.common import MemoryTestCase, write_legacy


class MigrationAndOpsTests(MemoryTestCase):
    def _legacy_fixture(self):
        roots = [self.project.parent / name for name in ("测试甲", "测试乙", "测试丙", "测试丁", "测试戊", "测试己", "测试庚", "测试辛", "测试壬")]
        for root in roots:
            root.mkdir(parents=True, exist_ok=True)
            write_legacy(self.base, root, root.name + " 当前状态", root.name + " 的可检索中文内容", "Users_fixture_Documents_")
        return roots

    def test_collision_migration_and_rebuild(self):
        roots = self._legacy_fixture()
        plan = migration_plan()
        self.assertEqual(plan["source_count"], 9)
        self.assertEqual(len(plan["collision_buckets"]["Users_fixture_Documents_"]), 9)
        report = migrate_legacy()
        self.assertEqual(report["migrated"], 9)
        self.assertEqual(report["failure_count"], 0)
        conn = connect()
        projects = conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
        migrations = conn.execute("SELECT COUNT(*) FROM migrations").fetchone()[0]
        conn.close()
        self.assertEqual(projects, 9)
        self.assertEqual(migrations, 9)
        rebuilt = rebuild_index()
        self.assertEqual(rebuilt["indexed"], 9)
        self.assertEqual(rebuilt["errors"], [])
        self.assertTrue(doctor()["healthy"])

    def test_install_shadow_and_rollback(self):
        target = Path(self.temp.name) / "skills" / "tiger_memory"
        target.mkdir(parents=True)
        (target / "old.txt").write_text("v1", encoding="utf-8")
        backup_root = Path(self.temp.name) / "skill-backups"
        old_target = os.environ.get("PLM_SKILL_DIR")
        old_backup = os.environ.get("PLM_SKILL_BACKUP_DIR")
        os.environ["PLM_SKILL_DIR"] = str(target)
        os.environ["PLM_SKILL_BACKUP_DIR"] = str(backup_root)
        try:
            installed = install_skill(target)
            self.assertEqual(installed["engine"], "shadow")
            self.assertTrue((target / "scripts" / "tiger_memory.py").exists())
            fixture = target / "examples" / "code_memory" / "cases.json"
            self.assertTrue(fixture.is_file())
            self.assertTrue(Path(installed["backup"]).joinpath("old.txt").exists())
            result = subprocess.run(
                [sys.executable, str(target / "scripts" / "tiger_memory.py"), "config"],
                env=dict(os.environ), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["engine"], "shadow")
            second_target = Path(self.temp.name) / "skills" / "reinstalled"
            reinstall_env = dict(os.environ)
            reinstall_env["PLM_INSTALL_VALIDATION"] = "1"
            reinstalled = subprocess.run(
                [sys.executable, str(target / "scripts" / "tiger_memory.py"), "install", "--target", str(second_target)],
                env=reinstall_env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertEqual(reinstalled.returncode, 0, reinstalled.stderr)
            self.assertTrue((second_target / "scripts" / "tiger_memory.py").exists())
            self.assertEqual((second_target / "examples" / "code_memory" / "cases.json").read_bytes(),
                             fixture.read_bytes())
            rolled = rollback_skill(target)
            self.assertTrue((target / "old.txt").exists())
            self.assertIn("recovery_export", rolled)
        finally:
            if old_target is None:
                os.environ.pop("PLM_SKILL_DIR", None)
            else:
                os.environ["PLM_SKILL_DIR"] = old_target
            if old_backup is None:
                os.environ.pop("PLM_SKILL_BACKUP_DIR", None)
            else:
                os.environ["PLM_SKILL_BACKUP_DIR"] = old_backup

    def test_cutover_rejects_stale_acceptance_report(self):
        write_event(self.project, "cutover", "stale report guard")
        write_config({"engine": "shadow", "write_mode": "v1"})
        report_dir = v2_root() / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "acceptance.json").write_text(
            json.dumps({
                "passed": True, "installed_version": "2.0.1", "code_fingerprint": "wrong",
                "event_count": 1,
            }),
            encoding="utf-8",
        )
        with self.assertRaises(PLMError):
            cutover()


if __name__ == "__main__":
    unittest.main()
