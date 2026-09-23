from __future__ import annotations

import json
import os
import sqlite3
import unittest
import uuid
from pathlib import Path
from unittest import mock

from tiger_memory.database import connect
from tiger_memory.events import atomic_write, parse_event, serialize_event
from tiger_memory.model import PLMError, display_slug, memory_base, project_id_for_root, v2_root
from tiger_memory.ops import export_v1
from tiger_memory.search import search
from tiger_memory.security import SecretDetected
from tiger_memory.service import (
    add_project_alias, consolidate, context_data, fact_history, forget, rebuild_index,
    render_context, review_candidate, submit_candidate, supersede_fact, usage_report, write_event, write_fact,
)
from tests.common import MemoryTestCase


class CoreTests(MemoryTestCase):
    def test_renamed_store_variable_preserves_legacy_store(self):
        legacy = self.base.parent / "legacy-store"
        with mock.patch.dict(os.environ, {"PROJECT_LONG_MEMORY_DIR": str(legacy)}):
            os.environ.pop("TIGER_MEMORY_DIR", None)
            self.assertEqual(memory_base(), legacy.resolve())
            os.environ["TIGER_MEMORY_DIR"] = str(self.base)
            self.assertEqual(memory_base(), self.base.resolve())

    def test_project_identity_avoids_chinese_slug_collision(self):
        one = self.project.parent / "肉鸽游戏"
        two = self.project.parent / "舆情推演系统"
        self.assertNotEqual(project_id_for_root(one), project_id_for_root(two))
        self.assertNotEqual(display_slug(one), display_slug(two))
        self.assertIn("肉鸽游戏", display_slug(one))

    def test_write_search_and_idempotency(self):
        first = write_event(
            self.project, "微信发布检查", "已运行构建检查，当前仍是测试版本。",
            tags="微信,发布", idempotency_key="same-operation",
        )
        second = write_event(
            self.project, "不会重复", "另一段内容", tags="重复",
            idempotency_key="same-operation",
        )
        self.assertEqual(first.event_id, second.event_id)
        self.assertEqual(parse_event(first.path).metadata["title"], "微信发布检查")
        project, results = context_data(self.project, "微信发布检查", 5, "current")
        self.assertEqual(results[0].ref_id, first.event_id)
        self.assertTrue(all(result.project_id == project["project_id"] for result in results))
        session = write_event(
            self.project, "会话私有", "只属于 session-1", scope="session", scope_id="session-1",
        )
        _, default_scope = context_data(self.project, "会话私有", 5, "current")
        _, session_scope = context_data(self.project, "会话私有", 5, "current", "session", "session-1")
        self.assertFalse(any(item.ref_id == session.event_id for item in default_scope))
        self.assertTrue(any(item.ref_id == session.event_id for item in session_scope))

    def test_unrelated_query_returns_no_result(self):
        event = write_event(
            self.project, "微信发布检查", "已运行构建检查，当前仍是测试版本。", tags="微信,发布",
        )
        _, results = context_data(self.project, "春节酒店机票预订", 5, "current")
        self.assertFalse(any(item.ref_id == event.event_id for item in results))

    def test_secret_rejection(self):
        with self.assertRaises(SecretDetected):
            write_event(self.project, "bad", "api_key=abcdef123456")
        with self.assertRaises(SecretDetected):
            write_event(self.project, "bad", "-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----")

    def test_strict_metadata_and_symlink_defense(self):
        event = write_event(self.project, "safe", "ordinary content")
        bad = dict(event.metadata)
        bad["project_override"] = "evil"
        with self.assertRaises(PLMError):
            serialize_event(bad, event.body)
        outside = Path(self.temp.name) / "outside"
        outside.mkdir()
        link = v2_root() / "linked"
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(outside, target_is_directory=True)
        with self.assertRaises(PLMError):
            atomic_write(link / "escape.md", "no")

    def test_fact_supersede_history_and_forget(self):
        evidence = self.confirmation("release.status", "测试中")
        old = write_fact(
            self.project, "release.status", "测试中", title="版本状态", tags="发布",
            evidence_kind="user_confirmation", evidence=evidence,
        )
        old_fact = old.metadata["fact_id"]
        new = supersede_fact(
            self.project, old_fact, "已验证", evidence_kind="user_confirmation",
            evidence=self.confirmation("release.status", "已验证"),
        )
        history = fact_history(self.project, "release.status")
        self.assertEqual([row["status"] for row in history], ["superseded", "active"])
        conn = connect()
        current = search(conn, old.project_id, "版本状态 已验证", 5, "current")
        historical = search(conn, old.project_id, "版本状态 测试中", 5, "history")
        conn.close()
        self.assertTrue(any(item.ref_id == new.metadata["fact_id"] for item in current))
        self.assertTrue(any(item.ref_id == old_fact for item in historical))
        forget(self.project, new.metadata["fact_id"])
        conn = connect()
        after = search(conn, old.project_id, "已验证", 5, "current")
        residue = conn.execute("SELECT COUNT(*) FROM vectors WHERE ref_id=?", (new.metadata["fact_id"],)).fetchone()[0]
        conn.close()
        self.assertFalse(any(item.ref_id == new.metadata["fact_id"] for item in after))
        self.assertEqual(residue, 0)

    def test_candidate_policy_and_entities(self):
        high = submit_candidate(self.project, {
            "title": "发布状态", "summary": "据推测已经发布", "kind": "episode",
            "facts": [{"fact_key": "release.status", "value": "已发布"}],
            "evidence_kind": "agent_inference",
        })
        result = consolidate()
        self.assertEqual(result["states"].get("needs_confirmation"), 1)
        approved = review_candidate(
            self.project, high["candidate_id"], "approve", "user_confirmation",
            self.confirmation("release.status", "已发布"),
        )
        self.assertEqual(approved["states"].get("active"), 1)
        low = submit_candidate(self.project, {
            "title": "构建流程", "summary": "运行本地 verify 命令。", "kind": "procedure",
            "facts": [{"fact_key": "build.command", "value": "npm run verify"}],
            "entities": [{"name": "verify", "type": "command", "aliases": ["检查命令"]}],
            "evidence_kind": "command_output",
            "evidence": self.output_evidence(),
        })
        result = consolidate()
        self.assertEqual(result["states"].get("active"), 1)
        conn = connect()
        aliases = conn.execute("SELECT COUNT(*) FROM entity_aliases").fetchone()[0]
        candidates = conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
        conn.close()
        self.assertGreaterEqual(aliases, 2)
        self.assertEqual(candidates, 2)
        rebuilt = rebuild_index()
        self.assertEqual(rebuilt["errors"], [])
        conn = connect()
        self.assertGreaterEqual(conn.execute("SELECT COUNT(*) FROM entity_aliases").fetchone()[0], 2)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0], 2)
        conn.close()

    def test_project_alias_survives_rebuild(self):
        event = write_event(self.project, "alias", "path move support")
        moved = self.project.parent / "移动后的项目"
        result = add_project_alias(self.project, moved)
        self.assertEqual(result["project_id"], event.project_id)
        rebuild_index()
        conn = connect()
        alias = conn.execute("SELECT project_id FROM project_aliases WHERE alias_path=?", (str(moved.resolve()),)).fetchone()
        conn.close()
        self.assertEqual(alias["project_id"], event.project_id)

    def test_scope_and_cross_project_mutation_guards(self):
        session = write_event(
            self.project, "会话记录", "只属于私有会话", scope="session", scope_id="session-1",
        )
        _, recent = context_data(self.project, "", 5, "current")
        self.assertFalse(any(item.ref_id == session.event_id for item in recent))
        with self.assertRaises(PLMError):
            context_data(self.project, "会话记录", 5, "current", "session", "")

        other = self.project.parent / "另一个项目"
        other.mkdir()
        fact = write_fact(self.project, "ordinary.phase", "one")
        with self.assertRaises(PLMError):
            supersede_fact(other, fact.metadata["fact_id"], "two")
        with self.assertRaises(PLMError):
            forget(other, session.event_id)

    def test_rebuild_preserves_retraction_order(self):
        original = write_event(
            self.project, "顺序测试", "应保持撤回",
            event_id="ffffffff-ffff-4fff-8fff-ffffffffffff",
            recorded_at="2026-01-01T00:00:00.100000Z",
        )
        write_event(
            self.project, "撤回顺序测试", "稍后撤回",
            event_id="00000000-0000-4000-8000-000000000000",
            recorded_at="2026-01-01T00:00:00.200000Z",
            extra={"operation": "retract", "target_id": original.event_id},
        )
        report = rebuild_index()
        self.assertTrue(report["swapped"])
        self.assertEqual(report["errors"], [])
        conn = connect()
        status = conn.execute("SELECT status FROM events WHERE event_id=?", (original.event_id,)).fetchone()[0]
        conn.close()
        self.assertEqual(status, "retracted")

    def test_purge_removes_source_and_derived_content(self):
        event = write_event(
            self.project, "清除测试", "这段隐私正文必须消失",
            extra={"entities": [{"name": "清除实体", "aliases": ["清除别名"]}]},
        )
        forget(self.project, event.event_id, purge=True)
        self.assertFalse(event.path.exists())
        conn = connect()
        self.assertIsNone(conn.execute("SELECT 1 FROM events WHERE event_id=?", (event.event_id,)).fetchone())
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0], 0)
        conn.close()
        destination = Path(self.temp.name) / "export"
        export_v1(destination)
        exported = "\n".join(path.read_text(encoding="utf-8") for path in destination.rglob("*.md"))
        self.assertNotIn("这段隐私正文必须消失", exported)

    def test_immutable_event_and_protected_metadata(self):
        event = write_event(
            self.project, "不可变", "原始正文", event_id="11111111-1111-4111-8111-111111111111",
            recorded_at="2026-01-01T00:00:00Z",
        )
        with self.assertRaises(PLMError):
            write_event(
                self.project, "不可变", "覆盖正文", event_id=event.event_id,
                recorded_at="2026-01-01T00:00:00Z",
            )
        with self.assertRaises(PLMError):
            write_event(self.project, "坏元数据", "普通正文", extra={"scope": "global"})
        with self.assertRaises(SecretDetected):
            write_event(
                self.project, "秘密元数据", "普通正文",
                extra={"evidence": {"api_key": "abcdef123456"}},
            )

    def test_fact_validity_single_current_and_evidence(self):
        with self.assertRaises(PLMError):
            write_fact(self.project, "release.status", "已发布")
        future = write_fact(
            self.project, "future.phase", "未来才生效", valid_from="2099-01-01T00:00:00Z",
        )
        conn = connect()
        current = search(conn, future.project_id, "future.phase 未来才生效", 5, "current")
        historical = search(conn, future.project_id, "future.phase 未来才生效", 5, "history")
        conn.close()
        self.assertFalse(any(item.ref_id == future.metadata["fact_id"] for item in current))
        self.assertTrue(any(item.ref_id == future.metadata["fact_id"] for item in historical))
        with self.assertRaises(PLMError):
            write_fact(self.project, "future.phase", "另一个当前值")

    def test_usage_audit_stores_no_query_text_and_survives_rebuild(self):
        write_event(self.project, "调用统计", "统计写入")
        secret_query = "只用于查询审计的原始问题"
        render_context(self.project, secret_query, 3)
        report = usage_report(1)
        operations = {item["operation"]: item["runs"] for item in report["summary"]}
        self.assertGreaterEqual(operations.get("write", 0), 1)
        self.assertGreaterEqual(operations.get("context", 0), 1)
        conn = connect()
        before = conn.execute("SELECT COUNT(*) FROM usage_runs").fetchone()[0]
        columns = [row[1] for row in conn.execute("PRAGMA table_info(usage_runs)")]
        conn.close()
        self.assertNotIn("query", columns)
        self.assertNotIn(secret_query, json.dumps(report, ensure_ascii=False))
        rebuild_index()
        conn = connect()
        after = conn.execute("SELECT COUNT(*) FROM usage_runs").fetchone()[0]
        conn.close()
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
