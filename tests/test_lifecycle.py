from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from unittest.mock import patch

from tiger_memory import database
from tiger_memory.database import connect
from tiger_memory.events import atomic_write, serialize_event
from tiger_memory.model import PLMError, v2_root
from tiger_memory.privacy import ledger_path
from tiger_memory.service import (consolidate, context_data, doctor, forget,
    rebuild_index, submit_candidate, supersede_fact, write_event, write_fact)
from tests.common import MemoryTestCase


class LifecycleTests(MemoryTestCase):
    def test_startup_includes_current_fact_and_json_source_event(self):
        fact = write_fact(self.project, "build.mode", "local")
        write_event(self.project, "最近事件", "recent filler")
        _, results = context_data(self.project, "", 1)
        self.assertEqual(results[0].ref_id, fact.metadata["fact_id"])
        self.assertEqual(results[0].as_dict()["source_event_id"], fact.event_id)

    def test_durable_marker_is_not_overwritten_by_retry_after_sql_loss(self):
        candidate = submit_candidate(self.project, {"summary": "source committed", "facts": [{"fact_key": "mode", "value": "local"}]})
        consolidate()
        marker = v2_root() / "batches" / (candidate["candidate_id"] + ".json")
        original = marker.read_bytes()
        conn = connect()
        conn.execute("DELETE FROM memory_fts")
        conn.execute("DELETE FROM vectors")
        conn.execute("DELETE FROM facts")
        conn.execute("DELETE FROM events")
        conn.execute("UPDATE candidates SET state='pending'")
        conn.commit()
        conn.close()
        result = consolidate()
        self.assertEqual(result["states"], {"pending": 1})
        self.assertEqual(marker.read_bytes(), original)
        self.assertEqual(len(list((v2_root() / "projects").glob("*/events/*/*/*.md"))), 2)
        self.assertTrue(rebuild_index()["swapped"])
        self.assertEqual(consolidate()["processed"], 0)

    def test_missing_committed_marker_blocks_rebuild_and_doctor(self):
        candidate = submit_candidate(self.project, {"summary": "committed summary"})
        consolidate()
        (v2_root() / "batches" / (candidate["candidate_id"] + ".json")).unlink()
        self.assertFalse(rebuild_index()["swapped"])
        self.assertFalse(doctor()["healthy"])

    def test_purge_candidate_fact_removes_summary_and_no_sql_cursor_lock(self):
        candidate = submit_candidate(self.project, {"summary": "独特隐私片段的摘要", "facts": [{"fact_key": "profile.preference", "value": "独特隐私片段"}]})
        consolidate()
        conn = connect()
        target = conn.execute("SELECT fact_id FROM facts").fetchone()[0]
        conn.close()
        forget(self.project, target, purge=True)
        self.assertFalse((v2_root() / "candidates" / (candidate["candidate_id"] + ".json")).exists())
        self.assertTrue(rebuild_index()["swapped"])
        conn = connect()
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM events WHERE body LIKE '%独特隐私片段%'").fetchone()[0], 0)
        conn.close()

    def test_batch_last_invalid_fact_leaves_no_events(self):
        submit_candidate(self.project, {"summary": "不可部分激活", "facts": [
            {"fact_key": "build.mode", "value": "local"},
            {"fact_key": "build.mode", "value": "cloud"}]})
        self.assertEqual(consolidate()["states"], {"rejected": 1})
        conn = connect()
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM events").fetchone()[0], 0)
        conn.close()
        self.assertEqual(list((v2_root() / "projects").glob("*/events/*/*/*.md")), [])
        self.assertEqual(rebuild_index()["indexed"], 0)

    def test_batch_io_failure_does_not_leave_committed_prefix(self):
        submit_candidate(self.project, {"summary": "完整批次", "facts": [{"fact_key": "mode", "value": "local"}]})
        from tiger_memory.service import atomic_write as real_write
        count = [0]
        def fail_second(path, text):
            if path.suffix == ".md":
                count[0] += 1
                if count[0] == 2:
                    raise OSError("injected private output must not reach logs")
            return real_write(path, text)
        with patch("tiger_memory.service.atomic_write", side_effect=fail_second):
            result = consolidate()
        self.assertNotIn("private output", json.dumps(result))
        self.assertEqual(result["states"], {"rejected": 1})
        self.assertEqual(rebuild_index()["indexed"], 0)

    def test_batch_rebuild_repairs_stale_candidate_status(self):
        candidate = submit_candidate(self.project, {"summary": "完整批次", "facts": [{"fact_key": "mode", "value": "local"}]})
        self.assertEqual(consolidate()["states"], {"active": 1})
        path = v2_root() / "candidates" / (candidate["candidate_id"] + ".json")
        atomic_write(path, json.dumps(candidate))  # crash before candidate file receipt
        self.assertTrue(rebuild_index()["swapped"])
        conn = connect()
        self.assertEqual(conn.execute("SELECT state FROM candidates").fetchone()[0], "active")
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM events").fetchone()[0], 2)
        conn.close()
        self.assertEqual(consolidate()["processed"], 0)

    def test_uncommitted_event_not_replayed(self):
        event = write_event(self.project, "源", "普通原文")
        meta = dict(event.metadata, id=str(uuid.uuid4()), batch_id=str(uuid.uuid4()))
        path = event.path.with_name("uncommitted.md")
        atomic_write(path, serialize_event(meta, "未完成批次"))
        self.assertEqual(rebuild_index()["indexed"], 1)
        self.assertEqual(doctor()["uncommitted_batch_files"], 1)

    def test_fact_identity_scope_and_future_supersession(self):
        old = write_fact(self.project, "build.mode", "本地构建", valid_from="2020-01-01", scope="session", scope_id="one")
        other = write_fact(self.project, "build.mode", "另一会话", scope="session", scope_id="two")
        new = supersede_fact(self.project, old.metadata["fact_id"], "未来云构建", valid_from="2099-01-01")
        self.assertEqual(new.metadata["scope_id"], "one")
        _, current = context_data(self.project, "构建", 5, "current", "session", "one")
        ids = {row.ref_id for row in current}
        self.assertIn(old.metadata["fact_id"], ids)
        self.assertNotIn(new.metadata["fact_id"], ids)
        self.assertNotIn(other.metadata["fact_id"], ids)
        self.assertTrue(doctor()["healthy"])

    def test_candidate_preserves_scope_times_source_and_conditions(self):
        source = write_event(self.project, "约束记录", "仅本地测试，真机未测，不可发布", scope="session", scope_id="s")
        submit_candidate(self.project, {"summary": "条件摘要", "scope": "session", "scope_id": "s", "source_event_id": source.event_id,
            "observed_at": "2024-01-01", "facts": [{"fact_key": "build.mode", "value": "local", "valid_from": "2024-01-02",
            "valid_to": "2099-01-01", "conditions": ["仅本地", "真机未测", "不可发布"]}]})
        # Conditions are retained, not interpreted as proof of a release.
        self.assertEqual(consolidate()["states"], {"active": 1})
        conn = connect()
        rows = conn.execute("SELECT * FROM events WHERE source='agent_candidate'").fetchall()
        self.assertTrue(all(row["scope_id"] == "s" and row["source_event_id"] == source.event_id for row in rows))
        self.assertIn("真机未测", rows[-1]["body"])
        conn.close()
        self.assertTrue(rebuild_index()["swapped"])

    def test_retract_and_purge_follow_transitive_derivations(self):
        source = write_event(self.project, "隐私来源", "删除所有派生的独特内容")
        summary = write_event(self.project, "派生摘要", "删除所有派生的独特内容摘要", extra={"source_event_id": source.event_id,
            "entities": [{"name": "独特实体", "aliases": ["别名"]}]})
        fact = write_fact(self.project, "private.detail", "删除所有派生的独特内容事实", source_event_id=summary.event_id)
        forget(self.project, source.event_id)
        conn = connect()
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0], 0)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0], 0)
        self.assertEqual(conn.execute("SELECT status FROM facts").fetchone()[0], "retracted")
        conn.close()
        with self.assertRaises(PLMError):
            write_fact(self.project, "bad.derived", "cannot resurrect", source_event_id=source.event_id)
        forget(self.project, source.event_id, purge=True)
        self.assertTrue(ledger_path().is_file())
        self.assertFalse(any(event.path.exists() for event in (source, summary, fact)))
        self.assertTrue(rebuild_index()["swapped"])
        self.assertNotIn("独特内容", database.db_path().read_bytes().decode("utf-8", errors="ignore"))

    def test_arbitrary_source_and_hash_do_not_prove_release(self):
        source = write_event(self.project, "推测", "没有实际证据")
        with self.assertRaises(PLMError):
            write_fact(self.project, "release.status", "已发布", source_event_id=source.event_id,
                evidence_kind="git_commit", evidence={"commit": "a" * 40})
