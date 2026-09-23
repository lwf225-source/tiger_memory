from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from tiger_memory.database import connect_readonly, db_path
from tiger_memory.model import PLMError, v2_root
from tiger_memory.passages import validate_passage
from tiger_memory.service import (
    context_data, evidence_data, forget, rebuild_index, submit_candidate, write_config,
    write_event, write_fact,
)
from tests.common import MemoryTestCase


class PassageRecoveryTests(MemoryTestCase):
    QUERY = "离线索引重建"

    def _corpus(self):
        body = "User: 离线索引重建只读取本地原文，不允许联网。\n" + "\n\n".join(
            "第 %d 段：保留原始出处与条件，不得把推断写成已确认事实。" % number
            for number in range(100)
        ) + "\nAssistant: 离线索引重建完成前不得改用失效来源。"
        episode = write_event(self.project, "离线索引重建原始会话", body)
        fact = write_fact(
            self.project, "retrieval.mode", "离线索引重建仅使用本地原文",
            source_event_id=episode.event_id, conditions={"联网": "不允许"},
        )
        derived = write_event(
            self.project, "离线索引重建派生流程", "离线索引重建先核对原始来源；条件不满足时不得使用。",
            kind="procedure", extra={"source_event_id": fact.event_id},
        )
        private = write_event(
            self.project, "私有离线索引重建", "离线索引重建仅属于指定私有会话。",
            scope="session", scope_id="session-recovery",
        )
        other_root = self.project.parent / "独立项目"
        other_root.mkdir()
        other = write_event(other_root, "其他项目离线索引重建", "离线索引重建的这份来源只属于其他项目。")
        return (episode, fact, derived), private, other_root, other

    def _references(self, cwd=None, scope="project", scope_id=""):
        _, results = context_data(
            cwd or self.project, self.QUERY, limit=12, view="current", scope=scope,
            scope_id=scope_id, retrieval_profile="passages",
        )
        return {item.source_event_id: item.evidence_spans for item in results}

    def _source_files(self):
        database = db_path()
        excluded = {database, Path(str(database) + "-wal"), Path(str(database) + "-shm")}
        return {
            str(path.relative_to(v2_root())): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in v2_root().rglob("*") if path.is_file() and path not in excluded
        }

    def _remove_only_temporary_index(self):
        database = db_path()
        self.assertEqual(database.parent.resolve(), (self.base / "v2").resolve())
        self.assertTrue(database.is_file())
        # No broad cleanup: exactly the isolated fixture's index and SQLite
        # sidecars are discarded after all connections have been closed.
        for target in (database, Path(str(database) + "-wal"), Path(str(database) + "-shm")):
            target.resolve(strict=False).relative_to(self.base.resolve())
            self.assertFalse(target.is_symlink())
            target.unlink(missing_ok=True)
        self.assertFalse(database.exists())

    def _expand_exact(self, span):
        return evidence_data(
            self.project, span["parent_id"], span["body_sha256"], span["start"], span["end"],
            before_chars=0, after_chars=0, max_chars=len(span["text"]),
        )

    def _logical_rows(self):
        conn = connect_readonly()
        try:
            return {
                table: [tuple(row) for row in conn.execute("SELECT * FROM " + table + " ORDER BY " + key)]
                for table, key in (("events", "event_id"), ("candidates", "candidate_id"), ("usage_runs", "run_id"))
            }
        finally:
            conn.close()

    def _cli(self, *arguments):
        script = Path(__file__).resolve().parents[1] / "scripts" / "tiger_memory.py"
        environment = dict(os.environ, TIGER_MEMORY_DIR=str(self.base), PYTHONDONTWRITEBYTECODE="1")
        return subprocess.run(
            [sys.executable, str(script), *arguments], cwd=str(self.project), env=environment,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, check=False,
        )

    def test_source_only_rebuild_preserves_passage_identity_offsets_and_current_isolation(self):
        events, private, other_root, other = self._corpus()
        before = self._references()
        private_before = self._references(scope="session", scope_id="session-recovery")
        other_before = self._references(other_root)
        self.assertEqual(set(before), {event.event_id for event in events})
        self.assertEqual(set(private_before), {private.event_id})
        self.assertEqual(set(other_before), {other.event_id})
        for event in events:
            self.assertTrue(before[event.event_id])
            for span in before[event.event_id]:
                self.assertEqual(validate_passage(event.body, span, event.event_id), span)
        sources = self._source_files()
        self._remove_only_temporary_index()
        rebuilt = rebuild_index()
        self.assertTrue(rebuilt["swapped"])
        self.assertEqual(rebuilt["errors"], [])
        self.assertEqual(rebuilt["indexed"], 5)
        self.assertEqual(self._source_files(), sources)
        self.assertEqual(self._references(), before)
        self.assertEqual(self._references(scope="session", scope_id="session-recovery"), private_before)
        self.assertEqual(self._references(other_root), other_before)
        for spans in before.values():
            for span in spans:
                self.assertEqual(self._expand_exact(span), span)

    def test_retraction_survives_source_only_rebuild_and_rejects_old_descendant_pointers(self):
        events, private, other_root, other = self._corpus()
        references = self._references()
        self.assertEqual(set(references), {event.event_id for event in events})
        forget(self.project, events[0].event_id)
        self.assertEqual(self._references(), {})
        sources = self._source_files()
        self._remove_only_temporary_index()
        rebuilt = rebuild_index()
        self.assertTrue(rebuilt["swapped"])
        self.assertEqual(rebuilt["errors"], [])
        self.assertEqual(self._source_files(), sources)
        self.assertEqual(self._references(), {})
        self.assertEqual(set(self._references(scope="session", scope_id="session-recovery")), {private.event_id})
        self.assertEqual(set(self._references(other_root)), {other.event_id})
        for spans in references.values():
            for span in spans:
                with self.subTest(parent_id=span["parent_id"]), self.assertRaises(PLMError):
                    self._expand_exact(span)
        self.assertTrue(all(event.path.is_file() for event in events))

    def test_real_cli_context_to_evidence_expands_without_event_candidate_or_usage_writes(self):
        events, _, _, _ = self._corpus()
        write_config({"engine": "v2", "write_mode": "v2", "auto_consolidate": False})
        submit_candidate(self.project, {
            "title": "未审核草稿", "summary": "这份候选仅用于检查读取操作不会改变审核队列。",
            "kind": "episode", "evidence_kind": "agent_inference",
        }, "passage-e2e-pending")
        initial = self._logical_rows()
        context = self._cli("context", "--cwd", str(self.project), "--query", self.QUERY,
                            "--retrieval-profile", "passages", "--view", "current", "--json")
        self.assertEqual(context.returncode, 0, context.stderr)
        result = json.loads(context.stdout)
        selected = next(item for item in result["results"] if item["source_event_id"] == events[0].event_id)
        span = selected["evidence_spans"][0]
        after_context = self._logical_rows()
        self.assertEqual(after_context["events"], initial["events"])
        self.assertEqual(after_context["candidates"], initial["candidates"])
        # Context intentionally logs one invocation; only the subsequent
        # evidence expansion is promised to be completely usage-write-free.
        self.assertEqual(len(after_context["usage_runs"]), len(initial["usage_runs"]) + 1)
        sources = self._source_files()
        evidence = self._cli(
            "evidence", "--cwd", str(self.project), "--event-id", span["parent_id"],
            "--body-sha256", span["body_sha256"], "--start", str(span["start"]), "--end", str(span["end"]),
            "--before-chars", "80", "--after-chars", "80", "--max-chars", "1200",
        )
        self.assertEqual(evidence.returncode, 0, evidence.stderr)
        expanded = json.loads(evidence.stdout)
        self.assertEqual(validate_passage(events[0].body, expanded, events[0].event_id), expanded)
        self.assertLessEqual(expanded["start"], span["start"])
        self.assertGreaterEqual(expanded["end"], span["end"])
        self.assertLessEqual(len(expanded["text"]), 1200)
        self.assertEqual(self._logical_rows(), after_context)
        self.assertEqual(self._source_files(), sources)
