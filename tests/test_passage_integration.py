import json
from unittest.mock import patch

from tiger_memory.context_pack import pack_context
from tiger_memory.database import connect
from tiger_memory.model import PLMError
from tiger_memory.passages import validate_passage
from tiger_memory.search import search
from tiger_memory.service import forget, write_event, write_fact
from tests.common import MemoryTestCase
from tests.test_retrieval import RecordingEmbedding


class PassageIntegrationTests(MemoryTestCase):
    def retrieve(self, project_id, query, **kwargs):
        conn = connect()
        try:
            return search(conn, project_id, query, profile="passages", **kwargs)
        finally:
            conn.close()

    def test_late_evidence_exact_spans_rebuildable_and_budgeted(self):
        body = ("普通背景，不涉及本次操作。\n\n" * 190 +
                "## user\n只有离线验收通过后才能切换缓存；目前尚未通过，不得切换。\n")
        event = write_event(self.project, "长记录", body)
        results = self.retrieve(event.project_id, "离线验收通过后缓存可以切换吗")
        self.assertEqual(results[0].ref_id, event.event_id)
        for span in results[0].evidence_spans:
            self.assertEqual(validate_passage(event.body, span, event.event_id), span)
        self.assertTrue(any("不得切换" in span["text"] for span in results[0].evidence_spans))
        self.assertEqual(results[0].evidence_spans, self.retrieve(event.project_id, "离线验收通过后缓存可以切换吗")[0].evidence_spans)
        packed = pack_context({"project_id": event.project_id}, self.project, "缓存", results, token_budget=6000)
        self.assertIn("不得切换", packed)
        self.assertIn("chars=[", packed)
        self.assertLessEqual(len(packed.encode()), 6000)
        for budget in (0, 20, 200, 700, 1800):
            small = pack_context({"project_id": event.project_id}, self.project, "缓存", results, token_budget=budget)
            self.assertLessEqual(len(small.encode()), budget)
        # Forged provenance must fail, never silently emit a misleading pointer.
        results[0].evidence_spans[0]["start"] += 1
        with self.assertRaises(ValueError):
            pack_context({"project_id": event.project_id}, self.project, "缓存", results, token_budget=6000)

    def test_eligibility_before_chunking_and_model(self):
        event = write_event(self.project, "公开", "完全离线运行")
        write_event(self.project, "隐藏", "私有会话正文", scope="session", scope_id="secret")
        other = self.project.parent / "other"
        other.mkdir()
        write_event(other, "隐藏", "其他项目正文")
        old = write_event(self.project, "撤回", "撤回正文")
        forget(self.project, old.event_id)
        future = write_fact(self.project, "engine", "未来事实正文", valid_from="2020-01-01T00:00:00Z")
        write_event(self.project, "依赖", "未来派生正文", extra={"source_event_id": future.event_id})
        conn = connect()
        conn.execute("UPDATE facts SET valid_from='2099-01-01T00:00:00.000000Z' WHERE fact_id=?", (future.metadata["fact_id"],))
        conn.commit()
        conn.close()
        provider = RecordingEmbedding()
        with patch("tiger_memory.search.split_passages", wraps=__import__("tiger_memory.passages", fromlist=["split_passages"]).split_passages) as splitter:
            results = self.retrieve(event.project_id, "断网", embedding_provider=provider)
        seen = "\n".join(provider.seen + [call.args[0] for call in splitter.call_args_list])
        for forbidden in ("私有会话正文", "其他项目正文", "撤回正文", "未来事实正文", "未来派生正文"):
            self.assertNotIn(forbidden, seen)
        self.assertEqual([result.ref_id for result in results], [event.event_id])
        forget(self.project, event.event_id)
        self.assertEqual(self.retrieve(event.project_id, "离线"), [])

    def test_unsupported_reranker_and_private_scope_fail_explicitly(self):
        event = write_event(self.project, "说明", "缓存工作方式")
        with self.assertRaises(PLMError):
            self.retrieve(event.project_id, "缓存", reranker=object())
        with self.assertRaises(ValueError):
            self.retrieve(event.project_id, "缓存", scope="session")

    def test_work_limit_is_not_silent_partial_search(self):
        event = write_event(self.project, "说明", "缓存工作方式")
        record = {"body": "x" * 8_000_001}
        with patch("tiger_memory.search._eligible_records", return_value={("event", event.event_id): record}), patch("tiger_memory.search._scoped_aliases", return_value=("缓存", [])):
            with self.assertRaisesRegex(PLMError, "work limit"):
                self.retrieve(event.project_id, "缓存")

    def test_long_parent_cannot_crowd_out_other_parents(self):
        long = write_event(self.project, "long", "alpha beta " * 20000)
        short = write_event(self.project, "short", "alpha")
        results = self.retrieve(long.project_id, "alpha beta")
        self.assertIn(short.event_id, [result.ref_id for result in results])
        self.assertLessEqual(len(results[0].evidence_spans), 2)

    def test_exact_fact_key_keeps_effective_current_label(self):
        original = write_fact(self.project, "cache.engine", "SQLite", valid_from="2020-01-01T00:00:00Z")
        write_fact(self.project, "cache.engine", "FutureCache", valid_from="2099-01-01T00:00:00Z", supersedes=original.metadata["fact_id"])
        result = self.retrieve(original.project_id, "cache.engine")[0]
        self.assertEqual(result.ref_id, original.metadata["fact_id"])
        self.assertEqual(result.status, "superseded")
        self.assertIn("effective-current", result.reasons)
