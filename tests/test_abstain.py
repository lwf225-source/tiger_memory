"""Abstention floor tests (batch 7): opt-in relevance threshold + empty-set signal.

The switch is OFF by default and must be byte-identical to previous behavior;
when on, candidates without an absolute-similarity channel at/above the floor
are withheld, and a fully withheld query returns an empty list plus the
``abstain.abstained`` diagnostics flag.
"""

from __future__ import annotations

import json
from pathlib import Path

from tiger_memory import aml_adapter
from tiger_memory.cli import main as cli_main
from tiger_memory.database import connect
from tiger_memory.search import search
from tiger_memory.service import write_event
from tests.common import MemoryTestCase


class _AxisEmbedding:
    """Deterministic 2-d embedding: query on [1,0]; documents containing 手机
    sit at cosine 0.8, everything else at cosine 0.0 (excluded by the >0 floor)."""

    def describe(self):
        return {"provider": "axis-embedding", "status": "ready", "dimensions": 2,
                "model_id": "fake", "artifact_fingerprint": "fake-hash"}

    def embed(self, texts):
        output = []
        for index, text in enumerate(texts):
            if index == 0:
                output.append([1.0, 0.0])  # query axis (embed([query] + docs) call shape)
            else:
                output.append([0.6, 0.8] if "手机" in text else [0.0, 1.0])
        return output


class AbstainThresholdTests(MemoryTestCase):
    def _write(self, title, body, day=1, **kwargs):
        return write_event(self.project, title, body,
                           recorded_at="2026-03-%02dT00:00:00Z" % day, **kwargs)

    def _project_id(self):
        conn = connect()
        try:
            return conn.execute("SELECT project_id FROM projects").fetchone()[0]
        finally:
            conn.close()

    def _search(self, query, **kwargs):
        conn = connect()
        try:
            return search(conn, self._project_id(), query, **kwargs)
        finally:
            conn.close()

    def test_disabled_matches_default_exactly(self):
        self._write("目标记录", "哨兵词 出现在这里。")
        default = self._search("哨兵词")
        disabled = self._search("哨兵词", abstain_threshold=0.0)
        self.assertEqual([item.as_dict() for item in default], [item.as_dict() for item in disabled])
        info = {}
        self._search("哨兵词", diagnostics=info)
        self.assertEqual(info["abstain"], {"status": "disabled"})

    def test_invalid_threshold_rejected(self):
        self._write("目标记录", "哨兵词 出现在这里。")
        for bad in (True, False, -0.1, 1.1, float("nan"), float("inf"), "0.5"):
            with self.assertRaises(ValueError, msg=repr(bad)):
                self._search("任意", abstain_threshold=bad)

    def test_threshold_filters_weak_candidates(self):
        strong = self._write("手机购买记录", "手机 预算 哨兵甲。")
        weak = self._write("远足记录", "山野 徒步 哨兵乙。")
        results = self._search("手机 预算", embedding_provider=_AxisEmbedding(),
                               abstain_threshold=0.5, limit=10)
        ids = [item.ref_id for item in results]
        self.assertIn(strong.event_id, ids)
        self.assertNotIn(weak.event_id, ids)
        info = {}
        self._search("手机 预算", embedding_provider=_AxisEmbedding(),
                     abstain_threshold=0.5, limit=10, diagnostics=info)
        self.assertEqual(info["abstain"]["status"], "enabled")
        self.assertEqual(info["abstain"]["threshold"], 0.5)
        self.assertFalse(info["abstain"]["abstained"])

    def test_all_below_threshold_abstains_with_empty_result(self):
        self._write("手机购买记录", "手机 预算 哨兵甲。")
        info = {}
        results = self._search("手机 预算", embedding_provider=_AxisEmbedding(),
                               abstain_threshold=0.99, diagnostics=info)
        self.assertEqual(results, [])
        self.assertEqual(info["abstain"]["status"], "enabled")
        self.assertTrue(info["abstain"]["abstained"])
        self.assertEqual(info["abstain"]["kept"], 0)
        self.assertGreater(info["abstain"]["candidates"], 0)

    def test_no_candidates_also_abstains(self):
        self._write("手机购买记录", "手机 预算 哨兵甲。")
        info = {}
        results = self._search("量子比特 超导", abstain_threshold=0.5, diagnostics=info)
        self.assertEqual(results, [])
        self.assertTrue(info["abstain"]["abstained"])

    def test_legacy_profile_ignores_threshold(self):
        self._write("目标记录", "哨兵词 出现在这里。")
        info = {}
        results = self._search("哨兵词", profile="legacy", abstain_threshold=0.99, diagnostics=info)
        self.assertTrue(results)
        self.assertEqual(info["abstain"], {"status": "ignored-by-legacy-profile"})

    def test_combines_with_adjacent_and_recency(self):
        hit = self._write("目标记录", "哨兵词丙 目标。", day=2)
        self._write("相邻记录", "相邻正文完全无关。", day=3)
        # Surviving hit keeps its adjacent supplement.
        results = self._search("哨兵词丙", abstain_threshold=0.0, adjacent=1, recency_boost=True)
        self.assertTrue(any("adjacent-context" in item.reasons for item in results))
        # Full abstention withholds supplements too.
        info = {}
        results = self._search("哨兵词丙", abstain_threshold=0.99, adjacent=1,
                               recency_boost=True, diagnostics=info)
        self.assertEqual(results, [])
        self.assertTrue(info["abstain"]["abstained"])
        self.assertNotIn(hit.event_id, [item.ref_id for item in results])

    def test_passages_profile_filter(self):
        self._write("手机购买记录", "手机 预算 哨兵甲。" * 5)
        info = {}
        results = self._search("手机 预算", profile="passages", embedding_provider=_AxisEmbedding(),
                               abstain_threshold=0.99, diagnostics=info)
        self.assertEqual(results, [])
        self.assertEqual(info["abstain"]["status"], "enabled")
        self.assertTrue(info["abstain"]["abstained"])
        # Below the floor the passage hit survives.
        info2 = {}
        kept = self._search("手机 预算", profile="passages", embedding_provider=_AxisEmbedding(),
                            abstain_threshold=0.5, diagnostics=info2)
        self.assertTrue(kept)
        self.assertFalse(info2["abstain"]["abstained"])

    def test_cli_json_end_to_end(self):
        self._write("手机购买记录", "手机 预算 哨兵甲。")
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            cli_main(["context", "--cwd", str(self.project), "--query", "手机 预算",
                      "--json", "--abstain-threshold", "0.99"])
        payload = json.loads(buffer.getvalue())
        self.assertEqual(payload["results"], [])
        # Default CLI invocation (switch off) still returns the hit.
        buffer2 = io.StringIO()
        with redirect_stdout(buffer2):
            cli_main(["context", "--cwd", str(self.project), "--query", "手机 预算", "--json"])
        self.assertTrue(json.loads(buffer2.getvalue())["results"])


class AdapterAbstainTests(MemoryTestCase):
    def setUp(self):
        super().setUp()
        self._previous = aml_adapter.AML_ABSTAIN_THRESHOLD

    def tearDown(self):
        aml_adapter.AML_ABSTAIN_THRESHOLD = self._previous
        super().tearDown()

    def _add_and_search(self, query):
        aml_adapter.add({
            "request_id": "r1", "user_id": "u-abstain", "session_id": "s1",
            "messages": [{"role": "user", "content": "手机 预算 哨兵甲。", "timestamp": 1704067200000}],
        })
        return aml_adapter.search({"query": query, "user_id": "u-abstain", "top_k": 10})

    def test_adapter_abstention_signal_shape(self):
        aml_adapter.AML_ABSTAIN_THRESHOLD = 0.99
        response = self._add_and_search("手机 预算")
        self.assertEqual(response["data"], [])
        self.assertIs(response["plm_abstained"], True)
        self.assertEqual(response["plm_abstain"]["status"], "enabled")
        self.assertTrue(response["plm_abstain"]["abstained"])

    def test_adapter_threshold_off_returns_normally(self):
        aml_adapter.AML_ABSTAIN_THRESHOLD = 0.0
        response = self._add_and_search("手机 预算")
        self.assertTrue(response["data"])
        self.assertNotIn("plm_abstained", response)

    def test_adapter_surviving_hits_carry_no_flag(self):
        aml_adapter.AML_ABSTAIN_THRESHOLD = 0.05
        response = self._add_and_search("手机 预算")
        self.assertTrue(response["data"])
        self.assertNotIn("plm_abstained", response)
