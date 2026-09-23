"""The default suite validates the harness without requiring CodeGraph installed."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import evaluate_code_memory as evaluation


class CodeMemoryEvaluationTests(unittest.TestCase):
    def test_utf8_budget_includes_headers_for_every_arm(self):
        for arm in evaluation.ARMS:
            packet = evaluation.evidence_packet(
                arm, "历史证据" * 1000, {"status": "ready", "text": "代码证据" * 1000}, 513)
            self.assertLessEqual(len(packet["reader_context"].encode("utf-8")), 513)
            self.assertEqual(packet["context_bytes"], len(packet["reader_context"].encode("utf-8")))
            self.assertNotIn("\ufffd", packet["reader_context"])

    def test_failed_code_text_is_not_delivered_or_scored(self):
        packet = evaluation.evidence_packet(
            "combined", "historical reason", {"status": "stale", "text": "return 1"}, 1000)
        score = evaluation.score_packet(packet, [
            {"id": "current", "source": "code", "all_of": ["return 1"]}])
        self.assertNotIn("return 1", packet["reader_context"])
        self.assertFalse(packet["code_text_exposed"])
        self.assertEqual(score["covered"], 0)
        self.assertEqual(score["semantic_quality"], "not_measured")

    def test_coverage_requires_all_markers_in_the_correct_source(self):
        packet = evaluation.evidence_packet(
            "combined", "return 1", {"status": "ready", "text": "reason"}, 1000)
        score = evaluation.score_packet(packet, [
            {"id": "current", "source": "code", "all_of": ["return 1"]},
            {"id": "reason", "source": "memory", "all_of": ["reason"]}])
        self.assertEqual(score["covered"], 0)

    def test_invalid_budget_and_arm_rejected(self):
        for arm, budget in (("unknown", 1000), ("combined", 511), ("combined", 64001)):
            with self.assertRaises(ValueError):
                evaluation.evidence_packet(arm, "", {}, budget)

    def test_harness_uses_isolated_repositories_and_restores_memory_environment(self):
        indexed = {}
        roots = set()

        def sync(root, **kwargs):
            roots.add(root)
            indexed[str(root)] = (root / "app.py").read_text()
            return {"status": "ready", "index_health": {"engine_version": "test-double"}}

        def query(root, text, **kwargs):
            if evaluation.shutil.which("codegraph") is None:
                return {"status": "unavailable"}
            source = (root / "app.py").read_text()
            if source != indexed[str(root)]:
                return {"status": "stale"}
            return {"status": "ready", "text": source}

        with tempfile.TemporaryDirectory() as parent:
            original = str(Path(parent) / "user-memory-do-not-create")
            with mock.patch.dict(os.environ, {"TIGER_MEMORY_DIR": original}), \
                    mock.patch.object(evaluation.shutil, "which", return_value="/fake/codegraph"), \
                    mock.patch.object(evaluation.codegraph, "sync", side_effect=sync), \
                    mock.patch.object(evaluation.codegraph, "query", side_effect=query), \
                    mock.patch.object(evaluation, "plm_main", wraps=evaluation.plm_main) as product_entry:
                report = evaluation.evaluate()
                self.assertEqual(sum("--with-code" in call.args[0] for call in product_entry.call_args_list), 4)
                self.assertEqual(os.environ["TIGER_MEMORY_DIR"], original)
                self.assertFalse(Path(original).exists())
        self.assertEqual(len(report["records"]), 12)
        self.assertTrue(report["operational_checks_passed"])
        self.assertTrue(all(record["freshness"]["status_matches"] for record in report["records"]))
        self.assertFalse(any(record["freshness"]["unsafe_code_exposed"] for record in report["records"]))
        self.assertTrue(all(not root.exists() for root in roots))
        decision = [record for record in report["records"]
                    if record["case_id"] == "decision-and-changed-code" and record["arm"] == "combined"][0]
        self.assertEqual(decision["metrics"]["covered"], 2)
        self.assertEqual(report["semantic_quality"], "not_measured")


if __name__ == "__main__":
    unittest.main()
