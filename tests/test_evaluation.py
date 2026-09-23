from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tiger_memory import evaluation as ev
from tiger_memory.database import connect_readonly
from tiger_memory.model import PLMError


FIXTURES = Path(__file__).resolve().parents[1] / "evaluation" / "synthetic-v1"


def tiny_sources():
    return [
        {"source_id": "a", "project": "one", "split": "dev", "title": "取样", "body": "取样前关闭风机。", "scope": "project", "scope_id": ""},
        {"source_id": "b", "project": "one", "split": "dev", "title": "收尾", "body": "取样后再次开启风机。", "scope": "project", "scope_id": ""},
        {"source_id": "c", "project": "one", "split": "dev", "title": "试验", "body": "私人试验风机保持开启。", "scope": "session", "scope_id": "private"},
    ]


def tiny_questions():
    return [
        {"question_id": "q1", "project": "one", "split": "dev", "question": "取样前后风机怎样操作？", "answer": "先关后开", "category": "cross_event", "required_source_ids": ["a", "b"], "unanswerable": False, "scope": "project", "scope_id": ""},
        {"question_id": "q2", "project": "one", "split": "dev", "question": "设备报废金额？", "answer": "", "category": "unanswerable", "required_source_ids": [], "unanswerable": True, "scope": "project", "scope_id": ""},
    ]


class EvaluationTests(unittest.TestCase):
    def test_frozen_split_and_source_separation(self):
        dev = ev.load_dataset(FIXTURES, "dev")
        heldout = ev.load_dataset(FIXTURES, "heldout")
        self.assertEqual((len(dev["questions"]), len(heldout["questions"])), (40, 40))
        self.assertEqual(len({source["project"] for source in dev["sources"]}), 4)
        self.assertFalse({source["source_id"] for source in dev["sources"]} & {source["source_id"] for source in heldout["sources"]})
        self.assertFalse({source["project"] for source in dev["sources"]} & {source["project"] for source in heldout["sources"]})
        self.assertEqual(set(dev["questions"][0]) & ev.SOURCE_FIELDS, {"project", "split", "scope", "scope_id"})

    def test_ingest_boundary_has_no_gold_and_restores_environment(self):
        with tempfile.TemporaryDirectory() as untouched:
            sentinel = Path(untouched) / "sentinel"
            sentinel.write_text("untouched", encoding="utf-8")
            with patch.dict(os.environ, {"TIGER_MEMORY_DIR": untouched}):
                with ev.isolated_memory() as root:
                    ingest = ev.ingest_sources(root, tiny_sources())
                    conn = connect_readonly()
                    rows = conn.execute("SELECT title,body,source FROM events").fetchall()
                    conn.close()
                    self.assertEqual(ingest["source_count"], 3)
                    text = json.dumps([dict(row) for row in rows], ensure_ascii=False)
                    self.assertNotIn(tiny_questions()[0]["question"], text)
                    self.assertNotIn("required_source_ids", text)
                    self.assertEqual({row["body"] for row in rows}, {source["body"] for source in tiny_sources()})
                self.assertEqual(os.environ["TIGER_MEMORY_DIR"], untouched)
                self.assertEqual([path.name for path in Path(untouched).iterdir()], ["sentinel"])

    def test_ingest_rejects_gold_fields(self):
        bad = dict(tiny_sources()[0], has_answer=True)
        with ev.isolated_memory() as root:
            with self.assertRaises(PLMError):
                ev.ingest_sources(root, [bad])

    def test_fixture_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            ev.write_dataset(folder, tiny_sources(), tiny_questions(), {"dataset_id": "tiny", "origin": "synthetic", "splits": ["dev"]})
            self.assertEqual(len(ev.load_dataset(folder, "dev")["questions"]), 2)
            (folder / "sources.json").write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(PLMError, "fingerprint mismatch"):
                ev.load_dataset(folder, "dev")

    def test_cross_split_project_rejected(self):
        source = dict(tiny_sources()[0], source_id="d", split="heldout")
        with tempfile.TemporaryDirectory() as directory:
            ev.write_dataset(Path(directory), tiny_sources() + [source], tiny_questions(), {"dataset_id": "bad", "origin": "synthetic", "splits": ["dev", "heldout"]})
            with self.assertRaisesRegex(PLMError, "more than one split"):
                ev.load_dataset(Path(directory), "dev")

    def test_metrics_denominators_and_partial_evidence(self):
        sources = {source["source_id"]: source for source in tiny_sources()}
        partial = ev.score_evidence(tiny_questions()[0], ["a", "a"], ["a"], sources)
        self.assertEqual(partial["evidence_recall"], .5)
        self.assertFalse(partial["full_evidence_coverage"])
        absent = ev.score_evidence(tiny_questions()[1], ["a"], ["a"], sources)
        combined = ev._aggregate([partial, absent])
        self.assertEqual(combined["evidence_recall_macro"], .5)
        self.assertEqual(combined["unanswerable_false_retrieval_rate"], 1)
        self.assertEqual(combined["full_evidence_coverage"], 0)

    def test_scope_and_budget_failures_are_separate(self):
        sources = {source["source_id"]: source for source in tiny_sources()}
        question = tiny_questions()[0]
        packed = ev.pack_evidence(question, ["a", "b", "c"], sources, 200)
        self.assertLessEqual(len(packed["context"].encode("utf-8")), 200)
        self.assertLessEqual(packed["budget_units"], 200)
        result = ev.score_evidence(question, ["a", "b"], [], sources)
        self.assertIn("context_budget_drop", result["failures"])
        self.assertIn("scope_error", ev.score_evidence(question, ["c"], ["c"], sources)["failures"])
        self.assertEqual(ev.pack_evidence(question, ["a"], sources, 1)["context"], "")

    def test_raw_scope_filter(self):
        question = tiny_questions()[0]
        self.assertNotIn("c", ev.raw_episode_search(tiny_sources(), question, 10))
        scoped = dict(question, scope="session", scope_id="private")
        self.assertEqual(ev.raw_episode_search(tiny_sources(), scoped, 10), ["c"])

    def test_heldout_report_does_not_expose_cases_or_reader_claims(self):
        report = ev.run_evaluation(FIXTURES, "heldout", ["none", "provider"], repeats=1)
        self.assertEqual(report["qa"]["status"], "not_measured")
        self.assertFalse(report["official_benchmark_score"])
        self.assertNotIn("cases", report["results"]["none"])
        self.assertEqual(report["results"]["none"]["overall"]["evidence_recall_macro"], 0)
        text = json.dumps(report, ensure_ascii=False)
        for question in ev.load_dataset(FIXTURES, "heldout")["questions"]:
            self.assertNotIn(question["question"], text)
            self.assertNotIn(question["question_id"], text)
        self.assertEqual(report["results"]["provider"]["status"], "not_measured")

    def test_longmemeval_adapter_strips_all_labels(self):
        record = {"question_id": "official1", "question_type": "single-session-user", "question": "What did I order?", "answer": "tea", "question_date": "2023/05/01", "haystack_session_ids": ["s1"], "haystack_dates": ["2023/04/01"], "haystack_sessions": [[{"role": "user", "content": "I ordered tea.", "has_answer": True}, {"role": "assistant", "content": "Noted.", "has_answer": False}]], "answer_session_ids": ["s1"]}
        adapted = ev.adapt_longmemeval([record], "fixture-revision")
        source = adapted["sources"][0]
        self.assertFalse(set(source) - ev.SOURCE_FIELDS)
        self.assertNotIn("has_answer", json.dumps(source))
        self.assertNotIn(record["question"], source["body"])
        self.assertIn("I ordered tea.", source["body"])
        self.assertEqual(adapted["questions"][0]["required_source_ids"], [source["source_id"]])
        self.assertEqual(adapted["questions"][0]["gold_answer_turns"][0]["turn_index"], 0)
        absent = ev.adapt_longmemeval([dict(record, question_id="official_abs")], "fixture-revision")
        self.assertEqual(absent["questions"][0]["required_source_ids"], [])
        with tempfile.TemporaryDirectory() as directory:
            ev.write_dataset(Path(directory), adapted["sources"], adapted["questions"], adapted["metadata"])
            self.assertEqual(len(ev.load_dataset(Path(directory), "official")["sources"]), 1)

    def test_reader_exact_match_requires_identity_and_penalizes_missing(self):
        metadata = {"reader_id": "fixed-test-reader", "revision": "1", "prompt_sha256": "a" * 64, "decoding": {"temperature": 0}, "dataset_fingerprint": "frozen-fixture"}
        result = ev.score_reader_predictions(tiny_questions(), [{"question_id": "q1", "hypothesis": "先关后开"}], metadata)
        self.assertEqual(result["normalized_exact_match"], .5)
        self.assertEqual(result["missing_predictions"], 1)
        self.assertEqual(result["semantic_qa_accuracy"]["status"], "not_measured")
        with self.assertRaises(PLMError):
            ev.score_reader_predictions(tiny_questions(), [], {})
        with self.assertRaises(PLMError):
            ev.score_reader_predictions(tiny_questions(), [{"question_id": "unknown", "hypothesis": "x"}], metadata)

    def test_duplicate_dataset_export_cannot_overwrite_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata = {"dataset_id": "tiny", "origin": "synthetic", "splits": ["dev"]}
            ev.write_dataset(Path(directory), tiny_sources(), tiny_questions(), metadata)
            with self.assertRaisesRegex(PLMError, "already exists"):
                ev.write_dataset(Path(directory), [], [], metadata)

    def test_model_fallback_is_degraded_and_components_are_separate(self):
        class StubProvider:
            model_id = "synthetic-test-only"
            def describe(self):
                return {"artifact_fingerprint": "a" * 64, "revision": "fixed", "cache_hits": 12}
        calls = []
        def fallback_search(conn, project_id, query, limit=5, view="current", scope="project", scope_id="", profile="lexical", embedding_provider=None, reranker=None, query_provider=None, diagnostics=None):
            calls.append((embedding_provider is not None, reranker is not None, query_provider is not None))
            diagnostics.update({"embedding": {"status": "unavailable", "fallback_reason": "missing-local-model"}, "reranker": {"status": "ready"}, "query_rewrite": {"status": "ready"}})
            return []
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            ev.write_dataset(folder, tiny_sources(), tiny_questions(), {"dataset_id": "tiny", "origin": "synthetic", "splits": ["dev"]})
            with patch.object(ev, "search", fallback_search):
                report = ev.run_evaluation(folder, "dev", ["embedding", "reranker", "provider", "recall-template"], repeats=1, embedding_provider=StubProvider(), reranker=StubProvider(), query_provider=StubProvider())
            self.assertEqual(report["results"]["embedding"]["status"], "degraded")
            self.assertEqual(report["results"]["reranker"]["status"], "measured")
            self.assertEqual(report["results"]["provider"]["status"], "degraded")
            self.assertEqual(report["results"]["recall-template"]["status"], "degraded")
            self.assertEqual(report["results"]["recall-template"]["requested_components"], ["embedding", "query_rewrite"])
            self.assertEqual(set(calls), {(True, False, False), (False, True, False), (True, True, False), (True, False, True)})
            self.assertEqual(report["configuration"]["embedding_provider"]["artifact_fingerprint"], "a" * 64)
            self.assertNotIn("cache_hits", report["configuration"]["embedding_provider"])

    def test_recall_template_requires_profile_capable_search(self):
        class StubProvider:
            def describe(self):
                return {"status": "ready"}

        # A legacy-compatible callable lacks the profile argument.  The
        # template baseline must report this as unmeasured rather than invoke
        # it with unsupported keyword arguments.
        def legacy_search(conn, project_id, query, limit=5, view="current", scope="project", scope_id=""):
            return []

        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            ev.write_dataset(folder, tiny_sources(), tiny_questions(),
                             {"dataset_id": "tiny", "origin": "synthetic", "splits": ["dev"]})
            with patch.object(ev, "search", legacy_search):
                report = ev.run_evaluation(folder, "dev", ["recall-template"], repeats=1,
                                           embedding_provider=StubProvider(), query_provider=StubProvider())
        result = report["results"]["recall-template"]
        self.assertEqual(result["status"], "not_measured")
        self.assertEqual(result["reason"], "search profile interface unavailable")


if __name__ == "__main__":
    unittest.main()
