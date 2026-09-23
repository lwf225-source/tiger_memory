import json
import unittest
from pathlib import Path
from unittest.mock import patch

from tiger_memory import qa_evaluation as qa
from tests import test_qa_evaluation as fixtures
from tests.test_qa_evaluation import FakeReader, questions


class PassageQATests(unittest.TestCase):
    setUp = fixtures.QAEvaluationTests.setUp
    tearDown = fixtures.QAEvaluationTests.tearDown
    run_fake = fixtures.QAEvaluationTests.run_fake

    def test_passage_citations_bound_to_parent_and_gold_never_forwarded(self):
        reader = FakeReader()
        report = self.run_fake(reader, baselines=["passages"], packing="excerpt")
        for messages in reader.messages:
            payload = json.loads(messages[1]["content"])
            for source in payload["sources"]:
                self.assertTrue(source["source_id"].startswith("psg-"))
                self.assertIn(source["parent_source_id"], {"source-a", "source-b"})
                self.assertNotIn("required_points", source)
        cases = qa._read_jsonl(self.root / "run" / "inputs.jsonl")
        case = cases[0]
        good = {"text": json.dumps({"answer": "先关后开", "abstained": False, "citations": case["citation_source_ids"]})}
        scored = qa.score_case(questions()[0], case, good)
        self.assertTrue(scored["contract_valid"])
        self.assertNotIn("no_gold_source_cited", scored["flags"])
        self.assertEqual(scored["reference_recall"], 1)
        bad = {"text": json.dumps({"answer": "先关后开", "abstained": False, "citations": ["source-a"]})}
        self.assertIn("unknown_citation", qa.score_case(questions()[0], case, bad)["flags"])
        self.assertEqual(report["status"], "complete")

    def test_old_excerpt_offsets_are_conservative(self):
        original = "重复\n重复\n只有通过后才能切换。"
        spans = qa._literal_spans(original, "重复\n…通过后才能…")
        self.assertEqual(len(spans), 1)
        self.assertEqual(original[spans[0]["start"]:spans[0]["end"]], "通过后才能")

    def test_model_required_and_import_rejected_for_passages(self):
        with self.assertRaisesRegex(qa.PLMError, "not configured"):
            self.run_fake(baselines=["passage_embedding"])
        with self.assertRaisesRegex(qa.PLMError, "direct source"):
            self.run_fake(baselines=["passages"], reader_inputs=self.root / "unused.jsonl")
