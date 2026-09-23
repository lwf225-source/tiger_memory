from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tiger_memory.evaluation import load_dataset
from tiger_memory.model import PLMError

spec = importlib.util.spec_from_file_location("prepare_longmemeval", Path(__file__).resolve().parents[1] / "scripts" / "prepare_longmemeval.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def synthetic_records():
    records = []
    for category in prepare.CATEGORIES:
        for index in range(5):
            kind = "single-session-user" if category == "abstention" else category
            qid = category + "-" + str(index) + ("_abs" if category == "abstention" else "")
            records.append({
                "question_id": qid, "question_type": kind,
                "question": "QUESTION-CANARY-" + qid, "answer": "GOLD-CANARY-" + qid,
                "question_date": "2023/06/01 (Thu) 10:00",
                "haystack_session_ids": ["answer-session", "filler-session"],
                "haystack_dates": ["2023/01/01 (Sun) 10:00", "2023/02/01 (Wed) 10:00"],
                "haystack_sessions": [
                    [{"role": "user", "content": "SOURCE user message " + str(index), "has_answer": True},
                     {"role": "assistant", "content": "SOURCE assistant message"}],
                    [{"role": "user", "content": "FILLER session must remain"}],
                ],
                "answer_session_ids": ["answer-session"],
            })
    return records


class LongMemEvalPreparationTests(unittest.TestCase):
    def test_selection_is_order_invariant_and_uses_only_ids_and_types(self):
        records = synthetic_records()
        selected = prepare.select_subset(records)
        changed = copy.deepcopy(records)
        for record in changed:
            record["answer"] = "modified gold"
            record["question"] = "modified question"
            record["answer_session_ids"] = ["filler-session"]
            record["haystack_sessions"] = []
        expected_ids = [record["question_id"] for record in selected]
        self.assertEqual(expected_ids, [record["question_id"] for record in prepare.select_subset(changed[::-1])])
        self.assertEqual(len(selected), 21)
        for category in prepare.CATEGORIES:
            self.assertEqual(sum(prepare._category(record) == category for record in selected), 3)

    def test_conversion_retains_all_history_and_separates_gold_from_reader(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input.json"
            source.write_text(json.dumps(synthetic_records()), encoding="utf-8")
            digest = prepare.file_sha256(source)
            output = root / "prepared"
            report = prepare.prepare_dataset(source, output, expected_sha256=digest, upstream_revision="synthetic-test-revision")
            loaded = load_dataset(output, "official")
            self.assertEqual(len(loaded["sources"]), 42)
            self.assertEqual(len(loaded["questions"]), 21)
            self.assertEqual(len({item["project"] for item in loaded["sources"]}), 21)
            source_text = json.dumps(loaded["sources"])
            for canary in ("QUESTION-CANARY", "GOLD-CANARY", "has_answer", "answer_session_ids", "answer-session"):
                self.assertNotIn(canary, source_text)
            self.assertEqual(sum("FILLER session must remain" in item["body"] for item in loaded["sources"]), 21)
            self.assertEqual(sum("SOURCE assistant message" in item["body"] for item in loaded["sources"]), 21)
            reader = json.loads((output / "reader-questions.json").read_text())
            for question in reader:
                self.assertLessEqual(set(question), prepare.READER_QUESTION_FIELDS)
                self.assertNotIn("answer", question)
                self.assertNotIn("required_source_ids", question)
                self.assertNotIn("unanswerable", question)
            self.assertNotIn("GOLD-CANARY", json.dumps(reader))
            for filename, expected in report["output_sha256"].items():
                self.assertEqual(prepare.file_sha256(output / filename), expected)
            self.assertEqual(report["input_sha256"], digest)
            self.assertFalse(report["official_grader_executed"])
            self.assertFalse(report["reader_executed"])
            self.assertEqual(report["variant"], "full_history_non_oracle")

    def test_wrong_input_checksum_never_creates_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "input.json"
            source.write_text("[]", encoding="utf-8")
            output = Path(temporary) / "prepared"
            with self.assertRaises(PLMError):
                prepare.prepare_dataset(source, output, expected_sha256="0" * 64)
            self.assertFalse(output.exists())

    def test_duplicate_ids_or_missing_categories_are_rejected(self):
        records = synthetic_records()
        with self.assertRaises(PLMError):
            prepare.select_subset(records + [records[0]])
        with self.assertRaises(PLMError):
            prepare.select_subset([record for record in records if not record["question_id"].endswith("_abs")])
        with self.assertRaises(PLMError):
            prepare.select_subset(records, 1)

    def test_cached_download_is_verified_without_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "public.json"
            payload = b"synthetic public dataset"
            destination.write_bytes(payload)
            with patch.object(prepare, "DATASET_BYTES", len(payload)), patch.object(prepare, "DATASET_SHA256", hashlib.sha256(payload).hexdigest()), patch.object(prepare.urllib.request, "build_opener", side_effect=AssertionError("unexpected network")):
                self.assertTrue(prepare.download_official(destination)["checksum_verified"])
            with self.assertRaises(PLMError):
                prepare.download_official(destination)
            self.assertEqual(destination.read_bytes(), payload)

    def test_download_redirects_are_https_and_official_hosts_only(self):
        for url in ("https://huggingface.co/path", "https://cas-bridge.xethub.hf.co/path"):
            prepare._safe_download_url(url)
        for url in ("http://huggingface.co/path", "https://example.com/path", "https://huggingface.co.evil.invalid/path", "https://user:password@huggingface.co/path"):
            with self.assertRaises(PLMError):
                prepare._safe_download_url(url)
