from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from tiger_memory import evaluation as ev
from tiger_memory import qa_evaluation as qa
from tiger_memory.model import PLMError

spec = importlib.util.spec_from_file_location("prepare_qa_audit", Path(__file__).resolve().parents[1] / "scripts/prepare_qa_audit.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def question_fixture(family):
    categories = ["answerable"] * 8 + ["abstention"] * 2 if family == "project-docs" else [category for category in audit.CATEGORIES for _ in range(2)]
    questions = []
    for index, category in enumerate(categories):
        qid = family + "-" + str(index)
        absent = category == "abstention"
        questions.append({"question_id": qid, "project": qid, "split": audit.SPLITS[family],
            "category": category, "unanswerable": absent, "question": "QUESTION-CANARY-" + qid,
            "answer": "GOLD-CANARY-" + qid, "required_source_ids": [] if absent else ["source-" + qid]})
    return questions


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="plm-audit-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()

    def fixtures(self):
        datasets, runs = {}, {}
        for family, split in audit.SPLITS.items():
            datasets[family] = self.root / (family + "-dataset")
            runs[family] = self.root / (family + "-run")
            questions = question_fixture(family)
            sources = [{"source_id": "source-" + row["question_id"], "project": row["project"], "split": split,
                "title": "Neutral title", "body": "BODY-CANARY-" + row["question_id"]} for row in questions]
            ev.write_dataset(datasets[family], sources, questions, {"dataset_id": family + "-fixture", "origin": "synthetic", "splits": [split]})
            dataset = ev.load_dataset(datasets[family], split)
            by_source = {row["source_id"]: row for row in sources}
            cases = [qa.prepare_case(row, baseline, ["source-" + row["question_id"]], by_source, "grounded", 6000, "full") for row in questions for baseline in audit.BASELINES]
            binding = {"dataset_fingerprint": dataset["fingerprint"], "split": split,
                "prompt_profile": "grounded", "baselines": list(audit.BASELINES),
                "retrieval_provenance": "direct_source_only_retrieval",
                "inputs_fingerprint": ev.fingerprint([qa._binding_case(case) for case in cases])}
            manifest = {"binding": binding, "run_fingerprint": ev.fingerprint(binding)}
            predictions = []
            for case in cases:
                row = {"question_id": case["question_id"], "baseline": case["baseline"], "run_fingerprint": manifest["run_fingerprint"],
                    "messages_sha256": case["messages_sha256"], "sources_sha256": ev.fingerprint(case["sources"]),
                    "text": json.dumps({"answer": "PREDICTION-CANARY", "abstained": False, "citations": case["source_ids"]}),
                    "finish_reason": "stop", "error_type": ""}
                row["record_sha256"] = qa._prediction_checksum(row)
                predictions.append(row)
            runs[family].mkdir()
            (runs[family] / "manifest.json").write_text(json.dumps(manifest))
            (runs[family] / "inputs.jsonl").write_text(qa._lines(cases))
            (runs[family] / "predictions.jsonl").write_text(qa._lines(predictions))
            (runs[family] / "report.json").write_text(json.dumps({"run_fingerprint": manifest["run_fingerprint"], "status": "complete"}))
        selection = self.root / "selection.json"
        audit.freeze_selection(datasets, selection)
        return datasets, runs, selection

    def pack(self):
        datasets, runs, selection = self.fixtures()
        directory = self.root / "pack"
        audit.prepare_pack(selection, runs, directory)
        return directory, runs

    def verdicts(self, directory):
        pack = json.loads((directory / "reviewer-pack.json").read_text())
        value = {"pack_sha256": pack["pack_sha256"], "reviewer_kind": "codex_agent_semantic_review",
            "verdicts": [{"audit_id": row["audit_id"], "semantic_correct": False,
                "evidence_support": "insufficient", "reason": "REASON-CANARY: conclusion lacks support."} for row in pack["entries"]]}
        path = self.root / "verdicts.json"
        path.write_text(json.dumps(value))
        return path

    def test_sampler_is_fixed_and_ignores_text_gold_and_results(self):
        for family in audit.SPLITS:
            questions = question_fixture(family)
            expected = [row["question_id"] for row in audit.select_questions(questions, family)]
            for row in questions:
                row.update(question="changed", answer="changed", required_source_ids=[], prediction="changed", correct=True)
            self.assertEqual(expected, [row["question_id"] for row in audit.select_questions(questions[::-1], family)])
            self.assertEqual(len(expected), 8 if family == "project-docs" else 7)
        with self.assertRaises(PLMError):
            audit.select_questions(question_fixture("project-docs")[:-1], "project-docs")
        with self.assertRaises(PLMError):
            audit.select_questions(question_fixture("longmemeval")[:-2], "longmemeval")

    def test_pack_has_thirty_blinded_source_bound_outputs(self):
        directory, _ = self.pack()
        pack = json.loads((directory / "reviewer-pack.json").read_text())
        mapping = json.loads((directory / "mapping.json").read_text())
        self.assertEqual(len(pack["entries"]), 30)
        self.assertEqual(len({row["audit_id"] for row in pack["entries"]}), 30)
        text = json.dumps(pack)
        self.assertNotIn('"baseline"', text)
        self.assertNotIn('"lexical"', text)
        self.assertNotIn('"embedding"', text)
        for marker in ("QUESTION-CANARY", "GOLD-CANARY", "BODY-CANARY", "PREDICTION-CANARY"):
            self.assertIn(marker, text)
        self.assertEqual(pack["pack_sha256"], mapping["pack_sha256"])
        self.assertEqual((directory / "reviewer-pack.json").stat().st_mode & 0o777, 0o600)
        for row in pack["entries"]:
            self.assertTrue(row["format_contract"]["format_valid"])

    def test_missing_or_changed_predictions_do_not_change_sample(self):
        _, runs, selection = self.fixtures()
        path = runs["project-docs"] / "predictions.jsonl"
        original = path.read_text()
        path.write_text("\n".join(original.splitlines()[:-1]) + "\n")
        with self.assertRaises(PLMError):
            audit.prepare_pack(selection, runs, self.root / "missing")
        path.write_text(original.replace("PREDICTION-CANARY", "CHANGED-PREDICTION"))
        with self.assertRaises(PLMError):
            audit.prepare_pack(selection, runs, self.root / "changed")
        self.assertFalse((self.root / "missing").exists())

    def test_grades_bind_originals_and_summary_is_content_free(self):
        directory, _ = self.pack()
        verdicts = self.verdicts(directory)
        summary = audit.finalize_grades(directory, verdicts, self.root / "grades", "Codex Agent synthetic-test")
        text = json.dumps(summary)
        for marker in ("QUESTION-CANARY", "GOLD-CANARY", "BODY-CANARY", "PREDICTION-CANARY", "REASON-CANARY", '"question_id"'):
            self.assertNotIn(marker, text)
        self.assertEqual(summary["outputs"], 30)
        self.assertFalse(summary["official_benchmark_score"])
        grades = json.loads((self.root / "grades/grades.evaluator-only.json").read_text())["grades"]
        self.assertTrue(all(row["run_fingerprint"] and row["prediction_sha256"] for row in grades))
        self.assertTrue(all(not row["semantic_correct"] and row["format_contract"]["format_valid"] for row in grades))

    def test_incomplete_verdicts_mapping_tamper_and_changed_original_rejected(self):
        directory, runs = self.pack()
        path = self.verdicts(directory)
        value = json.loads(path.read_text())
        missing = copy.deepcopy(value)
        missing["verdicts"].pop()
        path.write_text(json.dumps(missing))
        with self.assertRaises(PLMError):
            audit.finalize_grades(directory, path, self.root / "missing-grades", "Agent")
        path.write_text(json.dumps(value))
        original = runs["project-docs"] / "predictions.jsonl"
        original.write_text(original.read_text().replace("PREDICTION-CANARY", "CHANGED-PREDICTION"))
        with self.assertRaises(PLMError):
            audit.finalize_grades(directory, path, self.root / "changed-grades", "Agent")
        mapping = directory / "mapping.json"
        mapping.write_text(mapping.read_text().replace('"lexical"', '"unknown"'))
        with self.assertRaises(PLMError):
            audit.finalize_grades(directory, path, self.root / "tampered-grades", "Agent")

    def test_cli_public_destination_and_symlinks_rejected(self):
        with self.assertRaises(PLMError):
            audit._private_cli_output(audit.ROOT / "evaluation/public-grades.json")
        with self.assertRaises(PLMError):
            audit._private_cli_output(audit.ROOT / ".codex_work/qa-audit/../public-grades.json")
        target = self.root / "target"
        target.mkdir()
        link = self.root / "link"
        link.symlink_to(target, target_is_directory=True)
        with self.assertRaises(PLMError):
            audit._write_new(link / "file.json", {})

    def test_complete_with_failure_keeps_case_but_partial_is_rejected(self):
        _, runs, selection = self.fixtures()
        path = runs["project-docs"] / "predictions.jsonl"
        rows = qa._read_jsonl(path)
        for row in rows:
            row["finish_reason"] = "length"
            row["record_sha256"] = qa._prediction_checksum(row)
        path.write_text(qa._lines(rows))
        report_path = runs["project-docs"] / "report.json"
        report = json.loads(report_path.read_text())
        report["status"] = "complete_with_reader_failures"
        report_path.write_text(json.dumps(report))
        directory = self.root / "failure-pack"
        audit.prepare_pack(selection, runs, directory)
        verdicts_path = self.verdicts(directory)
        verdicts = json.loads(verdicts_path.read_text())
        for row in verdicts["verdicts"]:
            row["semantic_correct"] = True
        verdicts_path.write_text(json.dumps(verdicts))
        summary = audit.finalize_grades(directory, verdicts_path, self.root / "failure-grades", "Agent fixture")
        self.assertEqual(summary["outputs"], 30)
        for baseline in audit.BASELINES:
            result = summary["results"]["project-docs/" + baseline]
            self.assertEqual(result["generation_failure_count"], 8)
            self.assertEqual(result["semantic_correct_fraction_all_selected"], 1)
            self.assertEqual(result["execution_qualified_semantic_correct_fraction"], 0)
        report["status"] = "reader_failed_partial_outputs_preserved"
        report_path.write_text(json.dumps(report))
        with self.assertRaises(PLMError):
            audit.prepare_pack(selection, runs, self.root / "partial-pack")
