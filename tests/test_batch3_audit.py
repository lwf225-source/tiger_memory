from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tiger_memory import evaluation as ev
from tiger_memory import qa_evaluation as qa
from tiger_memory.model import PLMError

spec = importlib.util.spec_from_file_location("audit_batch3", Path(__file__).resolve().parents[1] / "scripts/audit_batch3.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def questions_fixture():
    return [{"question_id": "question-" + str(index), "project": "project-" + str(index),
        "split": "heldout", "category": "abstention" if index >= 10 else "conditions",
        "unanswerable": index >= 10, "question": "QUESTION-CANARY-" + str(index),
        "answer": "GOLD-CANARY-" + str(index),
        "required_source_ids": [] if index >= 10 else ["source-" + str(index)]}
        for index in range(12)]


class Batch3AuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="plm-batch3-audit-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()

    def write(self, path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def fixtures(self):
        questions = questions_fixture()
        sources = [{"source_id": "source-" + str(index), "project": row["project"], "split": "heldout",
            "title": "Neutral title", "body": "BODY-CANARY-" + str(index)} for index, row in enumerate(questions)]
        folder = self.root / "dataset"
        ev.write_dataset(folder, sources, questions, {"dataset_id": audit.DATASET_ID, "origin": "synthetic", "splits": ["heldout"]})
        dataset = ev.load_dataset(folder, "heldout")
        by_source = {row["source_id"]: row for row in dataset["sources"]}
        cases = []
        for index, question in enumerate(dataset["questions"]):
            for baseline in (*audit.BASELINES, "passage_embedding"):
                source_id = "source-" + str(index)
                case = qa.prepare_case(question, baseline, [source_id], by_source, "baseline", 6000, "excerpt")
                if baseline != "lexical":
                    payload = json.loads(case["messages"][1]["content"])
                    record = payload["sources"][0]
                    record.update(source_id="slice-" + str(index), parent_source_id=source_id,
                        start=0, end=len(record["body"]), body_sha256=qa._hash_text(record["body"]))
                    case["sources"][0].update(source_id=record["source_id"], parent_source_id=source_id)
                    case["citation_source_ids"] = [record["source_id"]]
                    case["citation_parent_map"] = {record["source_id"]: source_id}
                    case["messages"][1]["content"] = json.dumps(payload)
                    case["messages_sha256"] = ev.fingerprint(case["messages"])
                cases.append(case)
        runs = {}
        for mode, decoder in audit.MODES.items():
            run = self.root / (mode + "-run")
            run.mkdir()
            runs[mode] = run
            binding = {"dataset_fingerprint": dataset["fingerprint"], "source_fingerprint": ev.fingerprint(dataset["sources"]),
                "split": "heldout", "question_ids": [row["question_id"] for row in questions],
                "prompt_profile": "baseline", "prompt_sha256": qa._hash_text(qa.PROMPTS["baseline"]),
                "packing": "excerpt", "reader": {"decoding": {"output_mode": decoder}},
                "baselines": [*audit.BASELINES, "passage_embedding"],
                "retrieval_provenance": "direct_source_only_retrieval",
                "inputs_fingerprint": ev.fingerprint([qa._binding_case(case) for case in cases])}
            manifest = {"binding": binding, "run_fingerprint": ev.fingerprint(binding)}
            predictions = []
            for case in cases:
                text = json.dumps({"answer": "PREDICTION-CANARY", "abstained": False,
                    "citations": case.get("citation_source_ids", case["source_ids"])})
                row = {"question_id": case["question_id"], "baseline": case["baseline"],
                    "run_fingerprint": manifest["run_fingerprint"], "messages_sha256": case["messages_sha256"],
                    "sources_sha256": ev.fingerprint(case["sources"]), "text": text,
                    "text_sha256": qa._hash_text(text), "error_type": "", "finish_reason": "stop"}
                row["record_sha256"] = qa._prediction_checksum(row)
                predictions.append(row)
            self.write(run / "manifest.json", manifest)
            (run / "inputs.jsonl").write_text(qa._lines(cases))
            (run / "predictions.jsonl").write_text(qa._lines(predictions))
            self.write(run / "report.json", {"run_fingerprint": manifest["run_fingerprint"], "status": "complete"})
        selection = self.root / "selection.json"
        audit.freeze_selection(folder, selection)
        return dataset, runs, selection

    def pack(self):
        _, runs, selection = self.fixtures()
        directory = self.root / "pack"
        audit.prepare_pack(selection, runs, directory)
        return directory, runs

    def verdicts(self, directory):
        pack = audit._read(directory / "reviewer-pack.json")
        path = self.root / "verdicts.json"
        self.write(path, {"pack_sha256": pack["pack_sha256"], "reviewer_kind": "codex_agent_semantic_review",
            "verdicts": [{"audit_id": row["audit_id"], "semantic_correct": True,
                "evidence_support": "not_applicable" if row["unanswerable"] else "sufficient",
                "reason": "REASON-CANARY: fixture verdict, not model evaluation."} for row in pack["entries"]]})
        return path

    def test_sampling_is_fixed_six_and_ignores_gold_text_predictions(self):
        questions = questions_fixture()
        expected = [row["question_id"] for row in audit.select_questions(questions)]
        self.assertEqual(len(expected), 6)
        self.assertEqual(sum(row["unanswerable"] for row in audit.select_questions(questions)), 2)
        for row in questions:
            row.update(question="changed", answer="changed", prediction="changed", correct=False, required_source_ids=[])
        self.assertEqual(expected, [row["question_id"] for row in audit.select_questions(questions[::-1])])
        with self.assertRaises(PLMError):
            audit.select_questions(questions[:-1])

    def test_freeze_reads_no_prediction_and_selection_has_no_contents(self):
        _, _, selection = self.fixtures()
        data = audit._read(selection)
        with patch.object(qa, "_read_jsonl", side_effect=AssertionError("prediction access forbidden")):
            audit.freeze_selection(data["dataset"]["path"], self.root / "second-selection.json")
        text = json.dumps(data)
        for marker in ("QUESTION-CANARY", "GOLD-CANARY", "BODY-CANARY", "PREDICTION-CANARY"):
            self.assertNotIn(marker, text)
        self.assertFalse(data["selection_uses_predictions"])
        self.assertEqual(selection.stat().st_mode & 0o777, 0o600)

    def test_pack_has_24_blinded_actual_citation_bound_outputs(self):
        directory, _ = self.pack()
        pack = audit._read(directory / "reviewer-pack.json")
        self.assertEqual(len(pack["entries"]), 24)
        self.assertEqual(len({row["question_key"] for row in pack["entries"]}), 6)
        for row in pack["entries"]:
            self.assertTrue(row["format_contract"]["contract_valid"])
            self.assertTrue(row["format_contract"]["citation_ids_valid"])
            self.assertEqual(set(row["citation_parent_map"]), {source["source_id"] for source in row["actually_provided_sources"]})
        text = json.dumps(pack)
        for marker in ('"baseline"', '"output_mode"', '"lexical"', '"passages"', '"free"', '"structured"', '"json_schema"'):
            self.assertNotIn(marker, text)
        for marker in ("QUESTION-CANARY", "GOLD-CANARY", "BODY-CANARY", "PREDICTION-CANARY"):
            self.assertIn(marker, text)
        self.assertEqual((directory / "reviewer-pack.json").stat().st_mode & 0o777, 0o600)

    def test_parent_id_is_not_a_legal_passage_citation(self):
        _, runs, selection = self.fixtures()
        path = runs["free"] / "predictions.jsonl"
        rows = qa._read_jsonl(path)
        for row in rows:
            if row["baseline"] == "passages":
                row["text"] = row["text"].replace("slice-", "source-")
                row["text_sha256"] = qa._hash_text(row["text"])
                row["record_sha256"] = qa._prediction_checksum(row)
        path.write_text(qa._lines(rows))
        directory = self.root / "invalid-citations"
        audit.prepare_pack(selection, runs, directory)
        entries = audit._read(directory / "reviewer-pack.json")["entries"]
        self.assertEqual(sum(not row["format_contract"]["citation_ids_valid"] for row in entries), 6)

    def test_requires_complete_runs_including_unselected_baseline(self):
        _, runs, selection = self.fixtures()
        path = runs["structured"] / "predictions.jsonl"
        original = path.read_text()
        path.write_text("\n".join(original.splitlines()[:-1]) + "\n")
        with self.assertRaises(PLMError):
            audit.prepare_pack(selection, runs, self.root / "missing")
        path.write_text(original.replace("PREDICTION-CANARY", "MUTATED-PREDICTION"))
        with self.assertRaises(PLMError):
            audit.prepare_pack(selection, runs, self.root / "mutated")
        path.write_text(original)
        report = audit._read(runs["free"] / "report.json")
        report["status"] = "reader_failed_partial_outputs_preserved"
        self.write(runs["free"] / "report.json", report)
        with self.assertRaises(PLMError):
            audit.prepare_pack(selection, runs, self.root / "partial")
        self.assertFalse((self.root / "missing").exists())

    def test_mode_swap_and_forged_partial_case_manifest_rejected(self):
        dataset, runs, selection = self.fixtures()
        with self.assertRaises(PLMError):
            audit.prepare_pack(selection, {"free": runs["structured"], "structured": runs["free"]}, self.root / "swapped")
        path = runs["free"]
        cases = qa._read_jsonl(path / "inputs.jsonl")[:-1]
        manifest = audit._read(path / "manifest.json")
        manifest["binding"]["inputs_fingerprint"] = ev.fingerprint([qa._binding_case(case) for case in cases])
        manifest["run_fingerprint"] = ev.fingerprint(manifest["binding"])
        self.write(path / "manifest.json", manifest)
        (path / "inputs.jsonl").write_text(qa._lines(cases))
        with self.assertRaises(PLMError):
            audit._load_run(path, dataset, "free")

    def test_actual_evidence_count_hash_and_canonical_slice_checked(self):
        dataset, runs, _ = self.fixtures()
        case = next(row for row in qa._read_jsonl(runs["free"] / "inputs.jsonl") if row["baseline"] == "passages")
        question = next(row for row in dataset["questions"] if row["question_id"] == case["question_id"])
        by_source = {row["source_id"]: row for row in dataset["sources"]}
        for mutation in ("count", "hash", "slice"):
            changed = copy.deepcopy(case)
            if mutation == "count":
                changed["sources"] = []
            elif mutation == "hash":
                changed["sources"][0]["body_sha256"] = "0" * 64
            else:
                payload = json.loads(changed["messages"][1]["content"])
                payload["sources"][0]["start"] = 1
                changed["messages"][1]["content"] = json.dumps(payload)
                changed["messages_sha256"] = ev.fingerprint(changed["messages"])
            with self.assertRaises(PLMError):
                audit._actual_evidence(changed, question, by_source)

    def test_finalize_binds_24_predictions_and_summary_is_content_free(self):
        directory, _ = self.pack()
        summary = audit.finalize_grades(directory, self.verdicts(directory), self.root / "grades", "Codex Agent fixture")
        self.assertEqual(summary["outputs"], 24)
        self.assertEqual(set(summary["results"]), {mode + "/" + baseline for mode in audit.MODES for baseline in audit.BASELINES})
        self.assertFalse(summary["official_benchmark_score"])
        for marker in ("QUESTION-CANARY", "GOLD-CANARY", "BODY-CANARY", "PREDICTION-CANARY", "REASON-CANARY", '"question_id"'):
            self.assertNotIn(marker, json.dumps(summary))
        grades = audit._read(self.root / "grades/grades.evaluator-only.json")["grades"]
        self.assertTrue(all(row["run_fingerprint"] and row["prediction_sha256"] and row["prediction_text_sha256"] for row in grades))

    def test_missing_verdict_changed_prediction_mapping_and_overwrite_rejected(self):
        directory, runs = self.pack()
        verdict_path = self.verdicts(directory)
        verdicts = audit._read(verdict_path)
        changed = copy.deepcopy(verdicts)
        changed["verdicts"].pop()
        self.write(verdict_path, changed)
        with self.assertRaises(PLMError):
            audit.finalize_grades(directory, verdict_path, self.root / "missing", "Agent")
        self.write(verdict_path, verdicts)
        path = runs["free"] / "predictions.jsonl"
        original = path.read_text()
        path.write_text(original.replace("PREDICTION-CANARY", "MUTATED-PREDICTION"))
        with self.assertRaises(PLMError):
            audit.finalize_grades(directory, verdict_path, self.root / "changed", "Agent")
        path.write_text(original)
        path = directory / "mapping.json"
        path.write_text(path.read_text().replace('"lexical"', '"unknown"'))
        with self.assertRaises(PLMError):
            audit.finalize_grades(directory, verdict_path, self.root / "tampered", "Agent")
        with self.assertRaises(PLMError):
            audit.prepare_pack(self.root / "selection.json", runs, directory)

    def test_truncated_output_kept_and_raw_semantics_separate_from_execution(self):
        _, runs, selection = self.fixtures()
        path = runs["free"] / "predictions.jsonl"
        rows = qa._read_jsonl(path)
        for row in rows:
            row["finish_reason"] = "length"
            row["record_sha256"] = qa._prediction_checksum(row)
        path.write_text(qa._lines(rows))
        report = audit._read(runs["free"] / "report.json")
        report["status"] = "complete_with_reader_failures"
        self.write(runs["free"] / "report.json", report)
        directory = self.root / "failure-pack"
        audit.prepare_pack(selection, runs, directory)
        summary = audit.finalize_grades(directory, self.verdicts(directory), self.root / "failure-grades", "Agent fixture")
        for baseline in audit.BASELINES:
            result = summary["results"]["free/" + baseline]
            self.assertEqual(result["outputs"], 6)
            self.assertEqual(result["generation_failure_count"], 6)
            self.assertEqual(result["raw_semantic_correct_fraction_all_selected"], 1)
            self.assertEqual(result["execution_qualified_semantic_correct_fraction"], 0)
            self.assertEqual(result["contract_qualified_semantic_correct_fraction"], 0)

    def test_private_paths_symlinks_and_traversal_rejected(self):
        for path in (audit.ROOT / "evaluation/public.json", audit.ROOT / ".codex_work/qa-audit/other.json",
                     audit.PRIVATE / "../outside.json"):
            with self.assertRaises(PLMError):
                audit._private_cli_output(path)
        target = self.root / "target"
        target.mkdir()
        link = self.root / "link"
        link.symlink_to(target, target_is_directory=True)
        with self.assertRaises(PLMError):
            audit._write_new(link / "file.json", {})


if __name__ == "__main__":
    unittest.main()
