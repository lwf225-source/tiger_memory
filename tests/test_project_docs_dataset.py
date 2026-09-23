from __future__ import annotations

import hashlib
import json
import re
import unittest
from pathlib import Path

from tiger_memory.evaluation import SOURCE_FIELDS, load_dataset


ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "evaluation" / "project-docs-v1"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(text):
    return re.sub(r"\s+", "", text.casefold())


class ProjectDocsDatasetTests(unittest.TestCase):
    def test_schema_and_pilot_counts_are_honest(self):
        dev = load_dataset(FOLDER, "dev")
        heldout = load_dataset(FOLDER, "heldout")
        self.assertEqual(len(dev["questions"]), 24)
        self.assertEqual(len(heldout["questions"]), 16)
        self.assertEqual(dev["manifest"]["real_project_count"], 1)
        self.assertFalse(dev["manifest"]["real_user_query_sample"])
        self.assertFalse(dev["manifest"]["official_benchmark_score"])
        questions = dev["questions"] + heldout["questions"]
        self.assertEqual(sum(question["unanswerable"] for question in questions), 5)
        self.assertEqual(sum(len(question["required_source_ids"]) > 1 for question in questions), 6)
        for source in dev["sources"] + heldout["sources"]:
            self.assertFalse(set(source) - SOURCE_FIELDS, source["source_id"])

    def test_every_source_is_an_exact_auditable_document_span(self):
        sidecar = read_json(FOLDER / "provenance-manifest.json")
        self.assertEqual(sidecar["provenance_sha256"], sha(FOLDER / "provenance.json"))
        self.assertEqual(sidecar["dataset_manifest_sha256"], sha(FOLDER / "manifest.json"))
        provenance = read_json(FOLDER / "provenance.json")
        source_by_id = {source["source_id"]: source for source in read_json(FOLDER / "sources.json")}
        documents = {entry["original_path"]: entry for entry in provenance["documents"]}
        self.assertEqual(set(source_by_id), {entry["source_id"] for entry in provenance["sources"]})
        for document in documents.values():
            path = FOLDER / document["snapshot_path"]
            self.assertTrue(path.is_relative_to(FOLDER))
            self.assertEqual(document["sha256"], sha(path), document["original_path"])
            self.assertEqual(document["line_count"], len(path.read_text(encoding="utf-8").splitlines()))
        for entry in provenance["sources"]:
            source = source_by_id[entry["source_id"]]
            lines = (FOLDER / entry["snapshot_path"]).read_text(encoding="utf-8").splitlines()
            span = "\n".join(lines[entry["line_start"] - 1:entry["line_end"]]).strip()
            self.assertEqual(source["body"], span, source["source_id"])
            self.assertEqual(entry["body_sha256"], hashlib.sha256(span.encode("utf-8")).hexdigest())
            self.assertEqual(entry["document_sha256"], documents[entry["original_path"]]["sha256"])

    def test_split_has_no_shared_spans_or_near_duplicate_sources(self):
        sources = read_json(FOLDER / "sources.json")
        by_id = {source["source_id"]: source for source in sources}
        provenance = read_json(FOLDER / "provenance.json")["sources"]
        for left in provenance:
            for right in provenance:
                if by_id[left["source_id"]]["split"] == by_id[right["source_id"]]["split"]:
                    continue
                if left["original_path"] == right["original_path"]:
                    self.assertTrue(left["line_end"] < right["line_start"] or right["line_end"] < left["line_start"],
                                    (left["source_id"], right["source_id"]))
        def grams(text):
            text = normalized(text)
            return {text[index:index + 5] for index in range(len(text) - 4)}
        dev = [source for source in sources if source["split"] == "dev"]
        heldout = [source for source in sources if source["split"] == "heldout"]
        for left in dev:
            for right in heldout:
                a, b = grams(left["body"]), grams(right["body"])
                similarity = len(a & b) / max(1, len(a | b))
                self.assertLess(similarity, 0.7, (left["source_id"], right["source_id"]))

    def test_questions_and_rubrics_never_enter_source_schema(self):
        sources = read_json(FOLDER / "sources.json")
        source_text = "\n".join(source["body"] for source in sources)
        for split in ("dev", "heldout"):
            data = load_dataset(FOLDER, split)
            for question in data["questions"]:
                self.assertTrue(question["answer"].strip(), question["question_id"])
                self.assertTrue(question["acceptable_answers"], question["question_id"])
                self.assertTrue(question["required_points"], question["question_id"])
                self.assertNotIn(question["question"], source_text, question["question_id"])
                self.assertNotIn(question["question_id"], source_text, question["question_id"])

    def test_document_gold_passes_its_own_rubric_without_exposing_text(self):
        for split in ("dev", "heldout"):
            data = load_dataset(FOLDER, split)
            for question in data["questions"]:
                answer = normalized(question["answer"])
                for point in question["required_points"]:
                    self.assertTrue(any(normalized(form) in answer for form in point["any_of"]),
                                    (question["question_id"], point["id"]))
                for claim in question["forbidden_claims"]:
                    for pattern in claim["patterns"]:
                        self.assertIsNone(re.search(pattern, question["answer"], re.I),
                                          (question["question_id"], claim["id"]))

    def test_synthetic_rubrics_bind_original_labels_and_cover_both_partitions(self):
        folder = ROOT / "evaluation" / "synthetic-v1"
        for split in ("dev", "heldout"):
            data = load_dataset(folder, split)
            sidecar = read_json(folder / ("rubric." + split + ".json"))
            self.assertEqual(sidecar["questions_sha256"], sha(folder / ("questions." + split + ".json")))
            self.assertEqual(sidecar["sources_sha256"], sha(folder / "sources.json"))
            self.assertEqual(sidecar["scoring_kind"], "local_lexical_rubric_not_semantic_judge")
            rubrics = {item["question_id"]: item for item in sidecar["rubrics"]}
            self.assertEqual(set(rubrics), {question["question_id"] for question in data["questions"]})
            for question in data["questions"]:
                rubric = rubrics[question["question_id"]]
                self.assertEqual(rubric["unanswerable"], question["unanswerable"])
                answer = "资料未提供，无法确定。" if question["unanswerable"] else question["answer"]
                for point in rubric["required_points"]:
                    self.assertTrue(any(normalized(form) in normalized(answer) for form in point["any_of"]),
                                    (question["question_id"], point["id"]))
                for claim in rubric["forbidden_claims"]:
                    self.assertFalse(any(re.search(pattern, answer, re.I) for pattern in claim["patterns"]),
                                     (question["question_id"], claim["id"]))
