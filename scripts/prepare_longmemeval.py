#!/usr/bin/env python3
"""Prepare a fixed public LongMemEval-S diagnostic; never run a reader or judge.

Network activity is limited to the pinned official data URL and its HTTPS HF
download hosts. The default output is an ignored workspace folder, not PLM memory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tiger_memory.evaluation import adapt_longmemeval, fingerprint, write_dataset
from tiger_memory.model import PLMError

UPSTREAM_REPOSITORY = "https://github.com/xiaowu0162/LongMemEval"
UPSTREAM_CODE_REVISION = "9e0b455f4ef0e2ab8f2e582289761153549043fc"
DATASET_REPOSITORY = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned"
DATASET_REVISION = "98d7416c24c778c2fee6e6f3006e7a073259d48f"
DATASET_FILENAME = "longmemeval_s_cleaned.json"
DATASET_BYTES = 277383467
DATASET_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
DATASET_URL = DATASET_REPOSITORY + "/resolve/" + DATASET_REVISION + "/" + DATASET_FILENAME
SELECTION_SEED = "plm-longmemeval-s-diagnostic-v1"
CATEGORIES = (
    "single-session-user", "single-session-assistant", "single-session-preference",
    "temporal-reasoning", "knowledge-update", "multi-session", "abstention",
)
READER_QUESTION_FIELDS = {"question_id", "project", "split", "question", "question_date", "scope", "scope_id"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".prepare-", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, str(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _safe_download_url(url: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    host = parsed.hostname or ""
    allowed = host == "huggingface.co" or host.endswith(".huggingface.co") or host.endswith(".hf.co")
    if parsed.scheme != "https" or not allowed or parsed.username or parsed.password:
        raise PLMError("download redirect is not an approved official HTTPS host")


class _OfficialRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        _safe_download_url(newurl)
        return super().redirect_request(request, fp, code, msg, headers, newurl)


def download_official(destination: Path, progress: bool = False) -> Dict[str, Any]:
    """Download only this fixed revision, checking both upstream size and SHA256."""
    destination = Path(destination)
    if destination.is_symlink():
        raise PLMError("symlinked dataset target rejected")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != DATASET_BYTES or file_sha256(destination) != DATASET_SHA256:
            raise PLMError("existing dataset does not match the pinned upstream checksum")
    else:
        _safe_download_url(DATASET_URL)
        descriptor, temporary = tempfile.mkstemp(prefix=".longmemeval-download-", dir=str(destination.parent))
        try:
            opener = urllib.request.build_opener(_OfficialRedirects())
            request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "PLM-public-evaluation/1"})
            digest = hashlib.sha256()
            count = 0
            last_progress = 0
            with os.fdopen(descriptor, "wb") as output, opener.open(request, timeout=60) as response:
                _safe_download_url(response.geturl())
                length = response.headers.get("Content-Length")
                if length is not None and int(length) != DATASET_BYTES:
                    raise PLMError("upstream dataset size differs from pinned size")
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    count += len(chunk)
                    if count > DATASET_BYTES:
                        raise PLMError("download exceeded the pinned size")
                    output.write(chunk)
                    digest.update(chunk)
                    if progress and count - last_progress >= 32 * 1024 * 1024:
                        print(json.dumps({"downloaded_bytes": count, "expected_bytes": DATASET_BYTES}), flush=True)
                        last_progress = count
                output.flush()
                os.fsync(output.fileno())
            if count != DATASET_BYTES or digest.hexdigest() != DATASET_SHA256:
                raise PLMError("download checksum or size mismatch")
            os.replace(temporary, str(destination))
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            raise PLMError("official dataset download failed; no unverified file was published") from None
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return {
        "dataset": "LongMemEval-S-cleaned", "variant": "full_history_non_oracle",
        "url": DATASET_URL, "revision": DATASET_REVISION, "bytes": DATASET_BYTES,
        "sha256": DATASET_SHA256, "checksum_verified": True,
        "upstream_repository": UPSTREAM_REPOSITORY, "upstream_code_revision": UPSTREAM_CODE_REVISION,
        "license": "MIT", "verified_at": datetime.now(timezone.utc).isoformat(),
    }


def _category(record: Dict[str, Any]) -> str:
    qid = record.get("question_id")
    kind = record.get("question_type")
    if not isinstance(qid, str) or not qid or kind not in CATEGORIES[:-1]:
        raise PLMError("unknown official question identity or type")
    return "abstention" if qid.endswith("_abs") else kind


def selection_key(record: Dict[str, Any]) -> str:
    """No question text, answer, evidence location or result influences selection."""
    payload = SELECTION_SEED + "\0" + record["question_type"] + "\0" + record["question_id"]
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_subset(records: Sequence[Dict[str, Any]], per_category: int = 3) -> List[Dict[str, Any]]:
    if per_category not in {3, 4}:
        raise PLMError("diagnostic uses 3 or 4 questions per category (21 or 28 total)")
    grouped: Dict[str, List[Dict[str, Any]]] = {category: [] for category in CATEGORIES}
    seen = set()
    for record in records:
        category = _category(record)
        if record["question_id"] in seen:
            raise PLMError("duplicate official question id")
        seen.add(record["question_id"])
        grouped[category].append(record)
    selected = []
    for category in CATEGORIES:
        if len(grouped[category]) < per_category:
            raise PLMError("official data lacks enough examples in a required category")
        selected.extend(sorted(grouped[category], key=lambda record: (selection_key(record), record["question_id"]))[:per_category])
    return selected


def prepare_dataset(input_path: Path, output: Path, *, per_category: int = 3,
                    expected_sha256: str = DATASET_SHA256, upstream_revision: str = DATASET_REVISION) -> Dict[str, Any]:
    """Split full source histories, reader questions and evaluator-only gold labels."""
    actual_hash = file_sha256(input_path)
    if actual_hash != expected_sha256:
        raise PLMError("input checksum does not match the fixed preparation contract")
    with Path(input_path).open("r", encoding="utf-8") as stream:
        records = json.load(stream)
    if not isinstance(records, list):
        raise PLMError("official input must be a JSON array")
    selected = select_subset(records, per_category)
    adapted = adapt_longmemeval(selected, upstream_revision)
    # Session-id naming conventions can distinguish generated evidence sessions
    # from fillers. Keep stable opaque source IDs and neutral ordinal titles.
    ordinal = Counter()
    for source in adapted["sources"]:
        ordinal[source["project"]] += 1
        source["title"] = "Conversation session %04d" % ordinal[source["project"]]
    expected_sources = sum(len(record["haystack_sessions"]) for record in selected)
    if len(adapted["sources"]) != expected_sources:
        raise PLMError("adapter failed to preserve every selected history session")
    metadata = {
        **adapted["metadata"], "dataset_id": "longmemeval-s-diagnostic-" + str(len(selected)),
        "variant": "full_history_non_oracle", "input_sha256": actual_hash,
        "input_bytes": Path(input_path).stat().st_size, "upstream_data_revision": upstream_revision,
        "upstream_code_revision": UPSTREAM_CODE_REVISION, "official_benchmark_score": False,
        "selection": {"seed": SELECTION_SEED, "per_category": per_category,
                      "rule": "within category, ascending SHA256(seed NUL question_type NUL question_id), then question_id",
                      "uses_gold_or_question_text": False},
        "all_haystack_sessions_preserved": True,
    }
    output = Path(output)
    write_dataset(output, adapted["sources"], adapted["questions"], metadata)
    reader_questions = [{key: question[key] for key in READER_QUESTION_FIELDS if key in question}
                        for question in adapted["questions"]]
    _write_json(output / "reader-questions.json", reader_questions)
    gold = [{"question_id": question["question_id"], "answer": question["answer"],
             "category": question["category"], "unanswerable": question["unanswerable"],
             "required_source_ids": question["required_source_ids"], "gold_answer_turns": question["gold_answer_turns"]}
            for question in adapted["questions"]]
    _write_json(output / "gold.evaluator-only.json", gold)
    counts = Counter(_category(record) for record in selected)
    selection = {
        "schema_version": 1, "input_sha256": actual_hash, "input_bytes": Path(input_path).stat().st_size,
        "upstream_data_revision": upstream_revision, "upstream_code_revision": UPSTREAM_CODE_REVISION,
        "upstream_url": DATASET_URL, "variant": "full_history_non_oracle",
        "total_input_questions": len(records), "selected_questions": len(selected),
        "selection": metadata["selection"], "category_counts": dict(counts),
        "selected": [{"question_id": record["question_id"], "question_type": record["question_type"],
                      "category": _category(record), "selection_hash": selection_key(record),
                      "history_sessions": len(record["haystack_sessions"])} for record in selected],
        "source_count": expected_sources, "adapter_input_fingerprint": fingerprint(selected),
        "output_sha256": {name: file_sha256(output / name) for name in (
            "sources.json", "questions.official.json", "reader-questions.json", "gold.evaluator-only.json", "manifest.json",
        )},
        "official_grader_executed": False, "reader_executed": False,
    }
    _write_json(output / "preparation.json", selection)
    return selection


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=ROOT / ".codex_work" / "longmemeval")
    parser.add_argument("--input", type=Path, help="existing exact upstream file; its pinned SHA256 is still required")
    parser.add_argument("--per-category", type=int, choices=(3, 4), default=3)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        work = args.work_dir.resolve()
        input_path = args.input or work / DATASET_FILENAME
        if args.input:
            if file_sha256(input_path) != DATASET_SHA256:
                raise PLMError("supplied input does not match the pinned upstream checksum")
        else:
            download = download_official(input_path, progress=True)
            _write_json(work / "download.json", download)
        if args.download_only:
            print(json.dumps({"download_verified": True, "bytes": DATASET_BYTES, "sha256": DATASET_SHA256}))
            return 0
        report = prepare_dataset(input_path, work / ("diagnostic-" + str(args.per_category * len(CATEGORIES))),
                                 per_category=args.per_category)
        print(json.dumps({"output": str(work / ("diagnostic-" + str(report["selected_questions"]))),
                          "questions": report["selected_questions"], "sessions": report["source_count"],
                          "category_counts": report["category_counts"], "input_sha256": report["input_sha256"],
                          "official_grader_executed": False, "reader_executed": False}, ensure_ascii=False))
        return 0
    except (PLMError, OSError, ValueError, KeyError, TypeError):
        print(json.dumps({"error": "LongMemEval preparation failed; no reader or external grader was run"}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
