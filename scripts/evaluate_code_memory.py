#!/usr/bin/env python3
"""Offline, opt-in three-arm evidence diagnostic using a real local CodeGraph CLI.

All writes/indexes are isolated in TemporaryDirectory. No reader model is called.
The resulting contexts can be reviewed or given to the same external reader.
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tiger_memory import codegraph
from tiger_memory.cli import main as plm_main
from tiger_memory.service import write_event

FIXTURES = Path(__file__).resolve().parents[1] / "examples" / "code_memory"
ARMS = ("memory-only", "code-only", "combined")


def clip(text, budget):
    return text.encode("utf-8")[:max(0, budget)].decode("utf-8", errors="ignore")


def evidence_packet(arm, memory_text, code_result, budget):
    """Budget includes labels and status, not just source bodies; UTF-8 safe."""
    if arm not in ARMS or not 512 <= budget <= 64000:
        raise ValueError("arm must be supported and budget must be 512..64000 bytes")
    channels = ("memory", "code") if arm == "combined" else (arm.split("-")[0],)
    sections = {}
    quota = budget // len(channels)
    status = code_result.get("status", "not_requested")
    for channel in channels:
        if channel == "memory":
            header, content = "[historical memory]\n", memory_text
        else:
            header = "[current code evidence; status=" + status + "]\n"
            content = code_result.get("text", "") if status == "ready" else ""
        # A trailing newline is part of each quota, including the last one.
        sections[channel] = clip(header + content, quota - 1) + "\n"
    context = "".join(sections.values())
    return {"reader_context": context, "sections": sections,
            "context_bytes": len(context.encode("utf-8")), "budget_bytes": budget,
            "code_status": status if "code" in channels else "not_requested",
            "code_text_exposed": bool(code_result.get("text")) and status == "ready" and "code" in channels}


def score_packet(packet, requirements):
    """Literal evidence availability only. This does not judge any answer."""
    checks = [{"id": requirement["id"], "source": requirement["source"],
               "covered": all(token in packet["sections"].get(requirement["source"], "")
                              for token in requirement["all_of"])}
              for requirement in requirements]
    hits = sum(check["covered"] for check in checks)
    return {"checks": checks, "covered": hits, "required": len(checks),
            "literal_evidence_coverage": hits / len(checks) if checks else None,
            "semantic_quality": "not_measured"}


def context_payload(root, query, budget, code_query=None):
    output = io.StringIO()
    args = ["context", "--cwd", str(root), "--query", query,
            "--token-budget", str(budget), "--json"]
    if code_query is not None:
        args += ["--with-code", "--code-query", code_query, "--code-budget", str(budget)]
    with contextlib.redirect_stdout(output):
        exit_code = plm_main(args)
    if exit_code:
        raise RuntimeError("memory context retrieval failed")
    return json.loads(output.getvalue())


def memory_evidence(root, payload):
    # Preserve evidence source pointers for review, normalizing ephemeral roots.
    text = json.dumps(payload["results"], ensure_ascii=False, indent=2)
    return text.replace(str(root.parent), "<temporary>")


def _git(root, *args):
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    subprocess.run(["git", *args], cwd=root, env=env, check=True, capture_output=True)


def evaluate(budget=8000):
    if not 512 <= budget <= 64000:
        raise ValueError("budget must be 512..64000 bytes")
    if not shutil.which("codegraph"):
        raise RuntimeError("Install CodeGraph explicitly before running this opt-in diagnostic")
    fixture = json.loads((FIXTURES / "cases.json").read_text())
    before, after = [(FIXTURES / name).read_text() for name in ("before.py", "after.py")]
    digest = hashlib.sha256()
    for name in ("cases.json", "before.py", "after.py"):
        digest.update(name.encode() + b"\0" + (FIXTURES / name).read_bytes())
    records = []
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="plm-code-memory-eval-") as temporary:
        base = Path(temporary).resolve()
        root = base / "project"
        root.mkdir()
        with mock.patch.dict(os.environ, {"TIGER_MEMORY_DIR": str(base / "memory")}):
            source = root / "app.py"
            source.write_text(before)
            (root / ".gitignore").write_text(".codegraph/\n")
            _git(root, "init", "-q")
            _git(root, "add", ".")
            _git(root, "-c", "user.name=fixture", "-c", "user.email=fixture@local", "commit", "-qm", "before")
            for event in fixture["memory"]:
                write_event(root, event["title"], event["body"],
                            observed_at="2026-09-01T00:00:00Z", recorded_at="2026-09-01T00:00:00Z",
                            source="synthetic-fixture:" + event["id"])
            # Index original code once, then explicitly sync changed source.
            codegraph.sync(root, initialize=True)
            source.write_text(after)
            if codegraph.query(root, "retry_limit")["status"] != "stale":
                raise RuntimeError("changed-code fixture did not invalidate the index")
            ready = codegraph.sync(root)
            if ready["status"] != "ready":
                raise RuntimeError("fixture index failed to become ready")
            engine_version = ready.get("index_health", {}).get("engine_version")
            for case_index, case in enumerate(fixture["cases"]):
                if case["state"] == "stale":
                    source.write_text(after.replace("return 1", "return 2"))
                elif source.read_text() != after:
                    source.write_text(after)
                    codegraph.sync(root)
                # Rotate arm order to expose, not hide, warm-cache effects.
                order = ARMS[case_index % 3:] + ARMS[:case_index % 3]
                for arm in order:
                    began = time.monotonic()
                    historical = ""
                    result = {"status": "not_requested"}
                    guard = (mock.patch.object(codegraph.shutil, "which", return_value=None)
                             if case["state"] == "missing_cli" else contextlib.nullcontext())
                    with guard:
                        if arm == "code-only":
                            result = codegraph.query(root, case["code_query"], max_bytes=budget)
                        else:
                            payload = context_payload(root, case["memory_query"], budget,
                                                      case["code_query"] if arm == "combined" else None)
                            historical = memory_evidence(root, payload)
                            if arm == "combined":
                                result = payload["code_evidence"]
                    packet = evidence_packet(arm, historical, result, budget)
                    expected = case["expected_code_status"] if arm != "memory-only" else "not_requested"
                    provenance = json.loads(json.dumps(
                        {key: value for key, value in result.items() if key != "text"},
                        ensure_ascii=False).replace(str(base), "<temporary>"))
                    record = {"case_id": case["id"], "arm": arm, "question": case["question"],
                              "packet": packet, "metrics": score_packet(packet, case["requirements"]),
                              "code_provenance": provenance,
                              "elapsed_s": round(time.monotonic() - began, 4),
                              "freshness": {"expected_code_status": expected,
                                            "status_matches": packet["code_status"] == expected,
                                            "unsafe_code_exposed": expected in ("stale", "unavailable") and packet["code_text_exposed"]}}
                    records.append(record)
    summary = {}
    for arm in ARMS:
        samples = [record for record in records if record["arm"] == arm]
        hits = sum(record["metrics"]["covered"] for record in samples)
        total = sum(record["metrics"]["required"] for record in samples)
        summary[arm] = {"literal_evidence_coverage": hits / total,
                        "covered": hits, "required": total,
                        "elapsed_s": round(sum(record["elapsed_s"] for record in samples), 4),
                        "context_bytes": sum(record["packet"]["context_bytes"] for record in samples)}
    return {"schema_version": 1, "fixture_sha256": digest.hexdigest(),
            "codegraph_engine_version": engine_version,
            "budget_bytes_per_question": budget, "semantic_quality": "not_measured",
            "operational_checks_passed": all(
                record["freshness"]["status_matches"]
                and not record["freshness"]["unsafe_code_exposed"]
                and record["packet"]["context_bytes"] <= budget for record in records),
            "method": {"coverage": "Exact fixture evidence strings in delivered channel; not answer accuracy.",
                       "entry_points": "memory-only: plm context; code-only: codegraph.query; combined: plm context --with-code. All outputs then receive identical final-byte packing.",
                       "combined_allocation": "Equal byte quota per channel; unused quota is not transferred.",
                       "budget_scope": "Complete reader_context including status labels; question and report metadata excluded equally.",
                       "timing": "Single sequential run; rotating arm order; index build excluded from arm times. No statistical performance claim.",
                       "negative": "Missing CLI is injected locally; stale source is an actual post-index edit.",
                       "scope": "Synthetic diagnostic, not AML or a representative coding benchmark. No model, network, or paid reader."},
            "summary": summary, "records": records,
            "elapsed_s_including_setup": round(time.monotonic() - started, 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-bytes", type=int, default=8000)
    parser.add_argument("--output", type=Path, help="Explicit local JSON report path; stdout if omitted")
    args = parser.parse_args()
    report = evaluate(args.budget_bytes)
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        print(json.dumps({"output": str(args.output.resolve()), "summary": report["summary"],
                          "operational_checks_passed": report["operational_checks_passed"],
                          "semantic_quality": report["semantic_quality"]}, ensure_ascii=False, indent=2))
    else:
        print(serialized, end="")
    return 0 if report["operational_checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
