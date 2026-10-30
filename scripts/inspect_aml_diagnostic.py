"""Read-only structural audit of explicitly supplied diagnostic exports.

Accept Recorder snapshots or JsonlSink files. Does not open a store, contact a
provider, recover text, guess official samples, or authenticate supplied data.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re


HEX64 = re.compile(r"^[0-9a-f]{64}$")


def inspect(documents):
    events, errors, export_losses = [], [], []
    for document in documents:
        if "events" in document:
            events.extend(document["events"])
            if document.get("export_complete") is not True:
                export_losses.append(document.get("counters", {}))
        else:
            events.append(document)
    bindings = {(e.get("binding_sha256"), e.get("key_id"), e.get("key_check"))
                for e in events}
    if len(bindings) > 1:
        errors.append("mixed_release_or_key_comparison_refused")
    seen, writes, eligible, outputs = set(), set(), set(), set()
    stages, operations = Counter(), Counter()
    for event in events:
        identity = (event.get("instance_id"), event.get("sequence"), event.get("request_id"))
        if identity in seen:
            errors.append("duplicate_request_event")
        seen.add(identity)
        if event.get("schema") != "aml-full-diagnostic-v1":
            errors.append("unsupported_schema")
        for key in ("binding_sha256", "key_check"):
            if not HEX64.fullmatch(str(event.get(key, ""))):
                errors.append("invalid_binding")
        if event.get("replay_complete") is not False or event.get("official_chain_complete") is not False:
            errors.append("unsupported_completeness_claim")
        operations[event.get("operation", "unknown")] += 1
        if event.get("incomplete") or event.get("observation_complete") is not True:
            errors.append("incomplete_observation")
        request_stages = event.get("stages", [])
        names = Counter(row.get("stage") for row in request_stages)
        stages.update(names)
        if event.get("http"):
            if names["serialization"] != 1:
                errors.append("http_serialization_count")
            transports = [row.get("phase") for row in request_stages if row.get("stage") == "transport"]
            if transports not in (["write_started", "write_returned_not_client_receipt"],
                                  ["write_started", "write_failed"]):
                errors.append("http_transport_incomplete")
        local_views = {}
        for row in request_stages:
            if row.get("stage") == "write":
                writes.add((row.get("source_token"), row.get("content_hmac")))
            if row.get("stage") == "eligible":
                members = row.get("members", [])
                if row.get("manifest_complete") is not True or len(members) != row.get("member_count"):
                    errors.append("manifest_incomplete")
                local_views[row.get("read_view_id")] = {
                    (m.get("source_token"), m.get("content_hmac")) for m in members}
                eligible.update(local_views[row.get("read_view_id")])
            if row.get("stage") == "fact_dereference" and row.get("found"):
                local_views[row.get("read_view_id")] = {(row.get("source_token"), row.get("content_hmac"))}
        for row in request_stages:
            if row.get("stage") != "adapter_output":
                continue
            results = row.get("results", [])
            if "result_count" in row and len(results) != row["result_count"]:
                errors.append("output_list_truncated")
            for index, result in enumerate(results):
                if result.get("rank") != index:
                    errors.append("output_rank")
                cursor = 0
                for segment in result.get("segments", []):
                    start, end = segment.get("start"), segment.get("end")
                    if type(start) is not int or type(end) is not int or start != cursor or end < start:
                        errors.append("content_span_gap_or_overlap")
                        break
                    cursor = end
                    if segment.get("source_token"):
                        version = (segment["source_token"], segment.get("content_hmac"))
                        outputs.add(version)
                        view = segment.get("read_view_id")
                        if view is not None and version not in local_views.get(view, set()):
                            errors.append("source_view_version_mismatch")
                        if view is None and segment.get("origin") != "derived_or_virtual_not_in_eligible_manifest":
                            errors.append("source_view_missing")
                    if segment.get("kind") == "adapter_content_unmapped":
                        errors.append("unmapped_output")
                if cursor != result.get("content_utf8_bytes"):
                    errors.append("content_span_not_full")
    if export_losses:
        errors.append("export_loss_reported")
    comparable = len(bindings) <= 1 and not any(e == "invalid_binding" for e in errors)
    return {
        "schema": "aml-diagnostic-inspection-v1",
        "structurally_valid": not errors,
        "event_count": len(events), "operations": dict(operations), "stages": dict(stages),
        "errors": dict(Counter(errors)), "export_loss_count": len(export_losses),
        "comparable": comparable,
        "version_links": {
            "unique_write_versions": len(writes), "unique_eligible_versions": len(eligible),
            "unique_output_versions": len(outputs),
            "write_to_eligible": len(writes & eligible) if comparable else None,
            "write_to_output": len(writes & outputs) if comparable else None,
            "write_eligible_output": len(writes & eligible & outputs) if comparable else None,
        },
        "replay_complete": False, "official_chain_complete": False,
        "authentication_verified": False,
        "limitations": [
            "Structural validation does not authenticate the exported records.",
            "Absent Add evidence can predate this collection window.",
            "HMAC metadata cannot recover source or query text.",
            "No official sample attribution or F1 conclusion is produced.",
            "Crash losses require an independent request/transport manifest.",
        ],
    }


def read_documents(paths):
    # Finite explicit inputs only; no directory crawling or implicit store reads.
    if not 1 <= len(paths) <= 4096:
        raise ValueError("input count limit")
    documents, total = [], 0
    for value in paths:
        path = Path(value)
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
            raise ValueError("invalid or oversized input")
        raw = path.read_bytes()
        total += len(raw)
        if total > 128 * 1024 * 1024:
            raise ValueError("total input limit")
        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError("object required")
            documents.append(parsed)
        except json.JSONDecodeError:
            documents.extend(json.loads(line) for line in raw.splitlines() if line.strip())
    return documents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()
    try:
        report = inspect(read_documents(args.inputs))
    except (OSError, ValueError, TypeError, KeyError):
        print(json.dumps({"structurally_valid": False, "errors": {"invalid_input": 1}}))
        return 2
    print(json.dumps(report, indent=2))
    return 0 if report["structurally_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
