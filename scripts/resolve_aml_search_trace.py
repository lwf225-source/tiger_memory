"""Resolve opt-in Search trace tokens against a local read-only catalog.

No query/body/raw ID export, application configuration, model calls or writes.
Only event records are resolved; virtual hits and Facts remain explicitly
unresolved. This does not identify an official capability or reconstruct an
old response, historical storage snapshot, Answer prompt or Judge decision.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sqlite3

# Source checkout, installed Skill, and wheel support resources have distinct
# layouts. Resolve only adjacent bundled source, never an arbitrary sys.path.
_root = Path(__file__).resolve().parents[1]
_candidates = [_root / "tiger_memory/aml_search_trace.py",
               _root / "lib/tiger_memory/aml_search_trace.py"]
if _root.name == "_bundle" and _root.parent.name == "tiger_memory":
    _candidates.append(_root.parent / "aml_search_trace.py")
_source = next((path for path in _candidates if path.is_file()), None)
if _source is None:
    raise FileNotFoundError("bundled AML search trace module is missing")
_spec = importlib.util.spec_from_file_location("trace_lookup_only", _source)
_trace = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_trace)
storage_lookup_token = _trace.storage_lookup_token


def digest_field(value, optional=False):
    if optional and value is None:
        return None
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("invalid digest field")
    return value


def resolve(snapshot, catalog):
    trace = snapshot.get("status", {}).get("search_trace", snapshot)
    if trace.get("storage_lookup", {}).get("schema") != "aml-storage-lookup-v1":
        raise ValueError("snapshot has no recoverable storage lookup")
    instance = trace["instance_id"]
    binding = digest_field(trace["binding_sha256"])
    # Validate public salt before any database access.
    storage_lookup_token(instance, "project", None)
    events = trace["events"]
    if not isinstance(events, list) or len(events) > 128:
        raise ValueError("invalid bounded event list")
    conn = sqlite3.connect(Path(catalog).resolve().as_uri() + "?mode=ro", uri=True, timeout=5)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA query_only=ON")
        conn.execute("BEGIN")
        projects = {}
        for row in conn.execute("SELECT project_id FROM projects"):
            token = storage_lookup_token(instance, "project", row["project_id"])
            if token is not None:
                projects.setdefault(token, []).append(row["project_id"])
        cache, output = {}, []
        for event in events:
            rows = event.get("results", [])
            if not isinstance(rows, list) or len(rows) > 100:
                raise ValueError("invalid bounded result list")
            sequence = event["sequence"]
            if type(sequence) is not int or sequence < 1:
                raise ValueError("invalid event sequence")
            expected = digest_field(event.get("expected_project_token"), optional=True)
            retrieved = digest_field(event.get("retrieved_project_token"), optional=True)
            item = {"sequence": sequence, "sources": [],
                    "recorded_results": len(rows)}
            for flag in ("adapter_completed", "diagnostic_incomplete", "results_metadata_truncated"):
                value = event.get(flag)
                if value is not None and type(value) is not bool:
                    raise ValueError("invalid diagnostic flag")
                item[flag] = value
            if expected and retrieved and expected != retrieved:
                item["status"] = "project_identity_mismatch"
                output.append(item)
                continue
            token = retrieved or expected
            matches = projects.get(token, [])
            item["project_token"] = token
            item["project_matches"] = len(matches)
            if len(matches) != 1:
                item["status"] = ("no_project_token" if token is None else
                                  "project_missing" if not matches else "project_ambiguous")
                output.append(item)
                continue
            project_id = matches[0]
            if project_id not in cache:
                sources = {}
                stored = conn.execute("""
                    SELECT event_id,title,length(CAST(body AS BLOB)) AS body_bytes
                    FROM events WHERE project_id=? ORDER BY rowid
                """, (project_id,))
                count = 0
                for count, row in enumerate(stored, 1):
                    source_token = storage_lookup_token(instance, "source", row["event_id"])
                    role, sep, date = row["title"].partition(" · ")
                    metadata = {
                        "position_in_current_retained_arrival": count,
                        "body_utf8_bytes": row["body_bytes"],
                        "role": role if role in {"user", "assistant", "system", "developer", "tool"} else "other",
                        "source_undated": not sep or date == "undated",
                    }
                    if source_token is not None:
                        sources.setdefault(source_token, []).append(metadata)
                cache[project_id] = count, sources
            count, sources = cache[project_id]
            item.update(status="project_resolved", retained_event_count=count,
                        routing_evidence="retrieved" if retrieved else "expected_only")
            for row in rows:
                rank = row["rank"]
                if type(rank) is not int or not 1 <= rank <= 100:
                    raise ValueError("invalid result rank")
                source_token = digest_field(row.get("source_token"), optional=True)
                found = sources.get(source_token, [])
                match = {"rank": rank, "source_token": source_token,
                         "matches_in_project": len(found)}
                if len(found) == 1:
                    match.update(status="event_resolved", **found[0])
                else:
                    match["status"] = ("no_uuid_token" if source_token is None else
                                       "event_not_found" if not found else "event_ambiguous")
                item["sources"].append(match)
            output.append(item)
        return {
            "schema": "aml-storage-resolution-v1", "instance_id": instance,
            "binding_sha256": binding, "events": output,
            "trace_evicted": trace.get("evicted"), "read_only": True,
            "body_query_or_raw_identifiers_exported": False,
            "scope": "current-retained-events-not-historical-visible-prefix-or-official-sample",
        }
    finally:
        conn.rollback()
        conn.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = resolve(json.loads(args.snapshot.read_text()), args.catalog)
    except Exception as exc:
        print(json.dumps({"resolved": False, "error_type": type(exc).__name__}))
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
