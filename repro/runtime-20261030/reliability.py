"""Locked source recovery and typed derived-index validation."""
from collections import Counter, OrderedDict
import copy
import stat
import threading
import json
import logging
import re

from .events import event_relative_path, parse_event
from .model import Event, PLMError, v2_root
from .security import assert_safe_path

LOG = logging.getLogger(__name__)
FORMAL_SOURCE = re.compile(r"[0-9]{8}T[0-9]{6}Z_[0-9a-fA-F-]{36}\.md\Z")


class SourceRecoveryError(PLMError):
    """Public code only; paths belong exclusively in the local log."""


def _reliable_key(path, project_id):
    """Only intact, unambiguous JSON routing fields can classify a damaged file."""
    try:
        if path.is_symlink():
            return None
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines or lines[0] != "---":
            return None
        end = lines.index("---", 1)
        routing = {}
        for field in ("project_id", "idempotency_key"):
            matches = [line.split(":", 1)[1].strip() for line in lines[1:end]
                       if line.split(":", 1)[0].strip() == field and ":" in line]
            if len(matches) != 1:
                return None
            routing[field] = json.loads(matches[0])
        if routing["project_id"] != project_id or not isinstance(routing["idempotency_key"], str):
            return None
        return routing["idempotency_key"]
    except (OSError, ValueError):
        return None


# Cache only validated routing metadata, never bodies, negative lookups,
# deletion decisions, batch commit decisions, or a directory's membership.
# Process-local eviction is safe: a miss simply reparses the source.
_SOURCE_METADATA = OrderedDict()
_SOURCE_METADATA_LOCK = threading.Lock()
_SOURCE_METADATA_LIMIT = 100000


def _fingerprint(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("nonregular source")
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
            info.st_ctime_ns, info.st_mode, info.st_uid, info.st_gid)


def _validate_source(event, path, project):
    meta = event.metadata
    if (event.project_id != project["project_id"] or
            meta["project_root"] != project["canonical_root"] or
            path != v2_root() / event_relative_path(project["folder_name"],
                                                   event.event_id, meta["recorded_at"]) or
            not isinstance(meta["idempotency_key"], str)):
        raise ValueError("source identity mismatch")


def _source_metadata(path, project):
    signature = _fingerprint(path)
    cache_key = (str(path), project["project_id"], project["canonical_root"], project["folder_name"])
    with _SOURCE_METADATA_LOCK:
        cached = _SOURCE_METADATA.get(cache_key)
        if cached is not None and cached[0] == signature:
            _SOURCE_METADATA.move_to_end(cache_key)
            return Event(copy.deepcopy(cached[1]), "", path)
    event = parse_event(path)
    _validate_source(event, path, project)
    if _fingerprint(path) != signature:
        raise ValueError("source changed during validation")
    with _SOURCE_METADATA_LOCK:
        _SOURCE_METADATA[cache_key] = (signature, copy.deepcopy(event.metadata))
        _SOURCE_METADATA.move_to_end(cache_key)
        while len(_SOURCE_METADATA) > _SOURCE_METADATA_LIMIT:
            _SOURCE_METADATA.popitem(last=False)
    return event


def find_orphan(project, key, committed, deleted_ids):
    """Caller holds project + maintenance locks; committed reads metadata only.

    Enumerate membership and lstat EVERY file on EVERY call, so another process,
    a crashed writer, deletion, replacement, or in-place modification is seen.
    Only unchanged validated metadata is reused. Bodies returned for recovery
    are always freshly parsed and revalidated. Symlink ancestors are checked
    once per directory per call, not once per file. No persistent index added.
    """
    root = v2_root()
    folder = root / "projects" / project["folder_name"] / "events"
    matches = []
    scanned = 0
    safe_parents = set()
    try:
        for path in sorted(folder.glob("[0-9][0-9][0-9][0-9]/[0-9][0-9]/*.md")):
            if not FORMAL_SOURCE.fullmatch(path.name):
                continue
            scanned += 1
            safe = False
            try:
                if path.parent not in safe_parents:
                    assert_safe_path(path, root)
                    safe_parents.add(path.parent)
                safe = True
                event = _source_metadata(path, project)
                if event.event_id in deleted_ids or not committed(event):
                    continue
            except Exception:
                identified = _reliable_key(path, project["project_id"]) if safe else None
                LOG.warning("SOURCE_RECOVERY_CORRUPT path=%s", path)
                if identified is None or identified == key:
                    raise SourceRecoveryError("SOURCE_RECOVERY_CORRUPT") from None
                continue
            if event.metadata["idempotency_key"] == key:
                # Never return the cached metadata-only Event. Preserve first
                # source bytes, body, and current routing checks on recovery.
                try:
                    current = parse_event(path)
                    _validate_source(current, path, project)
                    if current.metadata != event.metadata:
                        raise ValueError("source changed during recovery")
                except Exception:
                    raise SourceRecoveryError("SOURCE_RECOVERY_CORRUPT") from None
                matches.append(current)
        if len(matches) > 1:
            LOG.warning("SOURCE_RECOVERY_AMBIGUOUS paths=%s", [str(e.path) for e in matches])
            raise SourceRecoveryError("SOURCE_RECOVERY_AMBIGUOUS")
        return matches[0] if matches else None
    finally:
        LOG.info("SOURCE_RECOVERY_SCAN project=%s scanned=%d", project["project_id"], scanned)


def reference_errors(conn):
    """Validate exact (type, ref, project) multiplicities against indexing rules.

    Non-fact live events get event refs, except retract operations. Fact refs
    use fact_id, not assertion event_id. Facts originally asserted active retain
    historical refs after supersession, but retracted lineages have none.
    Originally inactive fact assertions do not get a search record.
    """
    expected = set()
    errors = []
    for row in conn.execute("SELECT * FROM events WHERE status <> 'retracted'"):
        if row["kind"] != "fact" and not (row["operation"] == "retract" and row["target_id"]):
            expected.add(("event", row["event_id"], row["project_id"]))
    for row in conn.execute(
            "SELECT f.fact_id,f.project_id,e.source_path FROM facts f "
            "JOIN events e ON e.event_id=f.assertion_event_id AND e.project_id=f.project_id "
            "WHERE f.status <> 'retracted' AND e.status <> 'retracted'"):
        try:
            from pathlib import Path
            event = parse_event(Path(row["source_path"]))
            if event.metadata["status"] == "active":
                expected.add(("fact", row["fact_id"], row["project_id"]))
        except Exception:
            errors.append({"code": "INDEX_SOURCE_INVALID", "ref_id": row["fact_id"]})
    for table in ("memory_fts", "vectors"):
        actual = Counter(tuple(row) for row in conn.execute(
            "SELECT record_type,ref_id,project_id FROM " + table))
        for ref, count in sorted(actual.items()):
            if ref[0] not in {"event", "fact"}:
                code = "INDEX_UNKNOWN_TYPE"
            elif ref not in expected:
                code = "INDEX_DANGLING_REFERENCE"
            elif count != 1:
                code = "INDEX_DUPLICATE_REFERENCE"
            else:
                continue
            errors.append({"code": code, "table": table, "reference": ref, "count": count})
        for ref in sorted(expected - set(actual)):
            errors.append({"code": "INDEX_MISSING_REFERENCE", "table": table, "reference": ref})
    return errors
