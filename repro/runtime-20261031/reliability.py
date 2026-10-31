"""Locked source recovery and typed derived-index validation."""
from collections import Counter, OrderedDict
import copy
import os
import time
from contextlib import contextmanager
import stat
import threading
import json
import logging
import re

from .events import event_relative_path, parse_event
from .recovery_watch import SourceWatch, WatchUnavailable, WatchInvalidated
from .stage_timings import timed
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


def _find_orphan_full(project, key, committed, deleted_ids):
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


# At most 32 live project indexes, each bounded by watch count and metadata bytes.
# A larger single project may use idle capacity, while process-wide accounting
# bounds the aggregate instead of multiplying the per-project allowance by 32.
# These are process-local hints: restart/eviction/loss rebuilds from source.
_INCREMENTAL = OrderedDict()
_INCREMENTAL_LOCK = threading.Lock()
_INCREMENTAL_LIMIT = 32
_INCREMENTAL_PID = os.getpid()
# Serialized metadata sizes are a stable budget proxy, not a Python RSS claim.
# Source count also bounds per-entry dictionaries and watcher bookkeeping.
_INDEX_BUDGET_LOCK = threading.Lock()
_INDEX_TOTAL_SOURCES = 0
_INDEX_TOTAL_METADATA_BYTES = 0
_INDEX_MAX_TOTAL_SOURCES = 65536
_INDEX_MAX_TOTAL_METADATA_BYTES = 128 * 1024 * 1024
_RECOVERY_STATS = Counter()
_FALLBACK_REASONS = Counter()
# Fixed public labels only: exceptions may contain paths or user-controlled
# text, so unknown values are never used as a status key.
_KNOWN_FALLBACK_REASONS = frozenset({
    "platform", "filesystem", "init", "nested_filesystem", "watch_limit",
    "watch_add", "mount_table_changed", "read", "closed", "event_limit",
    "truncated", "truncated_name", "lost_or_replaced", "directory_missing",
    "directory_change", "source_replaced", "source_limit", "metadata_limit",
    "cooldown", "source_removed", "unstable_sources", "changed_during_lookup",
    "cache_capacity", "index_resource_limit", "unexpected",
})


def _count_fallback(exc=None, reason=None):
    if reason is None:
        reason = str(exc) if isinstance(exc, (WatchUnavailable, WatchInvalidated)) else "unexpected"
    if reason not in _KNOWN_FALLBACK_REASONS:
        reason = "unexpected"
    with _INCREMENTAL_LOCK:
        _RECOVERY_STATS["full_scan_fallbacks"] += 1
        _FALLBACK_REASONS[reason] += 1


def _count(name, amount=1):
    with _INCREMENTAL_LOCK:
        _RECOVERY_STATS[name] += amount


def _maximum(name, value):
    with _INCREMENTAL_LOCK:
        _RECOVERY_STATS[name] = max(_RECOVERY_STATS[name], value)


def recovery_status():
    with _INCREMENTAL_LOCK:
        return {"mode": "linux-local-inotify-with-full-scan-fallback",
                "cached_projects": len(_INCREMENTAL),
                "counters": dict(_RECOVERY_STATS),
                "fallback_reasons": dict(_FALLBACK_REASONS),
                "capacity": _index_capacity_status()}


def _index_capacity_status():
    with _INDEX_BUDGET_LOCK:
        return {"cached_sources": _INDEX_TOTAL_SOURCES,
                "cached_metadata_bytes": _INDEX_TOTAL_METADATA_BYTES,
                "max_sources_per_project": _SourceIndex.MAX_SOURCES,
                "max_metadata_bytes_per_project": _SourceIndex.MAX_METADATA_BYTES,
                "max_sources_process": _INDEX_MAX_TOTAL_SOURCES,
                "max_metadata_bytes_process": _INDEX_MAX_TOTAL_METADATA_BYTES}


class _SourceIndex:
    # The former 2048-source ceiling made every later write in larger projects
    # repeat the authoritative full scan, including throughout its cooldown.
    # Permit those projects to grow, but retain per-project AND aggregate bounds.
    MAX_METADATA_BYTES = 16 * 1024 * 1024
    MAX_SOURCES = 8192
    FULL_VERIFY_INTERVAL_SECONDS = 60

    def __init__(self, project):
        self.project = dict(project)
        self.folder = v2_root() / "projects" / project["folder_name"] / "events"
        self.lock = threading.Lock()
        self.users = 0
        self.watch = None
        self.retry_after = 0
        self.last_full_verify = 0
        self.entries = {}
        self.by_key = {}
        self.sizes = {}
        self.bytes = 0
        self._budget_sources = 0
        self._budget_bytes = 0

    def close(self):
        global _INDEX_TOTAL_SOURCES, _INDEX_TOTAL_METADATA_BYTES
        if self.watch is not None:
            self.watch.close()
        self.watch = None
        with _INDEX_BUDGET_LOCK:
            _INDEX_TOTAL_SOURCES -= self._budget_sources
            _INDEX_TOTAL_METADATA_BYTES -= self._budget_bytes
            self._budget_sources = 0
            self._budget_bytes = 0
        self.entries.clear()
        self.by_key.clear()
        self.sizes.clear()
        self.bytes = 0

    def put(self, path):
        global _INDEX_TOTAL_SOURCES, _INDEX_TOTAL_METADATA_BYTES
        # Subscribe to the inode before observing its identity/metadata. Directory
        # watches cover atomic replacement, inode watches cover hard-link edits.
        self.watch.add(path)
        assert_safe_path(path, v2_root())
        event = _source_metadata(path, self.project)
        size = len(json.dumps(event.metadata).encode("utf-8"))
        if len(self.entries) >= self.MAX_SOURCES and path not in self.entries:
            raise WatchUnavailable("source_limit")
        if self.bytes - self.sizes.get(path, 0) + size > self.MAX_METADATA_BYTES:
            raise WatchUnavailable("metadata_limit")
        prior = self.entries.get(path)
        entry = Event(copy.deepcopy(event.metadata), "", path)
        count_delta = int(prior is None)
        size_delta = size - self.sizes.get(path, 0)
        with _INDEX_BUDGET_LOCK:
            if (_INDEX_TOTAL_SOURCES + count_delta > _INDEX_MAX_TOTAL_SOURCES or
                    _INDEX_TOTAL_METADATA_BYTES + size_delta > _INDEX_MAX_TOTAL_METADATA_BYTES):
                raise WatchUnavailable("index_resource_limit")
            # Reserve before publishing dict state. If allocation/publication
            # fails, close() releases the reservation even for a partial entry.
            _INDEX_TOTAL_SOURCES += count_delta
            _INDEX_TOTAL_METADATA_BYTES += size_delta
            self._budget_sources += count_delta
            self._budget_bytes += size_delta
            if prior is not None:
                old_key = prior.metadata["idempotency_key"]
                self.by_key[old_key].discard(path)
                if not self.by_key[old_key]:
                    del self.by_key[old_key]
            self.bytes += size_delta
            self.sizes[path] = size
            self.entries[path] = entry
            self.by_key.setdefault(event.metadata["idempotency_key"], set()).add(path)
        _maximum("snapshot_sources_observed_max", len(self.entries))

    def build(self):
        self.close()
        root = v2_root()
        # Missing/unsafe paths use the established full scanner semantics.
        assert_safe_path(self.folder / "guard", root)
        self.watch = SourceWatch(root, self.folder)
        # Watch ancestors before walking membership. Any rename/replacement or
        # directory membership event invalidates the snapshot, including a new
        # year/month directory created by an old writer during graceful drain.
        ancestors = [self.folder]
        while ancestors[-1] != root:
            ancestors.append(ancestors[-1].parent)
        for directory in reversed(ancestors):
            self.watch.add(directory, directory=True)
        for year in sorted(self.folder.glob("[0-9][0-9][0-9][0-9]")):
            assert_safe_path(year / "guard", root)
            self.watch.add(year, directory=True)
            for month in sorted(year.glob("[0-9][0-9]")):
                assert_safe_path(month / "guard", root)
                self.watch.add(month, directory=True)
                for path in sorted(month.glob("*.md")):
                    if FORMAL_SOURCE.fullmatch(path.name):
                        self.put(path)
        self.last_full_verify = time.monotonic()
        _count("snapshot_builds")
        _count("snapshot_files", len(self.entries))

    def _source_changes(self):
        # Use the same formal-source membership as the authoritative full scan
        # in BOTH reconciliation and the final race guard. Temporary files and
        # unrelated shared-root files must not consume reconciliation attempts.
        # SourceWatch raises on structural/loss events before returning paths.
        result = set()
        for path in self.watch.changes():
            try:
                rel = path.relative_to(self.folder)
            except ValueError:
                continue
            if (len(rel.parts) == 3 and re.fullmatch(r"[0-9]{4}", rel.parts[0])
                    and re.fullmatch(r"[0-9]{2}", rel.parts[1])
                    and FORMAL_SOURCE.fullmatch(rel.parts[2])):
                result.add(path)
        return result

    def sync(self):
        if self.watch is not None and time.monotonic() - self.last_full_verify >= self.FULL_VERIFY_INTERVAL_SECONDS:
            self.close()
            _count("periodic_full_verifications")
        if self.watch is None:
            if time.monotonic() < self.retry_after:
                raise WatchUnavailable("cooldown")
            self.build()
        # Caller holds maintenance + project locks. Cooperating writes are
        # excluded; bounded reconciliation also refuses unstable external edits.
        for _ in range(3):
            changed = self._source_changes()
            if not changed:
                _count("incremental_checks")
                return
            for path in sorted(changed):
                # Deleted/replaced inode notifications require rebuilding watches.
                if not path.exists():
                    raise WatchInvalidated("source_removed")
                self.put(path)
                _count("changed_files")
        raise WatchInvalidated("unstable_sources")

    def find(self, key, committed, deleted_ids):
        self.sync()
        matches = []
        for path in sorted(self.by_key.get(key, ())):
            event = self.entries[path]
            if event.event_id in deleted_ids:
                continue
            try:
                if not committed(event):
                    continue
                # No cached body, commit decision, or deletion decision is used.
                assert_safe_path(path, v2_root())
                current = parse_event(path)
                _validate_source(current, path, self.project)
                if current.metadata != event.metadata:
                    raise ValueError("source changed during recovery")
            except Exception:
                raise SourceRecoveryError("SOURCE_RECOVERY_CORRUPT") from None
            matches.append(current)
        if self._source_changes():
            raise WatchInvalidated("changed_during_lookup")
        if len(matches) > 1:
            raise SourceRecoveryError("SOURCE_RECOVERY_AMBIGUOUS")
        return matches[0] if matches else None


@contextmanager
def _borrow_index(project):
    global _INCREMENTAL_PID
    identity = (str(v2_root()), project["project_id"], project["canonical_root"], project["folder_name"])
    with _INCREMENTAL_LOCK:
        if os.getpid() != _INCREMENTAL_PID:
            for entry in _INCREMENTAL.values():
                entry.close()
            _INCREMENTAL.clear()
            _INCREMENTAL_PID = os.getpid()
        entry = _INCREMENTAL.get(identity)
        if entry is None:
            if len(_INCREMENTAL) >= _INCREMENTAL_LIMIT:
                for cache_key, idle in list(_INCREMENTAL.items()):
                    if not idle.users:
                        idle.close()
                        del _INCREMENTAL[cache_key]
                        break
            if len(_INCREMENTAL) >= _INCREMENTAL_LIMIT:
                entry = None
            else:
                entry = _SourceIndex(project)
                _INCREMENTAL[identity] = entry
        if entry is not None:
            entry.users += 1
            _INCREMENTAL.move_to_end(identity)
    try:
        yield entry
    finally:
        if entry is not None:
            with _INCREMENTAL_LOCK:
                entry.users -= 1


@timed("source_recovery")
def find_orphan(project, key, committed, deleted_ids):
    """Incremental validated routing on supported local Linux filesystems.

    A full source scan remains authoritative on cold start, invalidation,
    unsupported systems, corrupt sources, or watcher/resource failure. Both old
    and new writers are observed; no persistent journal/schema change required.
    """
    with _borrow_index(project) as index:
        if index is not None:
            with index.lock:
                try:
                    return index.find(key, committed, deleted_ids)
                except (SourceRecoveryError,):
                    # Preserve public corruption/ambiguity semantics; invalidate
                    # hints so a later repaired file cannot retain stale state.
                    index.close()
                    raise
                except Exception as exc:
                    if (isinstance(exc, WatchUnavailable) and str(exc) in {
                            "source_limit", "metadata_limit", "index_resource_limit",
                            "watch_limit", "watch_add"}):
                        # Validated prefix at failure, not total project size:
                        # a watch can fail on an ancestor before any source.
                        _maximum("snapshot_sources_at_capacity_failure_max", len(index.entries))
                    index.close()
                    if isinstance(exc, WatchUnavailable):
                        # Resource/platform failure only disables acceleration;
                        # full scans continue and never cache absence/results.
                        if time.monotonic() >= index.retry_after:
                            index.retry_after = time.monotonic() + 60
                    _count_fallback(exc)
                    return _find_orphan_full(project, key, committed, deleted_ids)
    _count_fallback(reason="cache_capacity")
    return _find_orphan_full(project, key, committed, deleted_ids)


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
