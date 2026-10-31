from __future__ import annotations

import json
import os
import sqlite3
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .stage_timings import phase, timed, timed_acquire, timing_status
from .model import Event, PLMError, json_compact, utc_now, v2_root
from .security import ensure_private_dir
from .vector import encode_vector
from .paths import restored_legacy_path


DATABASE_VERSION = 5

# Platform session identifiers are private routing metadata.  AML derives a
# one-way digest from them for session-level RRF, but that digest must never
# become a lexical/semantic retrieval feature.
INTERNAL_RETRIEVAL_TAG_PREFIXES = ("aml-session:",)


def searchable_tags(tags: Iterable[object]) -> List[str]:
    """Return public retrieval tags, excluding internal routing metadata."""
    return [str(tag) for tag in tags
            if not str(tag).lower().startswith(INTERNAL_RETRIEVAL_TAG_PREFIXES)]


SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS projects (
    project_id TEXT PRIMARY KEY,
    canonical_root TEXT NOT NULL,
    display_name TEXT NOT NULL,
    folder_name TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS project_aliases (
    alias_path TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(project_id),
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(project_id),
    scope TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    title TEXT NOT NULL,
    tags_json TEXT NOT NULL,
    body TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    observed_at TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    source TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    source_legacy_path TEXT NOT NULL DEFAULT '',
    source_path TEXT NOT NULL UNIQUE,
    visibility TEXT NOT NULL,
    created_by TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    operation TEXT NOT NULL DEFAULT '',
    target_id TEXT NOT NULL DEFAULT ''
);
CREATE UNIQUE INDEX IF NOT EXISTS events_idempotency
ON events(project_id, idempotency_key) WHERE idempotency_key <> '';
CREATE INDEX IF NOT EXISTS events_project_time ON events(project_id, recorded_at DESC);
CREATE TABLE IF NOT EXISTS facts (
    fact_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(project_id),
    fact_key TEXT NOT NULL,
    value_json TEXT NOT NULL,
    status TEXT NOT NULL,
    valid_from TEXT NOT NULL,
    valid_to TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    retracted_at TEXT NOT NULL,
    supersedes TEXT NOT NULL,
    assertion_event_id TEXT NOT NULL REFERENCES events(event_id),
    source_event_id TEXT NOT NULL,
    confidence REAL NOT NULL,
    verified_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS facts_current ON facts(project_id, fact_key, status, recorded_at DESC);
CREATE TABLE IF NOT EXISTS entities (
    entity_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(project_id),
    entity_type TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(project_id, entity_type, canonical_name)
);
CREATE TABLE IF NOT EXISTS entity_aliases (
    project_id TEXT NOT NULL REFERENCES projects(project_id),
    alias TEXT NOT NULL,
    entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    PRIMARY KEY(project_id, alias)
);
CREATE TABLE IF NOT EXISTS event_entities (
    event_id TEXT NOT NULL REFERENCES events(event_id),
    entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    relation TEXT NOT NULL,
    PRIMARY KEY(event_id, entity_id, relation)
);
CREATE TABLE IF NOT EXISTS vectors (
    record_type TEXT NOT NULL,
    ref_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY(record_type, ref_id)
);
CREATE TABLE IF NOT EXISTS candidates (
    candidate_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(project_id),
    payload_json TEXT NOT NULL,
    state TEXT NOT NULL,
    reason TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(project_id, idempotency_key)
);
CREATE TABLE IF NOT EXISTS migrations (
    source_path TEXT PRIMARY KEY,
    source_hash TEXT NOT NULL,
    event_id TEXT NOT NULL,
    migrated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS shadow_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    project_id TEXT NOT NULL,
    query_hash TEXT NOT NULL,
    legacy_refs_json TEXT NOT NULL,
    v2_refs_json TEXT NOT NULL,
    overlap REAL NOT NULL,
    latency_ms REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS usage_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    project_id TEXT NOT NULL,
    operation TEXT NOT NULL,
    client TEXT NOT NULL,
    query_hash TEXT NOT NULL,
    result_count INTEGER NOT NULL,
    success INTEGER NOT NULL,
    latency_ms REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS usage_runs_created ON usage_runs(created_at DESC);
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    ref_id UNINDEXED,
    project_id UNINDEXED,
    record_type UNINDEXED,
    title,
    tags,
    body,
    tokenize='trigram'
);
"""


def db_path() -> Path:
    return v2_root() / "catalog.sqlite3"


class BusyRetryConnection(sqlite3.Connection):
    """Retry a single SQLite BUSY operation, never replay a transaction/body.

    BUSY_SNAPSHOT requires rollback by the caller and is not retryable here.
    LOCKED, IOERR, CORRUPT, FULL and all other errors propagate unchanged.
    The native busy handler waits briefly between attempts; there is no total
    request deadline or retry count. Statements remain inside SQLite's original
    transaction. commit() BUSY leaves that transaction intact for retry.
    """
    def _busy_call(self, operation, *args, **kwargs):
        attempts = 0
        while True:
            try:
                return operation(*args, **kwargs)
            except sqlite3.OperationalError as exc:
                code = getattr(exc, "sqlite_errorcode", None)
                if code is None or (code & 255) != sqlite3.SQLITE_BUSY or code == 517:
                    raise
                attempts += 1
                if attempts == 1 or attempts % 120 == 0:
                    logging.getLogger(__name__).warning(
                        'sqlite_busy_wait code=%d attempts=%d', code, attempts)
                with phase("db_busy_backoff"):
                    time.sleep(min(0.01 * attempts, 0.25))

    def execute(self, *args, **kwargs):
        if args and isinstance(args[0], str) and args[0].strip().upper() == "BEGIN IMMEDIATE":
            with phase("db_begin_wait"):
                return self._busy_call(super().execute, *args, **kwargs)
        return self._busy_call(super().execute, *args, **kwargs)

    @timed("db_commit")
    def commit(self):
        return self._busy_call(super().commit)


_LOOKUP_OBJECTS = ('fts_ref_lookup', 'fts_ref_lookup_key',
                   'fts_ref_lookup_vectors_insert', 'fts_ref_lookup_vectors_delete')


def _write_schema_ready(conn):
    try:
        row = conn.execute("SELECT value FROM metadata WHERE key='write_optimization_version'").fetchone()
    except sqlite3.OperationalError as exc:
        if getattr(exc, 'sqlite_errorcode', None) == sqlite3.SQLITE_ERROR and 'no such table' in str(exc):
            return False
        raise
    if row is None or row[0] != '1':
        return False
    return conn.execute("SELECT count(*) FROM sqlite_master WHERE name IN (?,?,?,?)",
                        _LOOKUP_OBJECTS).fetchone()[0] == len(_LOOKUP_OBJECTS)


def _initialize_write_schema(conn):
    conn.execute("BEGIN IMMEDIATE")
    try:
        if not _write_schema_ready(conn):
            # Execute individual DDL statements so schema, lookup backfill and
            # the ready marker commit atomically (executescript commits early).
            for statement in SCHEMA.split(';'):
                if statement.strip():
                    conn.execute(statement)
            columns = {row[1] for row in conn.execute("PRAGMA table_info(events)")}
            if 'source_event_id' not in columns:
                conn.execute("ALTER TABLE events ADD COLUMN source_event_id TEXT NOT NULL DEFAULT ''")
            conn.execute("CREATE INDEX IF NOT EXISTS events_source ON events(source_event_id)")
            conn.execute("CREATE TABLE IF NOT EXISTS fts_ref_lookup (fts_rowid INTEGER PRIMARY KEY, record_type TEXT NOT NULL, ref_id TEXT NOT NULL)")
            conn.execute("CREATE INDEX IF NOT EXISTS fts_ref_lookup_key ON fts_ref_lookup(record_type,ref_id)")
            conn.execute("DELETE FROM fts_ref_lookup")
            conn.execute("INSERT INTO fts_ref_lookup SELECT rowid,record_type,ref_id FROM memory_fts")
            # All supported writers (including the draining old binary) delete
            # FTS then vectors, and insert FTS then vectors in ONE transaction.
            # Normal table triggers preserve that compatibility without touching
            # FTS shadow tables or changing any FTS rowid/content/ranking.
            conn.execute("""CREATE TRIGGER IF NOT EXISTS fts_ref_lookup_vectors_delete
                AFTER DELETE ON vectors BEGIN
                DELETE FROM fts_ref_lookup WHERE record_type=OLD.record_type AND ref_id=OLD.ref_id;
                END""")
            conn.execute("""CREATE TRIGGER IF NOT EXISTS fts_ref_lookup_vectors_insert
                AFTER INSERT ON vectors BEGIN
                DELETE FROM fts_ref_lookup WHERE record_type=NEW.record_type AND ref_id=NEW.ref_id;
                INSERT INTO fts_ref_lookup
                  SELECT rowid,record_type,ref_id FROM memory_fts
                  WHERE rowid=(SELECT rowid FROM memory_fts ORDER BY rowid DESC LIMIT 1)
                    AND record_type=NEW.record_type AND ref_id=NEW.ref_id;
                INSERT INTO fts_ref_lookup
                  SELECT rowid,record_type,ref_id FROM memory_fts
                  WHERE record_type=NEW.record_type AND ref_id=NEW.ref_id
                  LIMIT CASE WHEN EXISTS (SELECT 1 FROM fts_ref_lookup
                         WHERE record_type=NEW.record_type AND ref_id=NEW.ref_id) THEN 0 ELSE -1 END;
                END""")
            conn.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES('database_version',?)", (str(DATABASE_VERSION),))
            conn.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES('write_optimization_version','1')")
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


@timed("db_connect")
def connect(path: Optional[Path] = None) -> sqlite3.Connection:
    target = path or db_path()
    ensure_private_dir(target.parent)
    conn = sqlite3.connect(str(target), timeout=0.25, factory=BusyRetryConnection)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=250")
        if conn.execute("PRAGMA journal_mode").fetchone()[0].lower() != 'wal':
            conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA foreign_keys=ON")
        if not _write_schema_ready(conn):
            _initialize_write_schema(conn)
        try:
            os.chmod(str(target), 0o600)
        except OSError:
            pass
        return conn
    except BaseException:
        conn.close()
        raise


def connect_readonly(path: Optional[Path] = None) -> sqlite3.Connection:
    """Open the derived search index without creating or mutating anything."""
    target = (path or db_path()).expanduser().resolve(strict=False)
    if not target.is_file():
        raise PLMError("memory index does not exist: %s" % target)
    conn = sqlite3.connect(target.as_uri() + "?mode=ro", uri=True,
                           timeout=30.0)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn
    except BaseException:
        conn.close()
        raise



def register_project(conn: sqlite3.Connection, project_id: str, canonical_root: str, display_name: str, folder_name: str) -> None:
    now = utc_now()
    conn.execute(
        "INSERT OR IGNORE INTO projects(project_id,canonical_root,display_name,folder_name,created_at) VALUES(?,?,?,?,?)",
        (project_id, canonical_root, display_name, folder_name, now),
    )
    row = conn.execute("SELECT project_id FROM projects WHERE project_id=?", (project_id,)).fetchone()
    if not row:
        raise PLMError("could not register project")
    conn.execute(
        "INSERT INTO project_aliases(alias_path,project_id,active,created_at) VALUES(?,?,1,?) "
        "ON CONFLICT(alias_path) DO UPDATE SET project_id=excluded.project_id, active=1 "
        "WHERE project_aliases.project_id IS NOT excluded.project_id OR project_aliases.active IS NOT 1",
        (canonical_root, project_id, now),
    )


def project_for_alias(conn: sqlite3.Connection, path: str) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT p.* FROM project_aliases a JOIN projects p ON p.project_id=a.project_id "
        "WHERE a.alias_path=? AND a.active=1",
        (path,),
    ).fetchone()


def event_for_idempotency(conn: sqlite3.Connection, project_id: str, key: str) -> Optional[sqlite3.Row]:
    if not key:
        return None
    return conn.execute("SELECT * FROM events WHERE project_id=? AND idempotency_key=? AND idempotency_key <> ''", (project_id, key)).fetchone()


def _remove_search_record(conn: sqlite3.Connection, record_type: str, ref_id: str) -> None:
    # Exact lookup supports all rows for a key (including duplicates present at
    # migration), and guards against deleting a different record on corruption.
    rowids = conn.execute("SELECT fts_rowid FROM fts_ref_lookup WHERE record_type=? AND ref_id=?",
                          (record_type, ref_id)).fetchall()
    for row in rowids:
        conn.execute("DELETE FROM memory_fts WHERE rowid=? AND record_type=? AND ref_id=?",
                     (row[0], record_type, ref_id))
    conn.execute("DELETE FROM fts_ref_lookup WHERE record_type=? AND ref_id=?", (record_type, ref_id))
    conn.execute("DELETE FROM vectors WHERE record_type=? AND ref_id=?", (record_type, ref_id))


@dataclass(frozen=True)
class _PreparedLexicalVector:
    """Ephemeral exact-input binding; never stored in metadata or shared cache."""
    text: str
    vector: bytes


@timed("lexical_prepare")
def prepare_lexical_vector(title: str, tags: str, body: str):
    text = " ".join((title, tags, body))
    return _PreparedLexicalVector(text, encode_vector(text))


def _index_search_record(
    conn: sqlite3.Connection,
    record_type: str,
    ref_id: str,
    project_id: str,
    title: str,
    tags: str,
    body: str,
    *, prepared_lexical=None,
) -> None:
    _remove_search_record(conn, record_type, ref_id)
    conn.execute(
        "INSERT INTO memory_fts(ref_id,project_id,record_type,title,tags,body) VALUES(?,?,?,?,?,?)",
        (ref_id, project_id, record_type, title, tags, body),
    )
    text = " ".join((title, tags, body))
    # A precomputed value is usable only for these exact lexical input bytes.
    # Generic writers and changed inputs keep the original computation path.
    vector = (prepared_lexical.vector
              if isinstance(prepared_lexical, _PreparedLexicalVector)
              and prepared_lexical.text == text else encode_vector(text))
    conn.execute(
        "INSERT OR REPLACE INTO vectors(record_type,ref_id,project_id,vector) VALUES(?,?,?,?)",
        (record_type, ref_id, project_id, vector),
    )


def index_event(conn: sqlite3.Connection, event: Event, *, prepared_lexical=None) -> None:
    meta = event.metadata
    path = str(event.path or "")
    tags = meta.get("tags", [])
    operation = str(meta.get("operation", ""))
    target_id = str(meta.get("target_id", ""))
    conn.execute(
        "INSERT OR REPLACE INTO events(event_id,project_id,scope,scope_id,kind,title,tags_json,body,status,"
        "observed_at,recorded_at,source,source_hash,source_legacy_path,source_path,visibility,created_by,idempotency_key,operation,target_id) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            meta["id"], meta["project_id"], meta["scope"], meta["scope_id"], meta["kind"],
            meta["title"], json_compact(tags), event.body, "active", meta["observed_at"],
            meta["recorded_at"], meta["source"], meta["source_hash"], restored_legacy_path(str(meta.get("source_legacy_path", ""))), path,
            meta["visibility"], meta["created_by"], meta["idempotency_key"], operation, target_id,
        ),
    )
    conn.execute("UPDATE events SET source_event_id=? WHERE event_id=?", (str(meta.get("source_event_id", "")), meta["id"]))
    if operation == "retract" and target_id:
        retract_lineage(conn, target_id, meta["project_id"], meta["recorded_at"])
        return
    if meta["kind"] != "fact":
        _index_search_record(conn, "event", meta["id"], meta["project_id"], meta["title"], " ".join(searchable_tags(tags)), event.body,
                             prepared_lexical=prepared_lexical)
        if meta.get("entities"):
            add_entities(conn, meta["project_id"], meta["id"], meta["entities"])
        return

    fact_id = str(meta["fact_id"])
    supersedes = str(meta.get("supersedes") or "")
    valid_from = str(meta.get("valid_from") or meta["observed_at"])
    if supersedes:
        conn.execute(
            "UPDATE facts SET status='superseded', valid_to=? WHERE fact_id=? AND project_id=?",
            (valid_from, supersedes, meta["project_id"]),
        )
    conn.execute(
        "INSERT OR REPLACE INTO facts(fact_id,project_id,fact_key,value_json,status,valid_from,valid_to,recorded_at,"
        "retracted_at,supersedes,assertion_event_id,source_event_id,confidence,verified_at,expires_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            fact_id, meta["project_id"], meta["fact_key"], json_compact(meta["value"]), meta["status"],
            valid_from, str(meta.get("valid_to") or ""), meta["recorded_at"],
            str(meta.get("retracted_at") or ""), supersedes, meta["id"], str(meta.get("source_event_id") or meta["id"]),
            float(meta.get("confidence", 0.8)), str(meta.get("verified_at") or ""),
            str(meta.get("expires_at") or ""),
        ),
    )
    if meta["status"] == "active":
        value_text = meta["value"] if isinstance(meta["value"], str) else json_compact(meta["value"])
        if meta.get("conditions") is not None:
            value_text += "\nConditions: " + json_compact(meta["conditions"])
        _index_search_record(conn, "fact", fact_id, meta["project_id"], meta["fact_key"], " ".join(searchable_tags(tags)), value_text)
    if meta.get("entities"):
        add_entities(conn, meta["project_id"], meta["id"], meta["entities"])


def lineage_ids(conn: sqlite3.Connection, target_id: str, project_id: str) -> List[str]:
    """Return source and all derived assertions, never another project."""
    row = conn.execute("SELECT assertion_event_id FROM facts WHERE fact_id=? AND project_id=?", (target_id, project_id)).fetchone()
    pending = [row[0] if row else target_id]
    seen = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        candidate = conn.execute("SELECT idempotency_key FROM events WHERE event_id=? AND project_id=?", (current, project_id)).fetchone()
        if candidate and candidate[0].startswith(("candidate-summary:", "candidate-fact:")):
            candidate_id = candidate[0].split(":")[1]
            # Until per-span dependencies are represented, a candidate is one
            # conservative privacy unit: its summary may repeat any sibling Fact.
            pending.extend(row[0] for row in conn.execute(
                "SELECT event_id FROM events WHERE project_id=? AND (idempotency_key=? OR idempotency_key LIKE ?)",
                (project_id, "candidate-summary:" + candidate_id, "candidate-fact:" + candidate_id + ":%")))
        pending.extend(row[0] for row in conn.execute(
            "SELECT event_id FROM events WHERE project_id=? AND source_event_id=? "
            "UNION SELECT assertion_event_id FROM facts WHERE project_id=? AND source_event_id=?",
            (project_id, current, project_id, current)))
    return sorted(seen)


def retract_lineage(conn: sqlite3.Connection, target_id: str, project_id: str, recorded_at: str) -> None:
    for event_id in lineage_ids(conn, target_id, project_id):
        conn.execute("UPDATE events SET status='retracted' WHERE event_id=? AND project_id=?", (event_id, project_id))
        _remove_search_record(conn, "event", event_id)
        for row in conn.execute("SELECT fact_id FROM facts WHERE assertion_event_id=?", (event_id,)).fetchall():
            conn.execute("UPDATE facts SET status='retracted', retracted_at=? WHERE fact_id=?", (recorded_at, row[0]))
            _remove_search_record(conn, "fact", row[0])
        conn.execute("DELETE FROM event_entities WHERE event_id=?", (event_id,))
    conn.execute("DELETE FROM entity_aliases WHERE entity_id NOT IN (SELECT entity_id FROM event_entities)")
    conn.execute("DELETE FROM entities WHERE entity_id NOT IN (SELECT entity_id FROM event_entities)")


def invalidate_superseded_derivations(conn, fact_id: str, project_id: str, recorded_at: str) -> None:
    row = conn.execute("SELECT assertion_event_id FROM facts WHERE fact_id=? AND project_id=?", (fact_id, project_id)).fetchone()
    if row:
        for event_id in lineage_ids(conn, row[0], project_id):
            if event_id != row[0]:
                retract_lineage(conn, event_id, project_id, recorded_at)


def add_entities(conn: sqlite3.Connection, project_id: str, event_id: str, entities: Sequence[Dict[str, Any]]) -> None:
    import uuid
    for item in entities:
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        entity_type = str(item.get("type", "concept")).strip() or "concept"
        row = conn.execute(
            "SELECT entity_id FROM entities WHERE project_id=? AND entity_type=? AND canonical_name=?",
            (project_id, entity_type, name),
        ).fetchone()
        entity_id = row["entity_id"] if row else str(uuid.uuid4())
        conn.execute(
            "INSERT OR IGNORE INTO entities(entity_id,project_id,entity_type,canonical_name,created_at) VALUES(?,?,?,?,?)",
            (entity_id, project_id, entity_type, name, utc_now()),
        )
        aliases = set(str(alias).strip() for alias in item.get("aliases", []) if str(alias).strip())
        aliases.add(name)
        for alias in aliases:
            conn.execute(
                "INSERT OR REPLACE INTO entity_aliases(project_id,alias,entity_id) VALUES(?,?,?)",
                (project_id, alias.lower(), entity_id),
            )
        conn.execute(
            "INSERT OR IGNORE INTO event_entities(event_id,entity_id,relation) VALUES(?,?,?)",
            (event_id, entity_id, str(item.get("relation", "mentions"))),
        )


def integrity(conn: sqlite3.Connection) -> Dict[str, Any]:
    result = conn.execute("PRAGMA integrity_check").fetchone()[0]
    counts = {}
    existing = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")}
    for table in ("projects", "project_aliases", "events", "facts", "entities", "candidates", "migrations", "shadow_runs", "usage_runs"):
        counts[table] = conn.execute("SELECT COUNT(*) FROM %s" % table).fetchone()[0] if table in existing else 0
    counts["fts"] = conn.execute("SELECT COUNT(*) FROM memory_fts").fetchone()[0]
    counts["vectors"] = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
    return {"integrity": result, "counts": counts}
