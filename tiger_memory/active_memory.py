"""Explicit, local Active Memory v1 over the existing Markdown/SQLite runtime.

Source declarations are caller claims, not authentication or proof of a chat.
The public read surface is current-state context only. Logical forgetting keeps
the native immutable audit sources, which must not be exposed as tenant APIs.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import sqlite3
import sys
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict

from . import database, service
from .events import event_relative_path, maintenance_lock, project_lock
from .model import PLMError, json_compact, parse_time, project_id_for_root, display_slug, scoped_memory_base
from .security import assert_no_secrets, ensure_private_dir

FORMAT = "active_memory.v1"
OWNER_FORMAT = "active_memory.owner.v1"
MARKER = "# Active Memory private workspace\n"
KINDS = {"preference", "goal", "project_context"}
CONFIRMATIONS = {"user_stated", "user_confirmed", "caller_asserted"}
STOP = set("a an and are as at be by for from in is it of on or that the this to with 我们 你们 他们 这个 那个 一个 什么 如何 请问 可以 需要 关于".split())
SOURCE_FIELDS = {"session_id", "message_id", "text", "asserted_at", "role"}
WRITE_FIELDS = {"kind", "topic_key", "source", "request_id", "important", "confirmation", "project_id", "applicability", "valid_until"}


class ActiveMemoryError(PLMError):
    """Fixed public error codes never include caller data or filesystem paths."""


def _fail(code="ACTIVE_INTEGRITY"):
    raise ActiveMemoryError(code)


def _hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _text(value, maximum=256, empty=False):
    if not isinstance(value, str) or (not empty and not value.strip()) or "\x00" in value:
        _fail("ACTIVE_INPUT")
    try:
        if len(value.encode("utf-8")) > maximum:
            _fail("ACTIVE_INPUT")
    except UnicodeError:
        _fail("ACTIVE_INPUT")
    return value


def _time(value, empty=False):
    _text(value, 256, empty=empty)
    if empty and value == "":
        return ""
    try:
        return parse_time(value)
    except (PLMError, ValueError, TypeError):
        _fail("ACTIVE_INPUT")


def _entry_id(value):
    _text(value)
    try:
        if str(uuid.UUID(value)) != value:
            _fail("ACTIVE_INPUT")
    except (ValueError, TypeError):
        _fail("ACTIVE_INPUT")
    return value


def _slot(value):
    return (value["project_id"], value["kind"], value["topic_key"])


def _fact_key(value):
    return "active-memory:" + _hash(json_compact(list(_slot(value))))


def _tokens(value):
    result = {word.lower() for word in re.findall(r"[A-Za-z0-9]+", value)}
    for run in re.findall(r"[\u4e00-\u9fff]+", value):
        result.update(run[index:index + 2] for index in range(len(run) - 1))
    return result - STOP


def _context(entries):
    return json_compact({"content_role": "untrusted_user_data", "entries": entries, "format": FORMAT}) if entries else ""


def _nbytes(value):
    return len(value.encode("utf-8"))


def _string_leaves(value):
    """Materialize original strings in deterministic sorted-key traversal."""
    if isinstance(value, str):
        return [value]
    result = []
    if isinstance(value, dict):
        for key in sorted(value):
            result.extend(_string_leaves(value[key]))
    elif isinstance(value, list):
        for item in value:
            result.extend(_string_leaves(item))
    return result


def _assert_payload(payload):
    # In particular source.text reaches the unchanged scanner before JSON
    # escaping can hide quoted or multiline credential shapes.
    assert_no_secrets(_string_leaves(payload))
    assert_no_secrets([json_compact(payload)])


def _no_links(path):
    # Check the supplied spelling before resolve can erase a symlink identity.
    for item in [path, *path.parents]:
        if item.is_symlink():
            _fail("ACTIVE_IDENTITY")


def _atomic_identity(path, value):
    fd, temporary = tempfile.mkstemp(prefix=".active-init-", dir=str(path.parent))
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class ActiveStore:
    def __init__(self, base_dir: Path, user_id: str):
        try:
            path = Path(base_dir)
        except (TypeError, ValueError):
            _fail("ACTIVE_INPUT")
        if not path.is_absolute() or ".." in path.parts:
            _fail("ACTIVE_INPUT")
        _no_links(path)
        self.base_dir = path.resolve(strict=False)
        self.user_id = _text(user_id)
        assert_no_secrets([str(path), self.user_id])
        self.user_hash = _hash(user_id)
        self.root = self.base_dir / "active" / "users" / self.user_hash
        self.workspace = self.root / "workspace"
        self.native_project_id = project_id_for_root(self.workspace)
        self._check_paths()

    def _check_paths(self):
        _no_links(self.root)
        forbidden = [Path.home() / ".codex" / "project-memory"]
        for name in ("TIGER_MEMORY_DIR", "PROJECT_LONG_MEMORY_DIR"):
            if os.environ.get(name):
                forbidden.append(Path(os.environ[name]).expanduser())
        # Resolve only paths, never open/read a private catalog or source tree.
        for raw in forbidden:
            path = raw.resolve(strict=False)
            if self.base_dir == path or self.base_dir in path.parents or path in self.base_dir.parents:
                _fail("ACTIVE_IDENTITY")
        for path in (self.workspace, self.root / "owner.json", self.workspace / "AGENTS.md", self.root / "owner.lock"):
            _no_links(path)
        ancestor = self.workspace
        while not ancestor.exists():
            ancestor = ancestor.parent
        git_root = service._git_project_root(ancestor)
        if git_root is not None and git_root != self.workspace:
            _fail("ACTIVE_IDENTITY")

    def _owner(self):
        return {"format": OWNER_FORMAT, "user_id": self.user_id, "user_sha256": self.user_hash,
                "workspace": str(self.workspace)}

    def _verify_owner(self):
        try:
            owner = self.root / "owner.json"
            if not owner.is_file() or owner.stat().st_size > 4096:
                _fail("ACTIVE_IDENTITY")
            if json.loads(owner.read_text(encoding="utf-8")) != self._owner():
                _fail("ACTIVE_IDENTITY")
            marker = self.workspace / "AGENTS.md"
            if not self.workspace.is_dir() or not marker.is_file() or marker.stat().st_size > 4096 or marker.read_text(encoding="utf-8") != MARKER:
                _fail("ACTIVE_IDENTITY")
            if service.find_project_root(self.workspace) != self.workspace:
                _fail("ACTIVE_IDENTITY")
        except (OSError, ValueError, TypeError):
            _fail("ACTIVE_IDENTITY")

    def _ensure(self, create=False, rebuilding=False):
        self._check_paths()
        if not self.root.exists() and not create:
            return False
        if create:
            ensure_private_dir(self.root)
            lock = os.open(str(self.root / "owner.lock"), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            try:
                fcntl.flock(lock, fcntl.LOCK_EX)
                self._check_paths()
                owner = self.root / "owner.json"
                if not owner.exists():
                    if any(path.name != "owner.lock" for path in self.root.iterdir()):
                        _fail("ACTIVE_IDENTITY")
                    _atomic_identity(owner, json_compact(self._owner()))
                    ensure_private_dir(self.workspace)
                    _atomic_identity(self.workspace / "AGENTS.md", MARKER)
                self._verify_owner()
                # Finish first-time schema creation under the owner lock so a
                # competing creator cannot preflight a half-created catalog.
                _no_links(database.db_path())
                if not database.db_path().exists():
                    initial = database.connect()
                    initial.close()
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
                os.close(lock)
        else:
            self._verify_owner()
        catalog = database.db_path()
        _no_links(catalog)
        if catalog.exists():
            conn = database.connect_readonly()
            try:
                self._catalog_identity(conn, allow_empty=True)
            finally:
                conn.close()
        elif not create and not rebuilding:
            # Owner-only stores are possible after an interrupted initial write;
            # never treat missing persisted indexes as a successful empty read.
            _fail("ACTIVE_REBUILD_REQUIRED")
        return True

    def _catalog_identity(self, conn, allow_empty=False):
        projects = conn.execute("SELECT * FROM projects").fetchall()
        aliases = conn.execute("SELECT * FROM project_aliases").fetchall()
        if allow_empty and not projects and not aliases:
            return
        if len(projects) != 1 or len(aliases) != 1:
            _fail("ACTIVE_IDENTITY")
        project, alias = projects[0], aliases[0]
        if (project["project_id"] != self.native_project_id or project["canonical_root"] != str(self.workspace)
                or project["folder_name"] != display_slug(self.workspace)
                or alias["project_id"] != self.native_project_id or alias["alias_path"] != str(self.workspace) or alias["active"] != 1):
            _fail("ACTIVE_IDENTITY")
        manifest_path = self.root / "v2" / "projects" / project["folder_name"] / "manifest.json"
        _no_links(manifest_path)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (manifest["project_id"] != self.native_project_id or manifest["canonical_root"] != str(self.workspace)
                    or manifest["aliases"] != [str(self.workspace)]):
                _fail("ACTIVE_IDENTITY")
        except (OSError, ValueError, KeyError, TypeError):
            _fail("ACTIVE_IDENTITY")

    def _payload(self, operation, arguments, entry_id=""):
        if set(arguments) != WRITE_FIELDS:
            _fail("ACTIVE_INPUT")
        payload = dict(arguments)
        for key in ("kind", "topic_key", "request_id", "confirmation", "applicability"):
            _text(payload[key])
        _text(payload["project_id"], empty=True)
        if (payload["kind"] not in KINDS or payload["important"] is not True
                or payload["confirmation"] not in CONFIRMATIONS
                or payload["applicability"] not in {"always", "topical"}
                or (payload["applicability"] == "always" and payload["kind"] != "preference")):
            _fail("ACTIVE_INPUT")
        source = payload["source"]
        if not isinstance(source, dict) or set(source) != SOURCE_FIELDS or source["role"] != "user":
            _fail("ACTIVE_INPUT")
        for key in ("session_id", "message_id"):
            _text(source[key])
        text = _text(source["text"], 8192)
        if text != text.strip():
            _fail("ACTIVE_INPUT")
        asserted = _time(source["asserted_at"])
        until = _time(payload["valid_until"], empty=True)
        if until and until <= asserted:
            _fail("ACTIVE_INPUT")
        payload["source"] = dict(source)
        payload.update({"operation": operation, "user_id": self.user_id})
        if operation == "supersede":
            payload["entry_id"] = _entry_id(entry_id)
        elif operation != "remember":
            _fail("ACTIVE_INPUT")
        # Full caller payload is checked before creating even an owner manifest.
        _assert_payload(payload)
        return payload

    def _value(self, payload, source_id):
        source = payload["source"]
        return {"kind": payload["kind"], "topic_key": payload["topic_key"], "project_id": payload["project_id"],
                "statement": source["text"], "applicability": payload["applicability"], "confirmation": payload["confirmation"],
                "asserted_at": _time(source["asserted_at"]), "valid_until": _time(payload["valid_until"], empty=True),
                "source_ref": {"event_id": source_id, "session_id": source["session_id"], "message_id": source["message_id"], "sha256": _hash(source["text"])}}

    @staticmethod
    def _record(kind, payload):
        return {"active_memory": {"format": FORMAT, "type": kind, "payload": payload, "payload_sha256": _hash(json_compact(payload))}}

    def _load(self, conn):
        """Validate the rebuildable declaration/request registry and live slots.

        Caller holds the native project lock and a SQL transaction. Markdown is
        authoritative; no process-local registry or connection cache is kept.
        """
        try:
            return self._load_checked(conn)
        except ActiveMemoryError:
            raise
        except (PLMError, OSError, ValueError, TypeError, KeyError, IndexError, sqlite3.Error):
            _fail()

    def _load_checked(self, conn):
        self._catalog_identity(conn)
        rows = conn.execute("SELECT * FROM events").fetchall()
        facts = {row["fact_id"]: dict(row) for row in conn.execute("SELECT * FROM facts")}
        sources, entries, requests, operations = {}, {}, {}, []
        seen_sources, batches = {}, {}
        for row in rows:
            path = Path(row["source_path"])
            expected_path = self.root / "v2" / event_relative_path(display_slug(self.workspace), row["event_id"], row["recorded_at"])
            if path != expected_path:
                _fail()
            event = service._read_evidence_event(path)
            meta = event.metadata
            if (meta["project_id"] != self.native_project_id or meta["project_root"] != str(self.workspace)
                    or meta["scope"] != "session" or meta["scope_id"] != self.user_id or meta["source"] != FORMAT
                    or row["body"] != event.body or not service.batch_committed(event)):
                _fail()
            batches.setdefault(meta["batch_id"], set()).add(event.event_id)
            for key in ("project_id", "scope", "scope_id", "kind", "title", "observed_at", "recorded_at", "source", "source_hash", "idempotency_key", "visibility", "created_by"):
                if row[key] != meta[key]:
                    _fail()
            if row["event_id"] != meta["id"] or row["source_event_id"] != meta.get("source_event_id", "") or row["operation"] != meta.get("operation", "") or row["target_id"] != meta.get("target_id", ""):
                _fail()
            conditions = meta["conditions"]
            if set(conditions) != {"active_memory"}:
                _fail()
            record = conditions["active_memory"]
            if record["format"] != FORMAT:
                _fail()
            record_type = record["type"]
            if record_type == "source":
                if set(record) != {"format", "type", "payload", "payload_sha256"}:
                    _fail()
                payload = record["payload"]
                operation = payload["operation"]
                arguments = {key: payload[key] for key in WRITE_FIELDS}
                checked = self._payload(operation, arguments, payload.get("entry_id", ""))
                if checked != payload or record["payload_sha256"] != _hash(json_compact(payload)) or payload["user_id"] != self.user_id:
                    _fail()
                source = payload["source"]
                if (meta["kind"] != "episode" or meta.get("operation") or meta.get("source_event_id")
                        or event.body != json_compact({"text": source["text"]}) or meta["source_hash"] != _hash(source["text"])
                        or meta["observed_at"] != _time(source["asserted_at"])):
                    _fail()
                source_key = (source["session_id"], source["message_id"])
                if source_key in seen_sources or payload["request_id"] in requests:
                    _fail()
                item = {"payload": payload, "meta": meta, "status": row["status"], "event": event}
                sources[event.event_id] = item
                seen_sources[source_key] = event.event_id
                requests[payload["request_id"]] = {"payload": payload, "source": event.event_id}
            elif record_type == "entry":
                if set(record) != {"format", "type"} or meta["kind"] != "fact" or meta.get("operation") or _hash(event.body) != meta["source_hash"]:
                    _fail()
                identity = _entry_id(meta["fact_id"])
                if identity in entries or identity not in facts:
                    _fail()
                entries[identity] = {"meta": meta, "fact": facts[identity], "event_status": row["status"]}
            elif record_type == "forget":
                if set(record) != {"format", "type", "payload", "payload_sha256"}:
                    _fail()
                payload = record["payload"]
                if (set(payload) != {"operation", "user_id", "request_id", "entry_id"} or payload["operation"] != "forget_slot"
                        or payload["user_id"] != self.user_id or record["payload_sha256"] != _hash(json_compact(payload))
                        or meta["kind"] != "episode" or meta.get("operation") != "retract" or not meta.get("target_id") or meta.get("source_event_id")
                        or _hash(event.body) != meta["source_hash"] or row["status"] != "active"):
                    _fail()
                _text(payload["request_id"])
                _entry_id(payload["entry_id"])
                operations.append((payload, meta))
            else:
                _fail()
        if set(facts) != set(entries):
            _fail()
        for batch_id, ids in batches.items():
            marker_path = self.root / "v2" / "batches" / (batch_id + ".json")
            _no_links(marker_path)
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if (marker["project_id"] != self.native_project_id or set(marker["event_ids"]) != ids
                    or len(marker["event_ids"]) != len(ids)):
                _fail("ACTIVE_REBUILD_REQUIRED")
        # Fail closed if a source-committed batch is not represented in SQLite.
        disk_ids = set()
        folder = self.root / "v2" / "projects" / display_slug(self.workspace) / "events"
        for path in folder.glob("*/*/*.md"):
            event = service._read_evidence_event(path)
            if service.batch_committed(event):
                if event.event_id in disk_ids:
                    _fail()
                disk_ids.add(event.event_id)
        if disk_ids != {row["event_id"] for row in rows}:
            _fail("ACTIVE_REBUILD_REQUIRED")
        by_source, children, active = {}, {}, {}
        for identity, entry in entries.items():
            meta, fact = entry["meta"], entry["fact"]
            source_id = meta["source_event_id"]
            if source_id not in sources or source_id in by_source:
                _fail()
            source = sources[source_id]
            value = self._value(source["payload"], source_id)
            if (meta["value"] != value or meta["fact_key"] != _fact_key(value) or meta["valid_from"] != value["asserted_at"]
                    or meta["valid_to"] != value["valid_until"] or meta["status"] != "active"
                    or meta.get("expires_at") or meta["batch_id"] != source["meta"]["batch_id"]):
                _fail()
            for key, expected in {"project_id": self.native_project_id, "fact_key": _fact_key(value), "value_json": json_compact(value),
                                  "assertion_event_id": meta["id"], "source_event_id": source_id,
                                  "valid_from": value["asserted_at"], "recorded_at": meta["recorded_at"],
                                  "supersedes": meta["supersedes"], "expires_at": ""}.items():
                if fact[key] != expected:
                    _fail()
            predecessor = source["payload"].get("entry_id", "")
            if meta["supersedes"] != predecessor:
                _fail()
            if predecessor:
                if predecessor not in entries or predecessor in children:
                    _fail()
                old = entries[predecessor]["meta"]
                if old["fact_key"] != meta["fact_key"] or meta["valid_from"] < old["valid_from"]:
                    _fail()
                children[predecessor] = identity
            entry["value"] = value
            entry["source"] = source
            by_source[source_id] = identity
        if set(by_source) != set(sources):
            _fail()
        # Even equal valid_from timestamps must not permit a corrupted cycle.
        complete = set()
        for identity in entries:
            trail, node = set(), identity
            while node and node not in complete:
                if node in trail:
                    _fail()
                trail.add(node)
                node = entries[node]["meta"]["supersedes"]
            complete.update(trail)
        retracted = set()
        for payload, meta in operations:
            if payload["entry_id"] not in entries or meta["target_id"] not in by_source:
                _fail()
            entry = entries[payload["entry_id"]]
            target = entries[by_source[meta["target_id"]]]
            if _slot(entry["value"]) != _slot(target["value"]):
                _fail()
            request_id = payload["request_id"]
            previous = requests.get(request_id)
            if previous and (previous["payload"] != payload or previous.get("batch_id") != meta["batch_id"]):
                _fail()
            requests[request_id] = {"payload": payload, "batch_id": meta["batch_id"]}
            retracted.add(meta["target_id"])
        for identity, entry in entries.items():
            meta, fact, source = entry["meta"], entry["fact"], entry["source"]
            removed = meta["source_event_id"] in retracted
            expected_status = "retracted" if removed else ("superseded" if identity in children else "active")
            if (fact["status"] != expected_status or source["status"] != ("retracted" if removed else "active")
                    or entry["event_status"] != ("retracted" if removed else "active")):
                _fail()
            expected_until = entries[children[identity]]["meta"]["valid_from"] if identity in children else entry["value"]["valid_until"]
            if fact["valid_to"] != expected_until:
                _fail()
            if expected_status == "active":
                slot = _slot(entry["value"])
                if slot in active:
                    _fail("ACTIVE_SLOT_CONFLICT")
                active[slot] = identity
        return {"entries": entries, "sources": sources, "source_keys": seen_sources, "requests": requests, "active": active, "by_source": by_source}

    @contextmanager
    def _read(self):
        with scoped_memory_base(self.root):
            if not self._ensure():
                yield None
                return
            with maintenance_lock():
                with project_lock(self.native_project_id):
                    self._check_paths()
                    self._verify_owner()
                    conn = database.connect_readonly()
                    try:
                        conn.execute("BEGIN")
                        yield self._load(conn)
                    finally:
                        conn.close()

    @staticmethod
    def _result(identity, entry):
        return {"entry_id": identity, "source_event_id": entry["value"]["source_ref"]["event_id"],
                "source_hash": entry["value"]["source_ref"]["sha256"], "status": "active"}

    def _replay(self, state, payload):
        previous = state["requests"].get(payload["request_id"])
        if not previous:
            return None
        if previous["payload"] != payload:
            _fail("ACTIVE_REQUEST_CONFLICT")
        if payload["operation"] == "forget_slot":
            return {"forgotten": True}
        source_id = previous["source"]
        if state["sources"][source_id]["status"] != "active":
            _fail("ACTIVE_REQUEST_FORGOTTEN")
        identity = state["by_source"][source_id]
        return self._result(identity, state["entries"][identity])

    def remember(self, *, kind, topic_key, source, request_id, important, confirmation="user_stated", project_id="", applicability="topical", valid_until=""):
        arguments = {key: value for key, value in locals().items() if key in WRITE_FIELDS}
        return self._write(self._payload("remember", arguments))

    def supersede(self, entry_id, *, kind, topic_key, source, request_id, important, confirmation="user_stated", project_id="", applicability="topical", valid_until=""):
        arguments = {key: value for key, value in locals().items() if key in WRITE_FIELDS}
        return self._write(self._payload("supersede", arguments, entry_id))

    def _write(self, payload):
        # Run the unchanged native high-risk eligibility rule before persistence.
        preview = self._value(payload, "")
        claim = service.fact_claim(_fact_key(preview), preview)
        if service.HIGH_RISK_FACT.search(claim + "Active memory"):
            checked = service.verify_evidence("", {}, self.workspace, claim)
            if not checked["high_risk_eligible"]:
                _fail("ACTIVE_HIGH_RISK_EVIDENCE_REQUIRED")
        with scoped_memory_base(self.root):
            self._ensure(create=True)
            with service.event_batch(self.workspace, str(uuid.uuid4())) as batch:
                self._check_paths()
                self._verify_owner()
                state = self._load(batch["conn"])
                replay = self._replay(state, payload)
                if replay is not None:
                    return replay
                source = payload["source"]
                if (source["session_id"], source["message_id"]) in state["source_keys"]:
                    _fail("ACTIVE_SOURCE_CONFLICT")
                slot = _slot(preview)
                predecessor = payload.get("entry_id", "")
                if predecessor:
                    if state["active"].get(slot) != predecessor:
                        _fail("ACTIVE_PREDECESSOR_CONFLICT")
                elif slot in state["active"]:
                    _fail("ACTIVE_SLOT_CONFLICT")
                source_event = service.write_event(
                    self.workspace, "Active memory source", json_compact({"text": source["text"]}),
                    kind="episode", scope="session", scope_id=self.user_id, observed_at=_time(source["asserted_at"]),
                    source=FORMAT, visibility="private", created_by="active-memory", source_hash=_hash(source["text"]),
                    idempotency_key="am-source:" + _hash(payload["request_id"]),
                    extra={"conditions": self._record("source", payload)})
                value = self._value(payload, source_event.event_id)
                fact_event = service.write_fact(
                    self.workspace, _fact_key(value), value, title="Active memory", source=FORMAT,
                    scope="session", scope_id=self.user_id, source_event_id=source_event.event_id,
                    supersedes=predecessor, valid_from=value["asserted_at"], valid_to=value["valid_until"],
                    observed_at=value["asserted_at"], idempotency_key="am-entry:" + _hash(payload["request_id"]),
                    conditions={"active_memory": {"format": FORMAT, "type": "entry"}})
                return {"entry_id": fact_event.metadata["fact_id"], "source_event_id": source_event.event_id,
                        "source_hash": value["source_ref"]["sha256"], "status": "active"}

    def forget_slot(self, entry_id, *, request_id):
        payload = {"operation": "forget_slot", "user_id": self.user_id, "entry_id": _entry_id(entry_id), "request_id": _text(request_id)}
        _assert_payload(payload)
        with scoped_memory_base(self.root):
            if not self._ensure():
                _fail("ACTIVE_ENTRY_UNAVAILABLE")
            with service.event_batch(self.workspace, str(uuid.uuid4())) as batch:
                self._check_paths()
                self._verify_owner()
                state = self._load(batch["conn"])
                replay = self._replay(state, payload)
                if replay is not None:
                    return replay
                if entry_id not in state["entries"]:
                    _fail("ACTIVE_ENTRY_UNAVAILABLE")
                slot = _slot(state["entries"][entry_id]["value"])
                targets = sorted(entry["meta"]["source_event_id"] for entry in state["entries"].values() if _slot(entry["value"]) == slot)
                for source_id in targets:
                    service.write_event(
                        self.workspace, "Active memory retraction", "Logical slot retraction.", kind="episode",
                        scope="session", scope_id=self.user_id, source=FORMAT, visibility="private", created_by="active-memory",
                        idempotency_key="am-forget:" + _hash(json_compact([request_id, source_id])),
                        extra={"operation": "retract", "target_id": source_id, "conditions": self._record("forget", payload)})
                return {"forgotten": True}

    def prepare_context(self, task, *, project_id="", as_of, budget_bytes=8192):
        task = _text(task, 8192, empty=True)
        project_id = _text(project_id, empty=True)
        as_of = _time(as_of)
        if type(budget_bytes) is not int or not 0 <= budget_bytes <= 65536:
            _fail("ACTIVE_INPUT")
        diagnostics = {"selected": 0, "always_selected": 0, "topical_selected": 0, "omitted_budget": 0,
                       "omitted_irrelevant": 0, "overridden": 0, "context_bytes": 0}
        with self._read() as state:
            if state is None:
                return {"context": "", "diagnostics": diagnostics}
            candidates = {}
            for identity in state["active"].values():
                value = state["entries"][identity]["value"]
                if value["project_id"] not in ({"", project_id} if project_id else {""}):
                    continue
                if value["asserted_at"] > as_of or (value["valid_until"] and as_of >= value["valid_until"]):
                    continue
                entry = dict(value, entry_id=identity, status="active")
                key = (entry["kind"], entry["topic_key"])
                previous = candidates.get(key)
                if previous:
                    diagnostics["overridden"] += 1
                    if previous["project_id"]:
                        continue
                candidates[key] = entry
            task_tokens = _tokens(task)
            always, topical = [], []
            for entry in candidates.values():
                score = len(task_tokens & _tokens(entry["topic_key"] + " " + entry["statement"]))
                if entry["applicability"] == "always":
                    always.append(entry)
                elif score:
                    topical.append((score, entry))
                else:
                    diagnostics["omitted_irrelevant"] += 1
            def tie_order(entries):
                entries.sort(key=lambda entry: entry["entry_id"])
                entries.sort(key=lambda entry: entry["asserted_at"], reverse=True)
                entries.sort(key=lambda entry: bool(entry["project_id"]), reverse=True)
            tie_order(always)
            topical_entries = [entry for _, entry in topical]
            scores = {entry["entry_id"]: score for score, entry in topical}
            tie_order(topical_entries)
            topical_entries.sort(key=lambda entry: scores[entry["entry_id"]], reverse=True)
            selected, always_cost = [], 0
            for entry in always:
                cost = _nbytes(json_compact(entry)) + (1 if diagnostics["always_selected"] else 0)
                if always_cost + cost > budget_bytes // 4 or _nbytes(_context(selected + [entry])) > budget_bytes:
                    diagnostics["omitted_budget"] += 1
                    continue
                selected.append(entry)
                always_cost += cost
                diagnostics["always_selected"] += 1
            for entry in topical_entries:
                if _nbytes(_context(selected + [entry])) > budget_bytes:
                    diagnostics["omitted_budget"] += 1
                    continue
                selected.append(entry)
                diagnostics["topical_selected"] += 1
            context = _context(selected)
            diagnostics["selected"] = len(selected)
            diagnostics["context_bytes"] = _nbytes(context)
            return {"context": context, "diagnostics": diagnostics}

    def run_task(self, task, handler, *, project_id="", as_of, budget_bytes=8192):
        if not callable(handler):
            _fail("ACTIVE_INPUT")
        prepared = self.prepare_context(task, project_id=project_id, as_of=as_of, budget_bytes=budget_bytes)
        # prepare_context has fully unwound its store override before callbacks.
        return handler(task, prepared["context"])

    def rebuild(self):
        with scoped_memory_base(self.root):
            if not self._ensure(rebuilding=True):
                _fail("ACTIVE_ENTRY_UNAVAILABLE")
            result = service.rebuild_index()
            with self._read():
                pass
            return result


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        # argparse normally repeats an invalid argument value in its error.
        _fail("ACTIVE_INPUT")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _fail("ACTIVE_INPUT")
        result[key] = value
    return result


def main(argv=None):
    parser = _ArgumentParser(description="Explicit local Active Memory v1")
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--user-id", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("remember", "supersede", "forget-slot"):
        commands.add_parser(command)
    context = commands.add_parser("context")
    context.add_argument("--task", required=True)
    context.add_argument("--project-id", default="")
    context.add_argument("--as-of", required=True)
    context.add_argument("--budget-bytes", type=int, default=8192)
    try:
        args = parser.parse_args(argv)
        store = ActiveStore(Path(args.base_dir), args.user_id)
        if args.command == "context":
            result = store.prepare_context(args.task, project_id=args.project_id, as_of=args.as_of, budget_bytes=args.budget_bytes)
        else:
            raw = sys.stdin.buffer.read(65537)
            if len(raw) > 65536:
                _fail("ACTIVE_INPUT")
            text = raw.decode("utf-8")
            assert_no_secrets([text])
            payload = json.loads(text, object_pairs_hook=_unique_object)
            if not isinstance(payload, dict):
                _fail("ACTIVE_INPUT")
            method = {"remember": store.remember, "supersede": store.supersede, "forget-slot": store.forget_slot}[args.command]
            result = method(**payload)
        print(json_compact(result))
        return 0
    except (PLMError, OSError, ValueError, TypeError, sqlite3.Error):
        print('{"error":"ACTIVE_REQUEST_REJECTED"}', file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
