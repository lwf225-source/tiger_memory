"""Explicitly initialized, shared, fail-closed embedding spending ledger.

No credentials, transport, automatic initialization, refunds or budget resets.
Prices use integer nano-CNY; the per-row token ceiling is a supplier-contract
assertion, NEVER inferred from chunk byte length. Evidence must be reviewed by
the operator. A digest binds that decision; it does not itself verify a tariff.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import uuid
from urllib.parse import urlsplit

from .providers import ProviderUnavailable
from . import request_budget as request_limits


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _hash(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _need(value, code):
    if not value:
        raise ProviderUnavailable("embedding-budget-" + code)


def _private_path(path, *, directory=False):
    _need(path.is_absolute() and not any(p.is_symlink() for p in (path, *path.parents)), "unsafe-path")
    info = path.stat()
    _need(stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode), "unsafe-path")
    _need(info.st_uid == os.getuid() and info.st_mode & 0o077 == 0, "permissions")
    if not directory:
        _need(info.st_nlink == 1, "unsafe-path")


def _sync_dir(path):
    fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@dataclass(frozen=True)
class BudgetPolicy:
    endpoint: str
    region: str
    model: str
    dimensions: int
    token_limit_per_row: int
    price_nano_cny_per_token: int
    max_cost_nano_cny: int
    max_requests: int
    evidence_sha256: str
    test_only: bool = False

    def validate(self):
        for key in ("dimensions", "token_limit_per_row", "price_nano_cny_per_token",
                    "max_cost_nano_cny", "max_requests"):
            _need(type(getattr(self, key)) is int and 0 < getattr(self, key) <= 2**53, "invalid-policy")
        _need(self.dimensions in {64, 128, 256, 512, 768, 1024, 1536, 2048}
              and self.model == "text-embedding-v4" and type(self.test_only) is bool, "invalid-policy")
        _need(isinstance(self.evidence_sha256, str)
              and re.fullmatch(r"[0-9a-f]{64}", self.evidence_sha256), "evidence-required")
        try:
            url = urlsplit(self.endpoint)
            _need(not (url.username or url.password or url.query or url.fragment)
                  and not any(c.isspace() for c in self.endpoint), "endpoint")
            if self.test_only:
                _need(self.region == "test-only" and url.scheme == "http"
                      and url.hostname in {"127.0.0.1", "::1"}, "test-policy-endpoint")
            else:
                aliyun = (self.region == "cn-beijing" and url.scheme == "https"
                      and url.port in (None, 443) and url.path == "/compatible-mode/v1/embeddings"
                      and (url.hostname == "dashscope.aliyuncs.com" or bool(re.fullmatch(
                          r"[a-zA-Z0-9-]+\.cn-beijing\.maas\.aliyuncs\.com", url.hostname or ""))))
                dmx = (self.endpoint == "https://www.dmxapi.cn/v1/embeddings"
                       and self.region == "dmxapi" and self.dimensions == 1024)
                _need(aliyun or dmx, "endpoint")
        except (ValueError, TypeError, AttributeError):
            raise ProviderUnavailable("embedding-budget-invalid-policy") from None
        return self

    @property
    def fingerprint(self):
        self.validate()
        return _hash(asdict(self))


def _is_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _create_claim_tables(conn):
    # Individual DDL statements preserve the surrounding transaction. Never
    # use executescript here: it can commit before an explicit migration.
    conn.execute("""CREATE TABLE work_sets (
        attempt TEXT PRIMARY KEY REFERENCES attempts(attempt),
        namespace TEXT NOT NULL, fingerprint TEXT NOT NULL,
        raw_keys TEXT NOT NULL, checksum TEXT NOT NULL)""")
    conn.execute("""CREATE TABLE work_claims (
        namespace TEXT NOT NULL, fingerprint TEXT NOT NULL,
        raw_key TEXT NOT NULL, attempt TEXT NOT NULL REFERENCES work_sets(attempt),
        PRIMARY KEY(namespace, fingerprint, raw_key))""")


class _BudgetConnection(sqlite3.Connection):
    """One request D for all preparation SQL, not a fresh wait per statement.

    Already admitted reservation commits and response settlement deliberately
    finish without cooperative interruption. This does not bound kernel I/O.
    """
    finishing = False

    def check(self):
        if not self.finishing:
            request_limits.check("embedding-budget-sql")

    def _prepare_wait(self):
        self.check()
        if not self.finishing and request_limits.current_request_budget() is not None:
            seconds = request_limits.timeout(5.0, "embedding-budget-busy")
            super().execute("PRAGMA busy_timeout=%d" % max(0, int(seconds * 1000)))

    def execute(self, *args, **kwargs):
        self._prepare_wait()
        try:
            result = super().execute(*args, **kwargs)
        except sqlite3.Error:
            self.check()  # Preserve RequestCancelled instead of generic SQL error.
            raise
        self.check()
        return result

    def executemany(self, *args, **kwargs):
        self._prepare_wait()
        try:
            result = super().executemany(*args, **kwargs)
        except sqlite3.Error:
            self.check()
            raise
        self.check()
        return result

    def progress(self):
        try:
            self.check()
            return 0
        except ProviderUnavailable:
            return 1


class PersistentEmbeddingBudget:
    """One SQLite journal for all users and processes of a spending envelope.

    Call initialize only as an explicit provisioning action. Save ledger_id in
    separately owned startup configuration. Opening never recreates anything.
    UNKNOWN includes reserved-but-not-sent: conservative after any crash. A
    stop prevents new reservations; cache-only use is independently possible.
    This protects cooperating processes on a local POSIX filesystem, not a
    privileged writer rolling back both the database and its external identity.
    """

    @classmethod
    def initialize(cls, root: Path, policy: BudgetPolicy):
        policy.validate()
        root = Path(root)
        _need(root.is_absolute() and not any(p.is_symlink() for p in (root, *root.parents)), "unsafe-path")
        _need(not root.exists(), "already-initialized")
        # Never remove a partial initialization: any failure requires explicit
        # operator review, rather than recreating full credit at next startup.
        try:
            root.mkdir(mode=0o700)
            ledger_id = uuid.uuid4().hex
            db = root / "ledger.sqlite3"
            fd = os.open(str(db), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(fd)
            conn = sqlite3.connect(str(db))
            try:
                conn.execute("PRAGMA synchronous=FULL")
                conn.executescript("""
                    CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                    CREATE TABLE attempts (
                        seq INTEGER PRIMARY KEY AUTOINCREMENT,
                        attempt TEXT UNIQUE NOT NULL, rows INTEGER NOT NULL,
                        reserved_tokens INTEGER NOT NULL, state TEXT NOT NULL,
                        usage_tokens INTEGER, request_hash TEXT NOT NULL,
                        reason TEXT NOT NULL, checksum TEXT NOT NULL);
                """)
                conn.executemany("INSERT INTO metadata VALUES (?,?)", [
                    ("schema", "1"), ("ledger_id", ledger_id),
                    ("policy", _canonical(asdict(policy)).decode()),
                    ("policy_hash", policy.fingerprint), ("stopped", "0"),
                    ("claims_version", "1"), ("claims_legacy_attempts", "0")])
                _create_claim_tables(conn)
                conn.commit()
            finally:
                conn.close()
            marker = {"schema": 1, "ledger_id": ledger_id, "policy_hash": policy.fingerprint}
            fd = os.open(str(root / "identity.json"), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(_canonical(marker))
                stream.flush()
                os.fsync(stream.fileno())
            _sync_dir(root)
            _sync_dir(root.parent)
            return cls(root, policy, ledger_id=ledger_id)
        except (OSError, sqlite3.Error, ValueError):
            raise ProviderUnavailable("embedding-budget-initialize-failed") from None

    def __init__(self, root: Path, policy: BudgetPolicy, *, ledger_id: str):
        self.root = Path(root)
        self.policy = policy.validate()
        _need(isinstance(ledger_id, str) and re.fullmatch(r"[0-9a-f]{32}", ledger_id), "identity-required")
        self.ledger_id = ledger_id
        self.snapshot()

    def receipt(self):
        """Metadata for a separately persisted, operator-owned startup binding."""
        return {"schema": 1, "ledger_id": self.ledger_id, "root": str(self.root),
                "policy": asdict(self.policy)}

    @classmethod
    def from_receipt(cls, path: Path, *, allow_test_policy=False):
        """Open a previously provisioned ledger; NEVER create one on startup.

        The receipt must live outside the ledger directory so deleting the
        ledger cannot erase the startup's expected identity. Neither artifact
        is a cryptographic authority against a privileged rollback of both.
        """
        try:
            path = Path(path)
            _private_path(path)
            _need(path.stat().st_size <= 16384, "receipt-size")
            receipt = json.loads(path.read_bytes())
            _need(set(receipt) == {"schema", "ledger_id", "root", "policy"}
                  and type(receipt["schema"]) is int and receipt["schema"] == 1, "receipt-schema")
            policy = BudgetPolicy(**receipt["policy"]).validate()
            _need(not policy.test_only or allow_test_policy, "test-policy-refused")
            root = Path(receipt["root"])
            _need(root not in path.parents, "receipt-must-be-external")
            return cls(root, policy, ledger_id=receipt["ledger_id"])
        except ProviderUnavailable:
            raise
        except (OSError, ValueError, TypeError, KeyError):
            raise ProviderUnavailable("embedding-budget-receipt-invalid") from None

    def bind(self, endpoint, model, dimensions):
        _need((endpoint, model, dimensions) == (self.policy.endpoint, self.policy.model, self.policy.dimensions),
              "provider-binding")

    def _paths(self):
        _private_path(self.root, directory=True)
        db, marker = self.root / "ledger.sqlite3", self.root / "identity.json"
        for path in (db, marker):
            _private_path(path)
        _need(db.stat().st_size <= 64 * 1024 * 1024 and marker.stat().st_size <= 4096, "size")
        _need(json.loads(marker.read_bytes()) == {"schema": 1, "ledger_id": self.ledger_id,
              "policy_hash": self.policy.fingerprint}, "identity-mismatch")
        return db

    @contextmanager
    def _transaction(self, *, write=False, finish_admitted=False):
        conn = None
        try:
            if not finish_admitted:
                request_limits.check("embedding-budget-open")
            db = self._paths()
            identity = (db.stat().st_dev, db.stat().st_ino)
            # mode=rw is essential: a missing database must NEVER be created.
            conn = sqlite3.connect(db.as_uri() + "?mode=rw", uri=True,
                timeout=5.0 if finish_admitted else request_limits.timeout(5.0, "embedding-budget-open"),
                factory=_BudgetConnection)
            conn.finishing = finish_admitted
            conn.set_progress_handler(conn.progress, 1000)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("BEGIN IMMEDIATE")
            _need(self._paths() == db and (db.stat().st_dev, db.stat().st_ino) == identity, "identity-drift")
            _need(conn.execute("PRAGMA quick_check").fetchone()[0] == "ok", "corrupt")
            meta = dict(conn.execute("SELECT key,value FROM metadata"))
            expected = {"schema": "1", "ledger_id": self.ledger_id,
                        "policy": _canonical(asdict(self.policy)).decode(),
                        "policy_hash": self.policy.fingerprint, "stopped": meta.get("stopped")}
            if "claims_version" in meta:
                expected.update(claims_version="1", claims_legacy_attempts=meta.get("claims_legacy_attempts"))
            _need(meta == expected and meta["stopped"] in {"0", "1"}, "metadata-corrupt")
            rows = [dict(row) for row in conn.execute("SELECT * FROM attempts ORDER BY seq")]
            total = 0
            for index, row in enumerate(rows, 1):
                conn.check()
                checksum = row.pop("checksum")
                _need(checksum == _hash(row) and row["seq"] == index
                      and re.fullmatch(r"[0-9a-f]{32}", row["attempt"])
                      and re.fullmatch(r"[0-9a-f]{64}", row["request_hash"]), "attempt-corrupt")
                _need(type(row["rows"]) is int and 1 <= row["rows"] <= 10
                      and row["reserved_tokens"] == row["rows"] * self.policy.token_limit_per_row, "attempt-corrupt")
                if row["state"] == "SETTLED":
                    _need(type(row["usage_tokens"]) is int and 0 <= row["usage_tokens"] <= row["reserved_tokens"], "attempt-corrupt")
                    total += row["usage_tokens"]
                else:
                    _need(row["state"] == "UNKNOWN" and row["usage_tokens"] is None, "attempt-corrupt")
                    total += row["reserved_tokens"]
            _need(len(rows) <= self.policy.max_requests
                  and total * self.policy.price_nano_cny_per_token <= self.policy.max_cost_nano_cny, "balance-corrupt")
            sequence = conn.execute("SELECT seq FROM sqlite_sequence WHERE name='attempts'").fetchone()
            _need((sequence[0] if sequence else 0) == len(rows), "attempt-sequence-corrupt")
            self._validate_claims(conn, rows, meta)
            conn.check()
            # After the last admission check this one short transaction owns
            # its commit. No network or checkpoint lock acquisition occurs here.
            if write:
                conn.finishing = True
            yield conn, rows, total, meta["stopped"] == "1"
            _need(conn.execute("PRAGMA page_count").fetchone()[0]
                  * conn.execute("PRAGMA page_size").fetchone()[0] <= 64 * 1024 * 1024, "size")
            conn.commit()
        except ProviderUnavailable:
            raise
        except (OSError, sqlite3.Error, ValueError, TypeError, KeyError, OverflowError):
            if not finish_admitted and (conn is None or not conn.finishing):
                request_limits.check("embedding-budget-failed")
            raise ProviderUnavailable("embedding-budget-unavailable") from None
        finally:
            if conn is not None:
                conn.close()

    def _work_checksum(self, row):
        return _hash({"ledger_id": self.ledger_id, "policy_hash": self.policy.fingerprint, **row})

    @staticmethod
    def _work_context(namespace, fingerprint, raw_keys):
        _need(_is_sha(namespace) and _is_sha(fingerprint), "claim-context")
        _need(isinstance(raw_keys, (list, tuple)) and all(_is_sha(key) for key in raw_keys)
              and len(set(raw_keys)) == len(raw_keys), "claim-keys")
        return sorted(raw_keys)

    def _validate_claims(self, conn, rows, meta):
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "claims_version" not in meta:
            _need(not ({"work_sets", "work_claims"} & tables), "claims-format-corrupt")
            return
        count = meta["claims_legacy_attempts"]
        _need(isinstance(count, str) and count.isdecimal() and str(int(count)) == count
              and 0 <= int(count) <= len(rows), "claims-format-corrupt")
        bound = {r["attempt"]: r for r in rows[int(count):]}
        _need(conn.execute("SELECT COUNT(*) FROM work_sets").fetchone()[0] == len(bound), "claims-corrupt")
        expected_count = sum(r["rows"] for r in bound.values())
        _need(conn.execute("SELECT COUNT(*) FROM work_claims").fetchone()[0] == expected_count, "claims-corrupt")
        expected = set()
        for row in conn.execute("SELECT * FROM work_sets"):
            conn.check()
            work = dict(row)
            checksum = work.pop("checksum")
            _need(checksum == self._work_checksum(work) and work["attempt"] in bound
                  and isinstance(work["raw_keys"], str) and len(work["raw_keys"]) <= 2048, "claims-corrupt")
            keys = json.loads(work["raw_keys"])
            ordered = self._work_context(work["namespace"], work["fingerprint"], keys)
            _need(len(keys) == bound[work["attempt"]]["rows"]
                  and work["raw_keys"] == _canonical(ordered).decode(), "claims-corrupt")
            expected.update((work["namespace"], work["fingerprint"], key, work["attempt"]) for key in keys)
        actual = set()
        for row in conn.execute("SELECT namespace,fingerprint,raw_key,attempt FROM work_claims"):
            conn.check()
            actual.add(tuple(row))
        _need(actual == expected and len(expected) == expected_count
              and len({item[:3] for item in expected}) == expected_count, "claims-corrupt")

    def _unclaimed(self, conn, namespace, fingerprint, raw_keys):
        meta = dict(conn.execute("SELECT key,value FROM metadata"))
        _need(meta.get("claims_version") == "1", "claims-upgrade-required")
        # Old records cannot be assigned a namespace from a body hash. Even
        # SETTLED legacy work may have lost its data; no speculative release.
        _need(meta["claims_legacy_attempts"] == "0", "legacy-work-blocked")
        for key in raw_keys:
            _need(conn.execute("SELECT 1 FROM work_claims WHERE namespace=? AND fingerprint=? AND raw_key=?",
                              (namespace, fingerprint, key)).fetchone() is None, "work-blocked")

    def check_unclaimed(self, namespace, fingerprint, raw_keys):
        """Observational whole-miss check; reserve repeats it atomically.

        Caller has already validated and removed usable checkpoints. No data,
        attempts or sending permits are created by this check.
        """
        keys = self._work_context(namespace, fingerprint, raw_keys)
        if keys:
            with self._transaction() as (conn, _rows, _total, _stopped):
                self._unclaimed(conn, namespace, fingerprint, keys)

    def snapshot(self):
        with self._transaction() as (conn, rows, total, stopped):
            meta = dict(conn.execute("SELECT key,value FROM metadata"))
            version = int(meta.get("claims_version", "0"))
            return {"schema": 1, "ledger_id": self.ledger_id, "policy_sha256": self.policy.fingerprint,
                    "requests": len(rows), "unknown": sum(r["state"] == "UNKNOWN" for r in rows),
                    "settled": sum(r["state"] == "SETTLED" for r in rows),
                    "occupied_nano_cny": total * self.policy.price_nano_cny_per_token,
                    "cap_nano_cny": self.policy.max_cost_nano_cny, "stopped": stopped,
                    "claims_version": version,
                    "claims": conn.execute("SELECT COUNT(*) FROM work_claims").fetchone()[0] if version else 0,
                    "legacy_attempts": int(meta.get("claims_legacy_attempts", str(len(rows)))),
                    "migration_state_sha256": _hash({"metadata": meta, "attempts": rows})}

    def upgrade_claims(self, expected_state_sha256):
        """Explicit transactional format upgrade, NEVER an unlock or refund.

        No normal startup calls this. The reviewed snapshot digest must still
        match under the lock. Receipt/marker/identity and all old charges stay.
        Any legacy attempts keep new misses globally blocked after migration.
        """
        _need(_is_sha(expected_state_sha256), "migration-binding")
        with self._transaction(write=True) as (conn, rows, _total, _stopped):
            meta = dict(conn.execute("SELECT key,value FROM metadata"))
            _need("claims_version" not in meta, "already-upgraded")
            _need(_hash({"metadata": meta, "attempts": rows}) == expected_state_sha256, "migration-binding")
            _create_claim_tables(conn)
            conn.executemany("INSERT INTO metadata VALUES (?,?)", [
                ("claims_version", "1"), ("claims_legacy_attempts", str(len(rows)))])
        return self.snapshot()

    def reserve(self, row_count: int, request_hash: str, claim=None):
        _need(type(row_count) is int and 1 <= row_count <= 10, "rows")
        _need(_is_sha(request_hash), "request-hash")
        _need(isinstance(claim, dict) and set(claim) == {"namespace", "fingerprint", "raw_keys"}, "claim-required")
        keys = self._work_context(claim["namespace"], claim["fingerprint"], claim["raw_keys"])
        _need(len(keys) == row_count, "claim-rows")
        with self._transaction(write=True) as (conn, rows, total, stopped):
            self._unclaimed(conn, claim["namespace"], claim["fingerprint"], keys)
            _need(not stopped, "stopped")
            tokens = row_count * self.policy.token_limit_per_row
            _need(len(rows) < self.policy.max_requests
                  and (total + tokens) * self.policy.price_nano_cny_per_token <= self.policy.max_cost_nano_cny,
                  "exhausted")
            row = {"seq": len(rows) + 1, "attempt": uuid.uuid4().hex, "rows": row_count,
                   "reserved_tokens": tokens, "state": "UNKNOWN", "usage_tokens": None,
                   "request_hash": request_hash, "reason": "reserved"}
            conn.execute("INSERT INTO attempts VALUES (?,?,?,?,?,?,?,?,?)", (*row.values(), _hash(row)))
            attempt = row["attempt"]
            work = {"attempt": attempt, "namespace": claim["namespace"],
                    "fingerprint": claim["fingerprint"], "raw_keys": _canonical(keys).decode()}
            conn.execute("INSERT INTO work_sets VALUES (?,?,?,?,?)", (*work.values(), self._work_checksum(work)))
            conn.executemany("INSERT INTO work_claims VALUES (?,?,?,?)",
                             [(work["namespace"], work["fingerprint"], key, attempt) for key in keys])
        # Both semantic claims and money are durable before this send permit.
        return attempt

    def settle(self, attempt: str, usage_tokens: int):
        with self._transaction(write=True, finish_admitted=True) as (conn, rows, _, _stopped):
            found = [r for r in rows if r["attempt"] == attempt]
            _need(len(found) == 1, "attempt-missing")
            row = found[0]
            _need(row["state"] == "UNKNOWN", "already-settled")
            _need(type(usage_tokens) is int and 0 <= usage_tokens <= row["reserved_tokens"], "usage-untrusted")
            row.update(state="SETTLED", usage_tokens=usage_tokens, reason="validated-response")
            conn.execute("UPDATE attempts SET state=?,usage_tokens=?,reason=?,checksum=? WHERE attempt=?",
                         (row["state"], usage_tokens, row["reason"], _hash(row), attempt))

    def stop(self):
        with self._transaction(write=True, finish_admitted=True) as (conn, _rows, _total, _stopped):
            conn.execute("UPDATE metadata SET value='1' WHERE key='stopped'")
