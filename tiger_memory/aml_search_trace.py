"""Opt-in, bounded in-memory Search diagnostics; never store source text.

This records adapter results, not HTTP delivery or the evaluator's prompt.
Content/query/user HMAC keys are random per recorder and never exported.
Optional storage-UUID join tokens use a separate public per-instance salt.
Neither mechanism reconstructs historical official questions or answers.
"""
from collections import deque
from contextvars import ContextVar
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import hmac
import json
import secrets
import threading
import time
import uuid

_CURRENT = ContextVar("aml_search_trace", default=None)
_COVERAGE_INTS = (
    "budget_bytes", "added_sources", "added_bytes", "direct_sources",
    "eligible_missing_sources", "omitted_sources", "returned_content_bytes",
    "inherited_sources", "inherited_retained", "relation_added_sources",
    "deferred_whole_sources", "recovered_whole_sources",
)
_EMBEDDING_FAILURES = frozenset({
    "embedding-budget-unavailable", "embedding-budget-work-blocked",
    "embedding-budget-exhausted", "embedding-budget-stopped",
    "embedding-checkpoint-busy", "embedding-checkpoint-unavailable",
    "embedding-request-budget-exhausted", "embedding-elapsed-budget-exhausted",
    "embedding-network-failed", "embedding-http-status",
    "embedding-http-429", "embedding-http-500", "embedding-http-502",
    "embedding-http-503", "embedding-http-504",
})


def storage_lookup_token(instance_id, domain, identifier):
    """Public per-instance salt, only for canonical storage UUIDs, not secrets.

    This is a join token for an operator holding the catalog, not encryption
    or an authentication proof. Never use this scheme for query/user/body text.
    """
    if (type(instance_id) is not str or len(instance_id) != 32
            or any(c not in "0123456789abcdef" for c in instance_id)):
        raise ValueError("invalid recorder instance")
    if domain not in {"project", "source"}:
        raise ValueError("invalid storage domain")
    if type(identifier) is not str or len(identifier) != 36:
        return None
    try:
        parsed = uuid.UUID(identifier)
    except (ValueError, AttributeError):
        return None
    if str(parsed) != identifier or parsed.version not in {4, 5}:
        return None
    return hashlib.sha256(
        ("aml-storage-lookup-v1\0" + instance_id + "\0" + domain + "\0" + identifier)
        .encode("ascii")).hexdigest()


def note(reason=None, *, inputs=None, diagnostics=None, storage_root=None, project=None):
    """Accept only internal reason labels and allowlisted numeric diagnostics."""
    current = _CURRENT.get()
    if current is None:
        return
    recorder, event = current
    try:
        if recorder.storage_links:
            if storage_root is not None:
                from .model import project_id_for_root
                token = storage_lookup_token(
                    recorder.instance_id, "project", project_id_for_root(storage_root))
                if token is not None:
                    # Expected routing after preflight; not evidence the store
                    # exists, the project was read, or any record was selected.
                    event["expected_project_token"] = token
            if project is not None:
                try:
                    identifier = project["project_id"]
                except (KeyError, TypeError, IndexError):
                    identifier = None
                token = storage_lookup_token(recorder.instance_id, "project", identifier)
                if token is not None:
                    event["retrieved_project_token"] = token
        if reason is not None:
            event["reason"] = reason
        if inputs is not None:
            query, user_id, top_k, options = inputs
            event.update(
                user_digest=recorder.digest("user", user_id),
                query_digest=recorder.digest("query", query),
                options_digest=recorder.digest("options", json.dumps(
                    options, ensure_ascii=False, separators=(",", ":"))),
                top_k=top_k,
            )
        if type(diagnostics) is dict:
            embedding = diagnostics.get("embedding")
            if type(embedding) is dict and embedding.get("status") == "unavailable":
                failure = embedding.get("fallback_reason")
                event["embedding_failure"] = (
                    failure if type(failure) is str and failure in _EMBEDDING_FAILURES
                    else "unclassified"
                )
            coverage = diagnostics.get("source_coverage")
            if type(coverage) is dict:
                event["coverage"] = {
                    key: coverage[key] for key in _COVERAGE_INTS
                    if type(coverage.get(key)) is int and coverage[key] >= 0
                }
    except Exception:
        # Diagnostics must not change retrieval behavior, including failures.
        event["diagnostic_incomplete"] = True


class SearchTrace:
    """Explicit launcher opt-in; no environment switch, disk writes or network."""

    def __init__(self, *, binding_sha256, capacity=64, storage_links=False):
        if (not isinstance(binding_sha256, str) or len(binding_sha256) != 64
                or any(c not in "0123456789abcdef" for c in binding_sha256)):
            raise ValueError("binding_sha256 must be a lowercase SHA-256 digest")
        if type(capacity) is not int or not 1 <= capacity <= 128:
            raise ValueError("capacity must be between 1 and 128")
        if type(storage_links) is not bool:
            raise ValueError("storage_links must be a bool")
        self.storage_links = storage_links
        self.binding_sha256 = binding_sha256
        self.capacity = capacity
        self.instance_id = secrets.token_hex(16)
        self._key = secrets.token_bytes(32)
        self._events = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._sequence = 0
        self._evicted = 0

    def digest(self, domain, value):
        return hmac.new(self._key, (domain + "\0" + value).encode("utf-8"),
                        hashlib.sha256).hexdigest()

    @contextmanager
    def capture(self):
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
        event = {"sequence": sequence, "started_unix_ns": time.time_ns(),
                 "reason": "unclassified", "adapter_completed": False}
        started = time.monotonic()
        token = _CURRENT.set((self, event))
        try:
            yield event
        finally:
            _CURRENT.reset(token)
            event["elapsed_ms"] = round((time.monotonic() - started) * 1000, 3)
            with self._lock:
                if len(self._events) == self.capacity:
                    self._evicted += 1
                self._events.append(event)

    def result(self, event, response):
        # The order is the actual adapter data order. No inference about inner
        # source blocks, tokenization, downstream clipping or answer quality.
        event["adapter_completed"] = True
        rows = response["data"]
        summaries = []
        for rank, row in enumerate(rows[:100], 1):
            body = row["content"]
            summaries.append({
                "rank": rank,
                "id_digest": self.digest("source-id", row["id"]),
                "content_digest": self.digest("content", body),
                "content_utf8_bytes": len(body.encode("utf-8")),
            })
            if self.storage_links:
                token = storage_lookup_token(self.instance_id, "source", row["id"])
                if token is not None:
                    summaries[-1]["source_token"] = token
        event.update(adapter_completed=True, result_count=len(rows),
                     results=summaries, results_metadata_truncated=len(rows) > 100,
                     total_content_utf8_bytes=sum(r["content_utf8_bytes"] for r in summaries))
        if self.storage_links:
            event["unlinked_result_ids"] = sum("source_token" not in row for row in summaries)

    def snapshot(self):
        with self._lock:
            result = {
                "schema": "aml-search-trace-v1",
                "instance_id": self.instance_id,
                "binding_sha256": self.binding_sha256,
                "capacity": self.capacity, "started": self._sequence,
                "evicted": self._evicted, "events": deepcopy(list(self._events)),
                "scope": "adapter-output-not-http-delivery-or-answer-input",
                "retention": "memory-only-until-eviction-clear-or-process-exit",
                "digests": "HMAC-SHA256-per-recorder-key-not-exported",
            }
            if self.storage_links:
                result["storage_lookup"] = {
                    "schema": "aml-storage-lookup-v1",
                    "salt": "instance_id",
                    "identifiers": "canonical-storage-uuid-v4-v5-only",
                    "scope": "catalog-join-not-official-sample-or-answer-mapping",
                }
            return result

    def clear(self):
        with self._lock:
            self._events.clear()
