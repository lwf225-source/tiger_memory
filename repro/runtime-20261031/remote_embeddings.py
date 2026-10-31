"""Explicit opt-in DashScope embeddings, using only the Python standard library.

Inputs leave the machine only when ``embed`` has uncached work. Long inputs are
split on Unicode boundaries at 8192 UTF-8 bytes; this is NOT a token bound.
All chunks contribute to a mean vector followed by L2
normalization. No input is silently truncated. Caches contain hashes and numbers.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from time import monotonic, sleep, time
import threading
import ssl
import http.client
import sys
from typing import Any, Dict, List, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from contextlib import nullcontext, contextmanager

from .stage_timings import phase, timed, timed_acquire, timing_status
from .providers import ProviderUnavailable, _digest
from .embedding_checkpoint import VectorCheckpoint, digest
from .transport_diagnostics import http_error_metadata


# Starts in legacy-compatible exclusive mode. Deployment enables shared mode
# only AFTER old connections and old nginx workers reach zero. A restart is safe
# by default and exposes the pending activation through authenticated status.
_CHECKPOINT_PARALLEL_ENABLED = False
_CHECKPOINT_PARALLEL_REQUESTED = False
_CHECKPOINT_LEGACY_ACTIVE = 0
_CHECKPOINT_MODE_CONDITION = threading.Condition()

def enable_checkpoint_parallel(*_):
    # Signal-safe: request only; never wait/acquire a Python lock in handler.
    global _CHECKPOINT_PARALLEL_REQUESTED
    _CHECKPOINT_PARALLEL_REQUESTED = True

def _activate_checkpoint_parallel_if_ready():
    global _CHECKPOINT_PARALLEL_ENABLED
    if _CHECKPOINT_PARALLEL_REQUESTED and _CHECKPOINT_LEGACY_ACTIVE == 0:
        _CHECKPOINT_PARALLEL_ENABLED = True
        _CHECKPOINT_MODE_CONDITION.notify_all()

@contextmanager
def checkpoint_access_mode():
    global _CHECKPOINT_LEGACY_ACTIVE
    with _CHECKPOINT_MODE_CONDITION:
        _activate_checkpoint_parallel_if_ready()
        while _CHECKPOINT_PARALLEL_REQUESTED and not _CHECKPOINT_PARALLEL_ENABLED:
            _CHECKPOINT_MODE_CONDITION.wait()
        parallel = _CHECKPOINT_PARALLEL_ENABLED
        if not parallel:
            _CHECKPOINT_LEGACY_ACTIVE += 1
    try:
        yield parallel
    finally:
        if not parallel:
            with _CHECKPOINT_MODE_CONDITION:
                _CHECKPOINT_LEGACY_ACTIVE -= 1
                _activate_checkpoint_parallel_if_ready()

def checkpoint_concurrency_status():
    with _CHECKPOINT_MODE_CONDITION:
        _activate_checkpoint_parallel_if_ready()
        return {"policy": "exclusive-until-drained-then-input-stripes-v1",
                "parallel_enabled": _CHECKPOINT_PARALLEL_ENABLED,
                "activation_requested": _CHECKPOINT_PARALLEL_REQUESTED,
                "legacy_active": _CHECKPOINT_LEGACY_ACTIVE,
                "input_lock_stripes": 256}

_RECOVERY_LOCK = threading.Lock()
_RECOVERY = {"transient_failures": 0, "recovered_batches": 0, "last_reason": None, "last_event_unix": None}

def network_recovery_status():
    with _RECOVERY_LOCK:
        return dict(_RECOVERY)

def _record_recovery(reason, attempt, recovered=False):
    with _RECOVERY_LOCK:
        _RECOVERY["recovered_batches" if recovered else "transient_failures"] += 1
        _RECOVERY["last_reason"] = reason
        _RECOVERY["last_event_unix"] = time()
    print(json.dumps({"event": "embedding_transport_recovery", "reason": reason,
                      "attempt": attempt, "recovered": recovered, "time_unix": time()}),
          file=sys.stderr, flush=True)

class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class DashScopeEmbeddingProvider:
    """OpenAI-compatible ``text-embedding-v4`` with explicit endpoint and key.

    ``endpoint`` is the complete /embeddings URL. Transient transport failures retry with backoff. There are no
    redirects or environment/system proxies. ``allow_test_loopback`` permits HTTP
    only for literal loopback IPs and is intended exclusively for offline tests.
    No local fee ceiling, call-count ceiling or elapsed-run expiry is enforced.
    No application socket deadline is imposed.
    """

    model_id = "text-embedding-v4"
    _CHUNK_BYTES = 8192
    _BATCH_SIZE = 10
    _RESPONSE_BYTES = 4 * 1024 * 1024

    def __init__(self, endpoint: str, api_key: str,
                 cache_dir: Optional[Path] = None, dimensions: int = 1024,
                 timeout: Optional[float] = None, *, allow_test_loopback: bool = False,
                 checkpoint_namespace: str = ""):
        try:
            parsed = urlsplit(endpoint)
            host = (parsed.hostname or "").lower().rstrip(".")
            _ = parsed.port
            allowed_http = allow_test_loopback and host in {"127.0.0.1", "::1"}
            if (not host or parsed.username is not None or parsed.password is not None
                    or parsed.query or parsed.fragment
                    or any(c.isspace() for c in endpoint)
                    or host == "lewen.woa.com" or host.endswith(".lewen.woa.com")
                    or (parsed.scheme != "https" and not (parsed.scheme == "http" and allowed_http))):
                raise ValueError()
        except (ValueError, TypeError, AttributeError):
            raise ProviderUnavailable("invalid-embedding-endpoint") from None
        if not isinstance(api_key, str) or not api_key.strip() or any(ord(c) < 33 or ord(c) > 126 for c in api_key):
            raise ProviderUnavailable("invalid-embedding-credential")
        if type(dimensions) is not int or dimensions not in {64, 128, 256, 512, 768, 1024, 1536, 2048}:
            raise ProviderUnavailable("invalid-embedding-dimensions")
        if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 0 < timeout):
            raise ProviderUnavailable("invalid-embedding-timeout")
        self._test_loopback = allowed_http
        self.last_attempt = None
        self.attempts = []
        self.chunk_cache_hits = 0
        self.endpoint = endpoint
        self._provider_name = ("dmx-openai-compatible" if host == "www.dmxapi.cn" else "dashscope-openai-compatible")
        self._configured_endpoint = endpoint
        self._api_key = api_key
        self.dimensions = dimensions
        self.timeout = None if timeout is None else float(timeout)
        self._started_at = monotonic()
        self.request_batches = 0
        self._failure = ""
        self.cache_hits = 0
        self.cache_misses = 0
        self._fingerprint = _digest({
            "provider": "dashscope-openai-compatible", "schema": 2,
            "endpoint": endpoint, "credential_hash": hashlib.sha256(api_key.encode()).hexdigest(),
            "model": self.model_id, "dimensions": dimensions,
            "chunk_bytes": self._CHUNK_BYTES, "aggregation": "ordered-fsum-mean-l2-v1",
            "region": "cn-beijing",  # Preserve existing DMX numeric cache identity.
        })
        self._checkpoint = (VectorCheckpoint(cache_dir, self._fingerprint,
            checkpoint_namespace or digest(str(Path(cache_dir).resolve())), dimensions)
            if cache_dir is not None else None)
        self._opener = build_opener(ProxyHandler({}), _NoRedirect())

    def describe(self) -> Dict[str, Any]:
        return {
            "provider": self._provider_name, "model_id": self.model_id,
            "dimensions": self.dimensions, "fingerprint": self._fingerprint,
            "endpoint_fingerprint": _digest(self.endpoint),
            "local_files_only": False, "trust_remote_code": False,
            "status": "unavailable" if self._failure else "ready",
            "fallback_reason": self._failure,
            "cache_hits": self.cache_hits, "cache_misses": self.cache_misses,
            "request_batches": self.request_batches,
            "chunk_cache_hits": self.chunk_cache_hits,
            "budget_enforced": False,
            "network_worker_enabled": False,
            "work_retry_policy": "cache-reuse-no-financial-claims",
            "last_attempt": self.last_attempt,
            "attempts": [dict(item) for item in self.attempts],
            "elapsed_seconds": max(0.0, monotonic() - self._started_at),
            "settings": {"batch_size": self._BATCH_SIZE, "chunk_bytes": self._CHUNK_BYTES,
                         "long_text_policy": "utf8-byte-chunks-mean-l2",
                         "timeout_seconds": self.timeout, "automatic_retries": "transient-with-backoff",
                         "max_request_batches": None,
                         "max_elapsed_seconds": None,
                         "elapsed_budget_policy": "none-user-authorized"},
        }

    @classmethod
    def _chunks(cls, text: str) -> List[str]:
        result, start, size = [], 0, 0
        for offset, character in enumerate(text):
            length = len(character.encode("utf-8"))
            if size + length > cls._CHUNK_BYTES:
                result.append(text[start:offset])
                start, size = offset, 0
            size += length
        result.append(text[start:])
        return result

    @timed('embedding_batch')
    def _request(self, texts: Sequence[str], publish=None, raw_keys=None) -> List[List[float]]:
        if self.endpoint != self._configured_endpoint:
            raise ProviderUnavailable("embedding-configuration-drift")
        body = json.dumps({"model": self.model_id, "input": list(texts),
                           "dimensions": self.dimensions, "encoding_format": "float"},
                          ensure_ascii=False).encode("utf-8")
        request = Request(self.endpoint, data=body, method="POST", headers={
            "Authorization": "Bearer " + self._api_key, "Content-Type": "application/json",
            "Accept": "application/json",
        })
        attempt = 0
        delay = 1.0
        while True:
            attempt += 1
            try:
                self.request_batches += 1
                with phase("embedding_http"):
                    with self._opener.open(request, timeout=self.timeout) as response:
                        if response.status != 200:
                            raise ProviderUnavailable("embedding-http-status")
                        raw = response.read(self._RESPONSE_BYTES + 1)
                break
            except HTTPError as exc:
                code = exc.code
                retry_after = exc.headers.get("Retry-After", "") if exc.headers else ""
                metadata = http_error_metadata(exc)
                exc.close()
                if code not in {408, 429, 500, 502, 503, 504}:
                    failure = ProviderUnavailable("embedding-http-" + str(code))
                    failure.http_error_metadata = metadata
                    raise failure from None
                reason = "embedding-http-" + str(code)
                if retry_after.isdigit():
                    delay = max(delay, float(retry_after))
            except (URLError, OSError, TimeoutError, http.client.HTTPException) as exc:
                if isinstance(exc, ssl.SSLCertVerificationError) or isinstance(getattr(exc, "reason", None), ssl.SSLCertVerificationError):
                    raise ProviderUnavailable("embedding-tls-verification-failed") from None
                reason = "embedding-network-failed"
            _record_recovery(reason, attempt)
            with phase("embedding_backoff"):
                sleep(delay)
            delay = min(delay * 2, 30.0)
        if attempt > 1:
            _record_recovery("embedding-transport-recovered", attempt, recovered=True)
        if len(raw) > self._RESPONSE_BYTES:
            raise ProviderUnavailable("embedding-response-too-large")
        try:
            payload = json.loads(raw)
        except (ValueError, UnicodeError):
            raise ProviderUnavailable("invalid-embedding-response") from None
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list) or len(payload["data"]) != len(texts):
            raise ProviderUnavailable("embedding-count-mismatch")
        if payload.get("model") != self.model_id:
            raise ProviderUnavailable("embedding-model-mismatch")
        ordered = {}
        for row in payload["data"]:
            if not isinstance(row, dict):
                raise ProviderUnavailable("invalid-embedding-response")
            index, vector = row.get("index"), row.get("embedding")
            if type(index) is not int or index < 0 or index >= len(texts) or index in ordered:
                raise ProviderUnavailable("invalid-embedding-index")
            if not isinstance(vector, list) or len(vector) != self.dimensions:
                raise ProviderUnavailable("embedding-dimension-mismatch")
            if any(type(value) not in (int, float) for value in vector):
                raise ProviderUnavailable("invalid-embedding-vector")
            try:
                numbers = [float(value) for value in vector]
                if any(not math.isfinite(value) for value in numbers):
                    raise ValueError()
            except (ValueError, OverflowError):
                raise ProviderUnavailable("nonfinite-embedding-vector") from None
            if not any(numbers):
                # Reject before publishing ANY row of the response: an invalid
                # upstream vector must not make all future replays fail cached.
                raise ProviderUnavailable("invalid-embedding-norm")
            ordered[index] = numbers
        values = [ordered[index] for index in range(len(texts))]
        # Publish validated vectors atomically; no financial reservation or
        # permanent work claim can poison a later retry after network failure.
        if publish is not None:
            with phase("embedding_cache_publish"):
                publish(values)
        return values

    @timed('embedding_total')
    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        try:
            if isinstance(texts, (str, bytes)):
                raise ProviderUnavailable("invalid-embedding-input")
            texts = list(texts)
            if any(not isinstance(text, str) or not text.strip() for text in texts):
                raise ProviderUnavailable("invalid-embedding-input")
            if self._checkpoint is None:
                return self._embed_locked(texts)
            with timed_acquire(checkpoint_access_mode(), "embedding_mode_wait") as parallel:
                if not parallel:
                    with timed_acquire(self._checkpoint.locked(), "embedding_checkpoint_wait"):
                        return self._embed_locked(texts)
                # Shared namespace guard preserves compatibility with legacy writers.
                # Warm cached inputs never take exclusive input stripes. Only misses
                # are rechecked and reconstructed under sorted per-input stripes.
                with timed_acquire(self._checkpoint.locked(shared=True), "embedding_checkpoint_wait"):
                    encoded, missing, _, _, _, _ = self._prepare(texts, repair_cache=False)
                    self.cache_hits += len(encoded)
                    if missing:
                        with timed_acquire(self._checkpoint.locked_inputs(_digest(t) for t in missing),
                                           "embedding_input_wait"):
                            with phase("embedding_input_held"):
                                rebuilt = self._embed_locked(list(missing))
                                encoded.update(zip(missing, rebuilt))
                    return [list(encoded[t]) for t in texts]
        except ProviderUnavailable as exc:
            self._failure = str(exc)
            raise
        except Exception:
            self._failure = "embedding-failed"
            raise ProviderUnavailable(self._failure) from None

    def _cached(self, kind, key, repair_cache):
        if self._checkpoint is None:
            return None
        try:
            return self._checkpoint.get(kind, key)
        except ProviderUnavailable as exc:
            if exc.args != ("embedding-checkpoint-corrupt",):
                raise
            if repair_cache:
                self._checkpoint.quarantine(kind, key)
            return None

    @timed('embedding_prepare')
    def _prepare(self, texts, *, repair_cache=False):
        unique = list(dict.fromkeys(texts))
        encoded, missing, chunks, keys = {}, {}, [], []
        for text in unique:
            key = _digest(text)
            cached = self._cached("pooled", key, repair_cache)
            if cached is not None:
                encoded[text] = cached

            else:

                parts = self._chunks(text)
                missing[text] = (len(chunks), len(parts))
                chunks.extend(parts)
                keys.extend(digest({"input": key, "position": i, "chunk": _digest(part)})
                            for i, part in enumerate(parts))
        vectors = []
        for key in keys:
            vectors.append(self._cached("raw", key, repair_cache))
        indices = [i for i, vector in enumerate(vectors) if vector is None]
        return encoded, missing, chunks, keys, vectors, indices

    def _embed_locked(self, texts):
        encoded, missing, chunks, keys, vectors, indices = self._prepare(texts, repair_cache=True)
        self.cache_hits += len(encoded)
        self.cache_misses += len(missing)
        self.chunk_cache_hits += len(vectors) - len(indices)
        for start in range(0, len(indices), self._BATCH_SIZE):
            batch = indices[start:start + self._BATCH_SIZE]
            def publish(values):
                for i, value in zip(batch, values):
                    if self._checkpoint:
                        self._checkpoint.put("raw", keys[i], value)
                    vectors[i] = value
            self._request([chunks[i] for i in batch], publish, [keys[i] for i in batch])
        for text, (start, count) in missing.items():
            mean = []
            for axis in range(self.dimensions):
                mean.append(math.fsum(vector[axis] / count for vector in vectors[start:start + count]))
            scale = max(abs(value) for value in mean)
            if not scale or not math.isfinite(scale):
                # Finite nonzero chunks can still cancel. Preserve the exact
                # derived inputs for diagnosis, but do not poison future calls.
                # This call fails; no hidden transport retry occurs here.
                if self._checkpoint:
                    for key in keys[start:start + count]:
                        self._checkpoint.quarantine("raw", key)
                raise ProviderUnavailable("invalid-embedding-norm")
            scaled = [value / scale for value in mean]
            norm = math.hypot(*scaled)
            encoded[text] = [value / norm for value in scaled]
            if self._checkpoint:
                with phase("embedding_cache_publish"):
                    self._checkpoint.put("pooled", _digest(text), encoded[text])
        self._failure = ""
        return [list(encoded[text]) for text in texts]
