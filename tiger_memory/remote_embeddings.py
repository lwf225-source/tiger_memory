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
from time import monotonic
from typing import Any, Dict, List, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from contextlib import nullcontext

from .providers import ProviderUnavailable, _digest
from . import request_budget as request_limits
from .embedding_budget import PersistentEmbeddingBudget
from .embedding_checkpoint import VectorCheckpoint, digest


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class DashScopeEmbeddingProvider:
    """OpenAI-compatible ``text-embedding-v4`` with explicit endpoint and key.

    ``endpoint`` is the complete /embeddings URL. There are no automatic retries,
    redirects or environment/system proxies. ``allow_test_loopback`` permits HTTP
    only for literal loopback IPs and is intended exclusively for offline tests.
    The elapsed budget is shared across this instance's calls and checked at
    batch boundaries. It cannot forcibly interrupt DNS or a slowly streaming
    response, so it is not a hard wall-clock limit on the enclosing HTTP request.
    """

    model_id = "text-embedding-v4"
    _CHUNK_BYTES = 8192
    _BATCH_SIZE = 10
    _RESPONSE_BYTES = 4 * 1024 * 1024

    def __init__(self, endpoint: str, api_key: str,
                 cache_dir: Optional[Path] = None, dimensions: int = 1024,
                 timeout: float = 30.0, *, allow_test_loopback: bool = False,
                 max_request_batches: int = 100, max_elapsed_seconds: float = 1200.0,
                 budget: Optional[PersistentEmbeddingBudget] = None,
                 checkpoint_namespace: str = "", network_supervisor=None):
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
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 0 < timeout <= 60:
            raise ProviderUnavailable("invalid-embedding-timeout")
        if type(max_request_batches) is not int or not 1 <= max_request_batches <= 10000:
            raise ProviderUnavailable("invalid-embedding-request-budget")
        if (isinstance(max_elapsed_seconds, bool) or not isinstance(max_elapsed_seconds, (int, float))
                or not math.isfinite(max_elapsed_seconds) or max_elapsed_seconds <= 0):
            raise ProviderUnavailable("invalid-embedding-elapsed-budget")
        if budget is not None:
            if not isinstance(budget, PersistentEmbeddingBudget):
                raise ProviderUnavailable("embedding-budget-required")
            budget.bind(endpoint, self.model_id, dimensions)
        if network_supervisor is not None:
            from .embedding_supervisor import NetworkSupervisor
            if not isinstance(network_supervisor, NetworkSupervisor):
                raise ProviderUnavailable("embedding-worker-configuration")
            if network_supervisor.config.enabled and (budget is None or cache_dir is None):
                raise ProviderUnavailable("embedding-worker-budget-checkpoint-required")
        self._network_supervisor = (network_supervisor if network_supervisor is not None
                                    and network_supervisor.config.enabled else None)
        self.budget = budget
        self._request_budget = request_limits.current_request_budget()
        self._test_loopback = allowed_http
        self.last_attempt = None
        self.attempts = []
        self.chunk_cache_hits = 0
        self.endpoint = endpoint
        self._configured_endpoint = endpoint
        self._api_key = api_key
        self.dimensions = dimensions
        self.timeout = float(timeout)
        self.max_request_batches = max_request_batches
        self.max_elapsed_seconds = float(max_elapsed_seconds)
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
            "region": budget.policy.region if budget is not None else "unbudgeted-test",
        })
        self._checkpoint = (VectorCheckpoint(cache_dir, self._fingerprint,
            checkpoint_namespace or digest(str(Path(cache_dir).resolve())), dimensions)
            if cache_dir is not None else None)
        self._opener = build_opener(ProxyHandler({}), _NoRedirect())

    def describe(self) -> Dict[str, Any]:
        return {
            "provider": "dashscope-openai-compatible", "model_id": self.model_id,
            "dimensions": self.dimensions, "fingerprint": self._fingerprint,
            "endpoint_fingerprint": _digest(self.endpoint),
            "local_files_only": False, "trust_remote_code": False,
            "status": "unavailable" if self._failure else "ready",
            "fallback_reason": self._failure,
            "cache_hits": self.cache_hits, "cache_misses": self.cache_misses,
            "request_batches": self.request_batches,
            "chunk_cache_hits": self.chunk_cache_hits,
            "budget_enforced": self.budget is not None,
            "network_worker_enabled": self._network_supervisor is not None,
            "work_retry_policy": "durable-per-key-claims-v1" if self.budget else "unbudgeted-test",
            "last_attempt": self.last_attempt,
            "attempts": [dict(item) for item in self.attempts],
            "elapsed_seconds": max(0.0, monotonic() - self._started_at),
            "settings": {"batch_size": self._BATCH_SIZE, "chunk_bytes": self._CHUNK_BYTES,
                         "long_text_policy": "utf8-byte-chunks-mean-l2",
                         "timeout_seconds": self.timeout, "automatic_retries": 0,
                         "max_request_batches": self.max_request_batches,
                         "max_elapsed_seconds": self.max_elapsed_seconds,
                         "elapsed_budget_policy": "batch-boundary-checks-not-hard-deadline"},
        }

    @classmethod
    def _chunks(cls, text: str) -> List[str]:
        result, start, size = [], 0, 0
        request_budget = request_limits.current_request_budget()
        for offset, character in enumerate(text):
            if request_budget is not None and offset % 1024 == 0:
                request_budget.check("embedding-chunks")
            length = len(character.encode("utf-8"))
            if size + length > cls._CHUNK_BYTES:
                result.append(text[start:offset])
                start, size = offset, 0
            size += length
        result.append(text[start:])
        return result

    def _request_limits(self):
        active = request_limits.current_request_budget()
        return tuple(dict.fromkeys(b for b in (self._request_budget, active) if b is not None))

    def _check_request(self, phase):
        for budget in self._request_limits():
            budget.check(phase)

    def _remaining_seconds(self) -> float:
        remaining = self.max_elapsed_seconds - (monotonic() - self._started_at)
        if remaining <= 0:
            raise ProviderUnavailable("embedding-elapsed-budget-exhausted")
        for budget in self._request_limits():
            remaining = min(remaining, budget.remaining("embedding-batch"))
        return remaining

    def _request(self, texts: Sequence[str], publish=None, raw_keys=None) -> List[List[float]]:
        if self.endpoint != self._configured_endpoint:
            raise ProviderUnavailable("embedding-configuration-drift")
        if self.budget is not None:
            self.budget.bind(self.endpoint, self.model_id, self.dimensions)
        if self.request_batches >= self.max_request_batches:
            raise ProviderUnavailable("embedding-request-budget-exhausted")
        body = json.dumps({"model": self.model_id, "input": list(texts),
                           "dimensions": self.dimensions, "encoding_format": "float"},
                          ensure_ascii=False).encode("utf-8")
        request = Request(self.endpoint, data=body, method="POST", headers={
            "Authorization": "Bearer " + self._api_key, "Content-Type": "application/json",
            "Accept": "application/json",
        })
        timeout = min(self.timeout, self._remaining_seconds())
        if self.budget is None and not self._test_loopback:
            raise ProviderUnavailable("embedding-budget-required")
        if self.budget is not None and self._checkpoint is None:
            raise ProviderUnavailable("embedding-checkpoint-required")
        attempt = None
        supervisor = self._network_supervisor
        handle = None
        try:
            if supervisor is not None:
                from .embedding_supervisor import RequestContext
                supervisor.validate_request(self.endpoint, self._api_key, body)
                limits = self._request_limits()
                context = RequestContext(limits[-1].generation if limits else 0,
                    self._fingerprint, hashlib.sha256(body).hexdigest(),
                    self._started_at + self.max_elapsed_seconds, limits)
                # All misses were checked under the checkpoint lock by _embed_locked.
                # READY consumes no durable permission; reserve repeats claim checks.
                handle = supervisor.prepare(context, allow_loopback=self._test_loopback)
            if self.budget is not None:
                attempt = self.budget.reserve(len(texts), hashlib.sha256(body).hexdigest(),
                    {"namespace": self._checkpoint.context["namespace"],
                     "fingerprint": self._fingerprint, "raw_keys": raw_keys})
                self.last_attempt = {"id": attempt, "state": "UNKNOWN", "rows": len(texts),
                                     "usage_tokens": None}
                self.attempts.append(self.last_attempt)
            # Expiry after durable reservation never refunds its claim.
            timeout = min(timeout, self._remaining_seconds())
            self.request_batches += 1
            if supervisor is not None:
                raw = supervisor.send(handle, attempt, self.endpoint, self._api_key, body, timeout)
            else:
                with self._opener.open(request, timeout=timeout) as response:
                    if response.status != 200:
                        raise ProviderUnavailable("embedding-http-status")
                    raw = response.read(self._RESPONSE_BYTES + 1)
            if len(raw) > self._RESPONSE_BYTES:
                raise ProviderUnavailable("embedding-response-too-large")
            payload = json.loads(raw)
        except HTTPError as exc:
            code = exc.code
            exc.close()
            raise ProviderUnavailable("embedding-http-" + str(code)) from None
        except (URLError, OSError, TimeoutError):
            raise ProviderUnavailable("embedding-network-failed") from None
        except (ValueError, UnicodeError):
            raise ProviderUnavailable("invalid-embedding-response") from None
        finally:
            if handle is not None:
                supervisor.cleanup(handle)
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list) or len(payload["data"]) != len(texts):
            raise ProviderUnavailable("embedding-count-mismatch")
        if ((self.budget is not None and payload.get("model") != self.model_id)
                or ("model" in payload and payload["model"] != self.model_id)):
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
            ordered[index] = numbers
        values = [ordered[index] for index in range(len(texts))]
        # Durable usable data and trusted billing are separate facts. If killed
        # after publish but before settle, data is reused and UNKNOWN stays held.
        if publish is not None:
            publish(values)
        if self.budget is not None:
            usage = payload.get("usage")
            total = usage.get("total_tokens") if isinstance(usage, dict) else None
            prompt = usage.get("prompt_tokens", total) if isinstance(usage, dict) else None
            if (type(total) is not int or type(prompt) is not int or total != prompt
                    or total < 0 or total > len(texts) * self.budget.policy.token_limit_per_row):
                if type(total) is int and total > len(texts) * self.budget.policy.token_limit_per_row:
                    # The asserted supplier ceiling has been contradicted.
                    # Preserve UNKNOWN and stop further reservations globally.
                    self.budget.stop()
                raise ProviderUnavailable("embedding-usage-untrusted")
            self.budget.settle(attempt, total)
            self.last_attempt.update(state="SETTLED", usage_tokens=total)
        self._remaining_seconds()
        return values

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        try:
            self._check_request("embedding-input")
            if isinstance(texts, (str, bytes)):
                raise ProviderUnavailable("invalid-embedding-input")
            texts = list(texts)
            if any(not isinstance(text, str) or not text.strip() for text in texts):
                raise ProviderUnavailable("invalid-embedding-input")
            lock = self._checkpoint.locked(timeout=min(5.0, max(0.0,
                self.max_elapsed_seconds - (monotonic() - self._started_at)))) if self._checkpoint else nullcontext()
            with lock:
                self._check_request("embedding-locked")
                result = self._embed_locked(texts)
                self._check_request("embedding-output")
                return result
        except ProviderUnavailable as exc:
            self._failure = str(exc)
            raise
        except Exception:
            self._failure = "embedding-failed"
            raise ProviderUnavailable(self._failure) from None

    def _prepare(self, texts):
        unique = list(dict.fromkeys(texts))
        encoded, missing, chunks, keys = {}, {}, [], []
        for text in unique:
            self._check_request("embedding-prepare")
            key = _digest(text)
            cached = self._checkpoint.get("pooled", key) if self._checkpoint else None
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
            self._check_request("embedding-cache")
            vectors.append(self._checkpoint.get("raw", key) if self._checkpoint else None)
        self._check_request("embedding-prepared")
        indices = [i for i, vector in enumerate(vectors) if vector is None]
        return encoded, missing, chunks, keys, vectors, indices

    def plan(self, texts: Sequence[str]) -> Dict[str, Any]:
        """Read-only capacity snapshot using exactly embed's chunk/cache rules.

        The caller supplies the actual first-write document representation;
        this provider knows no source store or request IDs. No lock files,
        cache entries, reservations or requests are created. Concurrent cache
        changes can invalidate the snapshot; embed ALWAYS checks again.
        Financial admission is deliberately deferred to durable reservations.
        """
        try:
            self._check_request("embedding-input")
            if isinstance(texts, (str, bytes)):
                raise ProviderUnavailable("invalid-embedding-input")
            texts = list(texts)
            if any(not isinstance(text, str) or not text.strip() for text in texts):
                raise ProviderUnavailable("invalid-embedding-input")
            if self.endpoint != self._configured_endpoint:
                raise ProviderUnavailable("embedding-configuration-drift")
            if self.budget is not None:
                self.budget.bind(self.endpoint, self.model_id, self.dimensions)
            encoded, missing, chunks, keys, vectors, indices = self._prepare(texts)
            batches = (len(indices) + self._BATCH_SIZE - 1) // self._BATCH_SIZE
            remaining = self.max_request_batches - self.request_batches
            self._check_request("embedding-plan-output")
            return {"schema": "embedding-plan-v1", "fingerprint": self._fingerprint,
                    "input_count": len(texts), "unique_inputs": len(encoded) + len(missing),
                    "total_chunks": sum(len(self._chunks(t)) for t in dict.fromkeys(texts)),
                    "pooled_cache_hits": len(encoded), "reusable_raw_chunks": len(vectors) - len(indices),
                    "uncached_chunks": len(indices), "required_batches": batches,
                    "remaining_request_batches": remaining, "within_request_batch_limit": batches <= remaining,
                    "cache_snapshot_atomic": False, "budget_admission": "deferred-to-reserve",
                    "new_sends": 0, "new_reservations": 0}
        except ProviderUnavailable:
            raise
        except (TypeError, ValueError, UnicodeError, OverflowError):
            raise ProviderUnavailable("invalid-embedding-input") from None

    def _embed_locked(self, texts):
        encoded, missing, chunks, keys, vectors, indices = self._prepare(texts)
        self.cache_hits += len(encoded)
        self.cache_misses += len(missing)
        self.chunk_cache_hits += len(vectors) - len(indices)
        needed = (len(indices) + self._BATCH_SIZE - 1) // self._BATCH_SIZE
        if needed > self.max_request_batches - self.request_batches:
            raise ProviderUnavailable("embedding-request-budget-exhausted")
        if self.budget is not None and indices:
            if self._checkpoint is None:
                raise ProviderUnavailable("embedding-checkpoint-required")
            # Check the entire miss set before doing any new paid work. Batch
            # reserve repeats this in the SAME transaction as claims + money.
            self.budget.check_unclaimed(self._checkpoint.context["namespace"],
                                        self._fingerprint, [keys[i] for i in indices])
        for start in range(0, len(indices), self._BATCH_SIZE):
            self._check_request("embedding-batch-admission")
            batch = indices[start:start + self._BATCH_SIZE]
            def publish(values):
                for i, value in zip(batch, values):
                    if self._checkpoint:
                        self._checkpoint.put("raw", keys[i], value)
                    vectors[i] = value
            self._request([chunks[i] for i in batch], publish, [keys[i] for i in batch])
        for text, (start, count) in missing.items():
            self._check_request("embedding-pooling")
            mean = []
            for axis in range(self.dimensions):
                self._check_request("embedding-pooling")
                mean.append(math.fsum(vector[axis] / count for vector in vectors[start:start + count]))
            scale = max(abs(value) for value in mean)
            if not scale or not math.isfinite(scale):
                raise ProviderUnavailable("invalid-embedding-norm")
            scaled = [value / scale for value in mean]
            norm = math.hypot(*scaled)
            encoded[text] = [value / norm for value in scaled]
            self._check_request("embedding-pooled-publish")
            if self._checkpoint:
                self._checkpoint.put("pooled", _digest(text), encoded[text])
        self._failure = ""
        return [list(encoded[text]) for text in texts]
