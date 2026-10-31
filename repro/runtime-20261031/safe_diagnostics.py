"""Privacy bounded Add/Search diagnostics; never alter handler outcomes.

Install after the existing safe_ingest privacy wrappers. Only fixed reason
codes and application source coordinates are emitted, never exception text,
request data, filesystem paths, locals, headers, credentials or payloads.
"""
from __future__ import annotations

import functools
import json
import re
import sys
import time

from tiger_memory.providers import ProviderUnavailable
from tiger_memory.transport_diagnostics import sanitize_http_error_metadata


# Exact values, not a prefix permission: a secret-shaped suffix is not a code.
REASON_CODES = frozenset("""
embedding-checkpoint-busy embedding-checkpoint-unavailable embedding-checkpoint-corrupt
embedding-checkpoint-write-failed embedding-checkpoint-conflict
embedding-checkpoint-unsafe-path embedding-checkpoint-key embedding-checkpoint-required
embedding-checkpoint-unsafe-file embedding-checkpoint-recovery-lock-required
embedding-tls-verification-failed embedding-network-failed embedding-http-status embedding-response-too-large
embedding-http-400 embedding-http-401 embedding-http-403 embedding-http-408
embedding-http-409 embedding-http-413 embedding-http-429 embedding-http-500
embedding-http-502 embedding-http-503 embedding-http-504
embedding-count-mismatch embedding-model-mismatch embedding-dimension-mismatch
invalid-embedding-response invalid-embedding-index invalid-embedding-vector
invalid-embedding-norm nonfinite-embedding-vector invalid-embedding-input
embedding-configuration-drift embedding-usage-untrusted embedding-failed
""".split())
REASON_PREFIXES = ("embedding-", "invalid-embedding-", "nonfinite-embedding-")
_CODE = re.compile(r"[a-z0-9-]{1,96}\Z")
_SYMBOL = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,95}\Z")
_APPLICATION_MODULE = re.compile(r"tiger_memory(?:\.[A-Za-z_][A-Za-z0-9_]*)+\Z")
_SHARED_MODULES = frozenset()
_TYPES = frozenset({"ProviderUnavailable", "RetrievalUnavailable",
                    "AddLockUnavailable", "StorageIdentityConflict", "SourceRecoveryError",
                    "ContractError",
                    "OperationalError", "IntegrityError", "TimeoutError", "OSError",
                    "ValueError", "RuntimeError", "AssertionError"})


def _reason(exc):
    # Read args directly: custom __str__ may expose data or itself fail.
    if not isinstance(exc, ProviderUnavailable) or len(exc.args) != 1:
        return None
    value = exc.args[0]
    if (type(value) is str and _CODE.fullmatch(value)
            and value.startswith(REASON_PREFIXES) and value in REASON_CODES):
        return value
    return None


def _frames(exc):
    frames = []
    tb = exc.__traceback__
    while tb is not None:
        module = tb.tb_frame.f_globals.get("__name__")
        function = tb.tb_frame.f_code.co_name
        if (type(module) is str and len(module) <= 160
                and (_APPLICATION_MODULE.fullmatch(module) or module in _SHARED_MODULES)
                and _SYMBOL.fullmatch(function)):
            frames.append({"module": module, "function": function, "line": tb.tb_lineno})
        tb = tb.tb_next
    return frames[-16:]


def exception_record(exc, operation, elapsed_seconds=0.0):
    if operation not in {"add", "search"}:
        raise ValueError("unsupported diagnostic operation")
    chain, seen = [], set()
    current = exc
    while current is not None and id(current) not in seen and len(chain) < 4:
        seen.add(id(current))
        name = type(current).__name__
        item = {"exception_type": name if name in _TYPES else "Exception",
                "reason_code": _reason(current), "frames": _frames(current)}
        errno = getattr(current, "errno", None)
        if type(errno) is int and 0 <= errno <= 65535:
            item["errno"] = errno
        sqlite_code = getattr(current, "sqlite_errorcode", None)
        if type(sqlite_code) is int and 0 <= sqlite_code <= 65535:
            item["sqlite_errorcode"] = sqlite_code
        if isinstance(current, ProviderUnavailable):
            transport = sanitize_http_error_metadata(
                getattr(current, "http_error_metadata", None))
            if transport is not None:
                item["transport"] = transport
        chain.append(item)
        # Search deliberately suppresses public causes; safe codes still help
        # diagnose internally without copying any arbitrary context messages.
        current = current.__cause__ if current.__cause__ is not None else current.__context__
    return {"event": "aml_exception_diagnostic_v2", "operation": operation,
            "time_unix": time.time(), "elapsed_ms": round(max(0.0, elapsed_seconds) * 1000, 3),
            "exceptions": chain}


def stderr_sink(record):
    print(json.dumps(record, sort_keys=True), file=sys.stderr, flush=True)


def wrap_operation(original, operation, sink=stderr_sink):
    if operation not in {"add", "search"}:
        raise ValueError("unsupported diagnostic operation")
    @functools.wraps(original)
    def observed(request):
        started = time.monotonic()
        try:
            return original(request)
        except Exception as exc:
            try:
                sink(exception_record(exc, operation, time.monotonic() - started))
            except Exception:
                pass
            raise
    observed._tiger_safe_diagnostics_v2 = True
    return observed


def install(aml, sink=stderr_sink):
    """Wrap existing privacy/validation handlers; idempotent on this instance."""
    for operation in ("add", "search"):
        original = getattr(aml, operation)
        if not getattr(original, "_tiger_safe_diagnostics_v2", False):
            setattr(aml, operation, wrap_operation(original, operation, sink))
