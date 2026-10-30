"""Opt-in Add exception diagnostics; preserve handler behavior and privacy.

Run with the existing candidate venv/environment. This launcher does not read
credentials, retain request data, change billing, or resume evaluation jobs.
It emits only exception type, errno and source coordinates, never messages,
locals, source lines, request identifiers, headers or payloads.
"""
from __future__ import annotations

import functools
import json
import sys
import time


def exception_record(exc):
    frames = []
    tb = exc.__traceback__
    while tb is not None:
        code = tb.tb_frame.f_code
        module = tb.tb_frame.f_globals.get("__name__", "")
        # Only installed application frames, not arbitrary filesystem paths.
        if isinstance(module, str) and module.startswith("tiger_memory."):
            frames.append({"module": module, "function": code.co_name,
                           "line": tb.tb_lineno})
        tb = tb.tb_next
    record = {"event": "add_exception_diagnostic_v1", "time_unix": time.time(),
              "exception_type": type(exc).__name__, "frames": frames[-16:]}
    errno = getattr(exc, "errno", None)
    if type(errno) is int:
        record["errno"] = errno
    return record


def stderr_sink(record):
    print(json.dumps(record, sort_keys=True), file=sys.stderr, flush=True)


def wrap_add(original, sink=stderr_sink):
    @functools.wraps(original)
    def observed_add(request):
        try:
            return original(request)
        except Exception as exc:
            try:
                sink(exception_record(exc))
            except Exception:
                # A broken diagnostic sink must not replace the real failure.
                pass
            raise
    return observed_add


def classify_permanent_rejection(original, rejection_type, contract_error):
    @functools.wraps(original)
    def classified_add(request):
        try:
            return original(request)
        except rejection_type:
            # Preserve rejection. Do not misreport permanent policy rejection
            # as temporary unavailability and induce repeated identical calls.
            raise contract_error("sensitive input rejected") from None
    return classified_add


# Uniform privacy transformation; never branch on dataset, user or question.
REDACTION = "[redacted credential]"


def redact_credentials(text):
    from tiger_memory.security import SECRET_PATTERNS, assert_no_secrets
    intervals = sorted((m.start(), m.end()) for pattern in SECRET_PATTERNS
                       for m in pattern.finditer(text))
    if not intervals:
        return text, 0
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    parts, cursor = [], 0
    for start, end in merged:
        parts.extend([text[cursor:start], REDACTION])
        cursor = end
    parts.append(text[cursor:])
    clean = "".join(parts)
    # Keep the original guard, including any future detector changes.
    assert_no_secrets([clean])
    return clean, len(merged)


def privacy_transform(original, operation, validator, sink=stderr_sink):
    @functools.wraps(original)
    def transformed(request):
        # Validate first; malformed requests must not acquire success semantics.
        validator(request)
        clean = dict(request)
        count = 0
        if operation == "add":
            clean["messages"] = []
            for message in request["messages"]:
                item = dict(message)
                item["content"], n = redact_credentials(message["content"])
                count += n
                clean["messages"].append(item)
        elif operation == "search":
            clean["query"], n = redact_credentials(request["query"])
            count += n
            if "options" in request:
                clean["options"] = []
                for option in request["options"]:
                    value, n = redact_credentials(option)
                    count += n
                    clean["options"].append(value)
        else:
            raise ValueError("unsupported privacy transformation")
        if count:
            try:
                sink({"event": "credential_redaction_v1", "operation": operation,
                      "spans": count, "time_unix": time.time()})
            except Exception:
                pass
        # Success only comes from the real durable write / retrieval path.
        return original(clean)
    return transformed


def install(aml):
    from tiger_memory.security import SecretDetected
    aml.add = classify_permanent_rejection(wrap_add(privacy_transform(
        aml.add, "add", aml._validate_add)), SecretDetected, aml.ContractError)
    aml.search = privacy_transform(aml.search, "search", aml._validate_search)


def main(argv=None):
    from tiger_memory import aml_adapter, aml_open_source
    install(aml_adapter)
    return aml_open_source.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
