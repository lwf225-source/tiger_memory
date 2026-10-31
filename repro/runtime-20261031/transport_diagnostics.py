"""Bounded, nonblocking metadata for an already failed HTTP response.

No request, endpoint, response message, raw body, header or identifier is
returned. Error codes/types use exact fixed allowlists; request IDs are hashed.
The helper never opens a connection, waits for response bytes or retries work.
"""
from __future__ import annotations

import hashlib
import http.client
import io
import json
import re
import socket

MAX_BODY_BYTES = 8192
ERROR_CODES = frozenset({
    "invalid_request_error", "invalid_api_key", "insufficient_quota",
    "insufficient_user_quota", "context_length_exceeded", "model_not_found",
    "rate_limit_exceeded", "InvalidParameter", "InvalidParameter.ValueError",
    "InvalidParameter.RangeError", "InvalidParameter.Input", "InvalidRequest",
    "InputLengthExceeded", "DataInspectionFailed", "InvalidApiKey",
})
ERROR_TYPES = frozenset({
    "invalid_request_error", "permission_error", "authentication_error",
    "rate_limit_error", "server_error", "invalid_parameter_error",
})
BODY_STATES = frozenset({
    "json", "empty", "oversized", "unavailable", "incomplete_or_invalid_json",
    "unstructured_json",
})
_ID = re.compile(r"[A-Za-z0-9_-]{8,128}\Z")
_HEX = re.compile(r"[a-f0-9]{64}\Z")


def _request_id_digest(value):
    if type(value) is str and _ID.fullmatch(value):
        return hashlib.sha256(value.encode("ascii")).hexdigest()
    return None


def sanitize_http_error_metadata(value):
    """Revalidate exception-attached metadata before any diagnostic emission."""
    if (type(value) is not dict or type(value.get("schema_version")) is not int
            or value.get("schema_version") != 1):
        return None
    status = value.get("http_status")
    if type(status) is not int or not 100 <= status <= 599:
        return None
    out = {"schema_version": 1, "http_status": status}
    for key, allowed in (("error_code", ERROR_CODES), ("error_type", ERROR_TYPES),
                         ("body_state", BODY_STATES)):
        item = value.get(key)
        if type(item) is str and item in allowed:
            out[key] = item
    hashed = value.get("request_id_sha256")
    if type(hashed) is str and _HEX.fullmatch(hashed):
        out["request_id_sha256"] = hashed
    length = value.get("body_bytes_observed")
    if type(length) is int and 0 <= length <= MAX_BODY_BYTES + 1:
        out["body_bytes_observed"] = length
    return out


def _available_error_body(error):
    """One nonblocking read, never a read-until-EOF or helper-thread wait.

    urllib HTTPError.fp is normally HTTPResponse -> BufferedReader -> SocketIO.
    Only that exact standard-library shape and in-memory BytesIO are accepted.
    Unknown transports are skipped, preserving their original failure path.
    Partial/fragmented JSON is expected and deliberately not guessed at.
    """
    response = getattr(error, "fp", None)
    if type(response) is io.BytesIO:
        return response.read(MAX_BODY_BYTES + 1)
    if type(response) is not http.client.HTTPResponse:
        return None
    stream = response.fp
    if type(stream) is not io.BufferedReader:
        return None
    raw = stream.raw
    if type(raw) is not socket.SocketIO:
        return None
    sock = getattr(raw, "_sock", None)
    if not isinstance(sock, socket.socket):
        return None
    old_timeout = sock.gettimeout()
    try:
        sock.settimeout(0.0)
        # read1 is a single buffered-read operation; the socket itself cannot
        # wait, even with a slow/chunked/fragmented remote error response.
        return response.read1(MAX_BODY_BYTES + 1)
    finally:
        try:
            sock.settimeout(old_timeout)
        except OSError:
            # EOF may have closed this failed response's descriptor already.
            pass


def http_error_metadata(error):
    """Best effort only: all parsing/IO failures reduce diagnostics, not work."""
    status = getattr(error, "code", None)
    if type(status) is not int or not 100 <= status <= 599:
        return None
    record = {"schema_version": 1, "http_status": status, "body_state": "unavailable"}
    try:
        headers = getattr(error, "headers", None)
        if headers is not None:
            for name in ("x-request-id", "request-id", "x-dashscope-request-id"):
                hashed = _request_id_digest(headers.get(name))
                if hashed:
                    record["request_id_sha256"] = hashed
                    break
    except Exception:
        pass
    try:
        body = _available_error_body(error)
        if type(body) is not bytes:
            return sanitize_http_error_metadata(record)
        record["body_bytes_observed"] = len(body)
        if not body:
            record["body_state"] = "empty"
        elif len(body) > MAX_BODY_BYTES:
            record["body_state"] = "oversized"
        else:
            try:
                data = json.loads(body)
            except (ValueError, UnicodeError, RecursionError):
                record["body_state"] = "incomplete_or_invalid_json"
            else:
                if type(data) is not dict:
                    record["body_state"] = "unstructured_json"
                else:
                    record["body_state"] = "json"
                    detail = data.get("error")
                    if type(detail) is not dict:
                        detail = data
                    for key, allowed in (("code", ERROR_CODES), ("type", ERROR_TYPES)):
                        value = detail.get(key)
                        if type(value) is str and value in allowed:
                            record["error_" + key] = value
                    if "request_id_sha256" not in record:
                        hashed = _request_id_digest(data.get("request_id"))
                        if hashed:
                            record["request_id_sha256"] = hashed
    except Exception:
        # Never copy error text, including exceptions from the diagnostics.
        pass
    return sanitize_http_error_metadata(record)
