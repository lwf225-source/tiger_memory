"""Small synthetic HTTP check; defaults to an offline plan, never retries.

This is a participant-side check, not the competition's official Smoke job.
Only AML_API_KEY is read; the server owns the embedding credential.
"""
import argparse
import ipaddress
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def validate_url(value):
    try:
        parts = urllib.parse.urlsplit(value)
        host = (parts.hostname or "").lower().rstrip(".")
        parts.port
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = False
        if (not host or host == "lewen.woa.com" or host.endswith(".lewen.woa.com")
                or parts.username is not None or parts.password is not None
                or parts.query or parts.fragment or any(c.isspace() for c in value)
                or "\\" in value or "%" in parts.netloc
                or not (parts.scheme == "https" or (parts.scheme == "http" and loopback))):
            raise ValueError
    except ValueError:
        raise ValueError("invalid_service_url") from None
    return value.rstrip("/")


def plan():
    return {"schema": "tiger-aml-client-smoke-v1", "executed": False,
            "official_smoke": False, "synthetic_users": 3,
            "max_http_requests": 12, "automatic_retries": 0,
            "expected_embedding_batches_with_fresh_users": 4,
            "billing_cap_enforced": False,
            "note": "Batch expectation assumes the frozen v4 profile; check provider billing separately.",
            "checks": ["health", "status_requires_auth", "search_requires_auth",
                       "candidate_profile", "add_a", "add_b", "search_a",
                       "search_b", "unknown_user", "idempotent_replay", "replay_readback",
                       "status_after"]}


def run(base_url, api_key):
    base_url = validate_url(base_url)
    if not api_key or not api_key.isascii() or any(c.isspace() for c in api_key):
        raise ValueError("valid_AML_API_KEY_required")
    report = dict(plan(), executed=True, passed=False, results=[])
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    tag = "tiger-smoke-" + uuid.uuid4().hex
    users = [tag + "-a", tag + "-b", tag + "-unknown"]
    texts = ["Synthetic archive marker: violet boxes.", "Synthetic archive marker: amber drawers."]
    report["synthetic_user_prefix"] = tag

    def check(name, method, path, payload=None, status=200, authorized=True, predicate=None):
        started = time.monotonic()
        headers = {"Content-Type": "application/json"}
        if authorized:
            headers["Authorization"] = "Bearer " + api_key
        request = urllib.request.Request(base_url + path, method=method, headers=headers,
            data=None if payload is None else json.dumps(payload).encode())
        actual = None
        passed = False
        body = None
        try:
            try:
                response = opener.open(request, timeout=30)
            except urllib.error.HTTPError as exc:
                response = exc
            with response:
                actual = response.code
                raw = response.read(1024 * 1024 + 1)
            if len(raw) <= 1024 * 1024:
                body = json.loads(raw)
                passed = actual == status and (predicate is None or predicate(body))
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            pass
        # Neither remote bodies, exception text, endpoint nor credentials enter reports.
        report["results"].append({"check": name, "passed": bool(passed),
            "http_status": actual, "elapsed_ms": round((time.monotonic() - started) * 1000, 1)})
        if not passed:
            raise RuntimeError("check_failed")
        return body

    def add(user, text):
        return {"request_id": tag, "user_id": user, "session_id": tag,
                "messages": [{"role": "user", "content": text, "timestamp": 1704067200000}]}

    def search(user):
        return {"query": "Synthetic archive marker", "user_id": user, "top_k": 100, "options": []}

    def content_is(text):
        return lambda body: (len(body["data"]) == 1 and body["data"][0]["content"] == text
                             and isinstance(body["data"][0]["id"], str))

    try:
        check("health", "GET", "/health", predicate=lambda b: b["status"] == "ok")
        check("status_requires_auth", "GET", "/status", status=401, authorized=False)
        check("search_requires_auth", "POST", "/search", search(users[2]), status=401, authorized=False)
        state = check("candidate_profile", "GET", "/status", predicate=lambda b:
            b["config"]["profile"] == "aml-open-source-v4-1"
            and b["config"]["embedding"]["model"] == "text-embedding-v4"
            and b["config"]["retrieval"]["failure_policy"] == "503_no_fallback"
            and len(b["config"]["config_sha256"]) == 64)
        report["config_sha256"] = state["config"]["config_sha256"]
        for i, name in enumerate(["add_a", "add_b"]):
            check(name, "POST", "/add", add(users[i], texts[i]), predicate=lambda b, u=users[i]:
                  b["success"] is True and b["user_id"] == u and b["request_id"] == tag and b["session_id"] == tag)
        for i, name in enumerate(["search_a", "search_b"]):
            check(name, "POST", "/search", search(users[i]), predicate=content_is(texts[i]))
        check("unknown_user", "POST", "/search", search(users[2]), predicate=lambda b: b == {"data": []})
        check("idempotent_replay", "POST", "/add", add(users[0], "Changed replay must not replace original."),
              predicate=lambda b: b["success"] is True)
        check("replay_readback", "POST", "/search", search(users[0]), predicate=content_is(texts[0]))
        check("status_after", "GET", "/status", predicate=lambda b:
              b["config"]["config_sha256"] == report["config_sha256"])
        report["passed"] = True
    except RuntimeError:
        pass
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Send requests; Add/Search may incur embedding fees.")
    parser.add_argument("--base-url", help="Service URL, including any /aml prefix. HTTPS except literal loopback.")
    parser.add_argument("--output", type=Path, help="New JSON report path; existing files are not overwritten.")
    args = parser.parse_args(argv)
    if args.output and args.output.exists():
        parser.error("output_exists")
    if args.execute:
        if not args.base_url:
            parser.error("base_url_required")
        try:
            result = run(args.base_url, os.environ.get("AML_API_KEY", ""))
        except ValueError as exc:
            parser.error(str(exc))
    else:
        result = plan()
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(rendered)
    print(rendered, end="")
    return 0 if not args.execute or result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
