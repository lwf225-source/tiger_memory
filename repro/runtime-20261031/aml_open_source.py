"""Explicit AML open-source candidate. Checking configuration never calls an API."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit

from . import __version__, aml_adapter
from .remote_embeddings import DashScopeEmbeddingProvider
from .providers import ProviderUnavailable
from .security import ensure_private_dir
from .aml_store_preflight import inspect_store, validate_store

PROFILE = "aml-open-source-v4-1"
MARKER = ".tiger-aml-open-source.json"
REQUIRED = ("TIGER_MEMORY_DIR", "TIGER_MEMORY_EMBEDDING_ENDPOINT",
            "AML_API_KEY", "TIGER_MEMORY_ALLOW_REMOTE")


def _embedding_credential(env, provider):
    # An explicitly supplied generic credential wins even when invalid/empty:
    # fail closed rather than silently switching accounts. DMX never reads the
    # legacy Aliyun credential, including when the generic variable is absent.
    name = ("TIGER_MEMORY_EMBEDDING_API_KEY" if provider == "dmx"
            or "TIGER_MEMORY_EMBEDDING_API_KEY" in env else "DASHSCOPE_API_KEY")
    return name, env.get(name, "")


def configuration(environ=None):
    """Validate metadata only. Returned report contains neither keys nor text."""
    env = os.environ if environ is None else environ
    issues = ["missing:" + name for name in REQUIRED if not env.get(name, "").strip()]
    if env.get("TIGER_MEMORY_ALLOW_REMOTE") != "1":
        issues.append("remote_access_requires_TIGER_MEMORY_ALLOW_REMOTE=1")
    service_key = env.get("AML_API_KEY", "")
    if service_key and any(ord(c) < 33 or ord(c) > 126 for c in service_key):
        issues.append("invalid_service_credential")
    endpoint = env.get("TIGER_MEMORY_EMBEDDING_ENDPOINT", "").strip()
    try:
        url = urlsplit(endpoint)
        host = url.hostname or ""
        official = host in {"dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com"} or bool(
            re.fullmatch(r"[a-zA-Z0-9-]+\.(cn-beijing|cn-hongkong|ap-southeast-1)\.maas\.aliyuncs\.com", host))
        dmx = host == "www.dmxapi.cn" and url.path == "/v1/embeddings"
        allowed = dmx or (official and url.path == "/compatible-mode/v1/embeddings")
        valid = (allowed and url.scheme == "https" and url.port in (None, 443)
                 and url.username is None and url.password is None and not (url.query or url.fragment)
                 and not any(c.isspace() for c in endpoint))
    except ValueError:
        valid = False
        dmx = False
    if not valid:
        issues.append("endpoint_must_be_official_https_embeddings_url")
    credential_name, credential = _embedding_credential(env, "dmx" if dmx else "dashscope")
    if not credential.strip():
        issues.append("missing:" + credential_name)
    if valid and credential:
        try:
            # Constructor validates syntax only: no cache directory, API calls,
            # model download or credential probe.
            DashScopeEmbeddingProvider(endpoint, credential)
        except ProviderUnavailable:
            issues.append("invalid_embedding_configuration")
    raw = env.get("TIGER_MEMORY_DIR", "")
    root = Path(raw).expanduser().resolve() if raw else None
    daily = (Path.home() / ".codex" / "project-memory").resolve()
    if root is not None and (not Path(raw).expanduser().is_absolute() or root == daily
                            or daily in root.parents or root in daily.parents):
        issues.append("dedicated_absolute_store_required")
    manifest = {
        "profile": PROFILE, "core_version": __version__,
        "reliability_revision": "search-runtime-repair-20261031",
        "adapter_version": aml_adapter.AML_ADAPTER_VERSION,
        "embedding": {"model": "text-embedding-v4", "dimensions": 1024,
                      "endpoint_sha256": hashlib.sha256(endpoint.encode()).hexdigest(),
                      "batch_rows": 10, "chunk_utf8_bytes": 8192,
                      "chunk_pooling": "mean_l2", "timeout_seconds": None,
                      "max_request_batches": None, "max_elapsed_seconds": None,
                      "elapsed_budget_policy": "none-user-authorized",
                      "add_capacity_policy": "cache-reuse-unlimited-batches-v1",
                      "cache_scope": "user_id",
                      "work_retry_policy": "cache-reuse-no-financial-claims",
                      "claims_version": None, "provider": "dmxapi",
                      "transport_retry_policy": "transient-backoff-no-attempt-cap",
                      "binding_expiry_policy": "none-user-authorized",
                      "checkpoint_lock_policy": "exclusive-until-drained-then-input-stripes-v1",
                      "error_diagnostics": "nonblocking-bounded-allowlist-v1"},
        "retrieval": {"profile": "lexical", "session_rrf": True, "session_rrf_k": 30,
                      "query_rewrite": "off", "recency_boost": False,
                      "fact_projection": "off", "reranker": "off",
                      "failure_policy": "503_no_fallback"},
        "request_policy": {"deadline_policy": "removed",
                           "deadline_deployment_enabled": False,
                           "short_body": "400-close-before-json",
                           "socket_idle_timeout_seconds": None,
                           "error_connection_policy": "close-on-error",
                           "add_admission_lock": "per-user-flock-release-before-embed-v1",
                           "add_lock_timeout_seconds": None,
                           "storage_identity": "fixed-root-project-scope-v1",
                           "sensitive_search": "actual-query-before-provider-503-v1"},
        "source_recovery": {"policy": "local-inotify-capacity-accounted-v3",
                            "periodic_full_verification_seconds": 60},
        "observability": {"policy": "fixed-stage-operation-split-v3"},
        "generated_answers": False,
        "embedding_budget": None,
        "financial_limits": "removed-user-authorized",
    }
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest["config_sha256"] = digest
    return {"configured": not issues, "issues": issues, "config": manifest,
            "remote_probe_performed": False, "official_smoke_passed": False,
            "official_full_passed": False}


def configure():
    report = configuration()
    if not report["configured"]:
        raise ValueError("; ".join(report["issues"]))
    identity = {"profile": PROFILE, "config_sha256": report["config"]["config_sha256"]}
    root = validate_store(os.environ["TIGER_MEMORY_DIR"], identity)
    marker = root / MARKER
    ensure_private_dir(root)
    if not marker.exists():
        fd = os.open(str(marker), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(identity, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    # Secrets are captured only in process memory. Each request gets its own
    # provider and only that user's numeric cache; no cross-user state.
    endpoint = os.environ["TIGER_MEMORY_EMBEDDING_ENDPOINT"].strip()
    _, key = _embedding_credential(os.environ,
        "dmx" if urlsplit(endpoint).hostname == "www.dmxapi.cn" else "dashscope")

    def provider(user_id):
        return DashScopeEmbeddingProvider(endpoint, key,
            cache_dir=aml_adapter.user_storage_dir(user_id) / "embedding-cache", dimensions=1024,
            timeout=None, checkpoint_namespace=aml_adapter.user_storage_dir(user_id).name)

    aml_adapter.AML_EMBEDDING_FACTORY = provider
    aml_adapter.AML_EMBEDDING = "remote-v4"
    aml_adapter.AML_STRICT_RETRIEVAL = True
    aml_adapter.AML_PROFILE = "lexical"
    aml_adapter.AML_RECENCY_BOOST = False
    aml_adapter.AML_QUERY_REWRITE = "off"
    aml_adapter.AML_FACT_PROJECTION = "off"
    aml_adapter.AML_RERANKER = "off"
    aml_adapter._MODEL_SINGLETONS.clear()
    aml_adapter.AML_RUNTIME_CONFIG = report["config"]
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument("--check-config", action="store_true", help="configuration syntax only; no writes or API calls")
    checks.add_argument("--check-startup", action="store_true", help="configuration and read-only store/Git checks; no listener or API calls")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18191)
    args = parser.parse_args(argv)
    if args.check_config or args.check_startup:
        report = configuration()
        if args.check_startup and report["configured"]:
            identity = {"profile": PROFILE, "config_sha256": report["config"]["config_sha256"]}
            report["store_preflight"] = inspect_store(os.environ["TIGER_MEMORY_DIR"], identity)
            report["configured"] = report["store_preflight"]["passed"]
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report["configured"] else 2
    try:
        configure()
    except (ValueError, OSError) as exc:
        # OS messages may include paths; expose only the error category.
        print("open-source configuration rejected: " + (str(exc) if isinstance(exc, ValueError) else "store_io_error"), file=sys.stderr)
        return 2
    return aml_adapter.main(["--host", args.host, "--port", str(args.port)])


if __name__ == "__main__":
    raise SystemExit(main())
