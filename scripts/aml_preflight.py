#!/usr/bin/env python3
"""Static AML submission preflight; never reads, prints, or sends credentials."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tiger_memory import aml_adapter  # noqa: E402


def _docker_context_guard(root: Path) -> dict:
    dockerfile = root / "Dockerfile"
    ignore = root / ".dockerignore"
    required = {".codex_work/", "output/", "aml_data_store/", ".env", ".env.*"}
    entries = set(ignore.read_text(encoding="utf-8").splitlines()) if ignore.is_file() else set()
    text = dockerfile.read_text(encoding="utf-8") if dockerfile.is_file() else ""
    return {
        "dockerfile_present": dockerfile.is_file(),
        "dockerignore_present": ignore.is_file(),
        "missing_sensitive_context_exclusions": sorted(required - entries),
        "copies_full_context": "COPY . " in text or "ADD . " in text,
    }


def _has_origin(root: Path) -> bool:
    try:
        result = subprocess.run(["git", "remote", "get-url", "origin"], cwd=root,
                                text=True, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, timeout=5, check=False)
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def preflight(root: Path = ROOT) -> dict:
    """Report only non-sensitive, submission-relevant readiness facts."""
    root = Path(root).resolve()
    try:
        config_warnings = aml_adapter.validate_model_config()
    except ValueError as exc:
        config_warnings = ["invalid-config:" + str(exc)]
    docker = _docker_context_guard(root)
    full_gate = {
        "fact_model": aml_adapter.AML_FACT_MODEL,
        "query_provider_can_use_openai_namespace": True,
        "offline_default_is_full_gate_candidate": False,
        "credentials_present": False,
        "credentials_policy": "intentionally_not_inspected",
    }
    checks = {
        "public_remote_configured": _has_origin(root),
        "license_present": (root / "LICENSE").is_file(),
        "docker_context_guard": docker["dockerfile_present"] and docker["dockerignore_present"]
                                and not docker["missing_sensitive_context_exclusions"]
                                and not docker["copies_full_context"],
        "local_contract_config_valid": not any(item.startswith("invalid-config:") for item in config_warnings),
        "official_full_result": False,
    }
    return {
        "tool": "plm-aml-preflight-v1",
        "adapter_version": aml_adapter.AML_ADAPTER_VERSION,
        "checks": checks,
        "config_warnings": config_warnings,
        "docker": docker,
        "full_gate": full_gate,
        "next_required_evidence": [
            "official-model-credential-injection-rule",
            "fixed-gpt-4o-mini-paired-evaluation",
            "public-repository-and-license",
            "docker-build-and-smoke-in-a-docker-environment",
            "official-compatibility-smoke-and-full-result",
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    print(json.dumps(preflight(args.root), ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
