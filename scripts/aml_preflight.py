#!/usr/bin/env python3
"""Static AML submission preflight; never reads, prints, or sends credentials.

This is an inventory of local source artifacts, not a deployment check or an
assertion of competition eligibility. Run aml_open_source --check-config
separately to validate a deployment's explicit environment without API calls.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tiger_memory import aml_adapter, aml_open_source  # noqa: E402


def _docker_context_guard(root: Path) -> dict:
    """Optional packaging diagnostic; Docker is not a submission prerequisite."""
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
        "required_for_submission": False,
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
    """Report source availability, leaving external evidence explicitly unknown.

    Do not call configuration() here: this check intentionally does not even
    inspect credential presence. A Git remote alone cannot prove public access.
    """
    root = Path(root).resolve()
    checks = {
        "git_origin_configured": _has_origin(root),
        "license_present": (root / "LICENSE").is_file(),
        "open_source_entrypoint_present": (root / "tiger_memory" / "aml_open_source.py").is_file(),
        "remote_embedding_provider_present": (root / "tiger_memory" / "remote_embeddings.py").is_file(),
        "preparation_notes_present": (root / "docs" / "OPEN_SOURCE_PREP.md").is_file(),
    }
    return {
        "tool": "plm-aml-preflight-v2",
        "scope": "local-source-artifacts-only",
        "adapter_version": aml_adapter.AML_ADAPTER_VERSION,
        "checks": checks,
        "profile": {
            "name": aml_open_source.PROFILE,
            "entrypoint": "python -m tiger_memory.aml_open_source",
            "config_check": "python -m tiger_memory.aml_open_source --check-config",
            "embedding_model": "text-embedding-v4",
            "embedding_dimensions": 1024,
            "generative_llm": None,
            "required_environment_names": list(aml_open_source.REQUIRED),
            "credentials_policy": "intentionally_not_inspected",
        },
        "optional_packaging": {"docker": _docker_context_guard(root)},
        "unverified_evidence": {
            "current_rules_and_model_disclosure": "not_checked",
            "public_source_and_fixed_commit": "not_checked",
            "paid_embedding_api_and_retrieval_quality": "not_checked",
            "deployed_candidate_config_hash": "not_checked",
            "official_smoke_result": "not_checked",
            "official_full_result": "not_checked",
        },
        "next_required_evidence": [
            "current-open-source-rules-and-complete-model-disclosure",
            "public-source-license-and-fixed-candidate-commit",
            "dedicated-store-and-explicit-runtime-configuration-check",
            "authorized-embedding-api-call-and-retrieval-quality-latency-cost-evidence",
            "deployed-candidate-version-config-hash-auth-and-add-search-readback",
            "official-smoke-and-full-run-results",
        ],
        "network_probe_performed": False,
        "submission_readiness_verified": False,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    print(json.dumps(preflight(args.root), ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
