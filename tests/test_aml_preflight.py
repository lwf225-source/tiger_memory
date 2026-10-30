from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SPEC = importlib.util.spec_from_file_location(
    "aml_preflight", Path(__file__).resolve().parents[1] / "scripts" / "aml_preflight.py")
preflight = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(preflight)


class AMLPreflightTests(unittest.TestCase):
    def test_sensitive_docker_context_guard_requires_explicit_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Dockerfile").write_text("FROM python:3.12-slim\nCOPY . /app\n", encoding="utf-8")
            (root / ".dockerignore").write_text(".codex_work/\n", encoding="utf-8")
            report = preflight._docker_context_guard(root)
        self.assertTrue(report["copies_full_context"])
        self.assertIn(".env", report["missing_sensitive_context_exclusions"])
        self.assertFalse(report["required_for_submission"])

    def test_source_inventory_does_not_require_docker_or_generative_llm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in ("LICENSE", "tiger_memory/aml_open_source.py",
                             "tiger_memory/remote_embeddings.py", "docs/OPEN_SOURCE_PREP.md"):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture", encoding="utf-8")
            with mock.patch.object(preflight, "_has_origin", return_value=True):
                report = preflight.preflight(root)
        self.assertTrue(all(report["checks"].values()))
        self.assertFalse(report["optional_packaging"]["docker"]["dockerfile_present"])
        self.assertIsNone(report["profile"]["generative_llm"])
        self.assertFalse(report["submission_readiness_verified"])
        self.assertEqual(report["unverified_evidence"]["official_full_result"], "not_checked")
        self.assertFalse(any("docker" in item or "paired-evaluation" in item
                             for item in report["next_required_evidence"]))

    def test_missing_source_artifacts_are_reported_without_remote_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(preflight, "_has_origin", return_value=False):
                report = preflight.preflight(Path(directory))
        self.assertFalse(any(report["checks"].values()))
        self.assertEqual(report["unverified_evidence"]["public_source_and_fixed_commit"], "not_checked")

    def test_static_preflight_never_inspects_runtime_credentials_or_calls_api(self):
        forbidden = AssertionError("static preflight must not inspect runtime configuration")
        with mock.patch.dict(os.environ, {"DASHSCOPE_API_KEY": "private-sentinel-one",
                                          "AML_API_KEY": "private-sentinel-two"}):
            with mock.patch.object(preflight.aml_open_source, "configuration", side_effect=forbidden), \
                    mock.patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("network forbidden")), \
                    mock.patch.object(preflight, "_has_origin", return_value=True):
                report = preflight.preflight()
        serialized = json.dumps(report)
        self.assertNotIn("private-sentinel", serialized)
        self.assertNotIn("credentials_present", serialized)
        self.assertEqual(report["profile"]["credentials_policy"], "intentionally_not_inspected")
        self.assertFalse(report["network_probe_performed"])
        # Provider credentials are conditional: DMX uses the generic embedding key;
        # DASHSCOPE_API_KEY is only a fallback for an explicit Aliyun endpoint.
        required = report["profile"]["required_environment_names"]
        self.assertIn("AML_API_KEY", required)
        self.assertIn("TIGER_MEMORY_EMBEDDING_ENDPOINT", required)
        self.assertNotIn("DASHSCOPE_API_KEY", required)


if __name__ == "__main__":
    unittest.main()
