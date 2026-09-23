from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


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

    def test_workspace_preflight_never_reports_credentials(self):
        root = Path(__file__).resolve().parents[1]
        report = preflight.preflight(root)
        self.assertEqual(report["adapter_version"], "1.3.4")
        # The installable skill bundle intentionally omits repository-level
        # Docker artifacts.  A source checkout must pass the guard; a bundle
        # may report that its unavailable Docker context cannot be certified.
        if (root / "Dockerfile").is_file():
            self.assertTrue(report["checks"]["docker_context_guard"])
        else:
            self.assertFalse(report["checks"]["docker_context_guard"])
        self.assertFalse(report["full_gate"]["credentials_present"])
        self.assertEqual(report["full_gate"]["credentials_policy"], "intentionally_not_inspected")


if __name__ == "__main__":
    unittest.main()
