from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

from tiger_memory.evidence import MAX_EVIDENCE_BYTES, fact_claim, verify_evidence
from tiger_memory.model import project_id_for_root
from tests.common import MemoryTestCase


class EvidenceTests(MemoryTestCase):
    def _file(self, name, content):
        path = self.project / name
        if isinstance(content, dict):
            content = json.dumps(content, ensure_ascii=False).encode("utf-8")
        elif isinstance(content, str):
            content = content.encode("utf-8")
        path.write_bytes(content)
        return path, hashlib.sha256(content).hexdigest()

    def _confirmation(self, claim):
        path, digest = self._file("confirmation.json", {
            "schema_version": 1, "receipt_type": "user_confirmation",
            "asserted_claim": claim, "decision": "confirmed",
        })
        return {"source_path": str(path), "source_hash": digest, "asserted_claim": claim}

    def _git(self, *args):
        return subprocess.run(["git", "-C", str(self.project), *args], check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.decode().strip()

    def test_file_hash_requires_actual_matching_file(self):
        path, digest = self._file("evidence.txt", "构建日志：验证完成")
        evidence = {"path": str(path), "sha256": digest}
        result = verify_evidence("file_hash", evidence, self.project, "构建命令")
        self.assertTrue(result["valid"])
        self.assertEqual(result["level"], "version_verified")
        self.assertFalse(result["supports_claim"])
        self.assertFalse(result["high_risk_eligible"])
        path.write_text("changed", encoding="utf-8")
        self.assertEqual(verify_evidence("file_hash", evidence, self.project)["reason"], "hash_mismatch")
        path.unlink()
        self.assertFalse(verify_evidence("file_hash", evidence, self.project)["valid"])

    def test_confirmation_binds_exact_fact_key_and_value(self):
        claim = fact_claim("release.status", "已发布")
        evidence = self._confirmation(claim)
        result = verify_evidence("user_confirmation", evidence, self.project, claim)
        self.assertTrue(result["valid"])
        self.assertTrue(result["supports_claim"])
        self.assertTrue(result["high_risk_eligible"])
        self.assertFalse(result["identity_verified"])
        self.assertFalse(result["external_state_verified"])
        for wrong in (fact_claim("other_release.status", "已发布"), fact_claim("release.status", "未发布"), ""):
            self.assertFalse(verify_evidence("user_confirmation", evidence, self.project, wrong)["valid"])
        self.assertFalse(verify_evidence("user_confirmation", {"confirmation": "confirmed"}, self.project, claim)["valid"])

    def test_source_event_id_cannot_substitute_for_evidence(self):
        source = {"project_id": project_id_for_root(self.project), "event_id": "source-1", "status": "active"}
        result = verify_evidence("user_confirmation", {"source_event_id": "source-1"}, self.project,
                                 fact_claim("payment.status", "已付款"), source)
        self.assertFalse(result["valid"])
        evidence = self._confirmation(fact_claim("payment.status", "已付款"))
        for bad_source in ({**source, "status": "retracted"}, {**source, "project_id": "different-project"}):
            self.assertFalse(verify_evidence("user_confirmation", evidence, self.project,
                                            evidence["asserted_claim"], bad_source)["valid"])

    def test_command_output_reads_bytes_and_never_executes_command(self):
        path, digest = self._file("output.txt", "All checks passed")
        marker = self.project / "must-not-exist"
        evidence = {"command": "touch " + str(marker), "output_path": str(path), "output_hash": digest}
        with patch("tiger_memory.evidence.subprocess.run", side_effect=AssertionError("must not run")):
            self.assertTrue(verify_evidence("command_output", evidence, self.project, "构建检查流程")["valid"])
        self.assertFalse(marker.exists())
        self.assertFalse(verify_evidence("command_output", {"command": "verify", "output_hash": digest}, self.project)["valid"])
        self.assertFalse(verify_evidence("test_result", {**evidence, "output_hash": "a" * 64}, self.project)["valid"])

    def test_real_commit_and_blob_are_verified_without_proving_release(self):
        self._git("init", "-q")
        self._git("config", "user.name", "Evidence Test")
        self._git("config", "user.email", "fixture@example.invalid")
        path, digest = self._file("artifact.txt", "version one")
        self._git("add", "artifact.txt")
        self._git("commit", "-qm", "fixture")
        commit = self._git("rev-parse", "HEAD")
        evidence = {"commit": commit, "path": "artifact.txt", "sha256": digest}
        self.assertTrue(verify_evidence("git_commit", evidence, self.project)["valid"])
        path.write_text("version two", encoding="utf-8")
        # Git evidence names the committed version, not mutable worktree bytes.
        self.assertTrue(verify_evidence("git_commit", evidence, self.project)["valid"])
        self.assertFalse(verify_evidence("git_commit", {**evidence, "sha256": "a" * 64}, self.project)["valid"])
        self.assertFalse(verify_evidence("git_commit", {"commit": "0" * 40}, self.project)["valid"])
        result = verify_evidence("git_commit", evidence, self.project, fact_claim("release.status", "已发布"))
        self.assertTrue(result["source_verified"])
        self.assertFalse(result["valid"])
        self.assertFalse(result["high_risk_eligible"])
        self.assertEqual(result["reason"], "claim_not_supported")

    def test_structured_tool_receipt_supports_only_recorded_state(self):
        claim = fact_claim("release.status", "已发布")
        receipt = {
            "schema_version": 1, "receipt_type": "tool_result", "asserted_claim": claim,
            "assertion": {"fact_key": "release.status", "value": "已发布"},
            "operation": "release", "status": "released", "tool": "release-status-reader",
            "subject": "fixture-version", "observed_at": "2026-09-10T01:00:00Z",
        }
        def inspect(payload, requested=claim):
            path, digest = self._file("tool.json", payload)
            return verify_evidence("command_output", {"output_path": str(path), "output_hash": digest,
                                                      "asserted_claim": requested}, self.project, requested)
        result = inspect(receipt)
        self.assertTrue(result["valid"])
        self.assertTrue(result["high_risk_eligible"])
        self.assertFalse(result["external_state_verified"])
        for modified in ({**receipt, "status": "success"}, {**receipt, "operation": "payment"},
                         {**receipt, "observed_at": "invalid"}, {**receipt, "assertion": {"fact_key": "release.status", "value": "未发布"}}):
            self.assertFalse(inspect(modified)["valid"])
        ready = fact_claim("release.ready", "可以发布")
        self.assertFalse(inspect({**receipt, "asserted_claim": ready,
                                  "assertion": {"fact_key": "release.ready", "value": "可以发布"}}, ready)["valid"])

    def test_plain_success_output_does_not_prove_high_risk_claim(self):
        path, digest = self._file("result.log", "All tests passed. Deployment was not performed.")
        claim = fact_claim("deployment.status", "已部署")
        result = verify_evidence("test_result", {"output_path": str(path), "output_hash": digest,
                                                "asserted_claim": claim}, self.project, claim)
        self.assertFalse(result["valid"])
        self.assertTrue(result["source_verified"])
        self.assertFalse(result["supports_claim"])

    def test_paths_reject_escape_symlink_and_nonregular_file(self):
        outside = Path(self.temp.name) / "outside.txt"
        outside.write_bytes(b"outside")
        digest = hashlib.sha256(b"outside").hexdigest()
        link = self.project / "link.txt"
        link.symlink_to(outside)
        directory_link = self.project / "linked-directory"
        directory_link.symlink_to(outside.parent, target_is_directory=True)
        fifo = self.project / "pipe"
        os.mkfifo(fifo)
        for name in (str(outside), "../../outside.txt", str(link), str(directory_link / "outside.txt"), str(fifo)):
            result = verify_evidence("file_hash", {"path": name, "sha256": digest}, self.project)
            self.assertFalse(result["valid"], name)
            self.assertNotIn(name, json.dumps(result))

    def test_oversized_and_secret_sources_return_sanitized_errors(self):
        path, digest = self._file("large.txt", b"x" * (MAX_EVIDENCE_BYTES + 1))
        self.assertEqual(verify_evidence("file_hash", {"path": str(path), "sha256": digest}, self.project)["reason"], "evidence_too_large")
        secret = "ghp_" + "x" * 32
        path, digest = self._file("sensitive.txt", secret)
        result = verify_evidence("file_hash", {"path": str(path), "sha256": digest}, self.project)
        self.assertEqual(result["reason"], "secret_detected")
        self.assertNotIn(secret, json.dumps(result))
        self.assertNotIn(str(path), json.dumps(result))

    def test_duplicate_receipt_keys_cannot_change_claim_meaning(self):
        claim = fact_claim("release.status", "已发布")
        raw = '{"schema_version":1,"receipt_type":"user_confirmation","decision":"denied","decision":"confirmed","asserted_claim":' + json.dumps(claim) + '}'
        path, digest = self._file("duplicate.json", raw)
        result = verify_evidence("user_confirmation", {"source_path": str(path), "source_hash": digest,
                                                       "asserted_claim": claim}, self.project, claim)
        self.assertFalse(result["valid"])
