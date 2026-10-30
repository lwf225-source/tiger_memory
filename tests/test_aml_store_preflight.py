"""No sockets, API credentials, real stores or user Git configuration."""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tiger_memory import aml_store_preflight as preflight


class StorePreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.env = {"PATH": os.defpath + os.pathsep + "/opt/homebrew/bin",
                    "HOME": str(self.base / "home"), "LC_ALL": "C",
                    "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull}
        self.env_patch = patch.dict(os.environ, self.env, clear=True)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        self.network = patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden"))
        self.network.start()
        self.addCleanup(self.network.stop)
        self.identity = {"profile": preflight.PROFILE, "config_sha256": "a" * 64}

    def inspect(self, root, identity=None):
        return preflight.inspect_store(str(root), identity, environ=self.env)

    def reject(self, root, code, identity=None):
        report = self.inspect(root, identity)
        self.assertFalse(report["passed"], report)
        self.assertEqual(report["issues"], [code])
        self.assertNotIn(str(self.base), json.dumps(report))

    def git(self, *args):
        if shutil.which("git") is None:
            self.fail("Git is required for these topology tests")
        result = subprocess.run(["git", *args], env=self.env, capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_empty_and_missing_external_store_are_readonly(self):
        empty = self.base / "empty with spaces"
        empty.mkdir()
        fresh = self.base / "new" / "store"
        for root in (empty, fresh):
            self.assertTrue(self.inspect(root)["passed"])
        self.assertFalse(fresh.parent.exists())
        self.assertEqual(list(empty.iterdir()), [])
        # A repository next door must not reject normal external storage.
        self.git("init", "-q", str(self.base / "sibling-repo"))
        self.assertTrue(self.inspect(empty)["passed"])

    def test_git_checkout_and_linked_worktree_marker_rejected(self):
        repo = self.base / "repo"
        self.git("init", "-q", str(repo))
        for root in (repo, repo / "new" / "store"):
            self.reject(root, "store_inside_git")
        linked = self.base / "linked"
        linked.mkdir()
        (linked / ".git").write_text("gitdir: synthetic-missing-admin-dir\n")
        self.reject(linked / "store", "store_inside_git")
        self.assertFalse((repo / "new").exists())

    def test_bare_repository_is_rejected(self):
        bare = self.base / "bare"
        self.git("init", "-q", "--bare", str(bare))
        self.reject(bare / "store", "store_inside_git")

    def test_symlinks_relative_daily_and_file_paths_rejected(self):
        target = self.base / "target"
        target.mkdir()
        link = self.base / "link"
        link.symlink_to(target, target_is_directory=True)
        broken = self.base / "broken"
        broken.symlink_to(self.base / "missing")
        for root in (link, link / "store", broken):
            self.reject(root, "store_canonical_required")
        self.reject("relative/store", "store_absolute_required")
        self.reject(target / ".." / "store", "store_canonical_required")
        self.reject(Path.home() / ".codex/project-memory/sub", "store_daily_overlap")
        self.reject(Path.home(), "store_daily_overlap")
        file = self.base / "file"
        file.write_text("preserve")
        self.reject(file, "store_not_directory")
        self.reject(file / "child", "store_not_directory")
        self.assertEqual(file.read_text(), "preserve")

    def test_marker_binding_and_errors_preserve_files(self):
        root = self.base / "owned"
        root.mkdir()
        marker = root / preflight.MARKER
        marker.write_text(json.dumps(self.identity))
        (root / "synthetic-source.txt").write_text("original")
        self.assertTrue(self.inspect(root, self.identity)["passed"])
        self.assertTrue(self.inspect(root)["passed"])
        self.assertFalse(self.inspect(root)["configuration_binding_checked"])
        before = marker.read_bytes()
        self.reject(root, "store_marker_conflict", dict(self.identity, config_sha256="b" * 64))
        self.assertEqual(marker.read_bytes(), before)
        for data in ("[]", "not-json", json.dumps({"profile": preflight.PROFILE}), "x" * 4097):
            marker.write_text(data)
            self.reject(root, "store_marker_conflict", self.identity)
            self.assertEqual(marker.read_text(), data)
        marker.unlink()
        self.reject(root, "store_marker_conflict", self.identity)
        other = self.base / "identity.json"
        other.write_text(json.dumps(self.identity))
        marker.symlink_to(other)
        self.reject(root, "store_marker_conflict", self.identity)
        self.assertEqual((root / "synthetic-source.txt").read_text(), "original")

    def test_git_errors_fail_closed_with_sanitized_output(self):
        fresh = self.base / "fresh"
        for error, code in ((FileNotFoundError("canary-path"), "store_git_probe_unavailable"),
                            (subprocess.TimeoutExpired("canary-path", 5), "store_git_probe_timeout"),
                            (OSError("canary-path"), "store_git_probe_failed")):
            with patch.object(preflight.subprocess, "run", side_effect=error):
                self.reject(fresh, code)
        failure = subprocess.CompletedProcess([], 128, "", "fatal: synthetic configuration error canary")
        with patch.object(preflight.subprocess, "run", return_value=failure):
            self.reject(fresh, "store_git_probe_failed")
        with patch.object(preflight.subprocess, "run", side_effect=AssertionError("must not probe")):
            report = preflight.inspect_store(str(fresh), environ=dict(self.env, GIT_DIR="canary"))
            self.assertEqual(report["issues"], ["store_git_environment"])
        self.assertFalse(fresh.exists())

    def test_git_config_context_and_timeout_are_preserved(self):
        outside = self.base / "fresh"
        nonrepo = subprocess.CompletedProcess([], 128, "", "git: warning: temporary directory unavailable\nfatal: not a git repository (or any of the parent directories): .git\n")
        with patch.object(preflight.subprocess, "run", return_value=nonrepo) as run:
            self.assertTrue(self.inspect(outside)["passed"])
        self.assertEqual(run.call_args.kwargs["timeout"], 5)
        self.assertEqual(run.call_args.kwargs["env"]["GIT_CONFIG_GLOBAL"], os.devnull)

    def test_cli_exit_codes_and_no_path_disclosure(self):
        for root, expected in ((self.base / "fresh", 0), (Path("relative"), 2)):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = preflight.main(["--store", str(root)])
            self.assertEqual(result, expected)
            report = json.loads(output.getvalue())
            self.assertFalse(report["network_probe_performed"])
            self.assertEqual(report["writability"], "not_tested")
            self.assertNotIn(str(self.base), output.getvalue())


if __name__ == "__main__":
    unittest.main()
