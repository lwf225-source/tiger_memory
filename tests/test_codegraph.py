"""Real Git snapshots and a fake CLI seam; no installed graph engine required."""
import contextlib
import io
import json
import subprocess
from unittest.mock import patch

from tiger_memory import codegraph
from tiger_memory.cli import main
from tiger_memory.model import PLMError
from tiger_memory.service import write_event
from tests.common import MemoryTestCase


class CodeGraphTests(MemoryTestCase):
    def setUp(self):
        super().setUp()
        self.git("init", "-q")
        self.git("config", "user.email", "fixture@local")
        self.git("config", "user.name", "fixture")
        self.source = self.project / "app.py"
        self.source.write_text("def caller():\n    return callee()\n", encoding="utf-8")
        self.git("add", ".")
        self.git("commit", "-qm", "fixture")

    def git(self, *args):
        return subprocess.run(["git", *args], cwd=self.project, check=True,
                              capture_output=True).stdout

    def fake_run(self, argv, root, timeout=30, max_bytes=24000):
        if argv[0] == "git":
            return self.real_run(argv, root, timeout, max_bytes)
        self.calls.append(argv)
        if argv[1] == "init":
            (root / ".codegraph" / "codegraph.db").touch()
        if argv[1] == "status":
            return json.dumps({"initialized": True, "lastIndexed": "2026-09-13",
                               "projectPath": str(root), "index": {"state": "complete"},
                               "version": "1.5.0", "fileCount": 1, "nodeCount": 2, "edgeCount": 1}), False
        if argv[1] == "version":
            return "1.5.0", False
        return "app.py:1 caller -> callee", False

    def bind(self):
        self.calls = []
        self.real_run = codegraph._run
        self.which = patch.object(codegraph.shutil, "which", return_value="/fixture/codegraph")
        self.runner = patch.object(codegraph, "_run", side_effect=self.fake_run)
        self.which.start()
        self.runner.start()
        self.addCleanup(self.which.stop)
        self.addCleanup(self.runner.stop)
        self.assertEqual(codegraph.sync(self.project, initialize=True)["status"], "ready")

    def test_unbound_query_never_initializes_or_calls_codegraph(self):
        with patch.object(codegraph.shutil, "which", return_value="/fixture/codegraph"):
            result = codegraph.query(self.project, "caller")
        self.assertEqual(result["status"], "unbound")
        self.assertFalse((self.project / ".codegraph").exists())

    def test_source_edit_blocks_query_until_explicit_sync(self):
        self.bind()
        self.assertIn("caller", codegraph.query(self.project, "caller")["text"])
        self.source.write_text("def changed(): pass\n")
        before = len(self.calls)
        self.assertEqual(codegraph.query(self.project, "caller")["status"], "stale")
        self.assertEqual(len(self.calls), before)
        codegraph.sync(self.project)
        self.assertEqual(codegraph.query(self.project, "changed")["status"], "ready")

    def test_untracked_source_and_branch_change_invalidate_binding(self):
        self.bind()
        extra = self.project / "new.py"
        extra.write_text("x = 1\n")
        self.assertEqual(codegraph.status(self.project)["status"], "stale")
        extra.unlink()
        self.git("checkout", "-qb", "another")
        self.assertEqual(codegraph.status(self.project)["status"], "stale")

    def test_binding_from_other_project_is_rejected(self):
        self.bind()
        path = self.project / ".codegraph" / "plm-binding.json"
        data = json.loads(path.read_text())
        data["project_root"] = "/another/project"
        path.write_text(json.dumps(data))
        self.assertEqual(codegraph.query(self.project, "caller")["status"], "unbound")

    def test_sync_timeout_does_not_refresh_binding(self):
        self.bind()
        self.source.write_text("x = 1")
        original = self.fake_run
        def fail(argv, root, *args, **kwargs):
            if argv[1] == "sync":
                raise PLMError("timeout")
            return original(argv, root, *args, **kwargs)
        with patch.object(codegraph, "_run", side_effect=fail):
            with self.assertRaises(PLMError):
                codegraph.sync(self.project)
        self.assertEqual(codegraph.status(self.project)["status"], "unbound")
        self.assertFalse((self.project / ".codegraph" / "plm-sync.lock").exists())

    def test_option_injection_is_passed_as_literal_query(self):
        self.bind()
        codegraph.query(self.project, "--path /another/project")
        command = next(c for c in self.calls if c[1] == "explore")
        self.assertEqual(command[-2:], ["--", "--path /another/project"])

    def test_query_failure_degrades_without_source_or_stderr(self):
        self.bind()
        original = self.fake_run
        def fail(argv, root, *args, **kwargs):
            if argv[1] == "explore":
                raise PLMError("secret stderr")
            return original(argv, root, *args, **kwargs)
        with patch.object(codegraph, "_run", side_effect=fail):
            result = codegraph.query(self.project, "caller")
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["text"], "")
        self.assertNotIn("secret stderr", json.dumps(result))

    def test_context_preserves_memory_results_and_default_shape(self):
        write_event(self.project, "caller decision", "Keep caller compatibility.")
        def context(extra):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(["context", "--cwd", str(self.project),
                                       "--query", "caller", "--json", *extra]), 0)
            return json.loads(output.getvalue())
        baseline = context([])
        combined = context(["--with-code"])
        self.assertNotIn("code_evidence", baseline)
        self.assertEqual(baseline["results"], combined["results"])
        self.assertEqual(combined["code_evidence"]["status"], "unbound")

    def test_during_query_edit_discards_code(self):
        self.bind()
        original = self.fake_run
        def edit(argv, root, *args, **kwargs):
            result = original(argv, root, *args, **kwargs)
            if argv[1] == "explore":
                self.source.write_text("x = 2")
            return result
        with patch.object(codegraph, "_run", side_effect=edit):
            result = codegraph.query(self.project, "caller")
        self.assertEqual(result["status"], "stale")
        self.assertEqual(result["text"], "")

    def test_budget_and_management_actions_rejected(self):
        for action in ("init", "sync", "uninit", "install"):
            with self.assertRaises(PLMError):
                codegraph.query(self.project, "caller", action=action)
        with self.assertRaises(PLMError):
            codegraph.query(self.project, "caller", max_bytes=0)

    def test_incomplete_index_never_becomes_ready(self):
        self.bind()
        original = self.fake_run
        def incomplete(argv, root, *args, **kwargs):
            text, truncated = original(argv, root, *args, **kwargs)
            if argv[1] == "status":
                data = json.loads(text)
                data["index"]["state"] = "partial"
                return json.dumps(data), False
            return text, truncated
        with patch.object(codegraph, "_run", side_effect=incomplete):
            self.assertEqual(codegraph.query(self.project, "caller")["status"], "unavailable")
            with self.assertRaises(PLMError):
                codegraph.sync(self.project)

    def test_secret_output_is_withheld(self):
        self.bind()
        with patch.object(codegraph, "assert_no_secrets", side_effect=PLMError("sensitive")):
            result = codegraph.query(self.project, "caller")
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["text"], "")

    def test_subprocess_timeout_and_utf8_budget(self):
        import sys
        with self.assertRaises(PLMError):
            codegraph._run([sys.executable, "-c", "import time; time.sleep(5)"],
                           self.project, timeout=0.05)
        text, truncated = codegraph._run([sys.executable, "-c", "print('中' * 100)"],
                                        self.project, max_bytes=32)
        self.assertTrue(truncated)
        self.assertLessEqual(len(text.encode("utf-8")), 32)
