import contextlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tiger_memory import aml_open_source as candidate, aml_adapter as aml
from tiger_memory.remote_embeddings import DashScopeEmbeddingProvider
from tests.aml_budget_fixture import receipt_fixture


class StartupIntegrationTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name).resolve()
        self.store = self.base / "store"
        self.env = {"PATH": os.defpath + ":/opt/homebrew/bin", "HOME": str(self.base / "home"),
                    "LC_ALL": "C", "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
                    "TIGER_MEMORY_DIR": str(self.store),
                    "TIGER_MEMORY_EMBEDDING_ENDPOINT": "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings",
                    "DASHSCOPE_API_KEY": "synthetic-embedding-only", "AML_API_KEY": "synthetic-service-only",
                    "TIGER_MEMORY_ALLOW_REMOTE": "1"}
        receipt, self.budget = receipt_fixture(self.base, self.env["TIGER_MEMORY_EMBEDDING_ENDPOINT"])
        self.env["TIGER_MEMORY_EMBEDDING_BUDGET_RECEIPT"] = str(receipt)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ, self.env, clear=True))
        for name in ("connect", "connect_ex", "bind"):
            self.stack.enter_context(patch.object(socket.socket, name, side_effect=AssertionError("network forbidden")))
        self.stack.enter_context(patch.object(socket, "getaddrinfo", side_effect=AssertionError("DNS forbidden")))
        self.stack.enter_context(patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("HTTP forbidden")))
        states = {name: value for name, value in vars(aml).items() if name.startswith("AML_")}
        self.addCleanup(lambda: [setattr(aml, name, value) for name, value in states.items()])
        models = dict(aml._MODEL_SINGLETONS)
        self.addCleanup(lambda: (aml._MODEL_SINGLETONS.clear(), aml._MODEL_SINGLETONS.update(models)))
        self.fake_requests = []
        def request(provider, texts, publish=None, raw_keys=None):
            self.fake_requests.append(len(texts))
            values = [[1.0] + [0.0] * 1023 for _ in texts]
            if publish is not None:
                publish(values)
            return values
        self.stack.enter_context(patch.object(DashScopeEmbeddingProvider, "_request", request))

    def test_git_store_rejected_before_listener_and_writes(self):
        repo = self.base / "repo"
        subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True, timeout=5)
        store = repo / "new" / "store"
        output = io.StringIO()
        with patch.dict(os.environ, TIGER_MEMORY_DIR=str(store)), patch.object(aml, "main") as listen, contextlib.redirect_stderr(output):
            self.assertEqual(candidate.main([]), 2)
        listen.assert_not_called()
        self.assertIn("store_inside_git", output.getvalue())
        self.assertNotIn(str(repo), output.getvalue())
        self.assertFalse(store.parent.exists())
        self.assertFalse(self.fake_requests)

    def test_check_startup_is_readonly_and_binds_marker(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch.object(aml, "main") as listen:
            self.assertEqual(candidate.main(["--check-startup"]), 0)
        listen.assert_not_called()
        self.assertFalse(self.store.exists())
        report = json.loads(output.getvalue())
        self.assertTrue(report["store_preflight"]["configuration_binding_checked"])
        self.assertFalse(self.fake_requests)
        for canary in ("synthetic-embedding-only", "synthetic-service-only", str(self.base)):
            self.assertNotIn(canary, output.getvalue())
        candidate.configure()
        marker = self.store / candidate.MARKER
        before = marker.read_bytes()
        with patch.dict(os.environ, TIGER_MEMORY_EMBEDDING_ENDPOINT="https://other.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/embeddings"), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(candidate.main(["--check-startup"]), 2)
        self.assertEqual(marker.read_bytes(), before)

    def test_external_start_replay_and_restart_without_listener(self):
        # Exercise main -> configure -> adapter boundary, but never bind a socket.
        with patch.object(aml, "main", return_value=0) as listen:
            self.assertEqual(candidate.main(["--port", "18191"]), 0)
        listen.assert_called_once_with(["--host", "127.0.0.1", "--port", "18191"])
        def add(user, text):
            return aml.add({"request_id": "synthetic-r", "user_id": user, "session_id": "synthetic-s",
                            "messages": [{"role": "user", "content": text}]})
        add("synthetic-u1", "violet boxes")
        add("synthetic-u2", "amber boxes")
        first = aml.search({"query": "violet", "user_id": "synthetic-u1", "top_k": 5})
        self.assertEqual([row["content"] for row in first["data"]], ["violet boxes"])
        add("synthetic-u1", "changed replay")
        count = len(self.fake_requests)
        marker = (self.store / candidate.MARKER).read_bytes()
        candidate.configure()  # new provider factory; same process, explicitly not SIGKILL proof
        self.assertEqual(aml.search({"query": "violet", "user_id": "synthetic-u1", "top_k": 5}), first)
        self.assertEqual(len(self.fake_requests), count)
        self.assertEqual((self.store / candidate.MARKER).read_bytes(), marker)
        self.assertNotEqual(aml.user_storage_dir("synthetic-u1"), aml.user_storage_dir("synthetic-u2"))

    def test_unowned_store_and_symlink_do_not_start(self):
        self.store.mkdir()
        keep = self.store / "keep.txt"
        keep.write_text("preserve")
        with patch.object(aml, "main") as listen, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(candidate.main([]), 2)
        listen.assert_not_called()
        self.assertEqual(keep.read_text(), "preserve")
        link = self.base / "link"
        link.symlink_to(self.store, target_is_directory=True)
        with patch.dict(os.environ, TIGER_MEMORY_DIR=str(link)), patch.object(aml, "main") as listen, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(candidate.main([]), 2)
        listen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
