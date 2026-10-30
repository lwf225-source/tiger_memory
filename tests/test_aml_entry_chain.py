"""Actual launcher -> local HTTP -> SIGKILL -> fresh launcher, synthetic only."""
import hashlib
import http.client
import json
import os
from pathlib import Path
import signal
import socket
import sqlite3
import subprocess
from tests.child_env import child_popen
import sys
import tempfile
import time
import unittest

from tests.aml_budget_fixture import receipt_fixture


def serve(directory):
    root = Path(directory)
    def audit(event, args):
        if event in ("socket.connect", "socket.bind"):
            if not isinstance(args[1], tuple) or args[1][0] not in {"127.0.0.1", "::1"}:
                raise RuntimeError("nonloopback denied")
        if event in ("socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr") and args[0] not in {"127.0.0.1", "::1", "localhost"}:
            raise RuntimeError("DNS denied")
    sys.addaudithook(audit)
    socket.getfqdn = lambda _host="": "localhost"
    from tiger_memory import aml_adapter as aml, aml_open_source as candidate
    from tests.test_embedding_budget import FakeTransport
    original = candidate.DashScopeEmbeddingProvider
    def provider(*args, **kwargs):
        instance = original(*args, **kwargs)
        instance._opener = FakeTransport(wire_file=root / "fake-wire.log")
        return instance
    candidate.DashScopeEmbeddingProvider = provider
    candidate.configure()
    server = aml.make_server("127.0.0.1", 0)
    (root / "port").write_text(str(server.server_port))
    server.serve_forever(poll_interval=.02)


class EntrypointProcessChainTests(unittest.TestCase):
    def test_two_users_replay_sigkill_new_process_source_cache_and_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            endpoint = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
            receipt, budget = receipt_fixture(root, endpoint)
            env = {"PATH": os.defpath, "HOME": str(root / "home"), "LC_ALL": "C",
                "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
                "TIGER_MEMORY_DIR": str(root / "store"), "TIGER_MEMORY_EMBEDDING_ENDPOINT": endpoint,
                "TIGER_MEMORY_EMBEDDING_BUDGET_RECEIPT": str(receipt), "TIGER_MEMORY_ALLOW_REMOTE": "1",
                "DASHSCOPE_API_KEY": "SYNTHETIC_ONLY", "AML_API_KEY": "SYNTHETIC_AUTH"}
            children = []
            def start():
                (root / "port").unlink(missing_ok=True)
                child = child_popen([sys.executable, "-B", "-m", "tests.test_aml_entry_chain", str(root)],
                    env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                children.append(child)
                end = time.monotonic()+15
                while not (root / "port").exists() and child.poll() is None and time.monotonic()<end:
                    time.sleep(.02)
                self.assertTrue((root / "port").exists(), "synthetic child did not start")
                return child, int((root / "port").read_text())
            def call(port, route, payload):
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
                try:
                    conn.request("POST", route, json.dumps(payload), {"Authorization": "Bearer SYNTHETIC_AUTH"})
                    response = conn.getresponse()
                    result = json.loads(response.read())
                    self.assertEqual(response.status, 200, result)
                    return result
                finally:
                    conn.close()
            def add(port, user, body):
                return call(port, "/add", {"request_id":"r", "user_id":user, "session_id":"s",
                    "messages":[{"role":"user", "content":body, "timestamp":1704067200000}]})
            def search(port,user):
                return call(port, "/search", {"query":"shared marker", "user_id":user, "top_k":10})
            def sources():
                return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root / "store").rglob("*.md")}
            try:
                first, port = start()
                originals = {"A":"shared marker violet only", "B":"shared marker amber only"}
                for user, body in originals.items():
                    add(port, user, body)
                found = {user:search(port,user) for user in originals}
                for user in originals:
                    self.assertEqual([row["content"] for row in found[user]["data"]], [originals[user]])
                    add(port,user,"changed replay")
                snapshot = sources()
                ledger = budget.snapshot()
                wire = (root / "fake-wire.log").read_bytes()
                self.assertEqual(len(snapshot), 2)
                self.assertEqual((ledger["requests"],ledger["unknown"]),(4,0))
                first.kill()
                self.assertEqual(first.wait(timeout=5), -signal.SIGKILL)
                second, port = start()
                for user in originals:
                    add(port,user,"another changed replay")
                    self.assertEqual(search(port,user),found[user])
                self.assertEqual(sources(),snapshot)
                self.assertEqual(budget.snapshot(),ledger)
                self.assertEqual((root / "fake-wire.log").read_bytes(),wire)
                conn = sqlite3.connect(root / "store/v2/catalog.sqlite3")
                rows = conn.execute("SELECT project_id, scope, scope_id FROM events").fetchall()
                self.assertEqual(len({row[0] for row in rows}),2)
                self.assertEqual({(row[1],row[2]) for row in rows},{("session","A"),("session","B")})
                self.assertEqual(conn.execute("PRAGMA integrity_check").fetchone()[0],"ok")
                conn.close()
            finally:
                for child in children:
                    if child.poll() is None:
                        child.kill()
                    stdout, stderr = child.communicate(timeout=5)
                    self.assertEqual((stdout,stderr),(b"",b""))


if __name__ == "__main__":
    serve(sys.argv[1])
