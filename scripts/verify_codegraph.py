#!/usr/bin/env python3
"""Opt-in integration acceptance using the installed CLI and disposable Git repos.

Runs no remote calls, creates no global memory, and leaves the user's repositories
unchanged. CLI/indexing costs are deliberately outside the ordinary test suite.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tiger_memory import codegraph
from tiger_memory.cli import main
from tiger_memory.service import write_event


def verify():
    started = time.monotonic()
    checks = []
    with tempfile.TemporaryDirectory(prefix="plm-codegraph-acceptance-") as temporary:
        base = Path(temporary).resolve()
        old_memory = os.environ.get("TIGER_MEMORY_DIR")
        os.environ["TIGER_MEMORY_DIR"] = str(base / "memory")
        try:
            def git(root, *args):
                subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)

            def repository(name, implementation):
                root = base / name
                root.mkdir()
                (root / "app.py").write_text(implementation, encoding="utf-8")
                (root / ".gitignore").write_text(".codegraph/\n", encoding="utf-8")
                git(root, "init", "-q")
                git(root, "config", "user.email", "fixture@local")
                git(root, "config", "user.name", "fixture")
                git(root, "add", ".")
                git(root, "commit", "-qm", "fixture")
                return root

            root = repository("中文代码项目", "def leaf():\n    return 'first'\n\ndef entry():\n    return leaf()\n")
            assert codegraph.query(root, "entry")["status"] == "unbound"
            assert codegraph.sync(root, initialize=True)["status"] == "ready"
            for action, symbol, expected in (("explore", "entry", "app.py"),
                                              ("callers", "leaf", "entry"),
                                              ("callees", "entry", "leaf"),
                                              ("impact", "leaf", "entry"),
                                              ("query", "entry", "app.py"),
                                              ("files", "", "app.py")):
                result = codegraph.query(root, symbol, action)
                assert result["status"] == "ready", (action, result)
                assert expected in result["text"], (action, result)
                checks.append(action)

            source = root / "app.py"
            source.write_text(source.read_text().replace("leaf", "replacement"))
            assert codegraph.query(root, "entry")["status"] == "stale"
            codegraph.sync(root)
            assert "replacement" in codegraph.query(root, "entry", "callees")["text"]
            checks.append("edit-stale-sync-new-call-edge")

            git(root, "checkout", "-qb", "branch-two")
            assert codegraph.query(root, "entry")["status"] == "stale"
            codegraph.sync(root)
            checks.append("branch-switch-stale")

            other = repository("second", "def entry():\n    return 'OTHER_ONLY_SENTINEL'\n")
            codegraph.sync(other, initialize=True)
            first = codegraph.query(root, "entry")
            second = codegraph.query(other, "entry")
            assert "OTHER_ONLY_SENTINEL" not in first["text"]
            assert "OTHER_ONLY_SENTINEL" in second["text"]
            checks.append("same-symbol-project-isolation")

            write_event(root, "entry decision", "Keep entry compatible for existing callers.")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                assert main(["context", "--cwd", str(root), "--query", "entry",
                             "--with-code", "--json"]) == 0
            combined = json.loads(output.getvalue())
            assert combined["code_evidence"]["status"] == "ready", combined
            assert "app.py" in combined["code_evidence"]["text"]
            assert any("Keep entry compatible" in r["body"] for r in combined["results"])
            checks.append("combined-history-and-current-code")
            clipped = codegraph.query(root, "entry", max_bytes=256)
            assert clipped["truncated"] and len(clipped["text"].encode("utf-8")) <= 256
            checks.append("utf8-code-budget")
        finally:
            if old_memory is None:
                os.environ.pop("TIGER_MEMORY_DIR", None)
            else:
                os.environ["TIGER_MEMORY_DIR"] = old_memory
    return {"passed": len(checks), "checks": checks,
            "elapsed_s": round(time.monotonic() - started, 2)}


if __name__ == "__main__":
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
