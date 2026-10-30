#!/usr/bin/env python3
"""Synthetic SPEC-only acceptance. Do not import candidate until explicit run."""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import sqlite3
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
from urllib.parse import unquote, urlsplit

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
T0 = "2026-01-01T00:00:00.000000Z"
T1 = "2026-02-01T00:00:00.000000Z"
T2 = "2026-03-01T00:00:00.000000Z"
DIAG = {"selected", "always_selected", "topical_selected", "omitted_budget",
        "omitted_irrelevant", "overridden", "context_bytes"}
ENTRY = {"entry_id", "kind", "topic_key", "project_id", "statement",
         "applicability", "confirmation", "status", "asserted_at",
         "valid_until", "source_ref"}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest_tree(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file() and not p.is_symlink()}


def install_guard(root):
    root = Path(root).resolve()

    def inside(path):
        if isinstance(path, int) or path is None:
            return True
        try:
            return Path(os.fsdecode(path)).resolve(strict=False).is_relative_to(root)
        except (TypeError, ValueError):
            return False

    def audit(event, args):
        if event in {"socket.connect", "socket.bind", "socket.getaddrinfo"}:
            raise RuntimeError("C guard: network disabled")
        if event == "import" and str(args[0]).split(".")[0] in {
            "torch", "transformers", "sentence_transformers", "openai", "anthropic"}:
            raise RuntimeError("C guard: model import disabled")
        if event == "open":
            path, mode, flags = args
            if not isinstance(path, int):
                opened = Path(os.fsdecode(path)).resolve(strict=False)
                private = Path.home() / ".codex" / "project-memory"
                if opened.is_relative_to(private):
                    raise RuntimeError("C guard: real private store read disabled")
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
                isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)))
            if writing and not inside(path) and os.fsdecode(path) != os.devnull:
                raise RuntimeError("C guard: write outside synthetic root")
        if event == "sqlite3.connect":
            audit.sqlite_calls += 1
            database_path = os.fsdecode(args[0])
            if database_path.startswith("file:"):
                uri = urlsplit(database_path)
                if (uri.scheme != "file" or uri.netloc not in {"", "localhost"}
                        or uri.fragment or not uri.path.startswith("/")
                        or uri.query not in {"", "mode=ro", "mode=rw", "mode=rwc"}
                        or any(ord(ch) < 32 for ch in database_path)
                        or any(ch == "%" and (len(uri.path[i + 1:i + 3]) != 2
                            or any(d not in "0123456789abcdefABCDEF" for d in uri.path[i + 1:i + 3]))
                            for i, ch in enumerate(uri.path))):
                    raise RuntimeError("C guard: invalid local database URI")
                database_path = unquote(uri.path, encoding="utf-8", errors="strict")
                if "\x00" in database_path:
                    raise RuntimeError("C guard: invalid local database URI")
            if not inside(database_path):
                raise RuntimeError("C guard: database outside synthetic root")
        if event in {"os.mkdir", "os.remove", "os.rmdir", "os.chmod", "os.utime", "os.truncate"}:
            if not inside(args[0]):
                raise RuntimeError("C guard: mutation outside synthetic root")
        if event in {"os.rename", "os.link", "os.symlink"}:
            # symlink source may be a target string; destination must be local.
            paths = args[1:2] if event == "os.symlink" else args[:2]
            if not all(inside(p) for p in paths):
                raise RuntimeError("C guard: link/rename outside synthetic root")
        if event == "subprocess.Popen":
            _exe, argv, _cwd, _env = args
            if not isinstance(argv, (list, tuple)):
                raise RuntimeError("C guard: shell process disabled")
            argv = [str(x) for x in argv]
            git_ok = (len(argv) == 5 and Path(argv[0]).name == "git"
                      and argv[1] == "-C" and inside(argv[2])
                      and argv[3:] == ["rev-parse", "--show-toplevel"])
            child_ok = (Path(argv[0]).resolve() == Path(sys.executable).resolve()
                        and len(argv) > 2 and argv[1] == str(Path(__file__).resolve())
                        and argv[2] in {"--worker", "--cli-child"})
            if not (git_ok or child_ok):
                raise RuntimeError("C guard: unapproved process")
    audit.sqlite_calls = 0
    sys.addaudithook(audit)
    return audit


def payload(n, text="cobalt synthetic preference", **kw):
    result = dict(kind="preference", topic_key="cobalt", important=True,
                  request_id="request-" + str(n), confirmation="user_stated",
                  project_id="", applicability="topical", valid_until="",
                  source=dict(session_id="session-" + str(n), message_id="message-" + str(n),
                              text=text, asserted_at=T0, role="user"))
    result.update(kw)
    return result


def bind_runtime(runtime_root, evidence_root, role):
    """Prove parent and each worker/CLI child use the requested runtime bytes."""
    runtime = Path(runtime_root).resolve()
    binding = {}
    for name in ("active_memory", "model"):
        module = importlib.import_module("tiger_memory." + name)
        actual = Path(module.__file__).resolve()
        expected = runtime / "tiger_memory" / (name + ".py")
        if actual != expected:
            raise RuntimeError("runtime import path mismatch")
        binding[name] = {"path": str(actual), "sha256": hashlib.sha256(actual.read_bytes()).hexdigest()}
    path = Path(evidence_root) / ("import-" + role + "-" + str(os.getpid()) + ".json")
    path.write_text(canonical(binding))


def worker(input_path):
    data = json.loads(Path(input_path).read_text())
    install_guard(data["temp_root"])
    sys.path.insert(0, data["runtime_root"])
    bind_runtime(data["runtime_root"], data["temp_root"], "child")
    ActiveStore = importlib.import_module("tiger_memory.active_memory").ActiveStore
    store = ActiveStore(Path(data["base"]), data["user"])
    if data.get("ready"):
        Path(data["ready"]).write_text("ready")
        deadline = time.monotonic() + 10
        while not Path(data["go"]).exists():
            if time.monotonic() > deadline:
                raise RuntimeError("race barrier timed out")
            time.sleep(0.01)
    try:
        if data["operation"] == "remember":
            result = store.remember(**data["payload"])
        elif data["operation"] == "supersede":
            result = store.supersede(data["entry_id"], **data["payload"])
        else:
            result = store.prepare_context(**data["payload"])
        print(canonical({"ok": True, "result": result}))
    except Exception as exc:
        print(canonical({"ok": False, "error_type": type(exc).__name__, "error": str(exc)}))


class Acceptance(unittest.TestCase):
    runtime_root = None
    temp_root = None

    def setUp(self):
        self.case_deadline = time.monotonic() + 25
        signal.setitimer(signal.ITIMER_REAL, min(25, max(0.01, self.run_deadline - time.monotonic())))
        self.addCleanup(lambda: signal.setitimer(signal.ITIMER_REAL, max(0.01, self.run_deadline - time.monotonic())))
        self.case = Path(tempfile.mkdtemp(prefix=self._testMethodName + "-", dir=self.temp_root))
        self.base = self.case / "active-base"
        self.legacy = self.case / "legacy-canary"
        self.legacy.mkdir()
        (self.legacy / "canary.txt").write_text("synthetic legacy canary")
        self.env = mock.patch.dict(os.environ, {"TIGER_MEMORY_DIR": str(self.legacy),
                                              "PROJECT_LONG_MEMORY_DIR": str(self.legacy)})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.am = importlib.import_module("tiger_memory.active_memory")
        self.store = self.am.ActiveStore(self.base, "synthetic-user")

    def reject(self, call, *args, **kwargs):
        with self.assertRaises(Exception) as caught:
            call(*args, **kwargs)
        return type(caught.exception).__name__, str(caught.exception)

    def put(self, n, text="cobalt synthetic preference", store=None, **kw):
        p = payload(n, text, **kw)
        return (store or self.store).remember(**p), p

    def context(self, task="cobalt", store=None, **kw):
        options = dict(project_id="", as_of=T2, budget_bytes=8192)
        options.update(kw)
        result = (store or self.store).prepare_context(task, **options)
        self.assertEqual(set(result), {"context", "diagnostics"})
        self.assertIsInstance(result["context"], str)
        self.assertEqual(set(result["diagnostics"]), DIAG)
        for value in result["diagnostics"].values():
            self.assertIs(type(value), int)
            self.assertGreaterEqual(value, 0)
        text = result["context"]
        self.assertLessEqual(len(text.encode()), options["budget_bytes"])
        self.assertEqual(result["diagnostics"]["context_bytes"], len(text.encode()))
        entries = []
        if text:
            body = json.loads(text)
            self.assertEqual(set(body), {"content_role", "entries", "format"})
            self.assertEqual(body["content_role"], "untrusted_user_data")
            self.assertEqual(body["format"], "active_memory.v1")
            self.assertEqual(text, canonical(body))
            entries = body["entries"]
            self.assertTrue(entries)
            for item in entries:
                self.assertEqual(set(item), ENTRY)
                self.assertEqual(item["status"], "active")
                self.assertEqual(set(item["source_ref"]), {"event_id", "session_id", "message_id", "sha256"})
        self.assertEqual(result["diagnostics"]["selected"], len(entries))
        self.assertEqual(result["diagnostics"]["always_selected"], sum(e["applicability"] == "always" for e in entries))
        self.assertEqual(result["diagnostics"]["topical_selected"], sum(e["applicability"] == "topical" for e in entries))
        return result, entries

    def events(self):
        from tiger_memory.events import parse_event
        events = []
        for path in self.base.rglob("*.md"):
            try:
                events.append(parse_event(path))
            except Exception:
                pass
        return events

    def test_01_source_schema(self):
        text = "cobalt 中文偏好🙂"
        p = payload(1, text)
        p["source"]["asserted_at"] = "2026-01-01T08:00:00+08:00"
        out = self.store.remember(**p)
        self.assertEqual(set(out), {"entry_id", "source_event_id", "source_hash", "status"})
        self.assertEqual(out["source_hash"], hashlib.sha256(text.encode()).hexdigest())
        _, entries = self.context()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0], dict(entry_id=out["entry_id"], kind="preference", topic_key="cobalt",
            project_id="", statement=text, applicability="topical", confirmation="user_stated",
            status="active", asserted_at=T0, valid_until="", source_ref=dict(event_id=out["source_event_id"],
                session_id="session-1", message_id="message-1", sha256=out["source_hash"])))

    def test_02_validation(self):
        mutations = [("important", False), ("important", 1), ("kind", "background"),
            ("kind", ""), ("topic_key", ""), ("topic_key", 1), ("request_id", ""),
            ("topic_key", "界" * 86), ("project_id", "x" * 257), ("request_id", "x" * 257),
            ("confirmation", "authenticated"), ("applicability", "automatic"),
            ("valid_until", "bad-time"), ("valid_until", T0)]
        for i, (key, value) in enumerate(mutations):
            p = payload("bad-" + str(i)); p[key] = value
            self.reject(self.store.remember, **p)
        for key, value in [("role", "assistant"), ("role", "tool"), ("text", " x"),
            ("text", "x "), ("text", ""), ("text", "界" * 2731), ("session_id", ""),
            ("message_id", 9), ("asserted_at", "invalid"), ("event_id", "forged"), ("user_id", "other")]:
            p = payload("bad-source-" + key); p["source"][key] = value
            self.reject(self.store.remember, **p)
        self.reject(self.store.remember, **payload("bad-always", kind="goal", applicability="always"))
        self.reject(self.store.remember, **payload("high-risk", "synthetic release approved", confirmation="user_confirmed"))
        for budget in (True, -1, 65537, "100", 1.5):
            self.reject(self.store.prepare_context, "cobalt", as_of=T2, budget_bytes=budget)
        self.reject(self.store.prepare_context, "界" * 2731, as_of=T2)
        self.reject(self.store.prepare_context, "cobalt", as_of="bad")
        self.assertEqual(self.events(), [])

    def test_03_identity_unknown_user(self):
        before = digest_tree(self.base) if self.base.exists() else {}
        self.context("unknown", store=self.am.ActiveStore(self.base, "unknown-user"))
        self.assertEqual(digest_tree(self.base) if self.base.exists() else {}, before)
        self.assertFalse((self.base / "active" / "users" / hashlib.sha256(b"unknown-user").hexdigest()).exists())
        self.put(1)
        expected = self.base / "active" / "users" / hashlib.sha256(b"synthetic-user").hexdigest()
        self.assertTrue(expected.is_dir())
        original, _ = self.context()
        other = self.am.ActiveStore(self.base, "synthetic-user ")
        self.put(2, "cobalt OTHER USER SYNTHETIC", store=other)
        self.assertEqual(self.context()[0], original)
        self.assertNotEqual(self.context(store=other)[0], original)

    def test_04_scopes_slots(self):
        self.put(1, "cobalt global")
        self.put(2, "cobalt project", project_id="project:A|B")
        self.put(3, "cobalt goal", kind="goal")
        self.put(4, "cobalt alternate topic", topic_key="alternate")
        self.put(5, "cobalt context", kind="project_context", project_id="project:A|B")
        self.assertEqual(len(self.context()[1]), 3)
        entries = self.context(project_id="project:A|B")[1]
        self.assertEqual(len(entries), 4)
        self.assertNotIn("cobalt global", [e["statement"] for e in entries])
        self.assertEqual(self.context(project_id="project:A|B ")[0], self.context()[0])

    def test_05_override_before_relevance(self):
        self.put(1, "cobalt global", topic_key="slot")
        self.put(2, "saffron project", topic_key="slot", project_id="P")
        result, entries = self.context("cobalt", project_id="P")
        self.assertEqual(entries, [])
        self.assertEqual(result["diagnostics"]["overridden"], 1)
        self.put(3, "cobalt " + "x" * 2000, topic_key="large", project_id="P")
        self.put(4, "cobalt global small", topic_key="large")
        self.assertEqual(self.context("cobalt", project_id="P", budget_bytes=700)[1], [])
        self.put(5, "cobalt expired project", topic_key="expired", project_id="P", valid_until=T1)
        self.put(6, "cobalt live global", topic_key="expired")
        self.assertEqual([e["statement"] for e in self.context("cobalt", project_id="P", budget_bytes=1000)[1]], ["cobalt live global"])

    def test_06_lexical_boundaries(self):
        # Third-discussion clarification: no token does not make a task invalid.
        for task in ("", " \t\r\n "):
            self.assertEqual(self.context(task)[1], [])
        self.put(1, "Be concise", topic_key="style", applicability="always")
        for task in ("", " \t\r\n "):
            self.assertEqual([e["applicability"] for e in self.context(task)[1]], ["always"])
        self.put(2, "cobalt alpha_beta 中文，方案 ＡＢＣ", topic_key="lexical")
        for task in ("", "unrelated", "the and 关于 我们", "文方", "中", "abc", "alphas"):
            self.assertEqual([e["applicability"] for e in self.context(task)[1]], ["always"], task)
        for task in ("COBALT", "alpha", "beta", "中文", "方案"):
            self.assertEqual(len(self.context(task)[1]), 2, task)

    def test_07_sorting(self):
        a, _ = self.put(1, "cobalt alpha", topic_key="a")
        b, _ = self.put(2, "cobalt beta", topic_key="b", project_id="P")
        c, _ = self.put(3, "cobalt alpha beta", topic_key="c")
        d, p = self.put(4, "cobalt alpha", topic_key="d")
        ids = [e["entry_id"] for e in self.context("cobalt alpha beta", project_id="P")[1]]
        self.assertEqual(ids, [c["entry_id"], b["entry_id"]] + sorted([a["entry_id"], d["entry_id"]]))
        p = payload(5, "cobalt alpha", topic_key="e"); p["source"]["asserted_at"] = T1
        e = self.store.remember(**p)
        self.assertEqual(self.context("cobalt alpha")[1][0]["entry_id"], e["entry_id"])

    def test_08_budget_exact_utf8(self):
        self.put(1, "cobalt 整条保留🙂" * 10)
        full, entries = self.context()
        n = len(full["context"].encode())
        self.assertEqual(self.context(budget_bytes=n)[0], full)
        for budget in (0, 1, n - 1):
            self.assertEqual(self.context(budget_bytes=budget)[0]["context"], "")
        self.assertEqual(entries[0]["statement"], "cobalt 整条保留🙂" * 10)

    def test_09_always_quarter_first_fit(self):
        self.put(1, "very long " + "界" * 1000, topic_key="long", applicability="always", project_id="P")
        small, _ = self.put(2, "concise", topic_key="small", applicability="always")
        topical, _ = self.put(3, "cobalt " + "字" * 150, topic_key="topic")
        _, all_entries = self.context(project_id="P", budget_bytes=65536)
        small_entry = next(e for e in all_entries if e["entry_id"] == small["entry_id"])
        n = len(canonical(small_entry).encode())
        result, entries = self.context(project_id="P", budget_bytes=4 * n)
        self.assertEqual([e["entry_id"] for e in entries], [small["entry_id"], topical["entry_id"]])
        self.assertEqual(result["diagnostics"]["always_selected"], 1)
        _, entries = self.context(project_id="P", budget_bytes=4 * n - 1)
        self.assertNotIn(small["entry_id"], [e["entry_id"] for e in entries])

    def test_10_correction(self):
        old, _ = self.put(1, "cobalt old")
        p = payload(2, "cobalt new"); p["source"]["asserted_at"] = T1
        new = self.store.supersede(old["entry_id"], **p)
        self.assertEqual(self.context(as_of=T0)[1], [])
        self.assertEqual(self.context()[1][0]["entry_id"], new["entry_id"])
        self.reject(self.store.supersede, old["entry_id"], **payload(3))
        self.reject(self.store.supersede, new["entry_id"], **payload(4))
        equal = payload(5, "cobalt equal"); equal["source"]["asserted_at"] = T1
        self.store.supersede(new["entry_id"], **equal)
        self.assertEqual(self.context()[1][0]["statement"], "cobalt equal")

    def test_11_explicit_time(self):
        self.put(1, valid_until=T1)
        self.assertEqual(len(self.context(as_of=T0)[1]), 1)
        self.assertEqual(self.context(as_of="2025-12-31T23:59:59Z")[1], [])
        self.assertEqual(self.context(as_of=T1)[1], [])
        from tiger_memory import model
        before = self.context(as_of=T0)[0]
        with mock.patch.object(model, "utc_now", return_value="2099-01-01T00:00:00.000000Z"):
            self.assertEqual(self.context(as_of=T0)[0], before)

    def test_12_request_idempotency(self):
        out, p = self.put(1)
        before = {str(e.path): e.path.read_bytes() for e in self.events()}
        self.assertEqual(self.store.remember(**copy.deepcopy(p)), out)
        self.assertEqual({str(e.path): e.path.read_bytes() for e in self.events()}, before)
        changed = copy.deepcopy(p); changed["source"]["text"] = "cobalt changed"
        self.reject(self.store.remember, **changed)
        self.reject(self.store.supersede, out["entry_id"], **p)
        self.reject(self.store.forget_slot, out["entry_id"], request_id=p["request_id"])
        self.assertEqual({str(e.path): e.path.read_bytes() for e in self.events()}, before)

    def test_13_source_binding(self):
        _, p = self.put(1)
        before = len(self.events())
        changed = copy.deepcopy(p); changed.update(request_id="new-request", topic_key="other")
        self.reject(self.store.remember, **changed)
        changed["topic_key"] = "cobalt"; changed["source"]["text"] = "cobalt mutated"
        self.reject(self.store.remember, **changed)
        self.assertEqual(len(self.events()), before)
        self.put(2, topic_key="new-slot")

    def test_14_forget(self):
        self.put(1, "cobalt global")
        old, p = self.put(2, "cobalt project old", project_id="P")
        newp = payload(3, "cobalt project new", project_id="P")
        self.store.supersede(old["entry_id"], **newp)
        self.assertEqual(self.store.forget_slot(old["entry_id"], request_id="forget"), {"forgotten": True})
        self.assertEqual(self.store.forget_slot(old["entry_id"], request_id="forget"), {"forgotten": True})
        self.assertEqual([e["statement"] for e in self.context(project_id="P")[1]], ["cobalt global"])
        first = self.reject(self.store.remember, **p)
        second = self.reject(self.store.remember, **p)
        self.assertEqual(first, second)
        self.assertNotIn(p["source"]["text"], first[1])
        changed = copy.deepcopy(p); changed["request_id"] = "new-request"
        self.reject(self.store.remember, **changed)
        self.put(4, "cobalt remembered again", project_id="P")

    def test_15_rebuild(self):
        out, p = self.put(1)
        before = self.context()[0]
        self.store.rebuild()
        self.assertEqual(self.context()[0], before)
        self.assertEqual(self.store.remember(**p), out)
        self.store.forget_slot(out["entry_id"], request_id="forget")
        self.store.rebuild()
        self.assertEqual(self.context()[1], [])
        changed = copy.deepcopy(p); changed["request_id"] = "reused-source"
        self.reject(self.store.remember, **changed)
        self.reject(self.store.remember, **p)
        self.put(2)

    def child(self, data, mode="--worker"):
        path = self.case / ("child-" + str(time.monotonic_ns()) + ".json")
        data.update(temp_root=str(self.temp_root), runtime_root=str(self.runtime_root),
                    base=str(self.base), user="synthetic-user")
        path.write_text(canonical(data))
        done = subprocess.run([sys.executable, str(Path(__file__).resolve()), mode, str(path)],
                              text=True, capture_output=True, timeout=10)
        self.assertEqual(done.returncode, 0, done.stderr)
        return json.loads(done.stdout)

    def test_16_process_cli_handler(self):
        added = self.child(dict(command="remember", payload=payload(1)), "--cli-child")
        self.assertEqual(added["status"], "active")
        expected = self.context()[0]
        actual = self.child(dict(payload=dict(task="cobalt", project_id="", as_of=T2, budget_bytes=8192)), "--cli-child")
        self.assertEqual(actual, expected)
        from tiger_memory.model import memory_base
        seen = []
        def handler(task, context):
            seen.append((task, context, memory_base()))
            return "handler-result"
        self.assertEqual(self.store.run_task("cobalt", handler, as_of=T2), "handler-result")
        self.assertEqual(seen, [("cobalt", expected["context"], self.legacy.resolve())])
        self.reject(self.store.run_task, "cobalt", handler, as_of="bad")
        self.assertEqual(len(seen), 1)

    def race(self, operation, entry_id=None):
        go = self.case / "go"
        procs = []
        try:
            for i in range(2):
                data = dict(temp_root=str(self.temp_root), runtime_root=str(self.runtime_root), base=str(self.base),
                    user="synthetic-user", operation=operation, payload=payload("race-" + str(i), "cobalt racer " + str(i)),
                    entry_id=entry_id, ready=str(self.case / ("ready-" + str(i))), go=str(go))
                inp = self.case / ("race-" + str(i) + ".json"); inp.write_text(canonical(data))
                procs.append(subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", str(inp)],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
            deadline = time.monotonic() + 10
            while not all((self.case / ("ready-" + str(i))).exists() for i in range(2)):
                self.assertLess(time.monotonic(), deadline, "race setup timed out")
                time.sleep(0.01)
            go.write_text("go")
            results = []
            for proc in procs:
                stdout, stderr = proc.communicate(timeout=10)
                self.assertEqual(proc.returncode, 0, stderr)
                results.append(json.loads(stdout))
            self.assertEqual(sum(r["ok"] for r in results), 1, results)
        finally:
            for proc in procs:
                if proc.poll() is None:
                    proc.kill(); proc.wait()

    def test_17_process_create_cas(self):
        self.race("remember")
        self.assertEqual(len(self.context()[1]), 1)
        self.assertEqual(len([e for e in self.events() if e.metadata.get("kind") == "fact"]), 1)
        self.assertEqual(len([e for e in self.events() if e.metadata.get("kind") == "episode"]), 1)

    def test_18_process_correct_cas(self):
        old, _ = self.put(1)
        self.race("supersede", old["entry_id"])
        self.assertEqual(len(self.context()[1]), 1)
        self.assertEqual(len([e for e in self.events() if e.metadata.get("kind") == "fact"]), 2)
        self.assertEqual(len([e for e in self.events() if e.metadata.get("kind") == "episode"]), 2)

    def test_19_threads_context_reset(self):
        env_before = dict(os.environ)
        users = [self.am.ActiveStore(self.base, "thread-user-" + str(i)) for i in range(2)]
        def action(i):
            store = users[i % 2]
            self.put("thread-" + str(i), "cobalt owner-" + str(i % 2), store=store, topic_key="slot-" + str(i))
            return [e["statement"] for e in self.context(store=store)[1]]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(action, range(8)))
        for i, statements in enumerate(results):
            self.assertTrue(all(s == "cobalt owner-" + str(i % 2) for s in statements))
        self.assertTrue(dict(os.environ) == env_before, "process environment changed")

    def test_20_corruption_failclosed(self):
        out, _ = self.put(1)
        source = next(e for e in self.events() if e.event_id == out["source_event_id"])
        original = source.path.read_bytes()
        from tiger_memory.events import serialize_event
        source.path.write_text(serialize_event(source.metadata, source.body + " TAMPERED"))
        called = []
        self.reject(self.store.run_task, "cobalt", lambda *x: called.append(x), as_of=T2)
        self.assertEqual(called, [])
        self.reject(self.store.rebuild)
        source.path.write_bytes(original)
        self.store.rebuild()
        self.store.supersede(out["entry_id"], **payload(2, "cobalt new"))
        catalogs = []
        for path in self.base.rglob("*"):
            if path.is_file() and path.suffix in {".sqlite", ".sqlite3", ".db"}:
                conn = sqlite3.connect(str(path))
                try:
                    if conn.execute("SELECT name FROM sqlite_master WHERE name='facts'").fetchone():
                        changed = conn.execute("UPDATE facts SET status='active',valid_to='' WHERE status='superseded'").rowcount
                        conn.commit()
                        if changed: catalogs.append(path)
                finally:
                    conn.close()
        self.assertEqual(len(catalogs), 1, "native synthetic catalog must be identifiable")
        self.reject(self.store.run_task, "cobalt", lambda *x: called.append(x), as_of=T2)
        self.assertEqual(called, [])

    def test_21_secrets_paths_owner(self):
        # Direct controls test the registered inner predicate; no outside DB opens.
        callback = self.guard_audit
        initial_count = callback.sqlite_calls
        expected_count = initial_count
        outside = self.temp_root.parent / ("synthetic-outside-" + self.temp_root.name + ".sqlite")
        outside_uri = outside.as_uri() + "?mode=rw"
        traversal_uri = self.temp_root.as_uri() + "/%2e%2e/" + outside.name + "?mode=rw"
        for uri in (outside_uri, traversal_uri):
            with self.assertRaisesRegex(RuntimeError, "C guard: database outside synthetic root"):
                callback("sqlite3.connect", (uri,))
            expected_count += 1
            self.assertEqual(callback.sqlite_calls, expected_count)
        for suffix in ("?mode=rw&vfs=synthetic", "?unknown=1", "?mode=ro&mode=rw"):
            with self.assertRaisesRegex(RuntimeError, "C guard: invalid local database URI"):
                callback("sqlite3.connect", ((self.temp_root / "synthetic-guard.sqlite").as_uri() + suffix,))
            expected_count += 1
            self.assertEqual(callback.sqlite_calls, expected_count)
        self.assertEqual(expected_count, initial_count + 5)
        # sqlite3.connect's audit event carries one database argument. This emits
        # that real event through all registered hooks, without opening a DB.
        sys.audit("sqlite3.connect", str(self.temp_root / "registered-positive.sqlite"))
        self.assertEqual(callback.sqlite_calls, initial_count + 6)
        (self.temp_root / "guard-registration-proof.json").write_text(canonical({
            "direct_negative_count": 5, "registered_positive_count": 1,
            "initial_count": initial_count, "final_count": callback.sqlite_calls,
            "same_registered_callback": True,
            "scope": "inner predicates directly; registration via valid audit event"}))
        # Original detectors must see raw quotes and CRLF, not only JSON escapes.
        negatives = [
            ('cobalt password="SYNTHETICQUOTEDVALUE123"', "SYNTHETICQUOTEDVALUE123"),
            ('cobalt\r\napi_key="SYNTHETICCRLFVALUE123"\r\nend', "SYNTHETICCRLFVALUE123"),
        ]
        for index, (text, marker) in enumerate(negatives):
            self.reject(self.store.remember, **payload("raw-negative-" + str(index), text))
            for path in self.case.rglob("*"):
                if path.is_file():
                    self.assertTrue(marker.encode() not in path.read_bytes(), "sensitive marker persisted")
        secret = "sk-SYNTHETICONLY00000000000000000000"
        p = payload(1, "cobalt " + secret)
        self.reject(self.store.remember, **p)
        for path in self.case.rglob("*"):
            if path.is_file(): self.assertNotIn(secret.encode(), path.read_bytes())
        for base in (self.legacy, self.legacy / "nested", self.case):
            def overlap():
                self.am.ActiveStore(base, "overlap").prepare_context("cobalt", as_of=T2)
            self.reject(overlap)
        link_base = self.case / "link-base"; link_base.mkdir()
        destination = self.case / "symlink-target"; destination.mkdir()
        (link_base / "active").symlink_to(destination, target_is_directory=True)
        self.reject(lambda: self.am.ActiveStore(link_base, "link").remember(**payload(2)))
        self.put(3)
        user_root = self.base / "active" / "users" / hashlib.sha256(b"synthetic-user").hexdigest()
        manifests = list(user_root.glob("*.json"))
        self.assertTrue(manifests, "owner manifest must exist in documented user root")
        for manifest in manifests:
            manifest.write_text('{"corrupt":"synthetic owner mismatch"}')
        self.reject(self.store.prepare_context, "cobalt", as_of=T2)

    def test_22_legacy_aml_canary(self):
        from tiger_memory import service, aml_adapter
        # Recency has continuous time decay: hold the control clock fixed.
        clock = mock.patch("tiger_memory.search.utc_now", return_value="2030-01-01T00:00:00.000000Z")
        clock.start()
        self.addCleanup(clock.stop)
        project = self.case / "ordinary-project"; project.mkdir()
        (project / "pyproject.toml").write_text("[project]\nname='synthetic-control'\nversion='0'\n")
        service.write_event(project, "synthetic cobalt", "cobalt ordinary evidence", idempotency_key="control")
        control = [x.as_dict() for x in service.context_data(project, "cobalt")[1]]
        add = dict(request_id="synthetic-aml-request", user_id="synthetic-aml-user", session_id="synthetic-aml-session",
                   messages=[dict(role="user", content="cobalt AML synthetic evidence", timestamp=1767225600000)])
        added = aml_adapter.add(add)
        self.assertEqual(added, dict(success=True, request_id=add["request_id"], user_id=add["user_id"], session_id=add["session_id"]))
        query = dict(query="cobalt", user_id="synthetic-aml-user", top_k=5)
        aml_before = aml_adapter.search(query)
        self.assertTrue(aml_before["data"])
        before = digest_tree(self.legacy)
        self.put(1)
        self.context()
        self.store.rebuild()
        self.assertEqual(digest_tree(self.legacy), before)
        self.assertEqual([x.as_dict() for x in service.context_data(project, "cobalt")[1]], control)
        self.assertEqual(aml_adapter.search(query), aml_before)


def cli_child(input_path):
    data = json.loads(Path(input_path).read_text())
    install_guard(data["temp_root"])
    sys.path.insert(0, data["runtime_root"])
    bind_runtime(data["runtime_root"], data["temp_root"], "child")
    import runpy
    p = data["payload"]
    command = data.get("command", "context")
    sys.argv = ["tiger_memory.active_memory", "--base-dir", data["base"], "--user-id", data["user"], command]
    if command == "context":
        sys.argv += ["--task", p["task"], "--project-id", p["project_id"], "--as-of", p["as_of"],
                     "--budget-bytes", str(p["budget_bytes"])]
    else:
        sys.stdin = io.TextIOWrapper(io.BytesIO(canonical(p).encode("utf-8")), encoding="utf-8")
    runpy.run_module("tiger_memory.active_memory", run_name="__main__")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--cli-child", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker); return 0
    if args.cli_child:
        cli_child(args.cli_child); return 0
    if not args.runtime_root:
        parser.error("--runtime-root is required; running is an explicit acceptance operation")
    runtime = args.runtime_root.resolve()
    root = Path(tempfile.mkdtemp(prefix="active-memory-C-")).resolve()
    # Deliberately retained on success/failure; no evidence cleanup in this runner.
    tempfile.tempdir = str(root)
    with mock.patch.object(sys, "addaudithook", wraps=sys.addaudithook) as registration:
        callback = install_guard(root)
    if registration.call_count != 1 or registration.call_args.args[0] is not callback:
        raise RuntimeError("guard registration object mismatch")
    Acceptance.guard_audit = staticmethod(callback)
    sys.path.insert(0, str(runtime))
    bind_runtime(runtime, root, "parent")
    Acceptance.runtime_root = runtime
    Acceptance.temp_root = root
    Acceptance.run_deadline = time.monotonic() + 120
    def timed_out(*_):
        raise TimeoutError("C acceptance time budget exceeded")
    signal.signal(signal.SIGALRM, timed_out)
    signal.setitimer(signal.ITIMER_REAL, 120)
    module = importlib.import_module("tiger_memory.active_memory")
    imported = Path(module.__file__).resolve()
    if imported != runtime / "tiger_memory" / "active_memory.py":
        raise RuntimeError("candidate import path mismatch")
    binding = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted((runtime / "tiger_memory").glob("*.py"))}
    (root / "runtime-binding.json").write_text(canonical(binding))
    print(canonical({"evidence_directory": str(root), "runtime_root": str(runtime),
                     "active_module_sha256": binding[str(imported)]}), flush=True)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Acceptance)
    with (root / "unittest.log").open("w") as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2, failfast=True).run(suite)
    signal.setitimer(signal.ITIMER_REAL, 0)
    summary = dict(tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        successful=result.wasSuccessful(), stopped_at_first_mismatch=True, synthetic_only=True,
        evidence_directory=str(root))
    (root / "result.json").write_text(canonical(summary))
    print((root / "unittest.log").read_text(), file=sys.stderr)
    print(canonical(summary))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
