#!/usr/bin/env python3
"""Explicit synthetic seed/task CLI. No external assistant integration."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.active_memory.local_guard import handler_guard, install_guard
from examples.active_memory.task_handler import render_task


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def binding():
    from tiger_memory import active_memory, model
    result = {}
    for module in (active_memory, model):
        path = Path(module.__file__).resolve()
        expected = ROOT / "tiger_memory" / (module.__name__.split(".")[-1] + ".py")
        if path != expected:
            raise RuntimeError("runtime binding mismatch")
        result[module.__name__] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    handler_path = ROOT / "examples" / "active_memory" / "task_handler.py"
    result["task_handler"] = {"path": str(handler_path), "sha256": hashlib.sha256(handler_path.read_bytes()).hexdigest()}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", required=True, type=Path)
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--session-id", required=True, help="Synthetic execution/session label")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("seed")
    task = commands.add_parser("task")
    task.add_argument("--task", default="cobalt checklist")
    task.add_argument("--project-id", default="")
    task.add_argument("--as-of", required=True)
    args = parser.parse_args(argv)
    if not args.base_dir.is_absolute():
        parser.error("base-dir must be absolute")
    base = args.base_dir.resolve()
    state = install_guard(base.parent, ROOT, deny_fixture_reads=args.command == "task")
    from tiger_memory.active_memory import ActiveStore
    from tiger_memory.model import memory_base
    proof = binding()
    store = ActiveStore(base, args.user_id)
    if args.command == "seed":
        path = ROOT / "examples" / "active_memory" / "fixtures" / "seed_session1.json"
        fixture = json.loads(path.read_text(encoding="utf-8"))
        if fixture["seed_session_id"] != args.session_id:
            raise RuntimeError("seed session mismatch")
        receipts = {item["label"]: store.remember(**item["payload"]) for item in fixture["entries"]}
        print(canonical({"command": "seed", "session_id": args.session_id, "pid": os.getpid(), "receipts": receipts,
                         "fixture_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "binding": proof}))
        return 0
    native_before = memory_base()
    evidence = {"command": "task", "session_id": args.session_id, "handler_calls": 0, "binding": proof,
                "diagnostics_available": False, "task_entry": "ActiveStore.run_task", "context": None,
                "pid": os.getpid(), "fixture_reads_denied": True}
    def handler(task_text, context):
        evidence["handler_calls"] += 1
        evidence["store_context_reset"] = memory_base() == native_before
        if not evidence["store_context_reset"]:
            raise RuntimeError("store context not reset")
        evidence["context"] = context
        evidence["context_sha256"] = hashlib.sha256(context.encode("utf-8")).hexdigest()
        evidence["handler_guard_enabled"] = True
        with handler_guard(state):
            return render_task(task_text, context)
    try:
        # This is the only task-start path: there is no direct context retrieval.
        evidence["output"] = store.run_task(args.task, handler, project_id=args.project_id,
                                            as_of=args.as_of, budget_bytes=8192)
        evidence["status"] = "completed"
        print(canonical(evidence))
        return 0
    except Exception:
        evidence["status"] = "rejected"
        evidence["error"] = "DEMO_TASK_REJECTED"
        print(canonical(evidence))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
