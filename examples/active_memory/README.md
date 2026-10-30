# A local task that uses memory from an earlier session

This synthetic example is an explicit application entry point. Each `task`
command starts a new process and calls `ActiveStore.run_task` before rendering
a fixed checklist. The renderer receives only task text and injected context;
it does not retrieve memory or call a model. No external assistant is connected.

Run from the Tiger memory checkout. These commands create a new temporary store
outside Git and use only the three synthetic statements in `fixtures/`:

```sh
DEMO_ROOT="$(python3 -B -c 'from pathlib import Path; import tempfile; print(Path(tempfile.mkdtemp(prefix="active-example-")).resolve())')"

env -i PATH=/usr/bin:/bin:/usr/local/bin PYTHONNOUSERSITE=1 \
  PYTHONDONTWRITEBYTECODE=1 TIGER_MEMORY_DIR="$DEMO_ROOT/native-canary" \
  PROJECT_LONG_MEMORY_DIR="$DEMO_ROOT/native-canary" \
  python3 -B scripts/demo_active_memory_task.py \
  --base-dir "$DEMO_ROOT/active-store" --user-id demo-user \
  --session-id demo-seed-session-1 seed

env -i PATH=/usr/bin:/bin:/usr/local/bin PYTHONNOUSERSITE=1 \
  PYTHONDONTWRITEBYTECODE=1 TIGER_MEMORY_DIR="$DEMO_ROOT/native-canary" \
  PROJECT_LONG_MEMORY_DIR="$DEMO_ROOT/native-canary" \
  python3 -B scripts/demo_active_memory_task.py \
  --base-dir "$DEMO_ROOT/active-store" --user-id demo-user \
  --session-id demo-task-session-2 task --task 'cobalt checklist' \
  --project-id demo-project-a --as-of 2026-10-02T00:00:00.000000Z
```

The seed returns receipts with generated UUIDs. The task returns JSON containing
`task_entry: "ActiveStore.run_task"`, `handler_calls: 1`, `style: "compact"`,
and `checklist: "check alpha | check beta | check gamma"`. Its source labels,
in order, are `demo.output_style`, `demo.cobalt.goal`, `demo.cobalt.context`.
UUIDs and absolute paths differ on every fresh run. The task command receives
no seed statement in its arguments or environment and cannot open the fixtures.

Actual frozen verification on 2026-10-13 used those seed/task interfaces in
separate processes and passed all 22 named subprocess steps once. This step
count includes setup and audit checks; it is separate from the 22-case Active
Memory acceptance suite.

| Verified task condition | Injected labels / result |
| --- | --- |
| Before seed | Empty context; default style |
| Same user and project in a new process | Global style, goal, project context; compact |
| Other user | Empty context; no new user directory |
| Other project or no project | Global style only |
| After a new-source correction | New style only for that slot; expanded |
| Earlier `as_of` after correction | Superseded style stays absent |
| Exactly at goal's `valid_until` | Goal absent; immediately before it, present |
| After forgetting project context | Its statement absent from context |
| After rebuilding | Context bytes unchanged; forgotten source/request cannot be reused |
| Artificial duplicate-active catalog | Exit 2; handler not invoked |

Original source/assertion Markdown hashes were identical before and after
forgetting. This is **logical forgetting, not physical deletion**. Expiration
is evaluated at the supplied `as_of`, without a background deletion service.
A still-active entry can be eligible at an earlier query time; superseded or
retracted entries cannot be revived that way. Global entries intentionally
remain visible across the same user's projects.

The demo has no diagnostics API (`diagnostics_available: false`). It proves
context behavior on this controlled task path, not diagnostics isolation,
automatic chat capture, arbitrary assistant integration, or model answer quality.
Its audit guard is not an OS sandbox. Temporary evidence is retained.

Full provenance, fixed predictions, commands and results are in
`output/active_memory_16/B/PREDICTIONS.md`,
`output/active_memory_16/A/demo-result.json`, and
`output/active_memory_16/REPORT.md`. The verification run also checked correction,
forgetting and deliberate corruption; use a fresh store for the short example
above rather than reusing that intentionally corrupted final verification store.
