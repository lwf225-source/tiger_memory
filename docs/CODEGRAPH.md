# CodeGraph evidence for tiger_memory

tiger_memory can retrieve code evidence alongside historical memory. The CodeGraph CLI
is optional; the Python core has no new dependencies. The adapter targets the CodeGraph 1.5.0 CLI interface. It works with local Git repositories
with at least one commit. Install the CLI separately before using this feature.

## Product scope and release acceptance

tiger_memory is the deliverable for competition and open-source users. Its optional
code-evidence capability must work independently of the author's Codex account,
installed skills, memory directory or MCP configuration. The product entry point
is tiger_memory's `context --with-code`: tiger_memory retrieves historical memory and current code
evidence and returns both with their distinct provenance. CodeGraph is an
external code-analysis dependency; tiger_memory owns memory retrieval, evidence assembly,
freshness checks, budgets and graceful degradation. An existing host MCP is an
additional integration route, not a prerequisite for using tiger_memory.

The current implementation provides a CLI-backed adapter and combined context.
It does not yet provide automatic decision-to-symbol links, semantic conflict
detection between memory and code, or a tiger_memory-managed MCP client. The host's direct
MCP calls also do not inherit the adapter's snapshot checks. Passing adapter
tests proves integration behavior, not improved memory-answer quality.

Before presenting this feature as release-ready or an evaluated improvement:

- Verify installation and the combined-context example in a clean environment
  without the author's Codex configuration; publish the supported dependency
  versions, installation steps and dependency/license attribution.
- Run a reproducible coding-memory comparison: memory only, code only, and
  combined context. Hold the dataset, reader model and total context budget
  fixed; report answer quality, stale-evidence errors, latency and cost.
- Include tasks requiring both a past decision and changed current code, plus
  non-code memory tasks to measure unnecessary retrieval and regression.
- Keep these results separate from the frozen AML configuration. This adapter
  is not wired into AML Add/Search, and no AML score gain has been measured for
  it. Report competition results only for the actual submitted configuration.

## Existing CodeGraph MCP users

When the host already has CodeGraph MCP, continue using it directly for code
queries and tiger_memory for historical decisions/constraints. The host combines the two
evidence sources in its answer. Do not execute both the MCP and CLI retrieval
for the same need. CodeGraph results are not automatically consolidated as
long-term memory; only separately justified decisions/results use tiger_memory's normal
candidate lifecycle.

The Python CLI adapter below is tiger_memory's standalone integration route for scripts,
CI or hosts without a callable MCP tool. tiger_memory does not launch another MCP server or create another code
database. Both routes use the repository's existing `.codegraph/`. An MCP-driven
index refresh can invalidate the CLI's snapshot receipt; explicit `code sync`
refreshes that receipt when switching to the fallback.

## Use

### Independent installation

The supported standalone path starts from this source checkout. It needs
Python 3.9+ and pip. The build backend and core runtime use the standard library:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --no-index --no-deps .
export TIGER_MEMORY_DIR="$HOME/.local/share/tiger_memory"
.venv/bin/tiger_memory --help
```

To enable code evidence, install the external CLI explicitly. One option, with
npm installed, pins the integration target version:

```bash
npm install -g @colbymchenry/codegraph@1.5.0
codegraph version
```

The package and install mechanism come from the [upstream project](https://github.com/colbymchenry/codegraph).
Version 1.5.0 is the integration target, not a claim about the latest
release. Run the opt-in verifier on your platform before relying on it. Installing tiger_memory does not install CodeGraph or alter host MCP settings.
Dependency provenance is recorded in [THIRD_PARTY.md](../THIRD_PARTY.md).

Replace `python3 scripts/tiger_memory.py` with the installed `.venv/bin/tiger_memory` in the commands
below. Source-checkout invocation remains supported:

```bash
# Explicitly build and bind the current repository's graph.
python3 scripts/tiger_memory.py code init --cwd /path/to/project

# CodeGraph source, locations and call paths, inside a provenance envelope.
python3 scripts/tiger_memory.py code explore --cwd /path/to/project --query 'write_event'
python3 scripts/tiger_memory.py code callers --cwd /path/to/project --query 'write_event'
python3 scripts/tiger_memory.py code impact --cwd /path/to/project --query 'write_event'
python3 scripts/tiger_memory.py code files --cwd /path/to/project

# One response with historical memory and separately identified current code.
python3 scripts/tiger_memory.py context --cwd /path/to/project \
  --query 'Why must write_event remain idempotent?' \
  --with-code --code-query 'write_event' --json

# After editing, committing, switching branches or pulling:
python3 scripts/tiger_memory.py code status --cwd /path/to/project
python3 scripts/tiger_memory.py code sync --cwd /path/to/project
```

Use `--with-code` for code-specific questions. The host may supply focused
symbols with `--code-query`; otherwise the original question is used. Ordinary
`context` output and the AML profile are unchanged. No keyword heuristic
silently routes personal-memory questions to a code index.

`code explore/query/callers/callees/impact/files` are query operations;
`code init/sync` are explicit mutations. Querying never creates or refreshes an
index. An existing graph requires an explicit tiger_memory sync to establish its binding.
The adapter disables telemetry, update checks, automatic binary downloads, and
the CodeGraph daemon in its child processes.

## Evidence and freshness

`.codegraph/plm-binding.json` records the canonical repository root, Git HEAD,
branch, dirty flag, working-tree content fingerprint, CLI version and index
time. It is a disposable local binding, outside the Event store. Fingerprinting
includes tracked files and non-ignored untracked files, including deletions;
it excludes `.codegraph/` and root Finder metadata. It scans at most 128 MiB;
large generated assets should be gitignored. It does not follow symlinks.

Before and after a query tiger_memory compares the working tree to the binding. A changed
tree yields `stale` with no code text until explicit sync. A missing or copied
binding yields `unbound`. A missing CLI, timeout, rejected secret-bearing output
or execution error yields `unavailable`. These states preserve ordinary memory
retrieval. Standalone unsuccessful code queries exit with status 2.

Returned code remains CodeGraph's source text and line references, with a
separate snapshot envelope. It is not an Event, user-confirmed fact, runtime
trace, or proof of a historical decision. A combined context allows the host to
compare memory evidence with current code; automatic semantic links between
decisions and symbols are not asserted. Derived graph edges are navigation
hints; verify dynamic/runtime behavior with the appropriate tests.

The code-text budget is independent of the memory budget: default 16,000 UTF-8
bytes, configurable with `context --code-budget` or `code --max-bytes` between
256 and 64,000. JSON metadata is additional. Truncation is marked explicitly.
Queries time out after 30 seconds (maximum 60); indexing after 180 seconds
(maximum 600). Command failures do not expose raw stderr.

tiger_memory also checks the actual CLI health report: initialized, correct repository,
complete extraction, no pending file changes, and a matching index revision.
A nonzero exit or interrupted/partial sync never creates a ready binding. Failed
sync invalidates the previous receipt until a successful retry.

## Verification

`python3 scripts/verify_install.py` builds a distribution and runs the installed
entry point in a disposable virtual environment with isolated HOME and memory,
outside the source checkout. Add `--with-codegraph` to also run the installed
combined-context path against the locally available CodeGraph CLI.

`python3 -m unittest tests.test_codegraph` exercises stale bindings, branch and
project isolation, degraded providers, incomplete indexes, output budgets and
unchanged ordinary memory results. `python3 scripts/verify_codegraph.py` runs
opt-in acceptance with the installed CLI in disposable Git repositories,
including all six query actions, a real renamed call edge after sync, branch
changes, same-name symbols in different projects and combined memory context.

### Public three-arm comparison

With CodeGraph already installed, run:

```bash
python3 scripts/evaluate_code_memory.py --budget-bytes 8000 \
  --output /tmp/plm-code-memory-evaluation.json
```

The public fixtures in `examples/code_memory/` include a previous decision and
a changed implementation, a non-code memory control, an outdated index and a
missing CLI. Each case runs memory-only, code-only and combined-context arms.
The exact UTF-8 byte cap applies to each exported `packet.reader_context`,
including its evidence labels. Other report fields and diagnostics are outside
that cap. It is a byte-controlled comparison, not tokenizer-controlled or a
complete measure of an Agent's total context/cost.

The report includes fixture identity, tool version, each actual context,
literal evidence coverage, freshness checks and local retrieval time. Literal
coverage means expected fixture text survived retrieval and packing. It does
not grade an answer or establish semantic support. `semantic_quality` remains
`not_measured`; no language model or paid service is invoked by this experiment.
Answer-quality evaluation with a fixed reader and independent coding tasks is
still required before claiming a measured product-quality improvement.

These local commands do not establish remote CI or release status. Preserve
the actual command output and environment details when reporting acceptance.

## Runtime boundary

This feature lives in the Python package and its `context`/`code` CLI. The
globally installed skill can contain a different `task-context` host workflow;
this change does not overwrite it. Use the independently installed `tiger_memory`
runtime or this checkout's `scripts/tiger_memory.py`. Skill installation, rollback and
cutover remain source/skill deployment operations; the standalone wheel does
not include a Codex skill bundle. CodeGraph
indexing is project-local and does not ingest code into global tiger_memory memory.
