# tiger_memory

Local-first project memory for agent workflows. Immutable Markdown events are
source records; SQLite, FTS5, facts and retrieval indexes are rebuildable.
Current source version: **2.2.0a5 (preview)**. The core requires Python 3.9+ and
has no third-party Python runtime dependencies.

## Explicit Active Memory component

[Active Memory](docs/ACTIVE_MEMORY.md) provides a local Python API and CLI for
explicitly important preferences, goals and project context. Applications can
retrieve current memory before a task with `run_task`; corrections and logical
forgetting preserve source lineage. This component is not automatically connected
to an external assistant or HTTP service. See the guide for storage isolation,
synthetic verification and the limits of logical forgetting.

## AML open-source track

**Current source snapshot:** the development package is **2.2.0a5 (preview)**.
A self-contained runtime snapshot is available in
[`repro/runtime-20261031`](repro/runtime-20261031/README.md), with isolated base
modules, an environment-based entry point and synthetic regression tests.
The root `python -m tiger_memory.aml_open_source` entry retains its earlier
budget-bound candidate behavior; the separate runtime uses a different policy.

The separate runtime uses `text-embedding-v4` at 1024 dimensions, durable
per-user caches, credential redaction and bounded diagnostic output. Transient
network/provider failures retry with backoff; authentication, TLS and invalid
response failures remain errors. Model calls may incur provider charges.
The latest runtime revision is `search-runtime-repair-20261031`; see its
[source identity](repro/runtime-20261031/SOURCE_IDENTITY.json) and
[verification](repro/runtime-20261031/VERIFICATION.json).

Keep the AML store **outside every Git checkout**, separate from the source tree
and your daily memory. Before configuring credentials or starting a listener,
check a proposed path without creating it or contacting an API:

```sh
python3 -m tiger_memory.aml_store_preflight \
  --store "$HOME/.local/share/tiger-memory-aml-v4"
```

Git is required for this local topology check. Exit 0 confirms only the checked
directory/marker topology; it does not confirm writability, model access or
deployment. See the [local launch and delivery checklist](docs/OPEN_SOURCE_PREP.md#本地启动与交付清单)
for configuration, launch and remaining external evidence.

This repository contains the source and reproduction materials for
**tiger_memory** (formerly Project Long Memory / PLM). See
[AML reproduction and submission notes](docs/AML_SUBMISSION.md) for the fixed
source baseline, Add/Search API, model disclosure, pinned optional dependencies,
and the changes included in this upload. The repository is licensed under
[Apache-2.0](LICENSE).

For the historical local BGE environment, use
[`repro/linux-cpu-requirements.lock`](repro/linux-cpu-requirements.lock) and the
installation order in the AML notes. The macOS model lock is a separate
measured environment.

The public repository supports review and reproduction. An externally reachable
Add/Search service must be deployed separately for participation; this source
upload does not establish that such a service is online or accepted by AML.
Add/Search endpoints require a configured service key. Local contract checks
do not establish official evaluation results.

## Standalone installation

From a complete source checkout, with Python and pip available:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --no-index --no-deps .
export TIGER_MEMORY_DIR="$HOME/.local/share/tiger_memory"
.venv/bin/tiger_memory --help
.venv/bin/tiger_memory write --cwd /path/to/project --title 'Cache decision' \
  --content 'Use a bounded cache to control memory usage.'
.venv/bin/tiger_memory context --cwd /path/to/project --query 'cache decision' --json
.venv/bin/tiger_memory doctor
```

Replace `/path/to/project` with your project directory. On Windows, use the
virtual environment's `Scripts` directory instead of `bin`. No Codex account,
MCP configuration, model download or API key is required. Source invocations
use `python3 scripts/tiger_memory.py` instead of the installed `tiger_memory` executable.

Existing stores remain at their current paths. `TIGER_MEMORY_DIR` takes
precedence when set; the former `PROJECT_LONG_MEMORY_DIR` setting is accepted
as a compatibility fallback. The default Codex store remains
`~/.codex/project-memory` so a name change does not create an empty store.

Both wheel and source archive supply the runtime, offline tests, public
verification tools, skill resources and synthetic examples. The wheel stores
support files inside the package; verification uses a temporary source layout
containing the installed runtime bytes.
Building or installing either does not upgrade an existing globally installed
Codex skill.

## Verification and migration

From the source checkout:

```sh
python3 -m unittest discover -s tests -v
python3 scripts/verify_install.py
python3 scripts/verify_release.py --output output/release-verification
```

The installation verifier builds a source archive and installs it offline into
an isolated temporary virtual environment. It requires Git and Python's `venv`
module; it exercises the installed console entry point, memory write/retrieval,
and graceful operation without CodeGraph. It does not modify global skills or
the user's memory store. A successful local run is not proof of a remote release.
The release verifier additionally checks a byte-identical wheel rebuild from
sdist, both verify exit paths and validated temporary Skill installation from
the wheel and source archive. It runs the suite several times and takes a few
minutes. Bundled tests use local fixtures (including historical project-document
snapshots) and local HTTP stubs; no paid model or external service is required.

`tiger_memory verify` is the migration/cutover acceptance gate, **not a simple installation
smoke test**. It runs the bundled offline tests and checks store health, legacy
migration evidence and shadow-comparison samples. A fresh store with no legacy
samples must report that migration was not tested; it cannot prove a migration
passed. Exit codes: `0` means acceptance passed, `1` means acceptance criteria
were not met, and `2` means an execution error. Inspect the JSON checks and test
summary to distinguish the cause. Do not use the hidden `--tests-passed` option
to bypass validation.

For an existing v1 store, inspect `migrate --dry-run` before `migrate`, collect
actual shadow comparisons, and run `verify` before an explicit `cutover`.
`cutover` changes the compatibility workflow's active engine and write mode;
normal standalone `write` and `context` usage does not require migration.

## Optional Codex skill

Skill installation works from the complete source checkout or an installed
wheel (`tiger_memory install --target ...`). Both validate the copied bundle using the
active Python interpreter before replacing the chosen target.

```sh
python3 scripts/tiger_memory.py install --target /path/to/codex/skills/tiger_memory
```

Choose the target explicitly. Installation validates the copied bundle, backs
up an existing target and sets the compatibility configuration to shadow mode
with v1 writes. It is a deployment operation, not part of the standalone quick
start. Use `rollback --help` to inspect the rollback interface. The skill's
instructions are in [skill/SKILL.md](skill/SKILL.md).

## Retrieval and evidence

The default retrieval profile is lexical. `context --retrieval-profile passages`
returns bounded source passages; `evidence --help` describes verified expansion.
Use `history`, `supersede` and `forget` for the fact lifecycle. Backup restoration
requires an explicit latest deletion ledger and a separate empty destination.
See each command's `--help` for its arguments.

`--abstain-threshold` is opt-in and defaults to zero. A score or a retrieved
passage is not answer accuracy, and an empty result means insufficient retrieved
evidence, not proof that something never happened. Optional retrieval switches
need evaluation against both answerable and unanswerable cases before choosing
new defaults.

Code evidence is optional and separate from historical memory. Read
[CodeGraph integration](docs/CODEGRAPH.md) for explicit index management and
[the synthetic three-arm diagnostic](examples/code_memory/README.md) for its
measurement limits. External attribution is in [THIRD_PARTY.md](THIRD_PARTY.md).
