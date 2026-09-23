# PLM v2 Architecture

## Trust boundary

Project files, Git commits, tests, and deployment evidence are the final truth.
PLM stores sourced working context and never upgrades an inference to a
high-risk current fact without strong evidence.

## Storage

`~/.codex/project-memory/v2/` contains:

- `projects/<display-name>--<project-id-prefix>/manifest.json`: stable project
  identity and path aliases.
- `projects/.../events/YYYY/MM/*.md`: immutable event and fact assertions.
- `candidates/*.json`: durable Agent-generated candidate queue.
- `catalog.sqlite3`: WAL catalog, current facts, FTS5, local vectors, entities,
  migration mappings, and shadow traces.
- `migration/`: checksum manifest and migration reports.
- `reports/acceptance.json`: the most recent cutover gate report.

Deleting the SQLite catalog does not delete memory. `rebuild-index` recreates
the catalog from manifests, Markdown events, candidate files, and the migration
checksum manifest. Assertions are replayed by their full timestamps, followed
by retractions, so rebuild preserves lifecycle state.

## Retrieval

Retrieval first enforces `project_id`, scope, and current/history status. It
then fuses FTS5 trigram/BM25 with a deterministic 512-dimensional hashed
character n-gram vector. Agent-supplied entity aliases expand known project
terms. Results always carry a source path and validity metadata.

The local vector is lexical and rebuildable; it is not presented as a neural
semantic embedding. The provider boundary can later accept a local embedding
model without changing the source record.

## Fact lifecycle

Facts are assertion events. A new version points to `supersedes`; the old fact
becomes `superseded` with a closed validity window but remains available in
history. Retraction is also an immutable event. Current retrieval excludes
superseded, expired, and retracted facts.

Event IDs are immutable. An identical retry is accepted; reusing an ID with
different content is rejected. Mutation commands resolve the current project
from `cwd` and cannot target another project's IDs.

`forget` records a retraction. `forget --purge` additionally removes the source
Event, Fact rows, search indexes, entity links, matching candidates, and any
legacy source note. Exports include active records only.
