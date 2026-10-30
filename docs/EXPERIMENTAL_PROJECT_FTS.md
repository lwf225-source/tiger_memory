# Experimental project FTS statistics

The Python `search.search` and `service.context_data` functions accept
`fts_statistics_scope="project"` as an experimental, explicit opt-in for the
`lexical` retrieval profile. The default is `"global"`. AML and CLI profiles do
not expose or enable this option. Unknown scope values and project statistics
with an unsupported profile are rejected. This experimental API may change.

Each lexical query channel copies all FTS rows belonging to the project into a
temporary in-memory FTS table to compute BM25 using that project's statistics.
The existing visibility predicates, candidate cutoff and ranking fusion remain
in use. Project statistics do not isolate different users sharing one project,
and pre-existing orphan FTS rows within a project still contribute statistics.

The default preflight limits are 100,000 rows and 64 MiB of UTF-8 payload across
the six copied columns. These are provisional bounds on copied source payload,
not SQLite index size, process RSS, latency or concurrent-search capacity.
Each lexical channel makes a fresh copy. Trigram index overhead is additional.
Before connecting this option to any profile, measure memory and latency near
the proposed limits and under expected concurrency. Those measurements have not
been completed.

Unsupported schemas, exceeded limits or copy/query failures raise explicit
errors; there is no fallback to global BM25. The helper releases its own read
snapshot before calling an external provider. An existing caller-owned
transaction is preserved, including uncommitted writes on errors; the caller
remains responsible for its duration, commit and rollback. No persistent cache
or source-file rewrite is introduced.
