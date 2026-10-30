# AML Search diagnostics candidate, 2026-10-20

This is an observability change, not a retrieval-quality fix. No official
evaluation, deployment, model call, or change to the ranked Search payload is
required to validate it.

## Always-on aggregate correction

The existing `outcomes.search_empty` now includes successful empty returns from
`top_k == 0`, missing storage, and explicit abstention. These also increment one
reason counter:

- `search_empty_top_k_zero`
- `search_empty_storage_missing`
- `search_empty_abstained`

Reason counters are subsets of `search_empty`, not additional requests.
Do not add them together when counting total requests. Normal empty retrieval
still increments only `search_empty`; legacy exceptions still increment
`search_failure`, even though their wire response is an empty HTTP 200.
HTTP auth, malformed JSON and transport failures are outside this accounting.
Existing historical counters must not be compared without this semantic change.

## Explicit, local recorder opt-in

`aml_adapter.AML_SEARCH_TRACE` defaults to `None`. No environment variable
silently enables it. A future separately versioned launcher can explicitly
assign a `aml_search_trace.SearchTrace(binding_sha256=..., capacity=64)`.

The binding must be a lowercase SHA-256 of a verified code/config manifest.
The constructor validates its format only: it does not independently prove
deployment or the manifest's truth. Validate files before assignment. A binding identifies source bytes; it is not deployment authorization.

When enabled, the existing authenticated `/status` adds `search_trace`.
There is no new public endpoint; `/status` still requires a nonempty configured
API key and authorized request. Search/Add/health payloads do not contain trace
metadata. Status snapshots and their callers remain responsible for access.

Each event contains:

- recorder instance ID and binding digest (at snapshot level), arrival sequence,
  wall-clock start and monotonic elapsed time, plus a per-request SHA-256 of
  `AML_RUNTIME_CONFIG` (sorted compact UTF-8 JSON, not the complete code/config
  binding and not proof that every global flag appears in that dictionary);
- fixed branch reason and whether the adapter returned;
- keyed digests of effective query, options, user and each returned result ID;
- each returned item's rank, content digest and UTF-8 byte length;
- a fixed `embedding_failure` category when context retrieval reports an
  unavailable embedding provider; unrecognized free-form reasons become
  `unclassified`, never copied into the event;
- allowlisted integer source-coverage diagnostics if the active retrieval
  implementation supplies them.

The effective query includes options when the existing adapter joins them.
No raw question, user ID, result ID, source body, exception message, model
payload, arbitrary diagnostic text or key is stored. Strict-profile sensitive
rejections and invalid schemas are recorded without request fingerprints.
These content/query/user/result-ID digests are domain-separated HMAC-SHA256 using a new private random key for each
recorder; the key is never exported. Cross-recorder comparisons and matching
against historical plain user hashes are intentionally unavailable.

### Optional catalog association (local proposal, not deployed)

`SearchTrace(..., storage_links=True)` adds a separate, explicit association
mechanism. The default is false and preserves the prior snapshot shape. This
proposal is **not** in the frozen `candidate92-integrated-1` release.

After identity preflight, `expected_project_token` records the UUID derived
from the expected storage root. It does not prove that storage exists. After
context retrieval returns, `retrieved_project_token` records that returned
project's UUID. Returned rows with canonical UUID v4/v5 IDs gain `source_token`.
Non-UUID IDs remain unlinked and are counted in `unlinked_result_ids`.

These tokens use SHA-256 with a public per-recorder salt (`instance_id`) and
separate project/source domains. They are catalog join keys, **not encryption,
anonymity guarantees or authentication proofs**. Someone holding the UUID and
snapshot can recompute them. They never accept query, user or content domains,
and are not substituted for the private HMAC fingerprints. A new recorder
changes the join tokens. No UUID, root path or source body is exported.

The read-only resolver can operate after the recorder process has exited:

```sh
python3 -I -B scripts/resolve_aml_search_trace.py \
  --snapshot /absolute/path/runtime_snapshot.json \
  --catalog /absolute/path/catalog.sqlite3
```

It accepts the trace itself or a saved `{"status":{"search_trace":...}}`
snapshot. It uses SQLite `mode=ro`, `query_only` and one read transaction; it
does not load application configuration, send requests or write the database.
It emits matched counts, positions in the **current retained** event order,
role/undated flags and body byte lengths. Raw IDs and bodies are not output.
Project conflicts, missing/ambiguous matches and non-UUID IDs stay explicit.
The resolver only supports event IDs: Facts, virtual hits and IDs embedded
inside context supplements are not resolved by this first association layer.
Eviction, incomplete diagnostics and result metadata truncation remain visible.

This can join a future recorded response to retained event metadata. It cannot
recover old HMAC-only traces, verify an old source body is unchanged, reconstruct
the historical visible prefix, associate F1 without official sample evidence,
or recover the official Answer/Judge inputs. It is not a quality acceptance test.

Storage is memory only, default 64 completed requests, configurable from 1 to
128. Each event stores at most 100 ordered result summaries and marks metadata
truncation if an altered adapter returns more. The total content-byte field
covers those recorded summaries. Old events are evicted, and eviction counts
are reported. A recorder's `clear()` removes retained events; process exit
removes the recorder and key. There is no disk sink, automatic export, TTL
purger, startup restoration or extra network/model dependency. Concurrent
completion order may differ from the arrival sequence.

## What this cannot establish

An adapter-completed event does not prove successful HTTP delivery. Wire
serialization, proxy buffering, tokenizer choice, official prompt composition,
downstream clipping, generated answers and Judge decisions remain outside this
trace. UTF-8 bytes are not model tokens.

Coverage counts are taken only when supplied, and do not prove semantic
sufficiency. The recorder does not parse embedded source-context JSON or invent
per-source omission reasons. Existing frozen patches expose only aggregate
selection counts; changes to those patches need separate versioning.

This cannot reconstruct prior Search responses, identify an official F1 item
without a legitimate mapping, or prove an official evaluation outcome.
Memory-only bounded retention can lose evidence; it is deliberately not an
unbounded historical archive.

## Local verification

`tests/test_aml_search_trace.py` exercises early returns, explicit abstention,
normal success/empty, strict and legacy failures, validation/sensitive-input
priority, unchanged response data, ordered fingerprints, privacy filtering,
concurrent isolation, bounded eviction, clear, and HTTP status authentication.
Inputs are local test fixtures. These are transport/diagnostic invariants, not
self-authored question-answering scores or official F1 acceptance.

`tests/test_aml_trace_lookup.py` validates local HTTP response transparency,
read-only catalog resolution and cross-project exclusion with synthetic inputs.
