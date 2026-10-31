# Search runtime source snapshot — 2026-10-31

This directory contains an isolated base package, runtime modules and a portable
entry point configured through environment variables. `MANIFEST.json` records
SHA-256 hashes of the public files. No deployment history, server configuration,
internal source paths or authorization records are included.

## Configuration and launch

Python 3.9 or newer and Git are required. No third-party Python package is needed
for this source snapshot. Inject these variables through deployment secret
management; do not commit their values:

- `TIGER_MEMORY_DIR`: an absolute canonical dedicated store outside every Git
  checkout, separate from daily Codex memory.
- `TIGER_MEMORY_EMBEDDING_ENDPOINT`: `https://www.dmxapi.cn/v1/embeddings`.
- `TIGER_MEMORY_EMBEDDING_API_KEY`: the embedding provider credential.
- `AML_API_KEY`: the service authentication credential.
- `TIGER_MEMORY_ALLOW_REMOTE`: `1`.

From the repository root:

```sh
python3 repro/runtime-20261031/entry.py --check-config
python3 repro/runtime-20261031/entry.py --check-startup
python3 repro/runtime-20261031/entry.py --host 127.0.0.1 --port 18191
```

The first two commands do not call the provider or start a listener. Normal
Add/Search cache misses call the provider and may incur charges. The runtime
has no monetary cap, budget receipt requirement, batch-count cap, elapsed-time
cap, five-second lock expiry, or provider socket timeout. Transient network
failures and HTTP 408/429/500/502/503/504 retry the same batch with backoff, without
a retry-count or total-duration quota. Normal backoff grows from one second to
30 seconds and honors Retry-After. Authentication errors, certificate failures,
invalid JSON, and invalid vectors still fail rather than retry indefinitely.
Only a complete validated embedding response can publish the cache and allow
successful Add completion.

Same-user locking, cache reuse, original-source first-write semantics,
authentication, user isolation, credential redaction, and model/vector validation
are preserved.
Reverse proxy settings are deployment configuration and are not reproduced by
this source launcher. The legacy base entry is not the repaired launch command.

This upload does not newly verify provider availability, account permissions,
production load, or official evaluation scores. No private store, credentials,
model cache, evaluation questions, answers, or deployment scripts are included.

## Changes and version binding

Runtime revision: `search-runtime-repair-20261031`. `SOURCE_IDENTITY.json`
binds the 15 unchanged release modules, 43 frozen base modules, and privacy
wrapper to their SHA-256 hashes. The portable launcher replaces private server
paths with environment configuration and preserves the deployed wrapper order
and SIGUSR1 handler. The older root package and runtime-20261030 remain historical
entries; use the launch commands above for this revision.

- Embedding cache uses ordered input stripe locks after explicit activation;
  repeated inputs share cached results and unrelated inputs can run concurrently.
- Recovery indexing supports 8192 sources / 16 MiB per project and 65536 sources /
  128 MiB process-wide serialized metadata. Exceeding this cache capacity falls
  back to a complete scan, without rejecting Add requests.
- HTTP failure diagnostics retain bounded allowlisted metadata without request
  text, response messages, credentials, or user paths. HTTP 400 is not retried.
- Operation-specific timing covers cache mode and input lock waits.

On startup the cache uses the legacy exclusive protocol. After all previous
writers and their in-flight requests have drained, the operator may send SIGUSR1
to this process to request the parallel transition. The transition waits for
legacy work to finish; restarting returns to exclusive mode. Do not activate
parallel mode while an older process can still write the cache.

## Synthetic verification

```sh
python3 repro/runtime-20261031/regression.py
```

Tests use temporary stores, synthetic vectors, and local sockets. Linux inotify
cases are skipped on other operating systems; see `VERIFICATION.json` for the
actual publication checks. These tests do not establish official AML scores or
current production availability. No private deployment logs or evaluation data
are included. The SQLite compatibility fixture uses runtime-20261030 from this
repository, so retain that directory when running the complete regression suite.
