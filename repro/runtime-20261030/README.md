# Portable runtime source snapshot

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
python3 repro/runtime-20261030/entry.py --check-config
python3 repro/runtime-20261030/entry.py --check-startup
python3 repro/runtime-20261030/entry.py --host 127.0.0.1 --port 18191
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

## Synthetic verification

```sh
python3 repro/runtime-20261030/test_transport.py
```

The five transport tests cover a real local HTTP response delayed beyond 31
seconds with only one send, timeout recovery and cache replay, HTTP 503/429
backoff, no retry on HTTP 401, and no retry on invalid response data. They use
synthetic embeddings and loopback HTTP only. The earlier revision's eight
reliability and six diagnostics test passes are not claimed for this revision;
obsolete copies of those tests are not included here. See `VERIFICATION.json`
for the checks actually performed on this snapshot. These are local regression
checks, not live provider tests or official AML evaluations.

## Add performance source update (2026-10-30)

This source revision adds an explicit non-null idempotency-key predicate so the
existing partial index can be selected. Recovery keeps a bounded process-local
cache of verified file metadata; it still enumerates and stats source files,
re-reads changed/new files, and re-reads recovered content before using it.
It does not cache negative lookups, directory membership, deletion decisions,
or transaction decisions. This preserves the recovery checks rather than
skipping them; an O(N) metadata pass remains.

The runtime revision is `add-performance-20261030`. Embedding model, retrieval
settings and transport retry policy retain the preceding snapshot configuration.
Local verification is recorded in `VERIFICATION.json`; synthetic measurements
do not establish production speedup or official evaluation quality.
