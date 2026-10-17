# Tiger memory — local candidate 92

**UNADOPTED / DECISION_PENDING.** This is a reproducible local engineering candidate for text-embedding-v4 with P3a transport. It is not a public deployment, accepted competition submission, official Smoke result, or quality score. The historical75 strict source-retention gate remains FAIL. Preparing this package does not select the candidate.

## Install and verify locally

Use a fresh environment outside all checkouts. Core runtime dependencies are Python standard library, SQLite with FTS5, and local Git executable. Distribution metadata retains Python >=3.9; this candidate's recorded actual installation environment is in the accompanying acceptance report. Linux/Windows and other Python versions are not certified by a pure-Python wheel tag.

```sh
python3 -m venv /absolute/disposable/venv
/absolute/disposable/venv/bin/python -I -B -m pip install --no-index --no-deps --no-compile /absolute/artifacts/tiger_memory-2.2.0a5+candidate92-py3-none-any.whl
/absolute/disposable/venv/bin/python -I -B -m tiger_memory.aml_reference --self-check
/absolute/disposable/venv/bin/python -I -B -m tiger_memory.aml_reference --rehearse
```

First verify artifact hashes and the source-manifest pin supplied separately by the publisher. A manifest stored alongside mutable files is consistency evidence, not an independent trust anchor. The installed self-check verifies all43 runtime files and the selected reference support documents; it reads no existing credential/receipt, starts no service, and performs only audit-guard negative controls. No workspace imports are needed.

`--rehearse` is deliberately synthetic: temporary home/store, test-only fake budget/receipt and loopback fake embeddings. It exercises HTTP Add/Search, cache reuse, two service-object restarts, UNKNOWN rejection/reuse and cleanup. It does not use actual credentials, pay a supplier, test production configure, SIGKILL recovery, ranking quality or official evaluation. Its temporary synthetic key is a fixture, not a deployable credential. The reference log wrapper applies only to this demonstration.

## Candidate identity

Distribution version: `2.2.0a5+candidate92`; runtime `__version__`: `2.2.0a5`. All43 runtime modules remain byte-identical to the accepted88v3/91 chain. The local distribution suffix distinguishes delivery without changing algorithm or runtime. `VERSION_MAP.json` and `SOURCE_MANIFEST.json` bind exact files. The original broad `tiger_memory.build_backend` remains an unchanged runtime file; this source distribution selects the separate `candidate_build` backend to enforce the smaller public allowlist. Do not invoke the original backend for this candidate.

P3a is the selected transmission path. P3b and deployment request deadline remain disabled; their dormant modules are retained solely to preserve the exact accepted runtime closure. Query rewriting, recency boost, reranker and fact projection remain off. No model weights, SDK, dependency wheel, real memory, DEV body/label/vector/cache, historical budget/receipt, private configuration or experiment output is shipped. Method description is in `docs/METHOD.md`; actual shipped-source and external-component attribution is in `THIRD_PARTY.md` and `NOTICE`.

## Real service prerequisites — not executed by this package

The operator must first resolve candidate selection, target OS/Python/resources, TLS/proxy/auth/limits, persistent storage, public fixed version and a lawful budget/receipt for future official data. The historical DEV STOP ledger is exhausted and cannot fund new work; copying/resetting it or removing UNKNOWN does not grant spending authority. This package creates no production budget, provides no real receipt and supplies no credential.

Use `repro/aml-open-source.env.example` as a blank template in a private external file. Keep the supplier key separate from the service bearer key. Select a dedicated canonical absolute store outside Git and daily-memory trees. Only after existing authorization and external prerequisites are satisfied:

```sh
python -I -B -m tiger_memory.aml_open_source --check-config
python -I -B -m tiger_memory.aml_open_source --check-startup
python -I -B -m tiger_memory.aml_open_source --host 127.0.0.1 --port 18191
```

These commands use operator-supplied process environment. Neither check sends supplier requests or starts a listener, but `configuration()` opens/validates an existing receipt and may perform runtime read/write transactions on its new target ledger. They are not a promise of whole-chain read-only behavior. Never run them against an unauthorized or historical ledger. Missing/invalid receipt fails closed; startup does not create credit. Provider timeout30 seconds covers socket operations;1200 seconds covers elapsed checks at batch boundaries, not an end-to-end HTTP deadline. Proxy timeout does not prove background termination.

The production CLI does not expose the reference log wrapper or a request-timeout/max-connections CLI setting. Target process supervision, concurrency limits, redacted durable logs and restoration procedures are external deployment work. `/health` is liveness, not proof of supplier/cache/budget readiness. See `docs/START_STOP_RECOVERY.md`.

## Build and publication boundary

From this extracted source directory, with a clean Python and no network:

```sh
python -B -c 'import candidate_build; print(candidate_build.build_sdist("../dist")); print(candidate_build.build_wheel("../dist"))'
```

The backend validates every source-manifest hash and packs only named files. An sdist rebuild must produce the exact same wheel bytes in the recorded toolchain. New unlisted files are never automatically included; export/scan tooling additionally rejects unexpected staging files and synthetic exclusion canaries. Original working tree and historical evidence are not modified.

No public commit has been created. `docs/PUBLICATION.md` describes the prepared content mapping and history boundary. A local archive is not the competition-required self-hosted API plus public fixed commit. Public push, deployment, application and official Smoke/Full are separate actions, not granted by this package. No Docker, cross-platform, full workload capacity,95-point readiness or top10 claim is made.
