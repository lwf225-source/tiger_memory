# Deployment status — 2026-10-18 (UTC+8)

This dated supplement records the current operator decision and deployment state. The original candidate92 source documents and manifests are preserved byte-for-byte as historical packaging evidence; their UNADOPTED, DECISION_PENDING, not-deployed, and unpublished labels describe the state when that export was frozen. They do not supersede the later facts below. No historical quality gate is changed by this supplement.

## Selected competition version

The operator has selected the latest available Tiger memory implementation for the open-source competition submission. The selected distribution is **2.2.0a5+candidate92**, runtime version **2.2.0a5**, with **text-embedding-v4 / 1024 dimensions / P3a**. Selection for submission is an operational choice; it is not proof of independent quality validation or official eligibility. The 43 runtime modules match both the frozen candidate92 export and the current development workspace byte-for-byte as checked on this date. There is no newer runtime module set in that workspace.

- Wheel: `tiger_memory-2.2.0a5+candidate92-py3-none-any.whl`
- Wheel SHA-256: `fb9572c322b6112a4f85f2e4cc0fa0eb48c437a65c9ecf30d067774714391f44`
- Original source-manifest SHA-256: `43e9dc908b3366e2d2d52044055cfceceabc6ddc2b0daef27f09d90f4cd4214d`
- Original sdist SHA-256: `3204520403ff09552b824bc50d807685f67e193e4781040892b09f82835f7aa2`
- Original runtime-manifest SHA-256: `de616935f16f21cbe636ff62f8c0d49f763a6698c91abbd474edba99d3e35900`

The original SOURCE_MANIFEST.json binds its original 61 files; it intentionally does not cover this later supplement or itself. The original wheel/sdist are unchanged. The public commit containing this supplement should be used as the fixed source version when registering the service; before that commit is actually published, publication remains pending. Do not use the development workspace's older HEAD as a substitute.

## Deployed service and observed checks

Current service URLs:

- Health: https://t.aigcfans.net/health
- Add: https://t.aigcfans.net/add
- Search: https://t.aigcfans.net/search
- Authenticated status: https://t.aigcfans.net/status

The installed distribution is 2.2.0a5+candidate92 in an isolated virtual environment on Ubuntu 26.04 LTS / Python 3.14.4. The target installation evidence binds the wheel, source manifest, and all 43 runtime files. Isolated synthetic verification covered SQLite FTS5/WAL, read-only snapshots, HTTP contracts, and process restart persistence. The service was observed active and enabled for startup.

The 2026-10-18 real-provider acceptance used **5 embedding requests**, all settled, with **0 unknown**. It passed authenticated profile checking, two real writes, searches and per-user isolation, unknown-user empty results, idempotent replay, and persisted search after a service restart. Unauthenticated status/add/search returned 401, and HTTPS certificate verification succeeded with normal system trust. An immediate restart probe ran before readiness and failed; subsequent readiness and persisted-search checks passed.

The application ledger recorded CNY 0.0000335 for this bounded acceptance, under its CNY 1 / 20-request limit. This is application accounting, not an independently reconciled supplier invoice, and this acceptance allowance is not a declaration of budget or capacity for official evaluation. Credentials, service tokens, receipts, and operational ledger contents are not published.

## Remaining competition gates and limitations

**Official competition Smoke and Full have not been run as of this supplement.** There is no official score, ranking, or accepted leaderboard result. Platform registration, the actual fixed public commit binding, available evaluation quota, and official workload/budget admission must be recorded separately. Service deployment and local/remote engineering checks do not imply those steps are complete.

The historical75 strict source-retention gate remains **FAIL**. Operationally selecting this candidate does not relabel that gate, establish non-inferiority, or replace independent answer-level validation. P3b and the deployment request deadline remain disabled. No claim is made about full workload throughput, maximum user size, hard HTTP latency, production saturation behavior, or competition eligibility; these require the corresponding evidence.

The original source/wheel exclude real memories, DEV or sealed evaluation payloads and labels, embedding caches, private environment files, credentials, receipts, historical budget data, and experiment outputs. This supplement adds only public version identifiers, service endpoints, and aggregate verification facts. Method, provenance, and reproduction instructions remain in the preserved source documents.
