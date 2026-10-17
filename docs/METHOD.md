# Method and evidence limits

Candidate: text-embedding-v4,1024 dimensions, user-isolated checkpoints, UTF8 chunks up to8192 bytes, batches of10 rows, pooled mean/L2 aggregation. Strict retrieval combines lexical evidence with cached/provider vectors and session RRF; query rewrite, recency, reranker and fact projection are off. Current HTTP contract accepts optional top-level options; this differs from historical75's raw evaluator. Return data is source evidence, not generated answers. Production has no fallback on provider failure.

Add serializes first-write admission for a store/user, validates capacity before source writes, releases admission before embedding, and relies on final provider checks and durable per-key budget claims. UNKNOWN is conservative occupied budget, not automatically refunded. Valid cached work is reusable; missing data for historical claimed work blocks implicit paid retry. Search preserves its SQLite read snapshot through retrieval, which can include embedding latency. Socket30s and elapsed1200s batch checks are not hard HTTP limits.

FlowGrid, ReFind and Chronicle are methodological influences credited by the original project description. This candidate does not claim those ideas as new. It distributes the project's own Python integration; no third-party implementation source or model weights are bundled.

Historical DEV summary, not QA accuracy or blind validation: Any384→416/449; All309→352/449; required sources469→538/675;74 gains and5 losses. The old strict zero-lost-source adoption gate FAIL is permanent. Candidate choice is still DECISION_PENDING, not changed by engineering verification. None is a research baseline, not a newly implemented deployment mode.

Evidence inherited only for unchanged frozen components:715 regression checks on their recorded freeze;89 fixed pooled-dependency kernel comparison;91 scoped L1/L2/path-wrapper and finite L3 acceptance (three200 responses, ten rows each; one miss503). None of those is official Smoke/Full, unseen-data performance or an E2E capacity/SLO claim. This package's fresh install/self-check/fake rehearsal is separately reported; no449/1774 rerun is part of packaging.
