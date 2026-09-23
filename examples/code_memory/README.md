# Code + memory evidence diagnostic

Run from a source checkout with Python 3, Git, and an explicitly installed
CodeGraph CLI (the real integration was developed against 1.5.0):

```sh
python3 scripts/evaluate_code_memory.py --budget-bytes 8000 --output /tmp/plm-code-memory-eval-report.json
```

This creates a disposable Git repository and a separate disposable tiger_memory memory
store, indexes `before.py`, changes it to `after.py`, and explicitly refreshes the
index. Neither the working repository's index nor the user's memory is changed.
No network or model calls are needed. Temporary files are removed on completion.

`cases.json` fixes two historical memories and four questions: a historical design
reason plus changed current code, a non-code memory control, a real stale-index
edit, and an injected missing-CLI failure. Every question runs memory-only,
code-only, and combined retrieval. All arms have the same final UTF-8 context
budget; combined divides it equally between memory and code. The report preserves
the exact `reader_context` delivered by each arm, along with evidence sources,
code readiness, context bytes, timing, and the fixture content hash.
The combined arm calls the actual `tiger_memory context --with-code` product entry point;
the other arms use `tiger_memory context` and the CodeGraph adapter respectively. Code
snapshot, revision, and readiness provenance is retained separately in every
record, with temporary root paths normalized.
The command exits nonzero if readiness differs from the fixture expectation,
unavailable/stale code is exposed, or a delivered context exceeds the byte budget.

`literal_evidence_coverage` checks explicit strings in the correct delivered
evidence channel. It is a diagnostic of evidence availability, **not answer
accuracy**, a competition score, or a claim of general coding improvement.
`semantic_quality` remains `not_measured`: this script generates no answers and
does not judge a reader. In the negative cases, missing current-code coverage is
expected; withholding stale or unavailable code is checked separately. One run
and a rotated arm order do not establish statistical latency improvements.

To assess answer quality later, give each saved question and `reader_context` to
the same fixed reader under the same settings, hide reference requirements from
that reader, and separately review correctness, unsupported statements, and
appropriate abstention. Preserve those answers and reader settings alongside this
report; do not relabel this script's string checks as semantic evaluation.

The default unit suite checks budgeting, scoring, isolation, and failure handling
without requiring CodeGraph:

```sh
python3 -m unittest tests.test_code_memory_evaluation
```
