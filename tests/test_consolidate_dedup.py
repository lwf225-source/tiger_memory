"""Consolidate-time near-duplicate detection tests (batch 9): opt-in ``dedup_similarity``.

Modelled on LLLMemoryAgent (AML #9: write-time cosine > 0.85 skips the write),
but PLM is an immutable-event governance system, so the DEFAULT action is the
conservative one: the candidate is marked ``needs_confirmation`` with the
suspected duplicate fact_id and a human decides via candidate-review.
``skip`` (keep existing, do not write) and ``supersede`` (new value replaces
the match, same fact_key only — the existing supersede validation is reused)
are explicit opt-ins. The switch is OFF by default and byte-identical to
previous behavior; the high-risk evidence gate runs before dedup and is never
bypassed by it.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

from tiger_memory.cli import main as cli_main
from tiger_memory.database import connect
from tiger_memory.model import PLMError
from tiger_memory.service import (
    DEDUP_SIMILARITY_THRESHOLD, _dedup_probe_text, consolidate, fact_history,
    review_candidate, submit_candidate, write_fact,
)
from tiger_memory.vector import cosine_blob, encode_vector
from tests.common import MemoryTestCase


class ConsolidateDedupTests(MemoryTestCase):
    KEY = "cache.engine"
    VALUE = "缓存引擎使用 SQLite，单机 WAL 模式，暂不考虑替换"
    DUP_KEY = "storage.engine"
    DUP_VALUE = "缓存引擎使用 SQLite，单机 WAL 模式，暂不考虑替换。"  # lexical cosine ≈ 0.8539

    def _existing(self):
        return write_fact(self.project, self.KEY, self.VALUE)

    def _submit(self, facts, key, **extra):
        payload = {"summary": "", "facts": facts}
        payload.update(extra)
        return submit_candidate(self.project, payload, idempotency_key=key)

    def _dup_fact(self, **overrides):
        fact = {"fact_key": self.DUP_KEY, "value": self.DUP_VALUE}
        fact.update(overrides)
        return fact

    def _lexical_similarity(self, key_a, value_a, key_b, value_b):
        return cosine_blob(
            encode_vector(_dedup_probe_text(key_a, value_a)),
            encode_vector(_dedup_probe_text(key_b, value_b)),
        )

    def _fact_rows(self):
        conn = connect()
        try:
            return [dict(row) for row in conn.execute(
                "SELECT fact_id,fact_key,status,supersedes FROM facts ORDER BY recorded_at,fact_id")]
        finally:
            conn.close()

    # -- default freeze -----------------------------------------------------

    def test_disabled_matches_default_exactly(self):
        self._existing()
        first = self._submit([self._dup_fact()], "off-a")
        default = consolidate()
        second = self._submit([self._dup_fact(fact_key="storage.engine.b")], "off-b")
        explicit_off = consolidate(20, dedup_similarity=0.0, dedup_action="skip")
        self.assertEqual(default["results"][0]["state"], explicit_off["results"][0]["state"])
        self.assertEqual(default["results"][0]["reason"], explicit_off["results"][0]["reason"])
        for result in (default["results"][0], explicit_off["results"][0]):
            self.assertEqual(set(result), {"candidate_id", "state", "reason"})
        self.assertEqual(default["states"], explicit_off["states"])
        self.assertEqual(default["processed"], explicit_off["processed"])
        self.assertEqual(
            [row["fact_key"] for row in self._fact_rows()],
            [self.KEY, self.DUP_KEY, "storage.engine.b"],
        )

    def test_invalid_parameters(self):
        self._existing()
        self._submit([self._dup_fact()], "bad-params")
        for bad in (1.5, -0.1, 2.0):
            with self.assertRaises(PLMError, msg=repr(bad)):
                consolidate(1, dedup_similarity=bad)
        with self.assertRaises(PLMError):
            consolidate(1, dedup_similarity=0.5, dedup_action="merge")
        # Switch off tolerates an irrelevant action value (frozen default path).
        consolidate(1, dedup_similarity=0.0, dedup_action="merge")

    # -- (a) review: the conservative default --------------------------------

    def test_review_marks_needs_confirmation_and_writes_nothing(self):
        existing = self._existing()
        candidate = self._submit([self._dup_fact()], "review-1")
        result = consolidate(20, dedup_similarity=DEDUP_SIMILARITY_THRESHOLD)
        entry = result["results"][0]
        self.assertEqual(entry["state"], "needs_confirmation")
        self.assertIn("near-duplicate", entry["reason"])
        self.assertIn(existing.metadata["fact_id"], entry["reason"])
        match = entry["dedup"]["matches"][0]
        self.assertEqual(match["fact_id"], existing.metadata["fact_id"])
        self.assertEqual(match["fact_key"], self.KEY)
        self.assertGreaterEqual(match["similarity"], DEDUP_SIMILARITY_THRESHOLD)
        self.assertEqual(entry["dedup"]["action"], "review")
        self.assertEqual(fact_history(self.project, self.DUP_KEY), [])
        conn = connect()
        state = conn.execute("SELECT state FROM candidates WHERE candidate_id=?", (candidate["candidate_id"],)).fetchone()[0]
        conn.close()
        self.assertEqual(state, "needs_confirmation")

    def test_review_then_approve_materializes_normally(self):
        self._existing()
        candidate = self._submit([self._dup_fact()], "review-approve")
        consolidate(20, dedup_similarity=0.85)
        approved = review_candidate(
            self.project, candidate["candidate_id"], "approve",
            "user_confirmation", self.confirmation(self.DUP_KEY, self.DUP_VALUE),
        )
        # Approval re-runs consolidate without dedup: a human already decided.
        self.assertEqual(approved["states"], {"active": 1})
        history = fact_history(self.project, self.DUP_KEY)
        self.assertEqual([row["status"] for row in history], ["active"])

    def test_below_threshold_writes_normally(self):
        self._existing()
        self._submit([{"fact_key": "snack.preference", "value": "团队下午茶偏好低糖点心"}], "below")
        result = consolidate(20, dedup_similarity=0.85)
        entry = result["results"][0]
        self.assertEqual(entry["state"], "active")
        self.assertEqual(entry["dedup"]["matches"], [])

    def test_threshold_boundary_is_inclusive(self):
        self._existing()
        similarity = self._lexical_similarity(self.DUP_KEY, self.DUP_VALUE, self.KEY, self.VALUE)
        self._submit([self._dup_fact()], "boundary-at")
        flagged = consolidate(20, dedup_similarity=similarity)
        self.assertEqual(flagged["results"][0]["state"], "needs_confirmation")
        self._submit([self._dup_fact()], "boundary-above")
        above = consolidate(20, dedup_similarity=min(1.0, similarity + 1e-6))
        self.assertEqual(above["results"][0]["state"], "active")
        self.assertEqual(above["results"][0]["dedup"]["matches"], [])

    def test_cross_scope_never_flagged(self):
        write_fact(self.project, self.KEY, self.VALUE, scope="session", scope_id="user-1")
        self._submit([self._dup_fact()], "scope-project")  # project scope candidate
        result = consolidate(20, dedup_similarity=0.85)
        self.assertEqual(result["results"][0]["state"], "active")
        self.assertEqual(result["results"][0]["dedup"]["matches"], [])
        self._submit([self._dup_fact(scope="session", scope_id="user-1")], "scope-session")
        flagged = consolidate(20, dedup_similarity=0.85)
        self.assertEqual(flagged["results"][0]["state"], "needs_confirmation")

    # -- (b) skip -------------------------------------------------------------

    def test_skip_keeps_existing_and_writes_the_rest(self):
        existing = self._existing()
        self._submit(
            [self._dup_fact(), {"fact_key": "snack.preference", "value": "团队下午茶偏好低糖点心"}],
            "skip-partial",
        )
        result = consolidate(20, dedup_similarity=0.85, dedup_action="skip")
        entry = result["results"][0]
        self.assertEqual(entry["state"], "active")
        self.assertIn("1 near-duplicate fact(s) skipped", entry["reason"])
        self.assertEqual(entry["dedup"]["matches"][0]["fact_id"], existing.metadata["fact_id"])
        self.assertEqual(fact_history(self.project, self.DUP_KEY), [])
        self.assertEqual([row["status"] for row in fact_history(self.project, "snack.preference")], ["active"])
        self.assertEqual(len(self._fact_rows()), 2)

    def test_skip_all_facts_returns_existing_fact_id(self):
        existing = self._existing()
        self._submit([self._dup_fact()], "skip-all")
        result = consolidate(20, dedup_similarity=0.85, dedup_action="skip")
        entry = result["results"][0]
        self.assertEqual(entry["state"], "active")
        self.assertEqual(entry["dedup"]["matches"][0]["fact_id"], existing.metadata["fact_id"])
        self.assertEqual([row["fact_key"] for row in self._fact_rows()], [self.KEY])

    # -- (c) supersede ----------------------------------------------------------

    def test_supersede_same_key_replaces_value(self):
        existing = self._existing()
        self._submit([{"fact_key": self.KEY, "value": self.VALUE}], "supersede-same")
        result = consolidate(20, dedup_similarity=0.85, dedup_action="supersede")
        entry = result["results"][0]
        self.assertEqual(entry["state"], "active")
        self.assertIn("1 near-duplicate fact(s) auto-superseded", entry["reason"])
        history = fact_history(self.project, self.KEY)
        self.assertEqual([row["status"] for row in history], ["superseded", "active"])
        self.assertEqual(history[1]["supersedes"], existing.metadata["fact_id"])

    def test_supersede_different_key_falls_back_to_review(self):
        self._existing()
        self._submit([self._dup_fact()], "supersede-diff")
        result = consolidate(20, dedup_similarity=0.85, dedup_action="supersede")
        entry = result["results"][0]
        self.assertEqual(entry["state"], "needs_confirmation")
        self.assertIn("same key", entry["reason"])
        self.assertEqual(fact_history(self.project, self.DUP_KEY), [])

    def test_explicit_supersedes_fact_is_not_scanned(self):
        existing = self._existing()
        self._submit(
            [{"fact_key": self.KEY, "value": "缓存引擎切换为 NextCache，单机模式保留",
              "supersedes": existing.metadata["fact_id"]}],
            "explicit-supersede",
        )
        result = consolidate(20, dedup_similarity=0.5, dedup_action="review")
        self.assertEqual(result["results"][0]["state"], "active")
        self.assertEqual(result["results"][0]["dedup"]["matches"], [])
        history = fact_history(self.project, self.KEY)
        self.assertEqual([row["status"] for row in history], ["superseded", "active"])

    # -- gates and providers -----------------------------------------------------

    def test_high_risk_evidence_gate_runs_before_dedup(self):
        write_fact(
            self.project, "release.status", "已发布",
            evidence_kind="user_confirmation", evidence=self.confirmation("release.status", "已发布"),
        )
        # High-risk near-duplicate WITHOUT claim-bound evidence: even in skip
        # mode the candidate must not be silently skipped/merged.
        self._submit([{"fact_key": "publish.status", "value": "已发布。"}], "high-risk")
        result = consolidate(20, dedup_similarity=0.5, dedup_action="skip")
        entry = result["results"][0]
        self.assertEqual(entry["state"], "needs_confirmation")
        self.assertIn("high-risk", entry["reason"])
        self.assertEqual(entry["dedup"]["matches"], [])
        self.assertEqual(fact_history(self.project, "publish.status"), [])

    def test_embedding_provider_overrides_lexical_similarity(self):
        self._existing()

        class IdenticalProvider:
            def embed(self, texts):
                return [[1.0, 0.0] for _ in texts]

        # Lexically unrelated (cosine 0.0) but the provider says identical.
        self._submit([{"fact_key": "snack.preference", "value": "团队下午茶偏好低糖点心"}], "embed-hit")
        flagged = consolidate(20, dedup_similarity=0.9, embedding_provider=IdenticalProvider())
        self.assertEqual(flagged["results"][0]["state"], "needs_confirmation")
        self.assertEqual(flagged["results"][0]["dedup"]["matches"][0]["similarity"], 1.0)

        class OrthogonalProvider:
            def embed(self, texts):
                return [[1.0, 0.0]] + [[0.0, 1.0] for _ in texts[1:]]

        # A verbatim duplicate that the provider scores as orthogonal passes.
        self._submit([self._dup_fact()], "embed-miss")
        passed = consolidate(20, dedup_similarity=0.9, embedding_provider=OrthogonalProvider())
        self.assertEqual(passed["results"][0]["state"], "active")
        self.assertEqual(passed["results"][0]["dedup"]["matches"], [])

    # -- CLI ------------------------------------------------------------------

    def test_cli_end_to_end(self):
        existing = self._existing()
        self._submit([self._dup_fact()], "cli-dedup")
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            cli_main(["consolidate", "--dedup-similarity", "0.85"])
        payload = json.loads(buffer.getvalue())
        entry = payload["results"][0]
        self.assertEqual(entry["state"], "needs_confirmation")
        self.assertEqual(entry["dedup"]["matches"][0]["fact_id"], existing.metadata["fact_id"])
        # Default CLI invocation carries no dedup instrumentation.
        self._submit([self._dup_fact(fact_key="storage.engine.c")], "cli-default")
        buffer2 = io.StringIO()
        with redirect_stdout(buffer2):
            cli_main(["consolidate"])
        entry2 = json.loads(buffer2.getvalue())["results"][0]
        self.assertEqual(entry2["state"], "active")
        self.assertNotIn("dedup", entry2)
