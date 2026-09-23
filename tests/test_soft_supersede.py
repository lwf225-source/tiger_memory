"""Guarded soft supersession tests (batch 8): opt-in ``soft_supersede``.

Modelled on FlowGrid (AML #8): with the switch ON, a Fact version whose
validity window was closed by a successor stays in the candidate set with a
fixed ranking demotion (``SOFT_SUPERSEDE_PENALTY``) instead of being excluded.
The guard: ranking-layer only — Fact lifecycle, forget/privacy semantics and
retraction are untouched; retracted or expired memories never come back.
The switch is OFF by default and must be byte-identical to previous behavior.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

from tiger_memory.cli import main as cli_main
from tiger_memory.database import connect
from tiger_memory.search import SOFT_SUPERSEDE_PENALTY, search
from tiger_memory.service import fact_history, forget, write_event, write_fact
from tests.common import MemoryTestCase


class SoftSupersedeTests(MemoryTestCase):
    def _project_id(self):
        conn = connect()
        try:
            return conn.execute("SELECT project_id FROM projects").fetchone()[0]
        finally:
            conn.close()

    def _search(self, query, **kwargs):
        conn = connect()
        try:
            return search(conn, self._project_id(), query, **kwargs)
        finally:
            conn.close()

    def _versions(self):
        """old: valid from 2026-01-01, window closed at 2026-03-01 by new."""
        old = write_fact(self.project, "cache.engine", "SQLite", valid_from="2026-01-01T00:00:00Z")
        new = write_fact(
            self.project, "cache.engine", "NextCache",
            supersedes=old.metadata["fact_id"], valid_from="2026-03-01T00:00:00Z",
        )
        return old, new

    def test_disabled_matches_default_exactly(self):
        self._versions()
        default = self._search("cache.engine", limit=10)
        disabled = self._search("cache.engine", soft_supersede=False, limit=10)
        self.assertEqual([item.as_dict() for item in default], [item.as_dict() for item in disabled])
        info = {}
        self._search("cache.engine", diagnostics=info)
        self.assertEqual(info["soft_supersede"], {"status": "disabled"})

    def test_invalid_flag_rejected(self):
        self._versions()
        for bad in (1, 0, "true", None):
            with self.assertRaises(ValueError, msg=repr(bad)):
                self._search("cache.engine", soft_supersede=bad)

    def test_current_view_recalls_old_version_with_demotion(self):
        old, new = self._versions()
        info = {}
        results = self._search("cache.engine", soft_supersede=True, limit=10, diagnostics=info)
        by_id = {item.ref_id: item for item in results}
        self.assertIn(new.metadata["fact_id"], by_id)
        self.assertIn(old.metadata["fact_id"], by_id)
        self.assertGreater(by_id[new.metadata["fact_id"]].score, by_id[old.metadata["fact_id"]].score)
        self.assertLess(results.index(by_id[new.metadata["fact_id"]]), results.index(by_id[old.metadata["fact_id"]]))
        self.assertIn("soft-supersede:-0.010000", by_id[old.metadata["fact_id"]].reasons)
        self.assertNotIn("effective-current", by_id[old.metadata["fact_id"]].reasons)
        self.assertIn("effective-current", by_id[new.metadata["fact_id"]].reasons)
        self.assertEqual(info["soft_supersede"]["status"], "enabled")
        self.assertEqual(info["soft_supersede"]["penalty"], SOFT_SUPERSEDE_PENALTY)
        self.assertEqual(info["soft_supersede"]["penalized"], 1)

    def test_penalty_magnitude_exact(self):
        old, _new = self._versions()
        off = {item.ref_id: item for item in self._search("cache.engine", view="history", limit=10)}
        on = {item.ref_id: item for item in self._search("cache.engine", view="history", soft_supersede=True, limit=10)}
        self.assertAlmostEqual(
            off[old.metadata["fact_id"]].score - on[old.metadata["fact_id"]].score,
            SOFT_SUPERSEDE_PENALTY, places=8,
        )

    def test_penalty_discipline_bounds(self):
        # Above the recency tie-breaker cap (visible demotion), below one
        # top-channel RRF step and the title bonus (cannot invent relevance).
        self.assertGreater(SOFT_SUPERSEDE_PENALTY, 0.004)
        self.assertLess(SOFT_SUPERSEDE_PENALTY, 1.0 / 61.0)
        self.assertLess(SOFT_SUPERSEDE_PENALTY, 0.03)

    def test_retracted_never_comes_back(self):
        old, new = self._versions()
        forget(self.project, old.event_id)
        ids = [item.ref_id for item in self._search("cache.engine", soft_supersede=True, limit=10)]
        self.assertNotIn(old.metadata["fact_id"], ids)
        self.assertIn(new.metadata["fact_id"], ids)

    def test_expired_stays_excluded(self):
        expired = write_fact(
            self.project, "trial.quota", "试用额度一百次",
            valid_from="2025-01-01T00:00:00Z", expires_at="2026-02-01T00:00:00Z",
        )
        ids = [item.ref_id for item in self._search("trial.quota", soft_supersede=True, limit=10)]
        self.assertNotIn(expired.metadata["fact_id"], ids)

    def test_asof_valid_version_is_not_penalized(self):
        old, new = self._versions()
        results = self._search("cache.engine", soft_supersede=True,
                               valid_at="2026-02-01T00:00:00Z", limit=10)
        by_id = {item.ref_id: item for item in results}
        self.assertIn(old.metadata["fact_id"], by_id)
        self.assertNotIn(new.metadata["fact_id"], by_id)
        self.assertNotIn("soft-supersede:-0.010000", by_id[old.metadata["fact_id"]].reasons)

    def test_scheduled_successor_predecessor_not_penalized(self):
        old = write_fact(self.project, "rollout.plan", "方案甲", valid_from="2026-01-01T00:00:00Z")
        write_fact(
            self.project, "rollout.plan", "方案乙",
            supersedes=old.metadata["fact_id"], valid_from="2099-01-01T00:00:00Z",
        )
        results = self._search("rollout.plan", soft_supersede=True, limit=10)
        old_hit = next(item for item in results if item.ref_id == old.metadata["fact_id"])
        self.assertNotIn("soft-supersede:-0.010000", old_hit.reasons)
        self.assertIn("effective-current", old_hit.reasons)

    def test_combines_with_recency_boost_and_adjacent(self):
        old, new = self._versions()
        results = self._search("cache.engine", soft_supersede=True, recency_boost=True, limit=10)
        old_hit = next(item for item in results if item.ref_id == old.metadata["fact_id"])
        self.assertTrue(any(reason.startswith("recency-boost:") for reason in old_hit.reasons))
        self.assertIn("soft-supersede:-0.010000", old_hit.reasons)
        new_hit = next(item for item in results if item.ref_id == new.metadata["fact_id"])
        self.assertGreater(new_hit.score, old_hit.score)
        write_event(self.project, "无关相邻记录", "完全无关的相邻正文", recorded_at="2026-08-01T00:00:00Z")
        with_adjacent = self._search("cache.engine", soft_supersede=True, adjacent=1, limit=10)
        self.assertTrue(any("adjacent-context" in item.reasons for item in with_adjacent))

    def test_combines_with_abstain_threshold(self):
        self._versions()
        info = {}
        results = self._search("cache.engine", soft_supersede=True, abstain_threshold=0.99,
                               limit=10, diagnostics=info)
        self.assertEqual(results, [])
        self.assertTrue(info["abstain"]["abstained"])

    def test_lifecycle_and_storage_untouched(self):
        old, new = self._versions()
        conn = connect()
        before = [dict(row) for row in conn.execute("SELECT fact_id,status,valid_to FROM facts ORDER BY valid_from")]
        conn.close()
        self._search("cache.engine", soft_supersede=True, limit=10)
        conn = connect()
        after = [dict(row) for row in conn.execute("SELECT fact_id,status,valid_to FROM facts ORDER BY valid_from")]
        conn.close()
        self.assertEqual(before, after)
        history = fact_history(self.project, "cache.engine")
        self.assertEqual([row["status"] for row in history], ["superseded", "active"])

    def test_legacy_profile_ignores_switch(self):
        old, new = self._versions()
        info = {}
        results = self._search("cache.engine", profile="legacy", soft_supersede=True, limit=10, diagnostics=info)
        self.assertNotIn(old.metadata["fact_id"], [item.ref_id for item in results])
        self.assertEqual(info["soft_supersede"], {"status": "ignored-by-legacy-profile"})

    def test_passages_profile_demotes_old_version(self):
        old, new = self._versions()
        info = {}
        results = self._search("cache.engine", profile="passages", soft_supersede=True, limit=10, diagnostics=info)
        by_id = {item.ref_id: item for item in results}
        self.assertIn(old.metadata["fact_id"], by_id)
        self.assertIn(new.metadata["fact_id"], by_id)
        self.assertIn("soft-supersede:-0.010000", by_id[old.metadata["fact_id"]].reasons)
        self.assertGreater(by_id[new.metadata["fact_id"]].score, by_id[old.metadata["fact_id"]].score)
        self.assertEqual(info["soft_supersede"]["penalized"], 1)

    def test_cli_json_end_to_end(self):
        old, new = self._versions()
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            cli_main(["context", "--cwd", str(self.project), "--query", "cache.engine",
                      "--json", "--soft-supersede", "--limit", "10"])
        payload = json.loads(buffer.getvalue())
        ids = [item["ref_id"] for item in payload["results"]]
        self.assertIn(old.metadata["fact_id"], ids)
        self.assertIn(new.metadata["fact_id"], ids)
        old_hit = next(item for item in payload["results"] if item["ref_id"] == old.metadata["fact_id"])
        self.assertIn("soft-supersede:-0.010000", old_hit["reasons"])
        # Default CLI invocation (switch off) still excludes the old version.
        buffer2 = io.StringIO()
        with redirect_stdout(buffer2):
            cli_main(["context", "--cwd", str(self.project), "--query", "cache.engine", "--json", "--limit", "10"])
        ids2 = [item["ref_id"] for item in json.loads(buffer2.getvalue())["results"]]
        self.assertNotIn(old.metadata["fact_id"], ids2)
