"""Temporal-intent tests (batch 10): deterministic query-side extraction and
its opt-in wiring into search/service/CLI/AML adapter.

Discipline under test:
- the extractor is conservative — no safe anchor, no intent ("宁可漏不可误判");
- ``after``/``since`` set intent but never an ``as_of`` (the valid_at mechanism
  is an upper bound only);
- relative anchors are reported for audit yet never applied by the wiring
  (historical-timestamp corpora would be emptied by a now-anchored filter);
- the wiring never overrides explicit caller parameters;
- flag off, or flag on with no intent found, is byte-identical to before.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from datetime import datetime, timezone
from unittest.mock import patch

from tiger_memory import aml_adapter
from tiger_memory.cli import main as cli_main
from tiger_memory.database import connect
from tiger_memory.search import search
from tiger_memory.service import context_data, write_event, write_fact
from tiger_memory.temporal_intent import extract_temporal_intent
from tests.common import MemoryTestCase


class ExtractorTests(MemoryTestCase):
    """Pure extractor behavior (no storage involved)."""

    def test_as_of_month_year_english(self):
        intent = extract_temporal_intent("What was the plan as of March 2026?")
        self.assertTrue(intent["has_temporal_intent"])
        self.assertEqual(intent["as_of"], "2026-03-31T23:59:59.999999Z")
        self.assertEqual(intent["matched"], "as of March 2026")

    def test_during_month_year(self):
        intent = extract_temporal_intent("Did James have a girlfriend during April 2022?")
        self.assertEqual(intent["as_of"], "2022-04-30T23:59:59.999999Z")

    def test_in_month_year_with_day(self):
        intent = extract_temporal_intent("What happened in March 15, 2026?")
        self.assertEqual(intent["as_of"], "2026-03-15T23:59:59.999999Z")

    def test_season_year(self):
        intent = extract_temporal_intent("What state did Joanna visit in summer 2021?")
        self.assertEqual(intent["as_of"], "2021-08-31T23:59:59.999999Z")

    def test_first_half_of_month(self):
        intent = extract_temporal_intent("Was the first half of September 2022 a good month?")
        self.assertEqual(intent["as_of"], "2022-09-15T23:59:59.999999Z")

    def test_between_months_uses_window_end(self):
        intent = extract_temporal_intent("did they meet in Boston between August and November 2023?")
        self.assertEqual(intent["as_of"], "2023-11-30T23:59:59.999999Z")

    def test_iso_date_and_month(self):
        self.assertEqual(extract_temporal_intent("as of 2026-03-15, who owned it?")["as_of"],
                         "2026-03-15T23:59:59.999999Z")
        self.assertEqual(extract_temporal_intent("status in 2026-03?")["as_of"],
                         "2026-03-31T23:59:59.999999Z")

    def test_before_resolves_to_period_start(self):
        self.assertEqual(extract_temporal_intent("what changed before March 2026?")["as_of"],
                         "2026-03-01T00:00:00.000000Z")
        self.assertEqual(extract_temporal_intent("what changed before 2026-03-15?")["as_of"],
                         "2026-03-15T00:00:00.000000Z")
        self.assertEqual(extract_temporal_intent("what changed before 2020?")["as_of"],
                         "2020-01-01T00:00:00.000000Z")

    def test_after_since_set_intent_without_as_of(self):
        for text in ("what changed after 2023-05-01?", "how has it evolved since 2020?",
                     "what came after March 2026?"):
            intent = extract_temporal_intent(text)
            self.assertTrue(intent["has_temporal_intent"], msg=text)
            self.assertIsNone(intent["as_of"], msg=text)
            self.assertTrue(intent["matched"], msg=text)

    def test_bare_year_with_preposition(self):
        self.assertEqual(extract_temporal_intent("what did we ship in 2023?")["as_of"],
                         "2023-12-31T23:59:59.999999Z")

    def test_chinese_forms(self):
        self.assertEqual(extract_temporal_intent("截至2026年3月，项目进展如何？")["as_of"],
                         "2026-03-31T23:59:59.999999Z")
        self.assertEqual(extract_temporal_intent("2026年3月之前完成了什么？")["as_of"],
                         "2026-03-01T00:00:00.000000Z")
        intent = extract_temporal_intent("2026年3月之后有什么变化？")
        self.assertTrue(intent["has_temporal_intent"])
        self.assertIsNone(intent["as_of"])

    def test_comparison_markers(self):
        for text in ("Did he previously live in Boston?", "what framework did we used to use?",
                     "what is the currently deployed version?", "他以前住在哪里？", "曾经的方案是什么？"):
            intent = extract_temporal_intent(text)
            self.assertTrue(intent["comparison"], msg=text)
            self.assertTrue(intent["has_temporal_intent"], msg=text)
            self.assertIsNone(intent["as_of"], msg=text)

    def test_relative_reported_but_not_as_of(self):
        now = datetime(2026, 9, 12, 10, 0, 0, tzinfo=timezone.utc)
        intent = extract_temporal_intent("what did I say last week?", now=now)
        self.assertTrue(intent["has_temporal_intent"])
        self.assertIsNone(intent["as_of"])
        self.assertEqual(intent["relative"]["expression"], "last week")
        self.assertTrue(intent["relative"]["as_of"].startswith("2026-09-06"))
        cn = extract_temporal_intent("上个月说了什么？", now=now)
        self.assertEqual(cn["relative"]["as_of"], "2026-08-31T23:59:59.999999Z")

    def test_rejections_no_false_positives(self):
        for text in ("Would Caroline be likely to pursue counseling?",
                     "Would Melanie go on another roadtrip soon?",
                     "What did Jean go through before meeting Maria?",
                     "What console does Nate own?",
                     "Tell me about the March 2026 report",  # bare month-year, no anchor preposition
                     "error code 2026-13-45",                # invalid month/day
                     ""):
            intent = extract_temporal_intent(text)
            self.assertFalse(intent["has_temporal_intent"], msg=text)
            self.assertIsNone(intent["as_of"], msg=text)

    def test_never_raises_on_odd_input(self):
        for value in (None, 123, "在2099年99月", "first half of Februtember 2026"):
            intent = extract_temporal_intent(value)
            self.assertIsInstance(intent, dict)
            self.assertIn("has_temporal_intent", intent)


class TemporalIntentWiringTests(MemoryTestCase):
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
        """old: valid from 2026-01-01; new: supersedes it valid from 2026-03-01."""
        old = write_fact(self.project, "cache.engine", "SQLite", valid_from="2026-01-01T00:00:00Z")
        new = write_fact(
            self.project, "cache.engine", "NextCache",
            supersedes=old.metadata["fact_id"], valid_from="2026-03-01T00:00:00Z",
        )
        return old, new

    def test_flag_off_byte_identical(self):
        old, new = self._versions()
        default = self._search("cache.engine as of 2026-02-01")
        off = self._search("cache.engine as of 2026-02-01", temporal_intent=False)
        self.assertEqual([item.as_dict() for item in default], [item.as_dict() for item in off])
        info = {}
        self._search("cache.engine", diagnostics=info)
        self.assertEqual(info["temporal_intent"], {"status": "disabled"})
        # The intent-laden query must NOT time-travel without the flag: the
        # current version is the only one returned.
        ids = [item.ref_id for item in default]
        self.assertNotIn(old.metadata["fact_id"], ids)

    def test_no_intent_query_byte_identical_results(self):
        self._versions()
        plain = self._search("cache.engine")
        info = {}
        armed = self._search("cache.engine", temporal_intent=True, diagnostics=info)
        self.assertEqual([item.as_dict() for item in plain], [item.as_dict() for item in armed])
        self.assertEqual(info["temporal_intent"]["status"], "enabled")
        self.assertFalse(info["temporal_intent"]["has_temporal_intent"])
        self.assertFalse(info["temporal_intent"]["valid_at_applied"])
        self.assertFalse(info["temporal_intent"]["soft_supersede_applied"])
        self.assertEqual(info["temporal"], {"status": "disabled"})

    def test_as_of_fills_valid_at(self):
        old, new = self._versions()
        info = {}
        results = self._search("cache.engine as of 2026-02-01", temporal_intent=True, diagnostics=info, limit=10)
        ids = [item.ref_id for item in results]
        self.assertIn(old.metadata["fact_id"], ids)
        self.assertNotIn(new.metadata["fact_id"], ids)
        self.assertTrue(info["temporal_intent"]["valid_at_applied"])
        self.assertEqual(info["temporal"]["valid_at"], "2026-02-01T23:59:59.999999Z")

    def test_explicit_valid_at_wins_over_extractor(self):
        old, new = self._versions()
        info = {}
        results = self._search("cache.engine as of 2026-02-01", temporal_intent=True,
                               valid_at="2026-04-01T00:00:00Z", diagnostics=info, limit=10)
        ids = [item.ref_id for item in results]
        self.assertIn(new.metadata["fact_id"], ids)
        self.assertNotIn(old.metadata["fact_id"], ids)
        self.assertFalse(info["temporal_intent"]["valid_at_applied"])

    def test_comparison_enables_soft_supersede(self):
        old, new = self._versions()
        baseline = self._search("cache.engine previously", limit=10)
        self.assertNotIn(old.metadata["fact_id"], [item.ref_id for item in baseline])
        info = {}
        armed = self._search("cache.engine previously", temporal_intent=True, diagnostics=info, limit=10)
        ids = [item.ref_id for item in armed]
        self.assertIn(old.metadata["fact_id"], ids)
        self.assertIn(new.metadata["fact_id"], ids)
        self.assertTrue(info["temporal_intent"]["soft_supersede_applied"])
        # New value still ranks above the demoted old one.
        self.assertLess(ids.index(new.metadata["fact_id"]), ids.index(old.metadata["fact_id"]))

    def test_explicit_soft_supersede_respected(self):
        self._versions()
        info = {}
        self._search("cache.engine", temporal_intent=True, soft_supersede=True, diagnostics=info, limit=10)
        self.assertFalse(info["temporal_intent"]["soft_supersede_applied"])
        self.assertEqual(info["soft_supersede"]["status"], "enabled")

    def test_relative_anchor_never_applied(self):
        old, new = self._versions()
        info = {}
        results = self._search("cache.engine last week", temporal_intent=True, diagnostics=info, limit=10)
        self.assertTrue(info["temporal_intent"]["has_temporal_intent"])
        self.assertFalse(info["temporal_intent"]["valid_at_applied"])
        self.assertFalse(info["temporal_intent"]["relative_applied"])
        self.assertEqual(info["temporal"], {"status": "disabled"})
        self.assertIn(new.metadata["fact_id"], [item.ref_id for item in results])

    def test_combines_with_recency_and_adjacent(self):
        self._versions()
        info = {}
        results = self._search("cache.engine as of 2026-02-01", temporal_intent=True,
                               recency_boost=True, adjacent=1, diagnostics=info, limit=10)
        self.assertEqual(info["temporal_intent"]["status"], "enabled")
        self.assertEqual(info["recency_boost"]["status"], "enabled")
        self.assertEqual(info["adjacent"]["status"], "ready")
        self.assertTrue(results)

    def test_legacy_profile_ignores_switch(self):
        old, new = self._versions()
        plain = self._search("cache.engine as of 2026-02-01", profile="legacy", limit=10)
        info = {}
        armed = self._search("cache.engine as of 2026-02-01", profile="legacy",
                             temporal_intent=True, diagnostics=info, limit=10)
        self.assertEqual([item.as_dict() for item in plain], [item.as_dict() for item in armed])
        self.assertEqual(info["temporal_intent"], {"status": "ignored-by-legacy-profile"})

    def test_invalid_flag_rejected(self):
        self._versions()
        with self.assertRaises(ValueError):
            self._search("cache.engine", temporal_intent="yes")

    def test_passages_profile_combines(self):
        self._versions()
        info = {}
        results = self._search("cache.engine during February 2026", temporal_intent=True,
                               profile="passages", diagnostics=info, limit=10)
        self.assertTrue(info["temporal_intent"]["valid_at_applied"])
        self.assertIsInstance(results, list)

    def test_cli_json_end_to_end(self):
        old, new = self._versions()
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            cli_main(["context", "--cwd", str(self.project), "--query", "cache.engine as of 2026-02-01",
                      "--json", "--temporal-intent", "--limit", "10"])
        ids = [item["ref_id"] for item in json.loads(buffer.getvalue())["results"]]
        self.assertIn(old.metadata["fact_id"], ids)
        self.assertNotIn(new.metadata["fact_id"], ids)
        # Without the flag the same query returns the current version only.
        buffer2 = io.StringIO()
        with redirect_stdout(buffer2):
            cli_main(["context", "--cwd", str(self.project), "--query", "cache.engine as of 2026-02-01",
                      "--json", "--limit", "10"])
        ids2 = [item["ref_id"] for item in json.loads(buffer2.getvalue())["results"]]
        self.assertIn(new.metadata["fact_id"], ids2)
        self.assertNotIn(old.metadata["fact_id"], ids2)


class AdapterTemporalIntentTests(MemoryTestCase):
    def tearDown(self):
        aml_adapter.AML_TEMPORAL_INTENT = False
        super().tearDown()

    def _seed(self):
        aml_adapter.add({
            "request_id": "r1", "user_id": "u1", "session_id": "s1",
            "messages": [
                {"role": "user", "timestamp": 1704067200000,
                 "content": "In January we picked SQLite for the cache engine."},
                {"role": "user", "timestamp": 1709251200000,
                 "content": "In March we migrated the cache engine to NextCache."},
            ],
        })

    def test_default_off_no_observability_key(self):
        self._seed()
        response = aml_adapter.search({"query": "cache engine as of 2024-02-01", "user_id": "u1", "top_k": 5})
        self.assertIn("data", response)
        self.assertNotIn("plm_temporal_intent", response)

    def test_switch_on_filters_and_reports(self):
        self._seed()
        aml_adapter.AML_TEMPORAL_INTENT = True
        response = aml_adapter.search({"query": "cache engine as of 2024-02-01", "user_id": "u1", "top_k": 5})
        self.assertIn("plm_temporal_intent", response)
        self.assertTrue(response["plm_temporal_intent"]["valid_at_applied"])
        bodies = [item["content"] for item in response["data"]]
        self.assertTrue(any("SQLite" in body for body in bodies))
        self.assertFalse(any("migrated" in body for body in bodies))
        # A no-intent query behaves exactly as the default path (no extra key).
        plain = aml_adapter.search({"query": "cache engine", "user_id": "u1", "top_k": 5})
        self.assertNotIn("plm_temporal_intent", plain)
        self.assertTrue(plain["data"])
