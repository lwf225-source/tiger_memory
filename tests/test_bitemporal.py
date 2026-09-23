"""Bitemporal query tests (batch 8): opt-in ``valid_at`` / ``known_at`` filters.

Fact carries valid time (valid_from/valid_to: when the fact holds), Event
carries system time (recorded_at: when the system learned it). The default
query interface never distinguished them; ``view=history`` and ``view=both``
are legacy aliases (both return active+superseded with no time filter) and are
frozen by existing tests. The new parameters are OFF by default and must be
byte-identical to previous behavior; when set they filter eligibility only —
ranking is untouched.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from tiger_memory.cli import main as cli_main
from tiger_memory.database import connect
from tiger_memory.search import search
from tiger_memory.service import context_data, forget, write_event, write_fact
from tests.common import MemoryTestCase


class BitemporalFilterTests(MemoryTestCase):
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

    def test_disabled_matches_default_exactly(self):
        self._versions()
        default = self._search("cache.engine")
        disabled = self._search("cache.engine", valid_at="", known_at="")
        self.assertEqual([item.as_dict() for item in default], [item.as_dict() for item in disabled])
        info = {}
        self._search("cache.engine", diagnostics=info)
        self.assertEqual(info["temporal"], {"status": "disabled"})

    def test_current_view_returns_only_new_version(self):
        old, new = self._versions()
        ids = [item.ref_id for item in self._search("cache.engine", limit=10)]
        self.assertIn(new.metadata["fact_id"], ids)
        self.assertNotIn(old.metadata["fact_id"], ids)

    def test_valid_at_time_travels_to_old_version(self):
        old, new = self._versions()
        info = {}
        results = self._search("cache.engine", valid_at="2026-02-01T00:00:00Z", limit=10, diagnostics=info)
        ids = [item.ref_id for item in results]
        self.assertIn(old.metadata["fact_id"], ids)
        self.assertNotIn(new.metadata["fact_id"], ids)
        self.assertEqual(info["temporal"]["status"], "enabled")
        self.assertEqual(info["temporal"]["valid_at"], "2026-02-01T00:00:00.000000Z")
        self.assertIsNone(info["temporal"]["known_at"])

    def test_valid_to_boundary_is_exclusive_valid_from_inclusive(self):
        old, new = self._versions()
        # valid_to == t: the old window is already closed at exactly t.
        ids = [item.ref_id for item in self._search("cache.engine", valid_at="2026-03-01T00:00:00Z", limit=10)]
        self.assertNotIn(old.metadata["fact_id"], ids)
        # valid_from == t: the new version is already valid at exactly t.
        self.assertIn(new.metadata["fact_id"], ids)

    def test_valid_at_future_selects_scheduled_version(self):
        old = write_fact(self.project, "rollout.plan", "方案甲", valid_from="2026-01-01T00:00:00Z")
        future = write_fact(
            self.project, "rollout.plan", "方案乙",
            supersedes=old.metadata["fact_id"], valid_from="2099-01-01T00:00:00Z",
        )
        now_ids = [item.ref_id for item in self._search("rollout.plan", limit=10)]
        self.assertIn(old.metadata["fact_id"], now_ids)
        self.assertNotIn(future.metadata["fact_id"], now_ids)
        future_ids = [item.ref_id for item in self._search("rollout.plan", valid_at="2100-01-01T00:00:00Z", limit=10)]
        self.assertIn(future.metadata["fact_id"], future_ids)
        self.assertNotIn(old.metadata["fact_id"], future_ids)

    def test_known_at_filters_facts_and_events_by_recorded_time(self):
        with patch("tiger_memory.model.utc_now", return_value="2026-02-01T00:00:00.000000Z"):
            early_fact = write_fact(self.project, "ui.theme", "浅色主题", valid_from="2026-01-01T00:00:00Z")
        with patch("tiger_memory.model.utc_now", return_value="2026-04-01T00:00:00.000000Z"):
            late_fact = write_fact(self.project, "ui.font", "宋体正文字体", valid_from="2026-01-01T00:00:00Z")
        early_event = write_event(self.project, "早期备忘", "主题 词甲", recorded_at="2026-02-01T00:00:00Z")
        late_event = write_event(self.project, "晚期备忘", "字体 词乙", recorded_at="2026-04-01T00:00:00Z")

        unfiltered = self._search("ui", limit=10)
        self.assertIn(early_fact.metadata["fact_id"], [item.ref_id for item in unfiltered])
        self.assertIn(late_fact.metadata["fact_id"], [item.ref_id for item in unfiltered])
        filtered = self._search("ui", known_at="2026-03-01T00:00:00Z", limit=10)
        filtered_ids = [item.ref_id for item in filtered]
        self.assertIn(early_fact.metadata["fact_id"], filtered_ids)
        self.assertNotIn(late_fact.metadata["fact_id"], filtered_ids)

        event_ids = [item.ref_id for item in self._search("备忘", known_at="2026-03-01T00:00:00", limit=10)]
        self.assertIn(early_event.event_id, event_ids)
        self.assertNotIn(late_event.event_id, event_ids)

    def test_combined_known_at_and_valid_at(self):
        """截至 T1 系统已知的、在 T2 有效的事实 — retroactive correction."""
        with patch("tiger_memory.model.utc_now", return_value="2026-01-15T00:00:00.000000Z"):
            old = write_fact(self.project, "metric.owner", "甲队负责", valid_from="2026-01-01T00:00:00Z")
        with patch("tiger_memory.model.utc_now", return_value="2026-06-01T00:00:00.000000Z"):
            new = write_fact(
                self.project, "metric.owner", "乙队负责",
                supersedes=old.metadata["fact_id"], valid_from="2026-02-01T00:00:00Z",
            )
        # At valid-time 2026-03 the retroactive new value holds...
        ids = [item.ref_id for item in self._search("metric.owner", valid_at="2026-03-01T00:00:00Z", limit=10)]
        self.assertIn(new.metadata["fact_id"], ids)
        # ...but as of what was known by 2026-02-01, the new value is invisible,
        # and the old one is no longer valid at 2026-03 — nothing for this key.
        both = self._search("metric.owner", valid_at="2026-03-01T00:00:00Z",
                            known_at="2026-02-01T00:00:00Z", limit=10)
        self.assertNotIn(new.metadata["fact_id"], [item.ref_id for item in both])
        self.assertNotIn(old.metadata["fact_id"], [item.ref_id for item in both])
        # Known by 2026-02-01 and valid at 2026-01-15: the old value answers.
        asof = self._search("metric.owner", valid_at="2026-01-15T00:00:00Z",
                            known_at="2026-02-01T00:00:00Z", limit=10)
        self.assertIn(old.metadata["fact_id"], [item.ref_id for item in asof])

    def test_retracted_never_returns_under_time_travel(self):
        fact = write_fact(self.project, "ui.color", "旧主题色", valid_from="2026-01-01T00:00:00Z")
        forget(self.project, fact.event_id)
        for kwargs in (
            {"valid_at": "2026-01-15T00:00:00Z"},
            {"known_at": "2099-01-01T00:00:00Z"},
            {"valid_at": "2026-01-15T00:00:00Z", "known_at": "2099-01-01T00:00:00Z"},
            {"view": "history", "valid_at": "2026-01-15T00:00:00Z"},
        ):
            ids = [item.ref_id for item in self._search("ui.color", limit=10, **kwargs)]
            self.assertNotIn(fact.metadata["fact_id"], ids, msg=repr(kwargs))

    def test_events_valid_at_uses_observed_at(self):
        event = write_event(
            self.project, "观察备忘", "观测 词丙",
            observed_at="2026-05-01T00:00:00Z", recorded_at="2026-05-02T00:00:00Z",
        )
        before = [item.ref_id for item in self._search("观察备忘", valid_at="2026-03-01T00:00:00Z", limit=10)]
        after = [item.ref_id for item in self._search("观察备忘", valid_at="2026-06-01T00:00:00Z", limit=10)]
        self.assertNotIn(event.event_id, before)
        self.assertIn(event.event_id, after)

    def test_invalid_timestamps_rejected(self):
        self._versions()
        for kwargs in ({"valid_at": "昨天中午"}, {"valid_at": "2026-13-45"},
                       {"known_at": "not-a-date"}, {"known_at": 12345}, {"valid_at": 0}):
            with self.assertRaises(ValueError, msg=repr(kwargs)):
                self._search("cache.engine", **kwargs)

    def test_view_semantics_locked_history_alias_both(self):
        """Frozen: history/both are aliases returning active+superseded versions."""
        old, new = self._versions()
        current_ids = [item.ref_id for item in self._search("cache.engine", view="current", limit=10)]
        history_ids = [item.ref_id for item in self._search("cache.engine", view="history", limit=10)]
        both_ids = [item.ref_id for item in self._search("cache.engine", view="both", limit=10)]
        self.assertIn(new.metadata["fact_id"], current_ids)
        self.assertNotIn(old.metadata["fact_id"], current_ids)
        self.assertIn(old.metadata["fact_id"], history_ids)
        self.assertIn(new.metadata["fact_id"], history_ids)
        self.assertEqual(sorted(history_ids), sorted(both_ids))

    def test_legacy_profile_ignores_temporal_params(self):
        old, new = self._versions()
        plain = self._search("cache.engine", profile="legacy", limit=10)
        info = {}
        ignored = self._search("cache.engine", profile="legacy", limit=10,
                               valid_at="2026-02-01T00:00:00Z", known_at="2026-01-01T00:00:00Z",
                               diagnostics=info)
        self.assertEqual([item.as_dict() for item in plain], [item.as_dict() for item in ignored])
        self.assertEqual(info["temporal"], {"status": "ignored-by-legacy-profile"})

    def test_startup_context_honours_temporal_filters(self):
        old, new = self._versions()
        _project, results = context_data(self.project, "", valid_at="2026-02-01T00:00:00Z")
        ids = [item.ref_id for item in results]
        self.assertIn(old.metadata["fact_id"], ids)
        self.assertNotIn(new.metadata["fact_id"], ids)

    def test_cli_json_end_to_end(self):
        old, new = self._versions()
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            cli_main(["context", "--cwd", str(self.project), "--query", "cache.engine",
                      "--json", "--valid-at", "2026-02-01", "--limit", "10"])
        payload = json.loads(buffer.getvalue())
        ids = [item["ref_id"] for item in payload["results"]]
        self.assertIn(old.metadata["fact_id"], ids)
        self.assertNotIn(new.metadata["fact_id"], ids)
        # Date shorthand normalized; --known-at composes.
        buffer2 = io.StringIO()
        with redirect_stdout(buffer2):
            cli_main(["context", "--cwd", str(self.project), "--query", "cache.engine",
                      "--json", "--valid-at", "2026-04-01", "--known-at", "2099-01-01", "--limit", "10"])
        ids2 = [item["ref_id"] for item in json.loads(buffer2.getvalue())["results"]]
        self.assertIn(new.metadata["fact_id"], ids2)
        # Invalid timestamp exits 2 with an error on stderr.
        with redirect_stderr(io.StringIO()):
            code = cli_main(["context", "--cwd", str(self.project), "--query", "cache.engine",
                             "--json", "--valid-at", "下周三"])
        self.assertEqual(code, 2)
