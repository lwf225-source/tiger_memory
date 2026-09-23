from __future__ import annotations

import io
from contextlib import redirect_stdout

from tiger_memory.cli import main
from tiger_memory.database import connect
from tiger_memory.search import (
    COVERAGE_MAX_BONUS,
    _coverage_rerank,
    _coverage_tokens,
    search,
)
from tiger_memory.model import SearchResult
from tiger_memory.service import write_event
from tests.common import MemoryTestCase


def _project_id():
    conn = connect()
    try:
        return conn.execute("SELECT project_id FROM projects").fetchone()[0]
    finally:
        conn.close()


class WindowReserveTests(MemoryTestCase):
    """Batch 12 opt-in ``window_reserve``: adjacent extras occupy reserved
    tail slots of the evidence window instead of trailing beyond it."""

    def _write(self, title, body, day, **kwargs):
        return write_event(self.project, title, body, recorded_at="2026-03-%02dT00:00:00Z" % day, **kwargs)

    def _search(self, project_id, query, **kwargs):
        conn = connect()
        try:
            return search(conn, project_id, query, **kwargs)
        finally:
            conn.close()

    @staticmethod
    def _split(results):
        hits = [item for item in results if "adjacent-context" not in item.reasons]
        extras = [item for item in results if "adjacent-context" in item.reasons]
        return hits, extras

    def test_reserve_disabled_matches_default_exactly(self):
        for day in range(1, 11):
            self._write("记录%02d" % day, "哨兵词乙 填充内容 %02d。" % day, day)
        default = self._search(_project_id(), "哨兵词乙", limit=4, adjacent=1)
        disabled = self._search(_project_id(), "哨兵词乙", limit=4, adjacent=1,
                              window_reserve=False, coverage_rerank=False)
        self.assertEqual([item.as_dict() for item in default], [item.as_dict() for item in disabled])

    def test_reserve_places_adjacent_inside_window(self):
        # 10 direct hits with limit=4: without reserve the window holds only
        # direct hits; with reserve a tail slot goes to a neighbour.
        # Interleave non-matching neighbours: equal-scoring direct hits may
        # legitimately rank by random event ID. With only matching events,
        # the top hit's timeline neighbours could already be direct hits and
        # thus be correctly excluded from the adjacent extras.
        events = []
        direct_ids = set()
        for day in range(1, 22):
            if day % 2 == 0:
                event = self._write("窗口%02d" % day, "锚点词丙 第 %02d 条。" % day, day)
                direct_ids.add(event.event_id)
            else:
                event = self._write("填充%02d" % day, "无关填充内容 %02d。" % day, day)
            events.append(event)
        plain = self._search(_project_id(), "锚点词丙", limit=4, adjacent=1)
        self.assertEqual(self._split(plain[:4])[1], [])
        self.assertEqual(len(plain[:4]), 4)
        self.assertTrue({item.ref_id for item in plain[:4]} <= direct_ids)
        info = {}
        conn = connect()
        try:
            reserved = search(conn, _project_id(), "锚点词丙", limit=4, adjacent=1,
                              window_reserve=True, diagnostics=info)
        finally:
            conn.close()
        window_hits, window_extras = self._split(reserved[:4])
        self.assertEqual(info["adjacent"]["window_reserve"], True)
        self.assertEqual(info["adjacent"]["in_window"], 1)  # share = max(1, 4 // 4)
        self.assertEqual(len(window_extras), 1)
        self.assertEqual(len(window_hits), 3)
        # The displaced direct hit is the lowest-ranked one; the top direct
        # hits keep their relative order.
        self.assertEqual([item.ref_id for item in window_hits],
                         [item.ref_id for item in self._split(plain)[0][:3]])
        # The in-window extra neighbours the top direct hit.
        top_position = events.index(next(e for e in events if e.event_id == window_hits[0].ref_id))
        neighbour_ids = {events[i].event_id for i in (top_position - 1, top_position + 1)
                         if 0 <= i < len(events)}
        self.assertIn(window_extras[0].ref_id, neighbour_ids)
        # Overflow extras still trail beyond the window for interactive use.
        self.assertEqual(reserved[4:], [item for item in reserved[4:] if "adjacent-context" in item.reasons])

    def test_reserve_prefers_neighbours_of_higher_ranked_hits(self):
        # Matching events interleave with non-matching fillers, so every hit
        # has non-hit timeline neighbours available for the reserved slots.
        day_of = {}
        matching = []
        for day in range(1, 17):
            if day % 2 == 1:
                event = self._write("种子%02d" % day, "锚点词丁 第 %02d 条。" % day, day)
                matching.append(event)
            else:
                event = self._write("填充%02d" % day, "无关填充内容 %02d。" % day, day)
            day_of[event.event_id] = day
        conn = connect()
        try:
            plain = search(conn, _project_id(), "锚点词丁", limit=8)
            reserved = search(conn, _project_id(), "锚点词丁", limit=8, adjacent=1,
                              window_reserve=True, diagnostics={})
        finally:
            conn.close()
        anchors = {plain[0].ref_id, plain[1].ref_id}
        anchor_days = {day_of[a] for a in anchors}
        neighbour_ids = {event_id for event_id, day in day_of.items()
                         if any(abs(day - anchor) == 1 for anchor in anchor_days)} - anchors
        in_window = {item.ref_id for item in reserved[:8] if "adjacent-context" in item.reasons}
        # share = max(1, 8 // 4) = 2; every in-window extra must neighbour one
        # of the two highest-ranked direct hits (seed-rank order).
        self.assertEqual(len(in_window), 2)
        self.assertTrue(in_window <= neighbour_ids)

    def test_reserve_without_adjacent_is_skipped_and_identical(self):
        for day in range(1, 6):
            self._write("空载%02d" % day, "哨兵词戊 内容 %02d。" % day, day)
        info = {}
        conn = connect()
        try:
            reserved = search(conn, _project_id(), "哨兵词戊", limit=4,
                              window_reserve=True, diagnostics=info)
            default = search(conn, _project_id(), "哨兵词戊", limit=4)
        finally:
            conn.close()
        self.assertEqual(info["adjacent"], {"status": "skipped-reserve-requires-adjacent"})
        self.assertEqual([item.as_dict() for item in reserved], [item.as_dict() for item in default])

    def test_reserve_bounds_are_validated(self):
        self._write("校验", "锚点词己 内容。", 1)
        conn = connect()
        try:
            with self.assertRaises(ValueError):
                search(conn, _project_id(), "锚点词己", window_reserve=1)
            with self.assertRaises(ValueError):
                search(conn, _project_id(), "锚点词己", coverage_rerank=0)
        finally:
            conn.close()

    def test_reserve_ignored_by_frozen_legacy_profile(self):
        self._write("遗留甲", "锚点词庚 第一条。", 1)
        self._write("遗留乙", "锚点词庚 第二条。", 2)
        info = {}
        conn = connect()
        try:
            legacy = search(conn, _project_id(), "锚点词庚", limit=4, profile="legacy",
                            adjacent=1, window_reserve=True, coverage_rerank=True, diagnostics=info)
        finally:
            conn.close()
        self.assertEqual(info["adjacent"], {"status": "ignored-by-legacy-profile"})
        self.assertEqual(info["coverage_rerank"], {"status": "ignored-by-legacy-profile"})
        self.assertTrue(all("adjacent-context" not in item.reasons for item in legacy))

    def test_passages_profile_reserve_in_window_with_valid_spans(self):
        body = "段落句。" * 80
        for day in range(1, 9):
            self._write("篇章%02d" % day, body + " 片段锚点辛 %02d" % day, day)
        conn = connect()
        try:
            results = search(conn, _project_id(), "片段锚点辛", limit=4, profile="passages",
                             adjacent=1, window_reserve=True, diagnostics={})
        finally:
            conn.close()
        in_window = [item for item in results[:4] if "adjacent-context" in item.reasons]
        self.assertTrue(in_window)
        for item in in_window:
            self.assertTrue(item.evidence_spans)



class CoverageRerankTests(MemoryTestCase):
    """Batch 12 opt-in ``coverage_rerank``: greedy query-side rare-word
    coverage over the ranking head; the result set is a pure permutation."""

    def _write(self, title, body, day, **kwargs):
        return write_event(self.project, title, body, recorded_at="2026-03-%02dT00:00:00Z" % day, **kwargs)

    def _search(self, project_id, query, diagnostics=None, **kwargs):
        conn = connect()
        try:
            return search(conn, project_id, query, diagnostics=diagnostics, **kwargs)
        finally:
            conn.close()

    def test_tokens_strip_stopwords_and_single_chars(self):
        tokens = _coverage_tokens("What is the A plan for Q2 2026?")
        self.assertIn("plan", tokens)
        self.assertIn("2026", tokens)
        self.assertNotIn("what", tokens)
        self.assertNotIn("the", tokens)
        self.assertNotIn("a", tokens)
        self.assertNotIn("is", tokens)

    def test_rerank_promotes_uncovered_rare_word(self):
        # Ranking scores: a(0.050) > b(0.049) > c(0.048); a and b cover only
        # "common", c alone covers the rare "quasar". With a 0.01 bonus cap,
        # c must overtake b but never a (a clear 0.02 gap to c is too large
        # once a also covers "common" first... a: 0.050+bonus covers common;
        # b: 0.049 no gain; c: 0.048+0.01 = 0.058 > b).
        records = {}
        for key, body in [(("event", "a"), "common anchor text"),
                          (("event", "b"), "common filler text"),
                          (("event", "c"), "quasar observation text")]:
            records[key] = {"title": "", "tags_json": "[]", "body": body}
        results = [
            SearchResult("a", "event", "ta", records[("event", "a")]["body"], "p", 0.050, [], "s", "active", "2026-01-01", ""),
            SearchResult("b", "event", "tb", records[("event", "b")]["body"], "p", 0.049, [], "s", "active", "2026-01-01", ""),
            SearchResult("c", "event", "tc", records[("event", "c")]["body"], "p", 0.048, [], "s", "active", "2026-01-01", ""),
        ]
        info = {}
        reranked = _coverage_rerank(results, records, "common quasar", 3, info)
        self.assertEqual([item.ref_id for item in reranked], ["a", "c", "b"])
        self.assertEqual(info["coverage_rerank"]["status"], "ready")
        self.assertEqual(info["coverage_rerank"]["info_words"], 2)
        self.assertEqual(info["coverage_rerank"]["covered_words"], 2)
        # The result set is a pure permutation — nothing added or dropped.
        self.assertEqual({item.ref_id for item in reranked}, {"a", "b", "c"})

    def test_rerank_bonus_cannot_bridge_clear_relevance_gap(self):
        records = {
            ("event", "a"): {"title": "", "tags_json": "[]", "body": "common anchor text"},
            ("event", "c"): {"title": "", "tags_json": "[]", "body": "quasar observation text"},
        }
        results = [
            SearchResult("a", "event", "ta", "common anchor text", "p", 0.050, [], "s", "active", "2026-01-01", ""),
            SearchResult("c", "event", "tc", "quasar observation text", "p", 0.050 - 2 * COVERAGE_MAX_BONUS, [], "s", "active", "2026-01-01", ""),
        ]
        reranked = _coverage_rerank(results, records, "common quasar", 2, {})
        self.assertEqual([item.ref_id for item in reranked], ["a", "c"])

    def test_rerank_no_info_words_keeps_order(self):
        self._write("停用词", "some body text here。", 1)
        info = {}
        default = self._search(_project_id(), "the what is", limit=4)
        reranked = self._search(_project_id(), "the what is", limit=4,
                                coverage_rerank=True, diagnostics=info)
        self.assertEqual(info["coverage_rerank"]["status"], "skipped-no-info-words")
        self.assertEqual([item.as_dict() for item in reranked], [item.as_dict() for item in default])

    def test_rerank_integration_is_permutation_with_diagnostics(self):
        for day in range(1, 9):
            self._write("覆盖%02d" % day, "lighthouse keeper note %02d" % day, day)
        info = {}
        default = self._search(_project_id(), "lighthouse keeper", limit=5)
        reranked = self._search(_project_id(), "lighthouse keeper", limit=5,
                                coverage_rerank=True, diagnostics=info)
        self.assertEqual(info["coverage_rerank"]["status"], "ready")
        self.assertEqual(info["coverage_rerank"]["max_bonus"], COVERAGE_MAX_BONUS)
        self.assertEqual(sorted(item.ref_id for item in reranked[:5]),
                         sorted(item.ref_id for item in default[:5]))

    def test_rerank_combines_with_window_reserve(self):
        for day in range(1, 11):
            self._write("组合%02d" % day, "anchor word telescope log %02d" % day, day)
        info = {}
        conn = connect()
        try:
            results = search(conn, _project_id(), "anchor telescope", limit=4, adjacent=1,
                             window_reserve=True, coverage_rerank=True, diagnostics=info)
        finally:
            conn.close()
        self.assertEqual(info["coverage_rerank"]["status"], "ready")
        self.assertEqual(info["adjacent"]["window_reserve"], True)
        self.assertEqual(len([item for item in results[:4] if "adjacent-context" in item.reasons]), 1)

    def test_rerank_combines_with_mmr(self):
        for day in range(1, 11):
            self._write("双开%02d" % day, "anchor word observatory log %02d" % day, day)
        info = {}
        conn = connect()
        try:
            results = search(conn, _project_id(), "anchor observatory", limit=4,
                             mmr=True, coverage_rerank=True, diagnostics=info)
        finally:
            conn.close()
        self.assertEqual(info["mmr"]["status"], "ready")
        self.assertEqual(info["coverage_rerank"]["status"], "ready")
        self.assertTrue(results)

    def test_cli_coverage_and_reserve_flags_end_to_end(self):
        for day in range(1, 9):
            self._write("命令%02d" % day, "anchor word quadrant log %02d" % day, day)
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["context", "--cwd", str(self.project), "--query", "anchor quadrant",
                                   "--adjacent", "1", "--window-reserve", "--coverage-rerank",
                                   "--limit", "4", "--json"]), 0)
        self.assertIn('"results"', output.getvalue())
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["context", "--cwd", str(self.project), "--query", "anchor quadrant",
                                   "--adjacent", "1", "--window-reserve", "--limit", "4"]), 0)
        self.assertIn("role=adjacent-context", output.getvalue())
