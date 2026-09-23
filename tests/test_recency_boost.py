from __future__ import annotations

import io
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone

from tiger_memory.cli import main
from tiger_memory.database import connect
from tiger_memory.passages import validate_passage
from tiger_memory.search import RECENCY_HALF_LIFE_DAYS, RECENCY_MAX_BONUS, search
from tiger_memory.service import write_event
from tests.common import MemoryTestCase


def _iso_days_ago(days: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="microseconds").replace("+00:00", "Z")


class RecencyBoostTests(MemoryTestCase):
    def _write(self, title, body, day, **kwargs):
        return write_event(self.project, title, body, recorded_at="2026-03-%02dT00:00:00Z" % day, **kwargs)

    def _search(self, project_id, query, **kwargs):
        conn = connect()
        try:
            return search(conn, project_id, query, **kwargs)
        finally:
            conn.close()

    def test_recency_disabled_matches_default_exactly(self):
        hit = self._write("独特标记", "独特标记词出现在这里。", 3)
        for kwargs in ({}, {"adjacent": 1}):
            default = self._search(hit.project_id, "独特标记词", **kwargs)
            disabled = self._search(hit.project_id, "独特标记词", recency_boost=False, **kwargs)
            self.assertEqual([item.as_dict() for item in default], [item.as_dict() for item in disabled])
            self.assertTrue(all(not any(reason.startswith("recency-boost:") for reason in item.reasons)
                                for item in default))

    def test_date_query_recalls_injected_month_text(self):
        march = self._write("三月记录", "哨兵正文不含任何日期字眼。", 6)
        write_event(self.project, "四月记录", "另一段哨兵正文也不含日期。",
                    recorded_at="2026-04-02T00:00:00Z")
        # Coarse phrasing: the CJK tokenizer reduces "2026年3月" to "2026",
        # which both 2026 records match; the March record must be recalled.
        boosted = self._search(march.project_id, "2026年3月", recency_boost=True)
        self.assertIn(march.event_id, [item.ref_id for item in boosted])
        plain = self._search(march.project_id, "2026年3月")
        self.assertNotIn(march.event_id, [item.ref_id for item in plain])

    def test_iso_date_query_pinpoints_single_record(self):
        march = self._write("三月记录", "哨兵正文不含任何日期字眼。", 6)
        write_event(self.project, "四月记录", "另一段哨兵正文也不含日期。",
                    recorded_at="2026-04-02T00:00:00Z")
        # "2026-03-06" is one precise lexical term. The fuzzy n-gram channel
        # still recalls the other 2026 record (shared digit n-grams), but the
        # injected exact date must rank the matching record first.
        boosted = self._search(march.project_id, "2026-03-06", recency_boost=True)
        self.assertEqual(boosted[0].ref_id, march.event_id)
        plain = self._search(march.project_id, "2026-03-06")
        self.assertNotIn(march.event_id, [item.ref_id for item in plain])

    def test_weekday_and_english_date_queries(self):
        friday = self._write("周五记录", "周五哨兵正文同样不含日期。", 6)  # 2026-03-06 is a Friday
        boosted_cn = self._search(friday.project_id, "星期五", recency_boost=True)
        self.assertIn(friday.event_id, [item.ref_id for item in boosted_cn])
        boosted_en = self._search(friday.project_id, "Friday", recency_boost=True)
        self.assertIn(friday.event_id, [item.ref_id for item in boosted_en])
        plain = self._search(friday.project_id, "星期五")
        self.assertNotIn(friday.event_id, [item.ref_id for item in plain])

    def test_recency_nudge_prefers_newer_near_tie(self):
        # Identical title and body produce a near-tie: every channel still
        # orders the pair deterministically, so the residual base gap is one
        # RRF position step per channel. A one-half-life recency gap (0.002)
        # is larger than that residual but stays far below structural bonuses.
        older = write_event(self.project, "并列记录", "并列哨兵词 完全相同的内容。",
                            recorded_at=_iso_days_ago(RECENCY_HALF_LIFE_DAYS))
        newer = write_event(self.project, "并列记录", "并列哨兵词 完全相同的内容。")
        plain = self._search(older.project_id, "并列哨兵词")
        self.assertEqual(len(plain), 2)
        base_gap = abs(plain[0].score - plain[1].score)
        self.assertLess(base_gap, RECENCY_MAX_BONUS)  # near-tie, not an exact tie
        boosted = self._search(older.project_id, "并列哨兵词", recency_boost=True)
        self.assertEqual(boosted[0].ref_id, newer.event_id)
        self.assertEqual(boosted[1].ref_id, older.event_id)
        gap = boosted[0].score - boosted[1].score
        self.assertGreater(gap, 0.0)
        # Tie-breaker only: the total gap never exceeds the capped amplitude
        # plus the residual position step it overrode.
        self.assertLessEqual(gap, RECENCY_MAX_BONUS + base_gap + 1e-9)
        for item in boosted:
            self.assertTrue(any(reason.startswith("recency-boost:") for reason in item.reasons))

    def test_recency_ignored_by_frozen_legacy_profile(self):
        hit = self._write("独特标记", "独特标记词出现在这里。", 3)
        plain = self._search(hit.project_id, "独特标记词", profile="legacy")
        diagnostics = {}
        boosted = self._search(hit.project_id, "独特标记词", profile="legacy",
                               recency_boost=True, diagnostics=diagnostics)
        self.assertEqual([item.as_dict() for item in plain], [item.as_dict() for item in boosted])
        self.assertEqual(diagnostics["recency_boost"]["status"], "ignored-by-legacy-profile")

    def test_recency_diagnostics_reported(self):
        hit = self._write("独特标记", "独特标记词出现在这里。", 3)
        enabled = {}
        self._search(hit.project_id, "独特标记词", recency_boost=True, diagnostics=enabled)
        self.assertEqual(enabled["recency_boost"]["status"], "enabled")
        self.assertEqual(enabled["recency_boost"]["half_life_days"], RECENCY_HALF_LIFE_DAYS)
        self.assertEqual(enabled["recency_boost"]["max_bonus"], RECENCY_MAX_BONUS)
        disabled = {}
        self._search(hit.project_id, "独特标记词", diagnostics=disabled)
        self.assertEqual(disabled["recency_boost"]["status"], "disabled")

    def test_recency_flag_is_validated(self):
        hit = self._write("独特标记", "独特标记词出现在这里。", 3)
        for invalid in ("yes", 1, 0):
            with self.subTest(recency_boost=invalid), self.assertRaises(ValueError):
                self._search(hit.project_id, "独特标记词", recency_boost=invalid)

    def test_passages_profile_recency_nudges_parent_order(self):
        older = write_event(self.project, "片段记录", "片段哨兵词 完全相同的正文内容。",
                            recorded_at=_iso_days_ago(RECENCY_HALF_LIFE_DAYS))
        newer = write_event(self.project, "片段记录", "片段哨兵词 完全相同的正文内容。")
        diagnostics = {}
        boosted = self._search(older.project_id, "片段哨兵词", profile="passages",
                               recency_boost=True, diagnostics=diagnostics)
        self.assertEqual(boosted[0].ref_id, newer.event_id)
        self.assertTrue(any(reason.startswith("recency-boost:") for reason in boosted[0].reasons))
        self.assertEqual(diagnostics["recency_boost"]["text_injection"], "skipped-exact-span-profile")
        for item in boosted:
            for span in item.evidence_spans:
                validate_passage(item.body, span, item.source_event_id)
        default = self._search(older.project_id, "片段哨兵词", profile="passages")
        disabled = self._search(older.project_id, "片段哨兵词", profile="passages", recency_boost=False)
        self.assertEqual([item.as_dict() for item in default], [item.as_dict() for item in disabled])

    def test_recency_boost_combines_with_adjacent(self):
        self._write("记录甲", "时间线第一条内容。", 1)
        hit = self._write("独特标记", "独特标记词出现在这里。", 2)
        self._write("记录丙", "时间线第三条内容。", 3)
        results = self._search(hit.project_id, "独特标记词", adjacent=1, recency_boost=True)
        hits = [item for item in results if "adjacent-context" not in item.reasons]
        extras = [item for item in results if "adjacent-context" in item.reasons]
        self.assertEqual(hits[0].ref_id, hit.event_id)
        self.assertTrue(any(reason.startswith("recency-boost:") for reason in hits[0].reasons))
        # Adjacent supplements stay scoreless supplements, never boosted hits.
        self.assertEqual(len(extras), 2)
        for item in extras:
            self.assertEqual(item.score, 0.0)
            self.assertFalse(any(reason.startswith("recency-boost:") for reason in item.reasons))

    def test_startup_context_unaffected_by_flag(self):
        self._write("记录甲", "启动上下文内容第一条。", 1)
        outputs = []
        for extra in ([], ["--recency-boost"]):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                self.assertEqual(main(["context", "--cwd", str(self.project)] + extra), 0)
            outputs.append(buffer.getvalue())
        self.assertEqual(outputs[0], outputs[1])
        self.assertIn("启动上下文内容第一条", outputs[1])

    def test_cli_recency_boost_end_to_end(self):
        march = self._write("三月记录", "命令行哨兵正文不含日期。", 6)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            self.assertEqual(main(["context", "--cwd", str(self.project), "--query", "2026年3月",
                                   "--recency-boost"]), 0)
        self.assertIn(march.body, buffer.getvalue())
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            self.assertEqual(main(["context", "--cwd", str(self.project), "--query", "2026年3月"]), 0)
        self.assertNotIn(march.body, buffer.getvalue())
        # Switches compose: recency boost + adjacent window in one call.
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            self.assertEqual(main(["context", "--cwd", str(self.project), "--query", "2026年3月",
                                   "--recency-boost", "--adjacent", "1"]), 0)
        self.assertIn(march.body, buffer.getvalue())
