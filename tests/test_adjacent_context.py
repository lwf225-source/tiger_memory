from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

from tiger_memory.cli import main
from tiger_memory.context_pack import context_size, pack_context
from tiger_memory.database import connect
from tiger_memory.model import SearchResult
from tiger_memory.passages import validate_passage
from tiger_memory.search import search
from tiger_memory.service import forget, write_event
from tests.common import MemoryTestCase


class AdjacentContextTests(MemoryTestCase):
    def _write(self, title, body, day, **kwargs):
        return write_event(self.project, title, body, recorded_at="2026-03-%02dT00:00:00Z" % day, **kwargs)

    def _timeline(self):
        return [
            self._write("豆浆早餐", "早餐吃了豆浆油条。", 1),
            self._write("羽毛球赛", "中午讨论了羽毛球比赛。", 2),
            self._write("独特标记", "独特标记词出现在这里。", 3),
            self._write("自行车修理", "傍晚修理了自行车链条。", 4),
            self._write("天文纪录", "夜里观看了天文纪录片。", 5),
        ]

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

    def test_adjacent_window_adds_requested_neighbours(self):
        first, second, hit, fourth, fifth = self._timeline()
        results = self._search(hit.project_id, "独特标记词", adjacent=1)
        hits, extras = self._split(results)
        self.assertEqual(hits[0].ref_id, hit.event_id)
        self.assertEqual([item.ref_id for item in extras], [second.event_id, fourth.event_id])
        for item in extras:
            self.assertEqual(item.reasons, ["adjacent-context"])
            self.assertEqual(item.score, 0.0)
        wider = self._search(hit.project_id, "独特标记词", adjacent=2)
        _, wide_extras = self._split(wider)
        self.assertEqual([item.ref_id for item in wide_extras],
                         [first.event_id, second.event_id, fourth.event_id, fifth.event_id])

    def test_adjacent_neighbours_are_deduplicated(self):
        first = self._write("共享甲", "共享锚点词 第一条。", 1)
        second = self._write("填充乙", "完全无关的填充内容乙。", 2)
        third = self._write("共享丙", "共享锚点词 第三条。", 3)
        fourth = self._write("填充丁", "另一段无关填充内容丁。", 4)
        results = self._search(first.project_id, "共享锚点词", adjacent=1)
        identities = [item.ref_id for item in results]
        self.assertEqual(len(identities), len(set(identities)))
        hits, extras = self._split(results)
        self.assertEqual({item.ref_id for item in hits}, {first.event_id, third.event_id})
        self.assertEqual({item.ref_id for item in extras}, {second.event_id, fourth.event_id})

    def test_adjacent_disabled_matches_default_exactly(self):
        hit = self._timeline()[2]
        default = self._search(hit.project_id, "独特标记词")
        disabled = self._search(hit.project_id, "独特标记词", adjacent=0)
        self.assertEqual([item.as_dict() for item in default], [item.as_dict() for item in disabled])
        self.assertTrue(all("adjacent-context" not in item.reasons for item in default))

    def test_adjacent_window_bounds_are_validated(self):
        hit = self._timeline()[2]
        for invalid in (-1, 9, 100):
            with self.subTest(adjacent=invalid), self.assertRaises(ValueError):
                self._search(hit.project_id, "独特标记词", adjacent=invalid)

    def test_adjacent_does_not_bypass_scope_or_status(self):
        first = self._write("边界甲", "公开正文第一条。", 1)
        write_event(self.project, "私密会话", "私密会话正文绝不外泄。", scope="session", scope_id="s1",
                    recorded_at="2026-03-02T00:00:00Z")
        retracted = self._write("已撤回", "撤回正文绝不外泄。", 3)
        forget(self.project, retracted.event_id)
        hit = self._write("目标丁", "哨兵目标词 命中条目。", 4)
        last = self._write("边界戊", "公开正文第五条。", 5)
        results = self._search(hit.project_id, "哨兵目标词", adjacent=2)
        _, extras = self._split(results)
        self.assertEqual({item.ref_id for item in extras}, {first.event_id, last.event_id})
        bodies = " ".join(item.body for item in results)
        self.assertNotIn("私密会话正文绝不外泄", bodies)
        self.assertNotIn("撤回正文绝不外泄", bodies)
        self.assertNotIn(retracted.event_id, [item.ref_id for item in results])

    def test_adjacent_ignored_by_frozen_legacy_profile(self):
        hit = self._timeline()[2]
        plain = self._search(hit.project_id, "独特标记词", profile="legacy")
        widened = self._search(hit.project_id, "独特标记词", profile="legacy", adjacent=2)
        self.assertEqual([item.as_dict() for item in plain], [item.as_dict() for item in widened])
        self.assertTrue(all("adjacent-context" not in item.reasons for item in widened))

    def test_adjacent_is_marked_in_packed_output(self):
        direct = SearchResult("direct-1", "event", "直接命中", "直接命中的正文内容。", "p", 0.5,
                              ["fts5-bm25"], "", "active", "2026-03-01T00:00:00Z", "")
        adjacent = SearchResult("adj-1", "event", "相邻补充", "相邻补充的正文内容。", "p", 0.0,
                                ["adjacent-context"], "", "active", "2026-03-02T00:00:00Z", "")
        packed = pack_context({"project_id": "p"}, Path("/synthetic"), "正文", [direct, adjacent], token_budget=1800)
        adjacent_line = next(row for row in packed.splitlines() if "[event:adj-1]" in row)
        direct_line = next(row for row in packed.splitlines() if "[event:direct-1]" in row)
        self.assertIn("role=adjacent-context", adjacent_line)
        self.assertNotIn("role=adjacent-context", direct_line)

    def test_adjacent_is_omitted_before_any_direct_hit(self):
        direct = SearchResult("direct-1", "event", "直接命中", "直接命中正文。" * 40, "p", 0.5,
                              ["fts5-bm25"], "", "active", "2026-03-01T00:00:00Z", "")
        adjacent = SearchResult("adj-1", "event", "相邻补充", "相邻补充正文。" * 40, "p", 0.0,
                                ["adjacent-context"], "", "active", "2026-03-02T00:00:00Z", "")
        omission_seen = False
        for budget in range(150, 1800, 25):
            for ordered in ([direct, adjacent], [adjacent, direct]):
                packed = pack_context({"project_id": "p"}, Path("/synthetic"), "正文", ordered, token_budget=budget)
                self.assertLessEqual(context_size(packed), budget)
                if "[event:adj-1]" in packed:
                    self.assertIn("[event:direct-1]", packed)
                elif "[event:direct-1]" in packed:
                    omission_seen = True
        self.assertTrue(omission_seen)

    def test_direct_hit_is_filled_before_adjacent_excerpt(self):
        direct = SearchResult("direct-1", "event", "直接命中", "直接命中正文。" * 40, "p", 0.5,
                              ["fts5-bm25"], "", "active", "2026-03-01T00:00:00Z", "")
        adjacent = SearchResult("adj-1", "event", "相邻补充", "相邻补充正文。" * 40, "p", 0.0,
                                ["adjacent-context"], "", "active", "2026-03-02T00:00:00Z", "")
        solo = pack_context({"project_id": "p"}, Path("/synthetic"), "正文", [direct], token_budget=1800)
        budget = context_size(solo) + 300
        packed = pack_context({"project_id": "p"}, Path("/synthetic"), "正文", [direct, adjacent], token_budget=budget)
        self.assertLessEqual(context_size(packed), budget)
        self.assertIn(direct.body, packed)
        self.assertNotIn(adjacent.body, packed)

    def test_passages_profile_adjacent_carries_valid_spans(self):
        first = self._write("记录甲", "时间线第一条内容。", 1)
        hit = self._write("片段目标", "片段哨兵词 目标正文。", 2)
        last = self._write("记录丙", "时间线第三条内容。", 3)
        results = self._search(hit.project_id, "片段哨兵词", profile="passages", adjacent=1)
        hits, extras = self._split(results)
        self.assertEqual(hits[0].ref_id, hit.event_id)
        self.assertEqual({item.ref_id for item in extras}, {first.event_id, last.event_id})
        for item in extras:
            self.assertTrue(item.evidence_spans)
            for span in item.evidence_spans:
                validate_passage(item.body, span, item.source_event_id)
        packed = pack_context({"project_id": hit.project_id}, Path("/synthetic"), "片段哨兵词",
                              results, token_budget=6000)
        self.assertIn("role=adjacent-context", packed)

    def test_cli_adjacent_flag_end_to_end(self):
        first = self._write("命令行甲", "时间线第一条内容。", 1)
        self._write("命令行目标", "检索锚点甲词 目标。", 2)
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["context", "--cwd", str(self.project), "--query", "检索锚点甲词",
                                   "--adjacent", "1"]), 0)
        rendered = output.getvalue()
        self.assertIn("role=adjacent-context", rendered)
        self.assertIn(first.body, rendered)
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["context", "--cwd", str(self.project), "--query", "检索锚点甲词"]), 0)
        self.assertNotIn("role=adjacent-context", output.getvalue())
