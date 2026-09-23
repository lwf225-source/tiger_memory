from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

from tiger_memory.cli import main
from tiger_memory.context_pack import context_size, pack_context
from tiger_memory.database import connect
from tiger_memory.model import SearchResult
from tiger_memory.search import search
from tiger_memory.service import write_event
from tests.common import MemoryTestCase


def _result(ref_id, body, reasons=None, title=""):
    return SearchResult(ref_id, "event", title or ("标题" + ref_id), body, "p", 0.5,
                        reasons or ["fts5-bm25"], "", "active", "2026-03-01T00:00:00Z", "")


class ExcerptFallbackTests(MemoryTestCase):
    def test_fallback_disabled_matches_default_exactly(self):
        results = [_result("one", "直接命中正文。" * 60), _result("two", "第二条正文内容。" * 60)]
        for budget in range(120, 2000, 40):
            default = pack_context({"project_id": "p"}, Path("/synthetic"), "正文", results, token_budget=budget)
            disabled = pack_context({"project_id": "p"}, Path("/synthetic"), "正文", results,
                                    token_budget=budget, excerpt_fallback=False)
            self.assertEqual(default, disabled)
            self.assertNotIn("role=excerpt", default)

    def test_fallback_keeps_marked_excerpt_for_dropped_hit(self):
        direct = _result("one", "直接命中正文。" * 60)
        second = _result("two", "第二条正文内容。" * 60)
        # The rescue band is narrow by design: a fallback pointer costs the
        # 8-byte floor plus the role marker, versus the usual 24-byte minimum.
        observed = False
        for budget in range(250, 700):
            off = pack_context({"project_id": "p"}, Path("/synthetic"), "正文", [direct, second], token_budget=budget)
            on = pack_context({"project_id": "p"}, Path("/synthetic"), "正文", [direct, second],
                              token_budget=budget, excerpt_fallback=True)
            self.assertLessEqual(context_size(off), budget)
            self.assertLessEqual(context_size(on), budget)
            if "[event:one]" in off:
                self.assertIn("[event:one]", on)  # fallback never loses what fit before
            if "[event:two]" not in off and "[event:two]" in on:
                line = next(row for row in on.splitlines() if "[event:two]" in row)
                self.assertIn("role=excerpt", line)
                self.assertNotIn(second.body, on)  # excerpt, never the whole body
                observed = True
        self.assertTrue(observed)

    def test_direct_hit_falls_back_before_adjacent(self):
        direct = _result("direct-1", "直接命中正文。" * 60)
        adjacent = _result("adj-1", "相邻补充正文。" * 60, reasons=["adjacent-context"])
        direct_only_seen = False
        for budget in range(150, 1800, 25):
            packed = pack_context({"project_id": "p"}, Path("/synthetic"), "正文", [adjacent, direct],
                                  token_budget=budget, excerpt_fallback=True)
            self.assertLessEqual(context_size(packed), budget)
            if "[event:adj-1]" in packed:
                self.assertIn("[event:direct-1]", packed)
                adjacent_line = next(row for row in packed.splitlines() if "[event:adj-1]" in row)
                self.assertIn("role=adjacent-context", adjacent_line)
            elif "[event:direct-1]" in packed:
                direct_only_seen = True
        self.assertTrue(direct_only_seen)

    def test_fallback_excerpt_is_truncated_not_relabelled_complete(self):
        direct = _result("one", "超长正文第一段。" * 80)
        packed = None
        for budget in range(150, 900, 10):
            candidate = pack_context({"project_id": "p"}, Path("/synthetic"), "正文", [direct],
                                     token_budget=budget, excerpt_fallback=True)
            if "[event:one]" in candidate and direct.body not in candidate:
                packed = candidate
                break
        self.assertIsNotNone(packed)
        line = next(row for row in packed.splitlines() if "[event:one]" in row)
        self.assertIn("role=excerpt", line)
        self.assertLessEqual(context_size(packed), budget)

    def test_passage_fallback_packs_marked_excerpt(self):
        event = write_event(self.project, "长文记录", "长文哨兵词 开头。" + "长文填充句子。" * 400,
                            recorded_at="2026-03-06T00:00:00Z")
        conn = connect()
        try:
            results = search(conn, event.project_id, "长文哨兵词", profile="passages")
        finally:
            conn.close()
        self.assertTrue(results and results[0].evidence_spans)
        observed = False
        marker = "[event:%s]" % event.event_id
        for budget in range(400, 9000, 100):
            off = pack_context({"project_id": "p"}, Path("/synthetic"), "长文哨兵词", results, token_budget=budget)
            on = pack_context({"project_id": "p"}, Path("/synthetic"), "长文哨兵词", results,
                              token_budget=budget, excerpt_fallback=True)
            self.assertLessEqual(context_size(off), budget)
            self.assertLessEqual(context_size(on), budget)
            if marker in off:
                continue
            if marker not in on:
                continue  # budget too small for even a marked excerpt pointer
            self.assertIn("role=excerpt-fallback", on)
            line = next(row for row in on.splitlines() if marker in row)
            self.assertIn("role=excerpt", line)
            observed = True
        self.assertTrue(observed)

    def test_fallback_flag_is_validated(self):
        direct = _result("one", "正文内容。")
        for invalid in ("yes", 1, 0):
            with self.subTest(excerpt_fallback=invalid), self.assertRaises(ValueError):
                pack_context({"project_id": "p"}, Path("/synthetic"), "正文", [direct],
                             token_budget=1800, excerpt_fallback=invalid)

    def test_cli_excerpt_fallback_end_to_end(self):
        # Passage packing is the true whole-or-drop path: a long source whose
        # anchor cannot fit is omitted by default and rescued by the fallback.
        event = write_event(self.project, "命令行长文", "命令行哨兵词 开头。" + "填充内容句子。" * 400,
                            recorded_at="2026-03-06T00:00:00Z")
        marker = "[event:%s]" % event.event_id
        rendered = {}
        for name, extra in (("off", []), ("on", ["--excerpt-fallback"])):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                self.assertEqual(main(["context", "--cwd", str(self.project), "--query", "命令行哨兵词",
                                       "--retrieval-profile", "passages", "--token-budget", "900"] + extra), 0)
            rendered[name] = buffer.getvalue()
            self.assertLessEqual(context_size(rendered[name]), 900)
        self.assertNotIn(marker, rendered["off"])
        self.assertIn(marker, rendered["on"])
        self.assertIn("role=excerpt-fallback", rendered["on"])
        self.assertNotIn("role=excerpt", rendered["off"])
        # Switches compose: fallback + adjacent window + recency boost in one call.
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            self.assertEqual(main(["context", "--cwd", str(self.project), "--query", "命令行哨兵词",
                                   "--excerpt-fallback", "--adjacent", "1", "--recency-boost",
                                   "--token-budget", "900"]), 0)
        self.assertLessEqual(context_size(buffer.getvalue()), 900)
