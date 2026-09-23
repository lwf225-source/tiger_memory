from __future__ import annotations

import unittest
from pathlib import Path

from tiger_memory.context_pack import context_size, pack_context
from tiger_memory.model import SearchResult


class ContextPackTests(unittest.TestCase):
    project = {"project_id": "project-fixture"}
    cwd = Path("/synthetic/workspace")

    def result(self, identity, body, title="项目证据", status="active"):
        return SearchResult(identity, "event", title, body, "project-fixture", 0.5,
                            ["fixture"], "/synthetic/" + identity + ".md", status,
                            "2026-01-01T00:00:00Z", "")

    def test_whole_context_budget_includes_rules_and_metadata(self):
        results = [self.result("one", "中文证据。" * 800), self.result("two", "第二条证据。" * 100)]
        rules = [("/synthetic/AGENTS.md", "完整规则。" * 200)]
        for budget in (0, 3, 12, 13, 40, 80, 180, 500, 1000, 1800):
            packed = pack_context(self.project, self.cwd, "证据", results, rules, budget, explain=True)
            self.assertLessEqual(context_size(packed), budget)
        full = pack_context(self.project, self.cwd, "证据", results, rules, 1800)
        self.assertIn("not exact tokens", full)
        self.assertIn("not fully loaded; read source", full)
        self.assertNotIn("完整规则。" * 200, full)

    def test_first_long_memory_cannot_consume_second_evidence_slot(self):
        results = [self.result("one", "第一条信息。" * 2000), self.result("two", "第二条说明采用 SQLite。")]
        packed = pack_context(self.project, self.cwd, "存储方案", results, token_budget=900)
        self.assertIn("[event:one]", packed)
        self.assertIn("[event:two]", packed)
        self.assertIn("SQLite", packed)

    def test_late_query_evidence_and_negative_conditions_survive(self):
        body = "背景材料。" * 600 + "\nSQLite 的选择原因是完整离线运行。\n条件：仅本地构建通过，尚未真机验证，不得发布。"
        packed = pack_context(self.project, self.cwd, "SQLite 的选择原因", [self.result("one", body)], token_budget=650)
        self.assertIn("SQLite", packed)
        self.assertIn("离线运行", packed)
        self.assertIn("不得发布", packed)
        self.assertIn("尚未真机验证", packed)
        self.assertIn("status=active", packed)
        self.assertIn("valid=2026-01-01T00:00:00Z..open", packed)
        self.assertIn("source=/synthetic/one.md", packed)

    def test_duplicate_body_keeps_both_source_ids_once(self):
        body = "经核对，采用本地 SQLite。"
        results = [self.result("one", body), self.result("two", body)]
        packed = pack_context(self.project, self.cwd, "存储方案", results, token_budget=900)
        self.assertEqual(packed.count(body), 1)
        self.assertIn("[event:one]", packed)
        self.assertIn("[event:two]", packed)

    def test_conditions_json_is_retained_as_evidence(self):
        body = "SQLite\n\nConditions: {\"scope\": \"仅本地\", \"constraint\": \"禁止联网\"}\n" + "背景说明。" * 200
        packed = pack_context(self.project, self.cwd, "数据库", [self.result("one", body)], token_budget=550)
        self.assertIn("Conditions:", packed)
        self.assertIn("禁止联网", packed)

    def test_small_budget_has_truthful_no_evidence_message(self):
        packed = pack_context(self.project, self.cwd, "", [self.result("one", "正文")], token_budget=80)
        self.assertIn("No evidence", packed)
        self.assertNotIn("[event:", packed)

    def test_short_rule_is_shown_as_complete_separate_source(self):
        packed = pack_context(self.project, self.cwd, "", [], [("/AGENTS.md", "Use Python.")], token_budget=1000)
        self.assertIn("Rule files (separate source documents)", packed)
        self.assertIn("complete source text", packed)
        self.assertIn("Use Python.", packed)
