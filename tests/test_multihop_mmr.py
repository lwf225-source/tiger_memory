from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

from tiger_memory.cli import main
from tiger_memory.context_pack import pack_context
from tiger_memory.database import connect
from tiger_memory.passages import validate_passage
from tiger_memory.search import MAX_ENTITY_EXPANSION, MMR_LAMBDA, search
from tiger_memory.service import forget, write_event
from tests.common import MemoryTestCase


def _entity(name, entity_type="place"):
    return {"name": name, "type": entity_type, "relation": "mentions", "aliases": []}


class _FakeEmbedding:
    """Deterministic stand-in embedding provider; no model runtime involved."""

    def describe(self):
        return {"provider": "fake-embedding", "status": "ready", "dimensions": 2,
                "model_id": "fake", "artifact_fingerprint": "fake-hash"}

    def embed(self, texts):
        return [
            [1.0, 0.0] if "手机" in text else [0.0, 1.0]
            for text in texts
        ]


class EntityExpansionTests(MemoryTestCase):
    def _write(self, title, body, day, entities=None, **kwargs):
        extra = {"entities": entities} if entities else None
        return write_event(self.project, title, body, recorded_at="2026-03-%02dT00:00:00Z" % day,
                           extra=extra, **kwargs)

    def _search(self, project_id, query, **kwargs):
        conn = connect()
        try:
            return search(conn, project_id, query, **kwargs)
        finally:
            conn.close()

    @staticmethod
    def _split(results):
        hits = [item for item in results if not any(r.startswith("entity-expansion") for r in item.reasons)]
        extras = [item for item in results if any(r.startswith("entity-expansion") for r in item.reasons)]
        return hits, extras

    def test_expansion_disabled_matches_default_exactly(self):
        self._write("目标记录", "哨兵词 出现在这里。", 1, entities=[_entity("阿尔卑斯")])
        self._write("关联记录", "完全不相关词面。", 2, entities=[_entity("阿尔卑斯")])
        default = self._search(self._project_id(), "哨兵词")
        disabled = self._search(self._project_id(), "哨兵词", expand_entities=0, mmr=False)
        self.assertEqual([item.as_dict() for item in default], [item.as_dict() for item in disabled])
        self.assertTrue(all(not any(r.startswith("entity-expansion") for r in item.reasons) for item in default))

    def _project_id(self):
        conn = connect()
        try:
            return conn.execute("SELECT project_id FROM projects").fetchone()[0]
        finally:
            conn.close()

    def test_expansion_adds_shared_entity_records(self):
        hit = self._write("目标记录", "哨兵词甲 出现在这里。", 1, entities=[_entity("阿尔卑斯")])
        linked = self._write("关联记录", "另一段没有查询词的记忆。", 2, entities=[_entity("阿尔卑斯")])
        noise = self._write("无关记录", "第三段同样没有查询词。", 3, entities=[_entity("安第斯")])
        results = self._search(hit.project_id, "哨兵词甲", expand_entities=1)
        hits, extras = self._split(results)
        self.assertEqual([item.ref_id for item in hits], [hit.event_id])
        self.assertEqual([item.ref_id for item in extras], [linked.event_id])
        self.assertEqual(extras[0].reasons, ["entity-expansion:hop1"])
        self.assertEqual(extras[0].score, 0.0)
        self.assertNotIn(noise.event_id, [item.ref_id for item in results])

    def test_expansion_two_hops_reaches_second_ring(self):
        first = self._write("一跳起点", "哨兵词乙 目标。", 1, entities=[_entity("甲实体", "concept")])
        middle = self._write("一跳中继", "中继正文无查询词。", 2,
                             entities=[_entity("甲实体", "concept"), _entity("乙实体", "concept")])
        far = self._write("二跳远端", "远端正文也无查询词。", 3, entities=[_entity("乙实体", "concept")])
        one_hop = self._search(first.project_id, "哨兵词乙", expand_entities=1)
        _, extras1 = self._split(one_hop)
        self.assertEqual({item.ref_id for item in extras1}, {middle.event_id})
        two_hop = self._search(first.project_id, "哨兵词乙", expand_entities=2)
        _, extras2 = self._split(two_hop)
        self.assertEqual({item.ref_id for item in extras2}, {middle.event_id, far.event_id})
        by_id = {item.ref_id: item for item in extras2}
        self.assertEqual(by_id[middle.event_id].reasons, ["entity-expansion:hop1"])
        self.assertEqual(by_id[far.event_id].reasons, ["entity-expansion:hop2"])

    def test_expansion_is_bounded_by_cap(self):
        hit = self._write("目标记录", "哨兵词丙 目标。", 1, entities=[_entity("枢纽实体", "concept")])
        for index in range(MAX_ENTITY_EXPANSION + 4):
            self._write("邻居%d" % index, "邻居正文 %d 无查询词。" % index, 2 + index,
                        entities=[_entity("枢纽实体", "concept")])
        results = self._search(hit.project_id, "哨兵词丙", expand_entities=2)
        _, extras = self._split(results)
        self.assertEqual(len(extras), MAX_ENTITY_EXPANSION)

    def test_expansion_does_not_bypass_scope_or_status(self):
        hit = self._write("目标记录", "哨兵词丁 目标。", 1, entities=[_entity("边界实体", "concept")])
        write_event(self.project, "私密会话", "私密扩展正文绝不外泄。", scope="session", scope_id="s1",
                    recorded_at="2026-03-02T00:00:00Z",
                    extra={"entities": [_entity("边界实体", "concept")]})
        retracted = self._write("已撤回", "撤回扩展正文绝不外泄。", 3, entities=[_entity("边界实体", "concept")])
        forget(self.project, retracted.event_id)
        results = self._search(hit.project_id, "哨兵词丁", expand_entities=2)
        bodies = " ".join(item.body for item in results)
        self.assertNotIn("私密扩展正文绝不外泄", bodies)
        self.assertNotIn("撤回扩展正文绝不外泄", bodies)
        self.assertEqual([item.ref_id for item in results], [hit.event_id])

    def test_expansion_deduplicates_against_adjacent(self):
        hit = self._write("目标记录", "哨兵词戊 目标。", 2, entities=[_entity("重叠实体", "concept")])
        neighbour = self._write("相邻重叠", "相邻正文无查询词。", 3, entities=[_entity("重叠实体", "concept")])
        results = self._search(hit.project_id, "哨兵词戊", adjacent=1, expand_entities=1)
        identities = [item.ref_id for item in results]
        self.assertEqual(len(identities), len(set(identities)))
        by_id = {item.ref_id: item for item in results}
        self.assertEqual(by_id[neighbour.event_id].reasons, ["adjacent-context"])

    def test_expansion_ignored_by_frozen_legacy_profile(self):
        hit = self._write("目标记录", "哨兵词己 目标。", 1, entities=[_entity("遗留实体", "concept")])
        self._write("关联记录", "关联正文无查询词。", 2, entities=[_entity("遗留实体", "concept")])
        plain = self._search(hit.project_id, "哨兵词己", profile="legacy")
        widened = self._search(hit.project_id, "哨兵词己", profile="legacy", expand_entities=2, mmr=True)
        self.assertEqual([item.as_dict() for item in plain], [item.as_dict() for item in widened])
        diagnostics = {}
        self._search(hit.project_id, "哨兵词己", profile="legacy", expand_entities=2, mmr=True,
                     diagnostics=diagnostics)
        self.assertEqual(diagnostics["entity_expansion"], {"status": "ignored-by-legacy-profile"})
        self.assertEqual(diagnostics["mmr"], {"status": "ignored-by-legacy-profile"})

    def test_expansion_diagnostics_reported(self):
        hit = self._write("目标记录", "哨兵词庚 目标。", 1, entities=[_entity("诊断实体", "concept")])
        self._write("盆栽记录", "隔壁房间里的盆栽长得很好。", 2, entities=[_entity("诊断实体", "concept")])
        diagnostics = {}
        self._search(hit.project_id, "哨兵词庚", expand_entities=2, diagnostics=diagnostics)
        self.assertEqual(diagnostics["entity_expansion"]["status"], "ready")
        self.assertEqual(diagnostics["entity_expansion"]["hops"], 2)
        self.assertEqual(diagnostics["entity_expansion"]["added"], 1)
        diagnostics_off = {}
        self._search(hit.project_id, "哨兵词庚", diagnostics=diagnostics_off)
        self.assertEqual(diagnostics_off["entity_expansion"], {"status": "disabled"})

    def test_expansion_flag_is_validated(self):
        hit = self._write("目标记录", "哨兵词辛 目标。", 1)
        for invalid in (-1, 3, 100, True, "1"):
            with self.subTest(expand_entities=invalid), self.assertRaises(ValueError):
                self._search(hit.project_id, "哨兵词辛", expand_entities=invalid)

    def test_passages_profile_expansion_carries_valid_spans(self):
        hit = self._write("片段目标", "片段哨兵词 目标正文。", 1, entities=[_entity("片段实体", "concept")])
        self._write("盆栽记录", "隔壁房间里的盆栽长得很好。", 2, entities=[_entity("片段实体", "concept")])
        results = self._search(hit.project_id, "片段哨兵词", profile="passages", expand_entities=1)
        _, extras = self._split(results)
        self.assertEqual(len(extras), 1)
        for item in extras:
            self.assertTrue(item.evidence_spans)
            for span in item.evidence_spans:
                validate_passage(item.body, span, item.source_event_id)
        packed = pack_context({"project_id": hit.project_id}, Path("/synthetic"), "片段哨兵词",
                              results, token_budget=6000)
        self.assertIn("role=entity-expansion", packed)


class MMRTests(MemoryTestCase):
    def _write(self, title, body, day, **kwargs):
        return write_event(self.project, title, body, recorded_at="2026-03-%02dT00:00:00Z" % day, **kwargs)

    def _search(self, project_id, query, **kwargs):
        conn = connect()
        try:
            return search(conn, project_id, query, **kwargs)
        finally:
            conn.close()

    def _duplicate_fixture(self):
        first = self._write("发布会快讯一", "苹果发布会 宣布了新款手机与手表 细节丰富。", 1)
        second = self._write("发布会快讯二", "苹果发布会 宣布了新款手机与手表 细节十分丰富。", 2)
        diverse = self._write("农场游记", "苹果园采摘季 之后的发布会见闻。", 3)
        return first, second, diverse

    def test_mmr_disabled_matches_default_exactly(self):
        first, _, _ = self._duplicate_fixture()
        default = self._search(first.project_id, "苹果发布会", limit=3)
        disabled = self._search(first.project_id, "苹果发布会", limit=3, mmr=False)
        self.assertEqual([item.as_dict() for item in default], [item.as_dict() for item in disabled])

    def test_mmr_prefers_complementary_over_near_duplicate(self):
        first, second, diverse = self._duplicate_fixture()
        baseline = self._search(first.project_id, "苹果发布会", limit=2)
        self.assertEqual({item.ref_id for item in baseline}, {first.event_id, second.event_id})
        reranked = self._search(first.project_id, "苹果发布会", limit=2, mmr=True)
        self.assertEqual(reranked[0].ref_id, baseline[0].ref_id)
        self.assertIn(diverse.event_id, [item.ref_id for item in reranked])
        self.assertNotEqual({item.ref_id for item in reranked}, {first.event_id, second.event_id})

    def test_mmr_diagnostics_reported(self):
        first, _, _ = self._duplicate_fixture()
        diagnostics = {}
        self._search(first.project_id, "苹果发布会", limit=2, mmr=True, diagnostics=diagnostics)
        self.assertEqual(diagnostics["mmr"]["status"], "ready")
        self.assertEqual(diagnostics["mmr"]["lambda"], MMR_LAMBDA)
        self.assertEqual(diagnostics["mmr"]["space"], "lexical-ngram-512")
        diagnostics_off = {}
        self._search(first.project_id, "苹果发布会", diagnostics=diagnostics_off)
        self.assertEqual(diagnostics_off["mmr"], {"status": "disabled"})

    def test_mmr_reuses_semantic_vectors_when_embedding_channel_ran(self):
        first, _, diverse = self._duplicate_fixture()
        diagnostics = {}
        reranked = self._search(first.project_id, "苹果发布会", limit=2, mmr=True,
                                embedding_provider=_FakeEmbedding(), diagnostics=diagnostics)
        self.assertEqual(diagnostics["mmr"]["space"], "semantic-reuse")
        self.assertIn(diverse.event_id, [item.ref_id for item in reranked])

    def test_mmr_flag_is_validated(self):
        hit = self._write("目标记录", "校验哨兵词 目标。", 1)
        for invalid in ("yes", 1, 0):
            with self.subTest(mmr=invalid), self.assertRaises(ValueError):
                self._search(hit.project_id, "校验哨兵词", mmr=invalid)

    def test_mmr_skipped_for_passages_profile(self):
        hit = self._write("片段目标", "片段校验词 目标正文。", 1)
        diagnostics = {}
        results = self._search(hit.project_id, "片段校验词", profile="passages", mmr=True,
                               diagnostics=diagnostics)
        self.assertEqual(diagnostics["mmr"], {"status": "skipped-passages-profile"})
        self.assertTrue(results)


class CombinedSwitchTests(MemoryTestCase):
    def _write(self, title, body, day, entities=None, **kwargs):
        extra = {"entities": entities} if entities else None
        return write_event(self.project, title, body, recorded_at="2026-03-%02dT00:00:00Z" % day,
                           extra=extra, **kwargs)

    def _search(self, project_id, query, **kwargs):
        conn = connect()
        try:
            return search(conn, project_id, query, **kwargs)
        finally:
            conn.close()

    def test_all_switches_combine_without_cross_contamination(self):
        first = self._write("时间线甲", "普通正文第一条。", 1)
        hit = self._write("组合目标", "组合哨兵词 目标正文。", 2, entities=[_entity("组合实体", "concept")])
        filler = self._write("填充记录", "完全无关的填充内容。", 3)
        linked = self._write("盆栽记录", "隔壁房间里的盆栽长得很好。", 4,
                             entities=[_entity("组合实体", "concept")])
        diagnostics = {}
        results = self._search(hit.project_id, "组合哨兵词", adjacent=1, recency_boost=True,
                               expand_entities=1, mmr=True, limit=4, diagnostics=diagnostics)
        roles = [(item.ref_id, [r for r in item.reasons if r in ("adjacent-context",) or r.startswith("entity-expansion")])
                 for item in results]
        direct = [ref for ref, marks in roles if not marks]
        adjacent = [ref for ref, marks in roles if "adjacent-context" in marks]
        expanded = [ref for ref, marks in roles if any(m.startswith("entity-expansion") for m in marks)]
        self.assertIn(hit.event_id, direct)
        self.assertEqual(adjacent, [first.event_id, filler.event_id])
        self.assertEqual(expanded, [linked.event_id])
        # Order: direct hits, then entity expansion (inside the window), then
        # adjacent supplements (frozen beyond-window semantics).
        order = [item.ref_id for item in results]
        self.assertLess(order.index(hit.event_id), order.index(linked.event_id))
        self.assertLess(order.index(linked.event_id), order.index(first.event_id))
        self.assertEqual(diagnostics["entity_expansion"]["status"], "ready")
        self.assertEqual(diagnostics["mmr"]["status"], "ready")
        self.assertEqual(diagnostics["adjacent"]["status"], "ready")
        self.assertEqual(diagnostics["recency_boost"]["status"], "enabled")

    def test_cli_expand_entities_and_mmr_end_to_end(self):
        hit = self._write("命令行目标", "检索锚点乙词 目标。", 1, entities=[_entity("命令行实体", "concept")])
        linked = self._write("命令行关联", "命令行关联正文无查询词。", 2, entities=[_entity("命令行实体", "concept")])
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["context", "--cwd", str(self.project), "--query", "检索锚点乙词",
                                   "--expand-entities", "1", "--mmr"]), 0)
        rendered = output.getvalue()
        self.assertIn("role=entity-expansion", rendered)
        self.assertIn(linked.body, rendered)
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["context", "--cwd", str(self.project), "--query", "检索锚点乙词"]), 0)
        plain = output.getvalue()
        self.assertNotIn("role=entity-expansion", plain)
        self.assertNotIn(linked.body, plain)
        self.assertIn(hit.body, plain)
