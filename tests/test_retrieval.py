from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tiger_memory.database import connect
from tiger_memory.providers import LocalCrossEncoderReranker, LocalSentenceTransformerEmbedding, ProviderUnavailable
from tiger_memory.search import _search_legacy, search
from tiger_memory.service import forget, write_event, write_fact
from tests.common import MemoryTestCase


class RecordingEmbedding:
    def __init__(self):
        self.seen = []

    def describe(self):
        return {"provider": "synthetic-test", "model_id": "test-only", "revision": "1", "dimensions": 2}

    def embed(self, texts):
        self.seen.extend(texts)
        return [[1.0, 0.0] if "断网" in text or "离线" in text else [0.0, 1.0] for text in texts]


class RetrievalTests(MemoryTestCase):
    def _search(self, project_id, query, **kwargs):
        conn = connect()
        try:
            return search(conn, project_id, query, **kwargs)
        finally:
            conn.close()

    def test_legacy_profile_reproduces_original_search(self):
        first = write_event(self.project, "微信发布检查", "当前是内部测试版本。")
        write_event(self.project, "项目安排", "下周继续验证微信构建。")
        conn = connect()
        try:
            expected = _search_legacy(conn, first.project_id, "微信发布检查")
            actual = search(conn, first.project_id, "微信发布检查", profile="legacy")
        finally:
            conn.close()
        self.assertEqual([item.as_dict() for item in actual], [item.as_dict() for item in expected])

    def test_chinese_short_words_and_continuous_question(self):
        answer = write_event(self.project, "缓存选择", "使用 SQLite 存储缓存，原因是支持离线运行。")
        write_event(self.project, "角色服饰", "角色穿着紫色披风。")
        short = self._search(answer.project_id, "缓存")
        question = self._search(answer.project_id, "我们为什么选择离线运行的缓存方案")
        self.assertEqual(short[0].ref_id, answer.event_id)
        self.assertIn("scoped-substring", short[0].reasons)
        self.assertEqual(question[0].ref_id, answer.event_id)

    def test_self_reported_confidence_cannot_improve_new_ranking(self):
        event = write_fact(self.project, "cache.engine", "SQLite", confidence=0.1)
        before = self._search(event.project_id, "cache.engine")[0].score
        old_before = self._search(event.project_id, "cache.engine", profile="legacy")[0].score
        conn = connect()
        conn.execute("UPDATE facts SET confidence=1.0 WHERE fact_id=?", (event.metadata["fact_id"],))
        conn.commit()
        conn.close()
        self.assertEqual(self._search(event.project_id, "cache.engine")[0].score, before)
        self.assertGreater(self._search(event.project_id, "cache.engine", profile="legacy")[0].score, old_before)

    def test_future_successor_keeps_effective_predecessor_and_conditions(self):
        original = write_fact(self.project, "cache.engine", "SQLite", valid_from="2020-01-01T00:00:00Z")
        successor = write_fact(
            self.project, "cache.engine", "NextCache", supersedes=original.metadata["fact_id"],
            valid_from="2099-01-01T00:00:00Z", conditions={"only_after": "验收完成"},
        )
        current = self._search(original.project_id, "cache.engine")
        self.assertEqual(current[0].ref_id, original.metadata["fact_id"])
        self.assertEqual(current[0].status, "superseded")
        self.assertEqual(current[0].valid_to, successor.metadata["valid_from"])
        self.assertNotIn(successor.metadata["fact_id"], [item.ref_id for item in current])
        historical = self._search(original.project_id, "cache.engine", view="history")
        new_result = next(item for item in historical if item.ref_id == successor.metadata["fact_id"])
        self.assertIn("验收完成", new_result.body)

    def test_models_only_receive_eligible_scope_and_unretracted_events(self):
        public = write_event(self.project, "运行选择", "完全离线运行")
        write_event(self.project, "私有指令", "私有会话正文", scope="session", scope_id="one",
                    extra={"entities": [{"name": "私有实体名称", "aliases": ["离线"]}]})
        other = self.project.parent / "另一项目"
        other.mkdir()
        write_event(other, "运行选择", "其他项目正文")
        retracted = write_event(self.project, "旧选择", "已被撤回正文")
        forget(self.project, retracted.event_id)
        provider = RecordingEmbedding()
        results = self._search(public.project_id, "离线", embedding_provider=provider)
        self.assertEqual(results[0].ref_id, public.event_id)
        combined = "\n".join(provider.seen)
        for forbidden in ("私有会话正文", "私有实体名称", "其他项目正文", "已被撤回正文"):
            self.assertNotIn(forbidden, combined)
        self.assertNotIn(str(public.path), combined)

    def test_future_expired_source_hides_derived_memories_before_model(self):
        original = write_fact(self.project, "cache.engine", "旧缓存依据", valid_from="2020-01-01T00:00:00Z")
        derived = write_event(self.project, "缓存说明", "从旧缓存依据派生的说明", extra={"source_event_id": original.event_id})
        derived_fact = write_fact(self.project, "cache.reason", "旧缓存依据派生事实必须随依据失效", source_event_id=original.event_id)
        successor = write_fact(self.project, "cache.engine", "新缓存依据", supersedes=original.metadata["fact_id"], valid_from="2099-01-01T00:00:00Z")
        with patch("tiger_memory.search.utc_now", return_value="2098-01-01T00:00:00.000000Z"):
            before = self._search(original.project_id, "缓存依据", limit=10)
        self.assertIn(derived.event_id, [item.ref_id for item in before])
        self.assertIn(derived_fact.metadata["fact_id"], [item.ref_id for item in before])
        provider = RecordingEmbedding()
        with patch("tiger_memory.search.utc_now", return_value="2100-01-01T00:00:00.000000Z"):
            after = self._search(original.project_id, "缓存依据", limit=10, embedding_provider=provider)
        identities = [item.ref_id for item in after]
        self.assertIn(successor.metadata["fact_id"], identities)
        self.assertNotIn(derived.event_id, identities)
        self.assertNotIn(derived_fact.metadata["fact_id"], identities)
        self.assertNotIn("从旧缓存依据派生的说明", "\n".join(provider.seen))
        self.assertNotIn("派生事实必须随依据失效", "\n".join(provider.seen))

    def test_cyclic_or_cross_scope_source_is_rejected_before_model(self):
        first = write_event(self.project, "来源一", "循环引用正文一")
        second = write_event(self.project, "来源二", "循环引用正文二")
        private = write_event(self.project, "私有来源", "私有来源正文", scope="session", scope_id="one")
        crossing = write_event(self.project, "越界引用", "越界派生正文")
        conn = connect()
        conn.execute("UPDATE events SET source_event_id=? WHERE event_id=?", (second.event_id, first.event_id))
        conn.execute("UPDATE events SET source_event_id=? WHERE event_id=?", (first.event_id, second.event_id))
        conn.execute("UPDATE events SET source_event_id=? WHERE event_id=?", (private.event_id, crossing.event_id))
        conn.commit()
        conn.close()
        provider = RecordingEmbedding()
        results = self._search(first.project_id, "引用正文", embedding_provider=provider)
        self.assertEqual(results, [])
        self.assertEqual(provider.seen, [])

    def test_semantic_channel_can_recall_without_shared_words(self):
        answer = write_event(self.project, "部署边界", "完全离线运行")
        provider = RecordingEmbedding()
        info = {}
        results = self._search(answer.project_id, "断网可以工作吗", embedding_provider=provider, diagnostics=info)
        self.assertEqual(results[0].ref_id, answer.event_id)
        self.assertTrue(any(reason.startswith("local-semantic-vector") for reason in results[0].reasons))
        self.assertEqual(info["embedding"]["status"], "ready")

    def test_missing_model_is_explicit_and_never_imported_or_downloaded(self):
        event = write_event(self.project, "缓存选择", "使用 SQLite 缓存")
        provider = LocalSentenceTransformerEmbedding(Path(self.temp.name) / "not-downloaded", model_id="example/model")
        info = {}
        with patch("tiger_memory.providers.importlib.import_module") as importing:
            results = self._search(event.project_id, "缓存", embedding_provider=provider, diagnostics=info)
            importing.assert_not_called()
        self.assertEqual(results[0].ref_id, event.event_id)
        self.assertEqual(info["embedding"]["fallback_reason"], "missing-local-model")
        self.assertIn("embedding-fallback:missing-local-model", results[0].reasons)

    def test_invalid_vectors_fall_back_without_mixing_dimensions(self):
        event = write_event(self.project, "缓存选择", "使用 SQLite 缓存")
        provider = RecordingEmbedding()
        provider.embed = lambda texts: [[1.0, 0.0]] + [[1.0] for _ in texts[1:]]
        info = {}
        results = self._search(event.project_id, "缓存", embedding_provider=provider, diagnostics=info)
        self.assertEqual(results[0].ref_id, event.event_id)
        self.assertEqual(info["embedding"]["fallback_reason"], "embedding-dimension-mismatch")
        self.assertFalse(any(reason.startswith("local-semantic-vector") for reason in results[0].reasons))

    def test_reranker_reorders_only_scoped_candidates_and_safe_failure(self):
        first = write_event(self.project, "缓存选择甲", "SQLite 缓存的旧解释")
        second = write_event(self.project, "缓存选择乙", "SQLite 缓存的关键依据")
        write_event(self.project, "缓存选择私有", "私有会话关键依据", scope="session", scope_id="one")
        seen = []

        def score(query, passages):
            seen.extend(passages)
            return [10.0 if "关键依据" in passage else 0.0 for passage in passages]

        reranker = SimpleNamespace(score=score)
        results = self._search(first.project_id, "缓存", reranker=reranker)
        self.assertEqual(results[0].ref_id, second.event_id)
        self.assertNotIn("私有会话", "\n".join(seen))
        reranker.score = lambda query, passages: [float("nan")] * len(passages)
        info = {}
        fallback = self._search(first.project_id, "缓存", reranker=reranker, diagnostics=info)
        self.assertTrue(fallback)
        self.assertEqual(info["reranker"]["fallback_reason"], "invalid-reranker-output")


class ProviderTests(MemoryTestCase):
    def _local_model(self):
        directory = Path(self.temp.name) / "local-model"
        directory.mkdir()
        (directory / "config.json").write_text('{"synthetic": true}', encoding="utf-8")
        return directory

    def test_offline_adapter_cache_is_bound_to_artifacts_and_revision(self):
        model_path = self._local_model()
        calls = []
        constructors = []

        class FakeModel:
            def __init__(self, path, **kwargs):
                constructors.append(kwargs)

            def get_sentence_embedding_dimension(self):
                return 2

            def encode(self, texts, **kwargs):
                calls.append(list(texts))
                return [[1.0, 0.0] for _ in texts]

        runtime = SimpleNamespace(SentenceTransformer=FakeModel, __version__="synthetic-test")
        cache = Path(self.temp.name) / "numeric-cache"
        with patch("tiger_memory.providers.importlib.import_module", return_value=runtime):
            provider = LocalSentenceTransformerEmbedding(model_path, "fixture", "rev1", cache)
            self.assertEqual(provider.embed(["不应该写入缓存的明文"]), [[1.0, 0.0]])
            provider.embed(["不应该写入缓存的明文"])
            same = LocalSentenceTransformerEmbedding(model_path, "fixture", "rev1", cache)
            same.embed(["不应该写入缓存的明文"])
            self.assertEqual(len(calls), 1)
            changed_revision = LocalSentenceTransformerEmbedding(model_path, "fixture", "rev2", cache)
            changed_revision.embed(["不应该写入缓存的明文"])
            self.assertEqual(len(calls), 2)
            original_stat = (model_path / "config.json").stat()
            (model_path / "config.json").write_text('{"synthetic": null}', encoding="utf-8")
            os.utime(model_path / "config.json", ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
            changed_weights = LocalSentenceTransformerEmbedding(model_path, "fixture", "rev1", cache)
            changed_weights.embed(["不应该写入缓存的明文"])
            self.assertEqual(len(calls), 3)
            self.assertNotEqual(provider.describe()["fingerprint"], changed_weights.describe()["fingerprint"])
        for kwargs in constructors:
            self.assertTrue(kwargs["local_files_only"])
            self.assertFalse(kwargs["trust_remote_code"])
            self.assertFalse(kwargs["token"])
        cache_contents = "".join(path.read_text(encoding="utf-8") for path in cache.rglob("*.json"))
        self.assertNotIn("不应该写入缓存的明文", cache_contents)
        self.assertTrue(all(path.stat().st_mode & 0o777 == 0o600 for path in cache.rglob("*.json")))

    def test_reranker_adapter_caches_numeric_pair_outputs(self):
        model_path = self._local_model()
        calls = []

        class FakeCrossEncoder:
            def __init__(self, path, **kwargs):
                self.assert_offline = kwargs["local_files_only"] and not kwargs["trust_remote_code"]

            def predict(self, pairs, **kwargs):
                calls.append(list(pairs))
                return [0.7 for _ in pairs]

        runtime = SimpleNamespace(CrossEncoder=FakeCrossEncoder, __version__="synthetic-test")
        with patch("tiger_memory.providers.importlib.import_module", return_value=runtime):
            reranker = LocalCrossEncoderReranker(model_path, "fixture", "rev1", Path(self.temp.name) / "cache")
            self.assertEqual(reranker.score("查询", ["候选一", "候选二"]), [0.7, 0.7])
            reranker.score("查询", ["候选一", "候选二"])
            self.assertEqual(len(calls), 1)
            reranker.score("不同查询", ["候选一"])
            self.assertEqual(len(calls), 2)

    def test_query_and_document_prefixes_are_explicit(self):
        provider = LocalSentenceTransformerEmbedding(Path(self.temp.name), query_prefix="query: ", document_prefix="passage: ")
        seen = []
        provider.embed = lambda texts: seen.extend(texts) or [[1.0] for _ in texts]
        provider.embed_query("中文问题")
        provider.embed_documents(["中文记忆"])
        self.assertEqual(seen, ["query: 中文问题", "passage: 中文记忆"])
        self.assertEqual(provider.describe()["settings"]["query_prefix"], "query: ")
