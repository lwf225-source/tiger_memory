"""Batch-13 LLM query-understanding provider tests: mocked HTTP success,
timeout, malformed responses, missing-credential degradation, disk cache,
and search-layer integration (extra channels, byte-identical fallback)."""

from __future__ import annotations

from contextlib import closing

import json
import os
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from tiger_memory import aml_adapter
from tiger_memory.database import connect
from tiger_memory.fact_provider import LLMFactProjector
from tiger_memory.providers import ProviderUnavailable
from tiger_memory.query_provider import DeterministicRecallQueryProvider, LLMQueryProvider
from tiger_memory.search import search
from tiger_memory.service import write_event
from tests.common import MemoryTestCase


def _chat_payload(content):
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


class _MockHTTPProvider(LLMQueryProvider):
    """Patches the single HTTP seam; no network, no credentials."""

    def __init__(self, responses, **kwargs):
        kwargs.setdefault("api_key", "test-key")
        kwargs.setdefault("base_url", "https://example.invalid")
        super().__init__(**kwargs)
        self._responses = list(responses)
        self.http_calls = 0

    def _http_post(self, url, payload):
        self.http_calls += 1
        if not self._responses:
            raise AssertionError("unexpected extra HTTP call")
        outcome = self._responses.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class ProviderUnitTests(MemoryTestCase):
    def test_deterministic_recall_provider_is_offline_and_first_person(self):
        provider = DeterministicRecallQueryProvider()
        self.assertEqual(provider.describe()["network"], False)
        self.assertEqual(provider.describe()["credentials"], False)
        self.assertEqual((provider.llm_calls, provider.cache_hits, provider.failures,
                          provider.total_latency_ms), (0, 0, 0, 0.0))
        self.assertEqual(provider.rewrite("Where did I go in spring?")["expansions"], [])
        self.assertTrue(provider.rewrite("Where did I go in spring?")["hyde"].startswith("I remember"))

    def test_success_parses_expansions_and_hyde(self):
        provider = _MockHTTPProvider([
            _chat_payload(json.dumps({
                "expansions": ["Where did I travel last spring?", "trip destinations in spring"],
                "hyde": "I remember telling Caroline that I went to Lisbon last spring.",
            })),
        ])
        result = provider.rewrite("Where did the user travel last spring?")
        self.assertEqual(len(result["expansions"]), 2)
        self.assertIn("Lisbon", result["hyde"])
        self.assertEqual(provider.llm_calls, 1)
        self.assertEqual(provider.failures, 0)

    def test_markdown_fenced_json_is_accepted(self):
        provider = _MockHTTPProvider([
            _chat_payload("```json\n" + json.dumps({"expansions": ["alt phrasing"], "hyde": ""}) + "\n```"),
        ])
        result = provider.rewrite("original question?")
        self.assertEqual(result["expansions"], ["alt phrasing"])

    def test_timeout_raises_provider_unavailable(self):
        provider = _MockHTTPProvider([ProviderUnavailable("transport-failed:TimeoutError")])
        with self.assertRaises(ProviderUnavailable):
            provider.rewrite("anything")
        self.assertEqual(provider.failures, 1)

    def test_http_error_raises_provider_unavailable(self):
        provider = _MockHTTPProvider([ProviderUnavailable("http-error:500")])
        with self.assertRaises(ProviderUnavailable):
            provider.rewrite("anything")

    def test_malformed_content_raises_provider_unavailable(self):
        for bad in ("not json at all", json.dumps(["a", "list"]),
                    json.dumps({"expansions": "not-a-list"}),
                    json.dumps({"expansions": [], "hyde": ""}),
                    json.dumps({"unexpected": True})):
            provider = _MockHTTPProvider([_chat_payload(bad)])
            with self.assertRaises(ProviderUnavailable, msg=bad):
                provider.rewrite("anything")

    def test_empty_envelope_raises_provider_unavailable(self):
        provider = _MockHTTPProvider([{"choices": []}])
        with self.assertRaises(ProviderUnavailable):
            provider.rewrite("anything")

    def test_missing_credentials_raise_before_any_http(self):
        provider = LLMQueryProvider(api_key="", base_url="", env_prefix="PLM_TEST_MISSING")
        with self.assertRaises(ProviderUnavailable) as caught:
            provider.rewrite("anything")
        self.assertEqual(str(caught.exception), "missing-credentials")
        self.assertEqual(provider.describe()["status"], "unavailable")

    def test_output_validation_filters_bad_entries(self):
        provider = _MockHTTPProvider([
            _chat_payload(json.dumps({
                "expansions": [
                    "original question?",            # duplicate of the query
                    42,                              # non-string
                    "x" * 500,                       # over-long
                    "  spaced   out   variant  ",    # normalized
                    "second variant",
                    "third variant",
                    "fourth variant (over the cap)",
                ],
                "hyde": "I recall mentioning this to someone.",
            })),
        ])
        result = provider.rewrite("original question?")
        self.assertEqual(result["expansions"],
                         ["spaced out variant", "second variant", "third variant"])

    def test_disk_cache_avoids_second_http_call(self):
        cache_dir = Path(self.temp.name) / "qcache"
        responses = [_chat_payload(json.dumps({"expansions": ["cached variant"], "hyde": ""}))]
        first = _MockHTTPProvider(list(responses), cache_dir=cache_dir)
        self.assertEqual(first.rewrite("cacheable question?")["expansions"], ["cached variant"])
        second = _MockHTTPProvider([], cache_dir=cache_dir)
        self.assertEqual(second.rewrite("cacheable question?")["expansions"], ["cached variant"])
        self.assertEqual(second.http_calls, 0)
        self.assertEqual(second.cache_hits, 1)
        # Cache key covers model + prompt version + modes: a different model misses.
        third = _MockHTTPProvider(list(responses), cache_dir=cache_dir, model="other-model")
        third.rewrite("cacheable question?")
        self.assertEqual(third.http_calls, 1)

    def test_cache_file_contains_no_credentials(self):
        cache_dir = Path(self.temp.name) / "qcache"
        provider = _MockHTTPProvider(
            [_chat_payload(json.dumps({"expansions": ["v"], "hyde": ""}))],
            cache_dir=cache_dir)
        provider.rewrite("secret-adjacent question?")
        for path in cache_dir.glob("*.json"):
            raw = path.read_text(encoding="utf-8")
            self.assertNotIn("test-key", raw)
            self.assertNotIn("example.invalid", raw)


class _StaticRewriteProvider:
    """Deterministic search-layer stand-in; no HTTP at all."""

    def __init__(self, expansions=(), hyde="", fail=False):
        self._expansions = list(expansions)
        self._hyde = hyde
        self._fail = fail
        self.calls = 0

    def describe(self):
        return {"provider": "static-rewrite", "status": "ready"}

    def rewrite(self, query):
        self.calls += 1
        if self._fail:
            raise ProviderUnavailable("transport-failed:TimeoutError")
        return {"expansions": list(self._expansions), "hyde": self._hyde}


class _FakeEmbedding:
    def __init__(self, keywords=("lisbon",)):
        self.keywords = keywords

    def describe(self):
        return {"provider": "fake-embedding", "status": "ready", "dimensions": 2,
                "model_id": "fake", "artifact_fingerprint": "fake-hash"}

    def embed_query(self, query):
        return self.embed([query])[0]

    def embed_documents(self, texts):
        return self.embed(texts)

    def embed(self, texts):
        return [
            [1.0, 0.0] if any(keyword in text.lower() for keyword in self.keywords) else [0.0, 1.0]
            for text in texts
        ]


class SearchIntegrationTests(MemoryTestCase):
    def _write(self, title, body, **kwargs):
        return write_event(self.project, title, body,
                           recorded_at="2026-03-01T00:00:00Z", **kwargs)

    def _search(self, project_id, query, **kwargs):
        conn = connect()
        try:
            return search(conn, project_id, query, **kwargs)
        finally:
            conn.close()

    def test_expansion_channel_recalls_otherwise_missed_record(self):
        target = self._write("旅行记录", "去年春天我去了里斯本，住在阿尔法玛区。")
        self._write("无关记录", "今天讨论了预算表和季度计划。")
        baseline = self._search(target.project_id, "spring travel destination?")
        provider = _StaticRewriteProvider(expansions=["里斯本 春天 旅行"])
        rewritten = self._search(target.project_id, "spring travel destination?",
                                 query_provider=provider)
        self.assertEqual(provider.calls, 1)
        self.assertIn(target.event_id, [item.ref_id for item in rewritten])
        self.assertTrue(any(reason.startswith("query-expansion-0-")
                            for item in rewritten if item.ref_id == target.event_id
                            for reason in item.reasons))

    def test_failing_provider_is_byte_identical_to_baseline(self):
        self._write("记录甲", "苹果 香蕉 橙子。")
        self._write("记录乙", "预算 季度 计划。")
        with closing(connect()) as conn:
            project_id = conn.execute("SELECT project_id FROM projects").fetchone()["project_id"]
        diagnostics_off = {}
        baseline = self._search(project_id, "预算", diagnostics=diagnostics_off)
        diagnostics_fail = {}
        provider = _StaticRewriteProvider(fail=True)
        degraded = self._search(project_id, "预算", query_provider=provider,
                                diagnostics=diagnostics_fail)
        self.assertEqual([item.ref_id for item in baseline],
                         [item.ref_id for item in degraded])
        self.assertEqual([item.score for item in baseline],
                         [item.score for item in degraded])
        stripped = [[r for r in item.reasons if not r.startswith("query-rewrite-fallback:")]
                    for item in degraded]
        self.assertEqual([item.reasons for item in baseline], stripped)
        self.assertEqual(diagnostics_off["query_rewrite"]["status"], "disabled")
        self.assertEqual(diagnostics_fail["query_rewrite"]["status"], "unavailable")
        self.assertIn("transport-failed", diagnostics_fail["query_rewrite"]["fallback_reason"])

    def test_hyde_adds_embedding_channel_only_with_embedding_provider(self):
        target = self._write("Lisbon trip", "I told Caroline I went to Lisbon in spring.")
        self._write("Budget notes", "Quarterly budget spreadsheet review.")
        hyde = "I remember telling Caroline that I went to Lisbon last spring."
        provider = _StaticRewriteProvider(hyde=hyde)
        # Without an embedding provider the HyDE text adds no channel.
        plain = self._search(target.project_id, "Where did I go in spring?",
                             query_provider=provider)
        self.assertFalse(any(reason.startswith("hyde-semantic")
                             for item in plain for reason in item.reasons))
        # With the fake embedding provider the HyDE channel fires and recalls.
        rewritten = self._search(target.project_id, "Where did I go in spring?",
                                 query_provider=provider,
                                 embedding_provider=_FakeEmbedding())
        self.assertEqual(rewritten[0].ref_id, target.event_id)
        self.assertTrue(any(reason.startswith("hyde-semantic")
                            for reason in rewritten[0].reasons))

    def test_legacy_profile_ignores_query_provider(self):
        self._write("legacy记录", "legacy 哨兵正文。")
        with closing(connect()) as conn:
            project_id = conn.execute("SELECT project_id FROM projects").fetchone()["project_id"]
        provider = _StaticRewriteProvider(expansions=["anything else"])
        diagnostics = {}
        results = self._search(project_id, "legacy", profile="legacy",
                               query_provider=provider, diagnostics=diagnostics)
        self.assertEqual(provider.calls, 0)
        self.assertEqual(diagnostics["query_rewrite"]["status"], "ignored-by-legacy-profile")
        self.assertTrue(results)


class AMLAdapterSwitchTests(MemoryTestCase):
    def setUp(self):
        super().setUp()
        self._saved = aml_adapter.AML_QUERY_REWRITE
        self._saved_model = aml_adapter.AML_QUERY_MODEL
        self._saved_env_prefix = aml_adapter.AML_QUERY_ENV_PREFIX
        self._saved_fact_projection = aml_adapter.AML_FACT_PROJECTION
        self._saved_fact_model = aml_adapter.AML_FACT_MODEL
        self._saved_fact_env_prefix = aml_adapter.AML_FACT_ENV_PREFIX
        self._saved_singletons = dict(aml_adapter._MODEL_SINGLETONS)
        # Adapter integration tests may legitimately populate the process-wide
        # deterministic provider.  These configuration tests must construct
        # their requested provider afresh, then restore the outer process.
        aml_adapter._MODEL_SINGLETONS.clear()

    def tearDown(self):
        aml_adapter.AML_QUERY_REWRITE = self._saved
        aml_adapter.AML_QUERY_MODEL = self._saved_model
        aml_adapter.AML_QUERY_ENV_PREFIX = self._saved_env_prefix
        aml_adapter.AML_FACT_PROJECTION = self._saved_fact_projection
        aml_adapter.AML_FACT_MODEL = self._saved_fact_model
        aml_adapter.AML_FACT_ENV_PREFIX = self._saved_fact_env_prefix
        aml_adapter._MODEL_SINGLETONS.clear()
        aml_adapter._MODEL_SINGLETONS.update(self._saved_singletons)
        super().tearDown()

    def test_default_template_serves_contract_without_external_credentials(self):
        self.assertEqual(aml_adapter.AML_QUERY_REWRITE, "recall-template")
        payload = {"request_id": "r1", "user_id": "u-query", "session_id": "s1",
                   "messages": [{"role": "user", "content": "我喜欢登山和摄影。"}]}
        aml_adapter.add(payload)
        result = aml_adapter.search({"query": "登山", "user_id": "u-query", "top_k": 5})
        self.assertIn("data", result)
        self.assertEqual(set(result), {"data"})

    def test_invalid_switch_fails_loudly(self):
        aml_adapter.AML_QUERY_REWRITE = "sometimes"
        with self.assertRaises(ValueError):
            aml_adapter.validate_model_config()

    def test_template_switch_needs_no_credentials(self):
        aml_adapter.AML_QUERY_REWRITE = "recall-template"
        import os
        saved = {key: os.environ.pop(key, None) for key in ("KIMI_API_KEY", "KIMI_BASE_URL")}
        try:
            warnings_out = aml_adapter.validate_model_config()
            self.assertFalse(any("AML_QUERY_REWRITE" in warning for warning in warnings_out))
            provider = aml_adapter._query_provider_singleton()
            self.assertEqual(provider.describe()["provider"], "deterministic-first-person-recall")
            payload = {"request_id": "template-r1", "user_id": "u-query-template", "session_id": "s1",
                       "messages": [{"role": "user", "content": "I took a spring trip to Lisbon."}]}
            aml_adapter.add(payload)
            result = aml_adapter.search({"query": "Where was the spring trip?", "user_id": "u-query-template", "top_k": 5})
            self.assertTrue(result["data"])
            self.assertEqual(set(result), {"data"})
        finally:
            for key, value in saved.items():
                if value is not None:
                    os.environ[key] = value

    def test_openai_namespace_prepares_gpt_query_provider_without_network(self):
        aml_adapter.AML_QUERY_REWRITE = "llm"
        aml_adapter.AML_QUERY_MODEL = "gpt-4o-mini"
        aml_adapter.AML_QUERY_ENV_PREFIX = "OPENAI"
        import os
        saved = {key: os.environ.get(key) for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL")}
        os.environ["OPENAI_API_KEY"] = "test-key"
        os.environ.pop("OPENAI_BASE_URL", None)
        try:
            self.assertFalse(any("AML_QUERY_REWRITE" in warning
                                 for warning in aml_adapter.validate_model_config()))
            provider = aml_adapter._query_provider_singleton()
            self.assertEqual(provider.describe()["model"], "gpt-4o-mini")
            self.assertTrue(provider.describe()["configured"])
            self.assertEqual(provider.base_url, "https://api.openai.com")
            self.assertEqual(provider.llm_calls, 0)
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_gpt_candidate_uses_same_model_for_add_and_search_without_network(self):
        aml_adapter.AML_QUERY_REWRITE = "llm"
        aml_adapter.AML_QUERY_MODEL = "gpt-4o-mini"
        aml_adapter.AML_QUERY_ENV_PREFIX = "OPENAI"
        aml_adapter.AML_FACT_PROJECTION = "llm"
        aml_adapter.AML_FACT_MODEL = "gpt-4o-mini"
        aml_adapter.AML_FACT_ENV_PREFIX = "OPENAI"
        saved = {key: os.environ.get(key) for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL")}
        os.environ["OPENAI_API_KEY"] = "test-key"
        os.environ.pop("OPENAI_BASE_URL", None)
        fact_payloads, query_payloads = [], []

        def fake_fact(_self, _url, payload):
            fact_payloads.append(payload)
            return _chat_payload(json.dumps({"facts": [
                {"key": "tea preference", "value": "The user prefers oolong.", "confidence": .9},
            ]}))

        def fake_query(_self, _url, payload):
            query_payloads.append(payload)
            return _chat_payload(json.dumps({"expansions": ["oolong preference"], "hyde": ""}))

        try:
            with mock.patch.object(LLMFactProjector, "_http_post", fake_fact), \
                    mock.patch.object(LLMQueryProvider, "_http_post", fake_query):
                aml_adapter.add({"request_id": "gpt-r1", "user_id": "gpt-user", "session_id": "s1",
                                 "messages": [{"role": "user", "content": "I prefer oolong tea."}]})
                result = aml_adapter.search({"query": "What tea does the user prefer?", "user_id": "gpt-user", "top_k": 5})
            self.assertTrue(result["data"])
            self.assertTrue(fact_payloads)
            self.assertTrue(query_payloads)
            self.assertEqual({item["model"] for item in fact_payloads + query_payloads}, {"gpt-4o-mini"})
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_llm_switch_without_credentials_degrades_and_serves(self):
        aml_adapter.AML_QUERY_REWRITE = "llm"
        import os
        saved = {k: os.environ.pop(k, None) for k in ("KIMI_API_KEY", "KIMI_BASE_URL")}
        try:
            warnings_out = aml_adapter.validate_model_config()
            self.assertTrue(any("AML_QUERY_REWRITE" in w for w in warnings_out))
            payload = {"request_id": "r2", "user_id": "u-query2", "session_id": "s1",
                       "messages": [{"role": "user", "content": "我喜欢登山和摄影。"}]}
            aml_adapter.add(payload)
            result = aml_adapter.search({"query": "登山", "user_id": "u-query2", "top_k": 5})
            self.assertTrue(result["data"])
            self.assertIn("plm_query_rewrite", result)
        finally:
            for key, value in saved.items():
                if value is not None:
                    os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
