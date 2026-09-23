"""Candidate AML fact-projection tests: provenance, deduplication, and fallback."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch
import uuid
from pathlib import Path

from tiger_memory import aml_adapter, service
from tiger_memory.fact_provider import LLMFactProjector
from tiger_memory.providers import ProviderUnavailable
from tests.common import MemoryTestCase


class _FakeProjector:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = 0

    def project(self, content, role, observed_at):
        self.calls += 1
        if self.fail:
            raise ProviderUnavailable("test failure")
        return [{"key": "preferred infusion", "value": "The user's preferred infusion is high-mountain oolong.",
                 "confidence": 0.9}]


class ProjectionCase(MemoryTestCase):
    def setUp(self):
        super().setUp()
        self.saved_switch = aml_adapter.AML_FACT_PROJECTION
        self.saved_singletons = dict(aml_adapter._MODEL_SINGLETONS)
        aml_adapter.AML_FACT_PROJECTION = "llm"
        aml_adapter._MODEL_SINGLETONS.clear()
        self.user_id = "projection-user-" + uuid.uuid4().hex

    def tearDown(self):
        aml_adapter.AML_FACT_PROJECTION = self.saved_switch
        aml_adapter._MODEL_SINGLETONS.clear()
        aml_adapter._MODEL_SINGLETONS.update(self.saved_singletons)
        super().tearDown()

    def add(self, request_id="r1"):
        return aml_adapter.add({
            "request_id": request_id, "user_id": self.user_id, "session_id": "s",
            "messages": [{"role": "user", "timestamp": 1704067200000,
                          "content": "My favorite tea is high-mountain oolong from Alishan."}],
        })

    def test_fact_hit_returns_original_message_and_event_id(self):
        fake = _FakeProjector()
        aml_adapter._MODEL_SINGLETONS["fact_projection"] = fake
        self.assertTrue(self.add()["success"])
        body = aml_adapter.search({"query": "preferred infusion", "user_id": self.user_id, "top_k": 1})
        self.assertEqual(len(body["data"]), 1)
        self.assertIn("high-mountain oolong", body["data"][0]["content"])
        self.assertNotIn("preferred infusion", body["data"][0]["content"].lower())
        self.assertTrue(body["data"][0]["id"])
        root = aml_adapter.user_storage_dir(self.user_id)
        _project, hits = service.context_data(root, "preferred infusion", limit=5, scope="session", scope_id=self.user_id)
        self.assertEqual(hits[0].record_type, "fact")
        self.assertEqual(hits[0].source_event_id, body["data"][0]["id"])

    def test_replay_does_not_duplicate_projected_fact(self):
        fake = _FakeProjector()
        aml_adapter._MODEL_SINGLETONS["fact_projection"] = fake
        self.add("same")
        self.add("same")
        root = aml_adapter.user_storage_dir(self.user_id)
        _project, hits = service.context_data(root, "preferred infusion", limit=10, scope="session", scope_id=self.user_id)
        self.assertEqual([hit.record_type for hit in hits].count("fact"), 1)

    def test_projection_failure_keeps_add_and_original_search_available(self):
        aml_adapter._MODEL_SINGLETONS["fact_projection"] = _FakeProjector(fail=True)
        self.assertTrue(self.add()["success"])
        body = aml_adapter.search({"query": "high-mountain oolong", "user_id": self.user_id, "top_k": 5})
        self.assertEqual(len(body["data"]), 1)
        self.assertIn("high-mountain oolong", body["data"][0]["content"])


class DefaultProjectionTests(MemoryTestCase):
    def test_default_add_and_search_never_instantiate_projection_provider(self):
        saved_switch = aml_adapter.AML_FACT_PROJECTION
        saved_singletons = dict(aml_adapter._MODEL_SINGLETONS)
        saved_key = os.environ.get("OPENAI_API_KEY")
        try:
            # A present process secret must be irrelevant to the frozen path.
            os.environ["OPENAI_API_KEY"] = "must-not-be-read"
            aml_adapter.AML_FACT_PROJECTION = "off"
            aml_adapter._MODEL_SINGLETONS.clear()
            with patch("tiger_memory.fact_provider.LLMFactProjector",
                       side_effect=AssertionError("default AML path invoked fact projector")):
                user_id = "default-projection-user-" + uuid.uuid4().hex
                response = aml_adapter.add({
                    "request_id": "default-r1", "user_id": user_id, "session_id": "s",
                    "messages": [{"role": "user", "timestamp": 1704067200000,
                                  "content": "The baseline has no external model dependency."}],
                })
                self.assertTrue(response["success"])
                result = aml_adapter.search({
                    "query": "external model dependency", "user_id": user_id, "top_k": 1,
                })
                self.assertEqual(len(result["data"]), 1)
        finally:
            aml_adapter.AML_FACT_PROJECTION = saved_switch
            aml_adapter._MODEL_SINGLETONS.clear()
            aml_adapter._MODEL_SINGLETONS.update(saved_singletons)
            if saved_key is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = saved_key


class FactProviderTests(unittest.TestCase):
    def test_response_is_bounded_cached_and_never_needs_a_second_http_call(self):
        with tempfile.TemporaryDirectory() as temporary:
            provider = LLMFactProjector(api_key="test", base_url="https://example.invalid", cache_dir=Path(temporary))
            calls = []

            def response(url, payload):
                calls.append((url, payload))
                return {"choices": [{"message": {"content": json.dumps({"facts": [
                    {"key": "tea preference", "value": "Prefers oolong", "confidence": 2},
                    {"key": "", "value": "ignored"},
                ]})}}]}

            provider._http_post = response
            first = provider.project("I prefer oolong", "user", "2024-01-01T00:00:00Z")
            second = provider.project("I prefer oolong", "user", "2024-01-01T00:00:00Z")
            self.assertEqual(first, second)
            self.assertEqual(len(calls), 1)
            self.assertEqual(first[0]["confidence"], 1.0)
            self.assertEqual(provider.cache_hits, 1)

    def test_official_base_url_is_used_when_only_an_api_key_is_supplied(self):
        provider = LLMFactProjector(api_key="test")
        self.assertTrue(provider.describe()["configured"])
        self.assertEqual(provider.base_url, "https://api.openai.com")


if __name__ == "__main__":
    unittest.main()
