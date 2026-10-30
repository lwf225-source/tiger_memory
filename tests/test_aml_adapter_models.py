"""AML adapter optional-model (plm-aml 1.1.0) tests: hybrid recall channels,
deterministic degradation to the lexical safety floor, and config validation."""

from __future__ import annotations

import http.client
import json
import threading
import unittest
from unittest import mock

from tiger_memory import aml_adapter

from tests.common import MemoryTestCase


class _FakeEmbedding:
    """Deterministic stand-in embedding provider; no model runtime involved."""

    def __init__(self, keywords=("cat", "pet")):
        self.keywords = keywords

    def describe(self):
        return {"provider": "fake-embedding", "status": "ready", "dimensions": 2,
                "model_id": "fake", "artifact_fingerprint": "fake-hash"}

    def embed(self, texts):
        return [
            [1.0, 0.0] if any(keyword in text.lower() for keyword in self.keywords) else [0.0, 1.0]
            for text in texts
        ]


class _FakeReranker:
    def __init__(self, boost_token=None, fail=False):
        self.boost_token = boost_token
        self.fail = fail

    def describe(self):
        return {"provider": "fake-reranker", "status": "ready", "model_id": "fake"}

    def score(self, query, passages):
        if self.fail:
            raise RuntimeError("reranker exploded")
        return [10.0 if self.boost_token and self.boost_token in passage else 1.0
                for passage in passages]


class ModelConfigCase(MemoryTestCase):
    """Restores every touched frozen constant and the provider singletons."""

    def setUp(self):
        super().setUp()
        self._saved = {
            name: getattr(aml_adapter, name)
            for name in (
                "AML_EMBEDDING", "AML_EMBEDDING_MODEL_DIR", "AML_EMBEDDING_MODEL_ID",
                "AML_EMBEDDING_QUERY_PREFIX", "AML_EMBEDDING_DOCUMENT_PREFIX",
                "AML_RERANKER", "AML_RERANKER_MODEL_DIR", "AML_RERANKER_MODEL_ID",
                "AML_SESSION_RRF",
            )
        }
        self._saved_singletons = dict(aml_adapter._MODEL_SINGLETONS)
        # Subclasses may fail while opening their HTTP fixture. unittest does
        # not run tearDown after a failed setUp, but always runs cleanups.
        self.addCleanup(self._restore_model_state)
        aml_adapter._MODEL_SINGLETONS.clear()
        # Model tests isolate reranker/embedding behavior from the separately
        # verified 1.3.0 session-RRF default.
        aml_adapter.AML_SESSION_RRF = False

    def _restore_model_state(self):
        for name, value in self._saved.items():
            setattr(aml_adapter, name, value)
        aml_adapter._MODEL_SINGLETONS.clear()
        aml_adapter._MODEL_SINGLETONS.update(self._saved_singletons)

    def tearDown(self):
        import os
        os.environ.pop("PLM_AML_EMBEDDING_MODEL_DIR", None)
        super().tearDown()

    def _seed(self, user_id, text):
        aml_adapter.add({
            "request_id": "req-1", "user_id": user_id, "session_id": "s",
            "messages": [{"role": "user", "content": text, "timestamp": 1704067200000}],
        })

    def _search(self, query, user_id):
        return aml_adapter.search({"query": query, "user_id": user_id, "top_k": 100})


class TestVersionAndValidation(ModelConfigCase):
    def test_adapter_version_is_1_4_0(self):
        self.assertEqual(aml_adapter.AML_ADAPTER_VERSION, "1.4.0")
        self.assertIn("1.4.0", aml_adapter.AMLRequestHandler.server_version)

    def test_embedding_freeze_defaults(self):
        # The retained BGE hybrid defaults: embedding on, reranker off.
        self.assertEqual(aml_adapter.AML_EMBEDDING, "local")
        self.assertEqual(aml_adapter.AML_EMBEDDING_MODEL_ID, "BAAI/bge-small-en-v1.5")
        self.assertEqual(aml_adapter.AML_EMBEDDING_QUERY_PREFIX,
                         "")
        self.assertEqual(aml_adapter.AML_EMBEDDING_DOCUMENT_PREFIX, "")
        self.assertEqual(aml_adapter.AML_RERANKER, "off")

    def test_validate_model_config_accepts_frozen_defaults_with_dir(self):
        import os
        # Frozen default is AML_EMBEDDING=local; a model dir must be supplied
        # (constant or env), otherwise startup fails loudly.
        os.environ["PLM_AML_EMBEDDING_MODEL_DIR"] = "/tmp/plm-model"
        aml_adapter.validate_model_config()

    def test_validate_model_config_rejects_bad_switch(self):
        aml_adapter.AML_EMBEDDING = "bogus"
        with self.assertRaises(ValueError):
            aml_adapter.validate_model_config()

    def test_validate_model_config_warns_but_never_crashes_without_dir(self):
        # A missing model dir is operational, not a config typo: startup must
        # still serve, degraded to the lexical safety floor.
        aml_adapter.AML_EMBEDDING = "local"
        aml_adapter.AML_EMBEDDING_MODEL_DIR = ""
        warnings = aml_adapter.validate_model_config()
        self.assertEqual(len(warnings), 1)
        self.assertIn("degraded", warnings[0])
        aml_adapter.AML_EMBEDDING_MODEL_DIR = "/tmp/somewhere"
        self.assertEqual(aml_adapter.validate_model_config(), [])

    def test_env_var_overrides_model_dir_constant(self):
        import os
        aml_adapter.AML_EMBEDDING_MODEL_DIR = "/constant/dir"
        self.assertEqual(aml_adapter._model_dir(aml_adapter.AML_EMBEDDING_MODEL_DIR,
                                                "PLM_AML_EMBEDDING_MODEL_DIR"), "/constant/dir")
        os.environ["PLM_AML_EMBEDDING_MODEL_DIR"] = "/env/dir"
        self.assertEqual(aml_adapter._model_dir(aml_adapter.AML_EMBEDDING_MODEL_DIR,
                                                "PLM_AML_EMBEDDING_MODEL_DIR"), "/env/dir")


class TestHybridRecall(ModelConfigCase):
    def test_off_switch_builds_no_provider(self):
        aml_adapter.AML_EMBEDDING = "off"
        aml_adapter.AML_RERANKER = "off"
        self.assertIsNone(aml_adapter._model_provider("embedding"))
        self.assertIsNone(aml_adapter._model_provider("reranker"))

    def test_missing_model_dir_degrades_deterministically_to_lexical(self):
        self._seed("user-hyb", "A ginger cat named Mochi sleeps all afternoon.")
        aml_adapter.AML_EMBEDDING = "off"
        baseline = self._search("xyzzy pet qvark", "user-hyb")
        aml_adapter.AML_EMBEDDING = "local"
        aml_adapter.AML_EMBEDDING_MODEL_DIR = "/nonexistent/plm-model-dir"
        degraded = self._search("xyzzy pet qvark", "user-hyb")
        self.assertEqual(baseline, degraded)
        status = aml_adapter.model_status()
        self.assertEqual(status["embedding"]["switch"], "local")
        self.assertFalse(status["embedding"]["active"])
        self.assertEqual(status["embedding"]["degraded_to"], "lexical")
        # The failure is cached: every later search degrades identically.
        self.assertIsNone(aml_adapter._model_provider("embedding"))

    def test_semantic_channel_bridges_zero_lexical_overlap(self):
        self._seed("user-sem", "A ginger cat named Mochi sleeps all afternoon.")
        aml_adapter.AML_EMBEDDING = "off"
        lexical = self._search("xyzzy pet qvark", "user-sem")
        self.assertEqual(lexical["data"], [])
        # Hybrid on: the injected provider bridges the semantic gap.
        aml_adapter.AML_EMBEDDING = "local"
        aml_adapter.AML_EMBEDDING_MODEL_DIR = "/configured-but-unused"
        aml_adapter._MODEL_SINGLETONS["embedding"] = _FakeEmbedding()
        hybrid = self._search("xyzzy pet qvark", "user-sem")
        self.assertEqual(len(hybrid["data"]), 1)
        self.assertIn("Mochi", hybrid["data"][0]["content"])
        status = aml_adapter.model_status()
        self.assertTrue(status["embedding"]["active"])

    def test_lexical_hits_survive_when_embedding_unavailable(self):
        self._seed("user-keep", "The deploy runbook token is zqvx-keep-42.")
        aml_adapter.AML_EMBEDDING = "local"
        aml_adapter.AML_EMBEDDING_MODEL_DIR = "/nonexistent/plm-model-dir"
        body = self._search("zqvx-keep-42", "user-keep")
        self.assertEqual(len(body["data"]), 1)
        self.assertIn("zqvx-keep-42", body["data"][0]["content"])

    def test_failing_reranker_falls_back_without_losing_results(self):
        self._seed("user-rer", "The deploy runbook token is zqvx-rer-7.")
        aml_adapter.AML_RERANKER = "local"
        aml_adapter.AML_RERANKER_MODEL_DIR = "/configured-but-unused"
        aml_adapter._MODEL_SINGLETONS["reranker"] = _FakeReranker(fail=True)
        body = self._search("zqvx-rer-7", "user-rer")
        self.assertEqual(len(body["data"]), 1)
        self.assertIn("zqvx-rer-7", body["data"][0]["content"])

    def test_reranker_reorders_candidates(self):
        self._seed("user-rank", "alpha runbook token zqvx-rank-1")
        aml_adapter.add({
            "request_id": "req-2", "user_id": "user-rank", "session_id": "s",
            "messages": [{"role": "user", "content": "beta runbook token zqvx-rank-2",
                          "timestamp": 1704067260000}],
        })
        aml_adapter.AML_RERANKER = "off"
        baseline = self._search("runbook token zqvx-rank", "user-rank")
        self.assertEqual(len(baseline["data"]), 2)
        aml_adapter.AML_RERANKER = "local"
        aml_adapter.AML_RERANKER_MODEL_DIR = "/configured-but-unused"
        aml_adapter._MODEL_SINGLETONS["reranker"] = _FakeReranker(boost_token="alpha runbook")
        reranked = self._search("runbook token zqvx-rank", "user-rank")
        self.assertEqual(len(reranked["data"]), 2)
        self.assertIn("alpha runbook", reranked["data"][0]["content"])
        self.assertNotEqual(
            [item["content"] for item in baseline["data"]],
            [item["content"] for item in reranked["data"]],
        )

    def test_search_contract_unchanged_with_models_configured(self):
        self._seed("user-shape", "Contract shape token zqvx-shape.")
        aml_adapter.AML_EMBEDDING = "local"
        aml_adapter.AML_EMBEDDING_MODEL_DIR = "/nonexistent/plm-model-dir"
        aml_adapter.AML_RERANKER = "local"
        aml_adapter.AML_RERANKER_MODEL_DIR = "/also-nonexistent"
        body = self._search("zqvx-shape", "user-shape")
        self.assertEqual(set(body), {"data"})
        item = body["data"][0]
        self.assertEqual(set(item), {"id", "content", "score", "created_at"})


class TestHybridOverHTTP(ModelConfigCase):
    def setUp(self):
        super().setUp()
        self.server = aml_adapter.make_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        super().tearDown()

    def _post(self, path, payload):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            conn.request("POST", path, body=json.dumps(payload))
            response = conn.getresponse()
            return response.status, json.loads(response.read().decode("utf-8"))
        finally:
            conn.close()

    def test_health_reports_1_4_0(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            conn.request("GET", "/health")
            response = conn.getresponse()
            body = json.loads(response.read().decode("utf-8"))
        finally:
            conn.close()
        self.assertEqual(response.status, 200)
        self.assertEqual(body["version"], "1.4.0")

    def test_degraded_hybrid_serves_lexical_results_over_http(self):
        aml_adapter.AML_EMBEDDING = "local"
        aml_adapter.AML_EMBEDDING_MODEL_DIR = "/nonexistent/plm-model-dir"
        status, _ = self._post("/add", {
            "request_id": "r", "user_id": "user-http", "session_id": "s",
            "messages": [{"role": "user", "content": "http hybrid token zqvx-http",
                          "timestamp": 1704067200000}],
        })
        self.assertEqual(status, 200)
        status, body = self._post("/search", {"query": "zqvx-http", "user_id": "user-http",
                                              "top_k": 100})
        self.assertEqual(status, 200)
        self.assertEqual(len(body["data"]), 1)
        self.assertIn("zqvx-http", body["data"][0]["content"])


class TestModelFixtureCleanup(unittest.TestCase):
    def test_failed_http_setup_restores_model_state(self):
        import os
        from unittest.mock import patch

        # Independent safety snapshot: a failing regression must not itself
        # contaminate the tests that follow it.
        original = {name: value for name, value in vars(aml_adapter).items()
                    if name.startswith("AML_")}
        singletons = dict(aml_adapter._MODEL_SINGLETONS)
        fixture = TestHybridOverHTTP("test_health_reports_1_4_0")
        result = unittest.TestResult()
        sentinel = object()
        try:
            aml_adapter.AML_SESSION_RRF = True
            aml_adapter._MODEL_SINGLETONS["cleanup-sentinel"] = sentinel
            expected = {name: getattr(aml_adapter, name) for name in original}
            expected_singletons = dict(aml_adapter._MODEL_SINGLETONS)
            # No socket is opened. unittest deliberately skips tearDown when
            # setUp fails, but must still invoke registered cleanups.
            with patch.dict(os.environ), patch.object(
                    aml_adapter, "make_server",
                    side_effect=PermissionError("fixture-cleanup-injected-bind-failure")):
                fixture.run(result)
            self.assertEqual(len(result.errors), 1)
            self.assertFalse(result.failures)
            self.assertIn("fixture-cleanup-injected-bind-failure", result.errors[0][1])
            self.assertEqual({name: getattr(aml_adapter, name) for name in original}, expected)
            self.assertEqual(set(aml_adapter._MODEL_SINGLETONS), set(expected_singletons))
            for key, value in expected_singletons.items():
                self.assertIs(aml_adapter._MODEL_SINGLETONS[key], value)
        finally:
            for name, value in original.items():
                setattr(aml_adapter, name, value)
            aml_adapter._MODEL_SINGLETONS.clear()
            aml_adapter._MODEL_SINGLETONS.update(singletons)
            if hasattr(fixture, "temp"):
                fixture.temp.cleanup()


if __name__ == "__main__":
    unittest.main()
