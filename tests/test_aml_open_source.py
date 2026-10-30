"""Candidate integration checks: only a local embedding stub, no paid calls."""
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from tiger_memory import aml_adapter as aml
from tiger_memory import aml_open_source as candidate
from tiger_memory.remote_embeddings import DashScopeEmbeddingProvider
from tiger_memory.embedding_budget import PersistentEmbeddingBudget
from tiger_memory.providers import ProviderUnavailable
from tests.test_aml_adapter import AMLServerCase
from tests import test_remote_embeddings as remote_fixture
from tests.aml_budget_fixture import receipt_fixture


class EmbeddingConfigurationRoutingTests(unittest.TestCase):
    """Synthetic receipt/configuration only; never open an HTTP connection."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.counter = 0

    def env(self, endpoint="https://www.dmxapi.cn/v1/embeddings"):
        self.counter += 1
        root = self.root / str(self.counter)
        root.mkdir()
        receipt, _ = receipt_fixture(root, endpoint)
        return {
            "TIGER_MEMORY_DIR": str(root / "store"),
            "TIGER_MEMORY_EMBEDDING_ENDPOINT": endpoint,
            "TIGER_MEMORY_EMBEDDING_API_KEY": "synthetic-dmx-key",
            "DASHSCOPE_API_KEY": "synthetic-aliyun-key-must-not-leak-to-dmx",
            "AML_API_KEY": "synthetic-service-key",
            "TIGER_MEMORY_ALLOW_REMOTE": "1",
            "TIGER_MEMORY_EMBEDDING_BUDGET_RECEIPT": str(receipt),
        }

    def test_dmx_selects_only_generic_embedding_key_without_probe(self):
        env = self.env()
        with patch.object(candidate, "DashScopeEmbeddingProvider",
                          wraps=DashScopeEmbeddingProvider) as provider:
            report = candidate.configuration(env)
        self.assertTrue(report["configured"], report["issues"])
        provider.assert_called_once_with(env["TIGER_MEMORY_EMBEDDING_ENDPOINT"],
                                         "synthetic-dmx-key")
        self.assertFalse(report["remote_probe_performed"])
        self.assertFalse(Path(env["TIGER_MEMORY_DIR"]).exists())
        for value in (env["TIGER_MEMORY_EMBEDDING_API_KEY"], env["DASHSCOPE_API_KEY"]):
            self.assertNotIn(value, json.dumps(report))

    def test_dmx_missing_or_invalid_generic_key_never_uses_dashscope_key(self):
        env = self.env()
        for value in (None, "", "bad\nkey"):
            changes = dict(env)
            if value is None:
                changes.pop("TIGER_MEMORY_EMBEDDING_API_KEY")
            else:
                changes["TIGER_MEMORY_EMBEDDING_API_KEY"] = value
            with self.subTest(value=value), patch.object(candidate, "DashScopeEmbeddingProvider",
                    wraps=DashScopeEmbeddingProvider) as provider:
                report = candidate.configuration(changes)
                self.assertFalse(report["configured"])
                for call in provider.call_args_list:
                    self.assertNotIn(env["DASHSCOPE_API_KEY"], call.args)

    def test_aliyun_explicit_rollback_keeps_legacy_key_and_generic_override(self):
        env = self.env("https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings")
        env.pop("TIGER_MEMORY_EMBEDDING_API_KEY")
        with patch.object(candidate, "DashScopeEmbeddingProvider",
                          wraps=DashScopeEmbeddingProvider) as provider:
            self.assertTrue(candidate.configuration(env)["configured"])
        provider.assert_called_once_with(env["TIGER_MEMORY_EMBEDDING_ENDPOINT"],
                                         env["DASHSCOPE_API_KEY"])
        env["TIGER_MEMORY_EMBEDDING_API_KEY"] = "synthetic-explicit-aliyun-key"
        with patch.object(candidate, "DashScopeEmbeddingProvider",
                          wraps=DashScopeEmbeddingProvider) as provider:
            self.assertTrue(candidate.configuration(env)["configured"])
        provider.assert_called_once_with(env["TIGER_MEMORY_EMBEDDING_ENDPOINT"],
                                         "synthetic-explicit-aliyun-key")
        env["TIGER_MEMORY_EMBEDDING_API_KEY"] = ""
        self.assertFalse(candidate.configuration(env)["configured"])

    def test_dmx_whitelist_rejects_wrong_host_path_protocol_and_authority(self):
        env = self.env()
        for endpoint in (
            "https://dmxapi.cn/v1/embeddings", "https://www.dmxapi.cn./v1/embeddings",
            "https://www.dmxapi.cn.example.com/v1/embeddings",
            "https://www.dmxapi.cn/compatible-mode/v1/embeddings",
            "https://www.dmxapi.cn/v1/embeddings/", "http://www.dmxapi.cn/v1/embeddings",
            "https://www.dmxapi.cn:444/v1/embeddings",
            "https://user@www.dmxapi.cn/v1/embeddings",
            "https://www.dmxapi.cn/v1/embeddings?key=synthetic",
            "https://www.dmxapi.cn/v1/embeddings#fragment",
        ):
            with self.subTest(endpoint=endpoint):
                report = candidate.configuration(dict(env, TIGER_MEMORY_EMBEDDING_ENDPOINT=endpoint))
                self.assertIn("endpoint_must_be_official_https_embeddings_url", report["issues"])

    def test_dmx_does_not_rebind_an_aliyun_budget_receipt(self):
        env = self.env("https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings")
        env["TIGER_MEMORY_EMBEDDING_ENDPOINT"] = "https://www.dmxapi.cn/v1/embeddings"
        self.assertIn("invalid_embedding_budget_receipt", candidate.configuration(env)["issues"])

    def test_dmx_policy_requires_exact_endpoint_region_dimension_and_evidence(self):
        env = self.env()
        budget = PersistentEmbeddingBudget.from_receipt(
            Path(env["TIGER_MEMORY_EMBEDDING_BUDGET_RECEIPT"]))
        self.assertEqual(budget.policy.region, "dmxapi")
        self.assertIs(budget.policy.validate(), budget.policy)
        for changes in (
            {"endpoint": "https://www.dmxapi.cn:443/v1/embeddings"},
            {"endpoint": "https://www.dmxapi.cn/v1/embeddings/"},
            {"endpoint": "https://www.dmxapi.cn.example.com/v1/embeddings"},
            {"endpoint": "https://user@www.dmxapi.cn/v1/embeddings"},
            {"endpoint": "https://www.dmxapi.cn/v1/embeddings?key=synthetic"},
            {"region": "cn-beijing"}, {"dimensions": 64},
            {"model": "wrong-model"}, {"evidence_sha256": ""},
        ):
            with self.subTest(changes=changes), self.assertRaises(ProviderUnavailable):
                replace(budget.policy, **changes).validate()
        env["TIGER_MEMORY_EMBEDDING_ENDPOINT"] = (
            "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings")
        self.assertIn("invalid_embedding_budget_receipt", candidate.configuration(env)["issues"])

    def test_configure_factory_captures_dmx_key(self):
        env = self.env()
        snapshot = {k: v for k, v in vars(aml).items() if k.startswith("AML_")}
        singletons = dict(aml._MODEL_SINGLETONS)
        self.addCleanup(lambda: [setattr(aml, k, v) for k, v in snapshot.items()])
        self.addCleanup(lambda: (aml._MODEL_SINGLETONS.clear(), aml._MODEL_SINGLETONS.update(singletons)))
        with patch.dict(os.environ, env, clear=True):
            candidate.configure()
            provider = aml.AML_EMBEDDING_FACTORY("synthetic-user")
        self.assertEqual(provider.endpoint, "https://www.dmxapi.cn/v1/embeddings")
        self.assertEqual(provider._api_key, "synthetic-dmx-key")
        self.assertEqual(provider.describe()["provider"], "dmx-openai-compatible")
        self.assertEqual(provider.dimensions, 1024)
        self.assertIsNotNone(provider.budget)


class OpenSourceProfileTests(AMLServerCase):
    def setUp(self):
        super().setUp()
        # Reuse the independent HTTP fixture, without inheriting its tests.
        self.remote = remote_fixture.RemoteEmbeddingsTests()
        self.remote.setUp()
        self.addCleanup(self.remote.doCleanups)
        names = [name for name in vars(aml) if name.startswith("AML_")]
        snapshot = {name: getattr(aml, name) for name in names}
        self.addCleanup(lambda: [setattr(aml, k, v) for k, v in snapshot.items()])
        singletons = dict(aml._MODEL_SINGLETONS)
        self.addCleanup(lambda: (aml._MODEL_SINGLETONS.clear(), aml._MODEL_SINGLETONS.update(singletons)))
        receipt, _budget = receipt_fixture(Path(self.temp.name).resolve(),
            "https://test.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/embeddings")
        self.env = patch.dict(os.environ, {
            "TIGER_MEMORY_EMBEDDING_BUDGET_RECEIPT": str(receipt),
            "TIGER_MEMORY_DIR": str(self.base.resolve()),
            "TIGER_MEMORY_EMBEDDING_ENDPOINT": "https://test.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/embeddings",
            "DASHSCOPE_API_KEY": "fixture-embedding-key", "AML_API_KEY": "fixture-service-key",
            "TIGER_MEMORY_ALLOW_REMOTE": "1",
        })
        self.env.start()
        # Exercise legacy Aliyun independently of an operator's generic key.
        os.environ.pop("TIGER_MEMORY_EMBEDDING_API_KEY", None)
        self.addCleanup(self.env.stop)
        candidate.configure()
        def provider(user):
            return DashScopeEmbeddingProvider(self.remote.endpoint, "fixture-embedding-key",
                cache_dir=aml.user_storage_dir(user) / "embedding-cache", allow_test_loopback=True)
        aml.AML_EMBEDDING_FACTORY = provider

    def request(self, method, path, payload=None, **kwargs):
        headers = kwargs.setdefault("headers", {})
        headers.setdefault("Authorization", "Bearer fixture-service-key")
        return super().request(method, path, payload=payload, **kwargs)

    def test_add_search_replay_cache_restart_and_user_isolation(self):
        self.assertEqual(self.add("r", "u1", "s", ["violet boxes"])[0], 200)
        self.assertEqual(len(self.remote.requests), 1)
        status, found = self.search("violet", "u1")
        self.assertEqual(status, 200)
        self.assertEqual(found["data"][0]["content"], "violet boxes")
        # Add prewarms exactly Search's document representation; only query is new.
        self.assertEqual(self.remote.requests[-1][1]["input"], ["violet"])
        count = len(self.remote.requests)
        self.add("r", "u1", "s", ["changed replay must not be embedded"])
        self.assertEqual(self.search("violet", "u1")[1], found)
        self.assertEqual(len(self.remote.requests), count)
        self.assertEqual(self.search("violet", "unknown")[1], {"data": []})
        self.add("r", "u2", "s", ["violet boxes"])
        self.assertEqual(len(self.remote.requests), count + 1)
        self.search("violet", "u2")
        self.assertEqual(len(self.remote.requests), count + 2)
        self.assertNotEqual(aml.user_storage_dir("u1"), aml.user_storage_dir("u2"))

    def test_remote_failure_is_503_and_retry_preserves_original_records(self):
        self.remote.respond = lambda p: (429, {}, {"private": "must-not-leak"})
        self.assertEqual(self.add("r", "u", "s", ["private stored memory"])[0], 503)
        status, body = self.search("uncached query", "u")
        self.assertEqual(status, 503)
        self.assertNotIn("private", json.dumps(body))
        self.remote.respond = self.remote.success
        self.assertEqual(self.add("r", "u", "s", ["different retry"])[0], 200)
        status, body = self.search("stored memory", "u")
        self.assertEqual(status, 200)
        self.assertEqual([r["content"] for r in body["data"]], ["private stored memory"])

    def test_status_requires_auth_and_never_calls_provider_or_exposes_keys(self):
        with patch.object(aml, "AML_EMBEDDING_FACTORY", side_effect=AssertionError("must not probe")):
            status, body = self.request("GET", "/status")
        self.assertEqual(status, 200)
        self.assertEqual(body["models"]["embedding"]["readiness"], "not_probed")
        text = json.dumps(body)
        for secret in ["fixture-service-key", "fixture-embedding-key"]:
            self.assertNotIn(secret, text)
        self.assertEqual(self.request("GET", "/status", headers={"Authorization": "Bearer wrong"})[0], 401)
        self.assertEqual(self.request("GET", "/health", headers={"Authorization": "Bearer wrong"})[0], 200)
        self.assertFalse(self.remote.requests)

    def test_config_check_is_readonly_and_rejects_missing_or_unsafe_settings(self):
        fresh = self.base.parent / "no-write-check"
        env = dict(os.environ, TIGER_MEMORY_DIR=str(fresh.resolve()))
        report = candidate.configuration(env)
        self.assertTrue(report["configured"])
        self.assertFalse(fresh.exists())
        self.assertFalse(self.remote.requests)
        for changes in [
            {"DASHSCOPE_API_KEY": ""}, {"TIGER_MEMORY_ALLOW_REMOTE": "0"},
            {"DASHSCOPE_API_KEY": "key\n"},
            {"AML_API_KEY": "key\n"},
            {"AML_API_KEY": ""}, {"TIGER_MEMORY_DIR": str(Path.home() / ".codex/project-memory")},
            {"TIGER_MEMORY_EMBEDDING_ENDPOINT": "https://example.com/compatible-mode/v1/embeddings"},
            {"TIGER_MEMORY_EMBEDDING_ENDPOINT": "https://@test.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/embeddings"},
            {"TIGER_MEMORY_EMBEDDING_ENDPOINT": "https://test.cn-beijing.maas.aliyuncs.com:bad/compatible-mode/v1/embeddings"},
        ]:
            with self.subTest(changes=changes):
                self.assertFalse(candidate.configuration(dict(env, **changes))["configured"])

    def test_store_identity_refuses_existing_unowned_or_changed_configuration(self):
        # A managed store can restart, but cannot mix model configurations.
        candidate.configure()
        with patch.dict(os.environ, TIGER_MEMORY_EMBEDDING_ENDPOINT="https://other.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/embeddings"):
            with self.assertRaisesRegex(ValueError, "invalid_embedding_budget_receipt"):
                candidate.configure()
        unowned = self.base.parent / "unowned"
        unowned.mkdir()
        (unowned / "keep.txt").write_text("keep")
        with patch.dict(os.environ, TIGER_MEMORY_DIR=str(unowned.resolve())):
            with self.assertRaisesRegex(ValueError, "nonempty"):
                candidate.configure()
        self.assertEqual((unowned / "keep.txt").read_text(), "keep")
