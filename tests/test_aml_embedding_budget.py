"""Actual local AML HTTP handlers with synthetic memory and in-memory transport.

Factory wiring is explicit here; it does not claim the startup entry owner's
configuration/receipt provisioning has already been integrated or deployed.
"""
import http.client
import hashlib
import os
from pathlib import Path
import threading
from unittest.mock import patch

from tiger_memory import aml_adapter as aml
from tiger_memory.embedding_budget import PersistentEmbeddingBudget
from tiger_memory.remote_embeddings import DashScopeEmbeddingProvider
from tests.test_aml_adapter import AMLServerCase
from tests.test_embedding_budget import policy, FakeTransport, evidence


class AMLBudgetHTTPTests(AMLServerCase):
    def setUp(self):
        super().setUp()
        self.configured = policy(dimensions=1024)
        self.budget = PersistentEmbeddingBudget.initialize(Path(self.temp.name).resolve() / "budget",
                                                         self.configured)
        self.fake = FakeTransport()
        self.providers = []
        self.logical_chunk_counts = []
        def factory(user):
            obj = DashScopeEmbeddingProvider(self.configured.endpoint, "SYNTHETIC_NOT_A_REAL_KEY",
                dimensions=1024, cache_dir=aml.user_storage_dir(user) / "embedding-cache",
                allow_test_loopback=True, budget=self.budget,
                checkpoint_namespace=aml.user_storage_dir(user).name)
            obj._opener = self.fake
            original_embed = obj.embed
            def embed(texts):
                self.logical_chunk_counts.append(sum(len(obj._chunks(text)) for text in dict.fromkeys(texts)))
                return original_embed(texts)
            obj.embed = embed
            self.providers.append(obj)
            return obj
        flags = patch.multiple(aml, AML_EMBEDDING_FACTORY=factory, AML_EMBEDDING="remote-v4",
            AML_STRICT_RETRIEVAL=True, AML_PROFILE="lexical", AML_QUERY_REWRITE="off",
            AML_FACT_PROJECTION="off", AML_RERANKER="off", AML_RECENCY_BOOST=False)
        flags.start()
        self.addCleanup(flags.stop)
        env = patch.dict(os.environ, {"AML_API_KEY": ""})
        env.start()
        self.addCleanup(env.stop)

    def test_add_partial_batch_http_retry_restart_search_and_user_isolation(self):
        self.fake.fail_batch = 2
        original = ["synthetic memory %s" % i for i in range(11)]
        self.assertEqual(self.add("retry", "user-a", "session", original)[0], 503)
        self.assertEqual([len(p["input"]) for p in self.fake.requests], [10, 1])
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)
        self.budget = PersistentEmbeddingBudget(self.budget.root, self.configured, ledger_id=self.budget.ledger_id)
        self.server = aml.make_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in self.base.rglob("*.md")}
        before = self.budget.snapshot()
        self.assertEqual(len(source_hashes), 11)
        # New HTTP instances/IDs and reordered batch inputs do not authorize
        # another paid send for the claimed last document with missing data.
        self.assertEqual(self.add("retry", "user-a", "session", ["changed replay"] * 11)[0], 503)
        self.assertEqual(source_hashes, {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in self.base.rglob("*.md")})
        self.assertEqual(self.add("new-id", "user-a", "session", list(reversed(original)))[0], 503)
        self.assertEqual(self.search("synthetic memory", "user-a")[0], 503)
        self.assertEqual(self.budget.snapshot(), before)
        self.assertEqual([len(p["input"]) for p in self.fake.requests], [10, 1])
        self.assertNotIn("changed replay", str(self.fake.requests))
        for name, value in source_hashes.items():
            self.assertEqual(hashlib.sha256(Path(name).read_bytes()).hexdigest(), value)
        self.assertEqual(self.add("retry", "user-b", "session", [original[0]])[0], 200)
        status, result = self.search("synthetic memory", "user-b")
        self.assertEqual(status, 200)
        self.assertEqual([row["content"] for row in result["data"]], [original[0]])
        self.assertEqual(self.search("synthetic memory", "unknown-user")[1], {"data": []})
        sends = len(self.fake.requests)
        self.budget.stop()
        self.assertEqual(self.search("uncached query", "user-b")[0], 503)
        self.assertEqual(len(self.fake.requests), sends)
        self.assertEqual(self.search("synthetic memory", "user-b")[0], 200)
        evidence("http-retry-restart", {"batch_sizes": [len(p["input"]) for p in self.fake.requests],
            "retained_original_sources": 11, "first_write_preserved": True,
            "same_id_retry_status": 503, "changed_id_reordered_status": 503,
            "unknown_retry_new_sends": 0, "unknown_retry_new_reserves": 0,
            "ledger": self.budget.snapshot(), "server_restart": "real listener restart in same process"})

    def test_http_1001_chunks_rejected_before_write_without_pinning_request_id(self):
        # 1000 distinct body blocks plus the actual native title/tag prefix
        # produce 1001 distinct provider chunks, below the HTTP byte limit.
        text = "".join(("block-%04d|" % i).ljust(8192, "x") for i in range(1000))
        plans = []
        actual_plan = DashScopeEmbeddingProvider.plan
        def observed_plan(provider, texts):
            result = actual_plan(provider, texts)
            chunks = [chunk for value in dict.fromkeys(texts) for chunk in provider._chunks(value)]
            plans.append({"result": result, "distinct_chunks": len(set(chunks))})
            return result
        with patch.object(DashScopeEmbeddingProvider, "plan", observed_plan), \
                patch.object(aml.service, "write_event", wraps=aml.service.write_event) as write:
            for _ in range(2):
                status, result = self.add("large", "large-user", "session", [text])
                self.assertEqual(status, 503)
                self.assertEqual(result["detail"]["reason"], "embedding request capacity exceeded")
                self.assertEqual(plans[-1]["result"]["total_chunks"], 1001)
                self.assertEqual(plans[-1]["distinct_chunks"], 1001)
                self.assertEqual(plans[-1]["result"]["uncached_chunks"], 1001)
                self.assertEqual(plans[-1]["result"]["required_batches"], 101)
                self.assertFalse(plans[-1]["result"]["within_request_batch_limit"])
                self.assertEqual(list(self.base.rglob("*.md")), [])
                self.assertEqual(self.logical_chunk_counts, [])  # embed not invoked
                self.assertEqual(self.fake.requests, [])
                self.assertEqual(self.budget.snapshot()["requests"], 0)
                write.assert_not_called()
            # Pre-write refusal must not permanently pin the request ID.
            self.assertEqual(self.add("large", "large-user", "session", ["small retry"])[0], 200)
            self.assertEqual(write.call_count, 1)
        self.assertEqual(len(list(self.base.rglob("*.md"))), 1)
        self.assertEqual(self.logical_chunk_counts, [1])
        self.assertEqual(len(self.fake.requests), 1)
        self.assertEqual(self.budget.snapshot()["requests"], 1)
        evidence("http-1001-prewrite-contract", {"body_utf8_bytes": len(text.encode()),
            "rejected_attempts": 2, "rejected_plans": plans[:2],
            "rejected_status": 503, "rejected_persisted_sources": 0,
            "rejected_embed_calls": 0, "rejected_embedding_sends": 0,
            "rejected_budget_reservations": 0, "same_id_small_retry_status": 200,
            "recovery_sources": 1, "recovery_fake_sends": 1,
            "official_input_envelope": "UNKNOWN"})

    def test_http_size_boundary_is_not_embedding_success_promise(self):
        cap = aml.AML_MAX_REQUEST_BYTES
        self.assertEqual(cap, 32 * 1024 * 1024)
        # Valid JSON at the byte limit reaches Add schema validation, not a
        # paid provider. It is deliberately not a valid 32MiB Add workload.
        status, result = self.request("POST", "/add", raw_body="{}" + " " * (cap - 2))
        self.assertEqual(status, 422)
        self.assertNotIn("too large", str(result))
        # Header-only +1 proves rejection before body read; do not send 32MiB
        # after the server has closed the rejected connection.
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            connection.request("POST", "/add", headers={"Content-Length": str(cap + 1)})
            response = connection.getresponse()
            self.assertEqual(response.status, 400)
            self.assertIn(b"too large", response.read())
        finally:
            connection.close()
        self.assertEqual(self.fake.requests, [])
        self.assertEqual(self.budget.snapshot()["requests"], 0)
        evidence("http-byte-boundary", {"body_limit_bytes": cap, "at_limit_schema_status": 422,
            "over_limit_status": 400, "embedding_sends": 0,
            "not_a_valid_full_size_add_success_test": True})
