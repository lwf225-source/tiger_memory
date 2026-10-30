"""Capacity snapshots use first-write source and exact document chunks, fake only."""
import hashlib
from pathlib import Path
from unittest.mock import patch
from tiger_memory import aml_adapter as aml, service, database
from tiger_memory.embedding_budget import PersistentEmbeddingBudget
from tiger_memory.remote_embeddings import DashScopeEmbeddingProvider
from tests.test_aml_adapter import AMLServerCase
from tests.test_embedding_budget import policy, FakeTransport

class CapacityFixture(AMLServerCase):
    def setUp(self):
        super().setUp()
        self.budget = PersistentEmbeddingBudget.initialize(Path(self.temp.name).resolve() / "budget", policy())
        self.fake = FakeTransport()
        self.plans = []
        self.providers = []
        def factory(user):
            provider = DashScopeEmbeddingProvider(self.budget.policy.endpoint, "SYNTHETIC_ONLY",
                dimensions=64, allow_test_loopback=True, budget=self.budget,
                cache_dir=aml.user_storage_dir(user) / "embedding-cache", checkpoint_namespace=user)
            provider._opener = self.fake
            original = provider.plan
            def plan(texts):
                result = original(texts)
                self.plans.append((list(texts), result))
                return result
            provider.plan = plan
            self.providers.append(provider)
            return provider
        flags = patch.multiple(aml, AML_EMBEDDING_FACTORY=factory, AML_EMBEDDING="remote-v4",
            AML_STRICT_RETRIEVAL=True, AML_PROFILE="lexical", AML_QUERY_REWRITE="off",
            AML_FACT_PROJECTION="off", AML_RERANKER="off", AML_RECENCY_BOOST=False,
            AML_NARRATIVE_TIME=False, AML_SESSION_RRF=False, AML_SESSION_VIEWS=False)
        flags.start()
        self.addCleanup(flags.stop)

    def text(self, chunks):
        prefix = aml._embedding_document("user · 2024-01-01", ["aml"], "x")[:-1]
        return "".join(("block-%04d|" % i).ljust(8192, "x") for i in range(chunks-1)) + "tail".ljust(8192-len(prefix.encode()), "z")

    def sources(self):
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in self.base.rglob("*.md")}


class CapacityTests(CapacityFixture):
    def test_1000_distinct_success_1001_prewrite_reject_and_retry_can_recover(self):
        large = self.text(1001)
        status, body = self.add("large", "A", "s", [large])
        self.assertEqual((status, body["detail"]["reason"]), (503, "embedding request capacity exceeded"))
        self.assertEqual(self.plans[-1][1]["uncached_chunks"], 1001)
        self.assertEqual(self.sources(), {})
        self.assertEqual(self.fake.requests, [])
        self.assertEqual(self.budget.snapshot()["requests"], 0)
        # A failed capacity request has no permanent request_id latch.
        small = self.text(1000)
        self.assertEqual(self.add("large", "A", "s", [small])[0], 200)
        self.assertEqual(self.plans[-1][1]["uncached_chunks"], 1000)
        parts = [x for request in self.fake.requests for x in request["input"]]
        self.assertEqual((len(parts), len(set(parts))), (1000, 1000))
        source = self.sources()
        self.assertEqual(len(source), 1)
        sends = len(self.fake.requests)
        self.assertEqual(self.add("large", "A", "s", [large])[0], 200)
        self.assertEqual(self.sources(), source)
        self.assertEqual(len(self.fake.requests), sends)
        self.assertEqual(self.plans[-1][1]["required_batches"], 0)
        self.assertEqual(self.plans[-1][0][0].split("\n", 2)[-1], small)

    def test_existing_oversized_source_retained_and_replay_uses_original(self):
        with patch.object(aml, "AML_EMBEDDING", "off"):
            self.assertEqual(self.add("r", "A", "s", [self.text(1001)])[0], 200)
        sources = self.sources()
        self.assertEqual(self.add("r", "A", "s", ["tiny changed replay"])[0], 503)
        self.assertEqual(self.sources(), sources)
        self.assertEqual(self.plans[-1][1]["uncached_chunks"], 1001)
        self.assertEqual(self.fake.requests, [])

    def test_cold_query_plus_documents_exceeds_cap_warm_document_succeeds(self):
        body = self.text(1000)
        with patch.object(aml, "AML_EMBEDDING", "off"):
            self.assertEqual(self.add("r", "A", "s", [body])[0], 200)
        self.assertEqual(self.search("block", "A", 1)[0], 503)
        self.assertEqual(self.fake.requests, [])
        self.assertEqual(self.add("r", "A", "s", ["changed"])[0], 200)
        count = len(self.fake.requests)
        self.assertEqual(self.search("block", "A", 1)[0], 200)
        self.assertEqual(len(self.fake.requests), count + 1)
        self.assertEqual(self.fake.requests[-1]["input"], ["block"])

    def test_valid_orphan_preview_is_first_write_and_is_not_imported_by_preview(self):
        with patch.object(aml, "AML_EMBEDDING", "off"):
            self.assertEqual(self.add("r", "A", "s", ["first write original"])[0], 200)
        conn = database.connect()
        conn.execute("DELETE FROM events")
        conn.commit()
        conn.close()
        root = aml.user_storage_dir("A")
        before = self.sources()
        event = service.preview_event_for_idempotency(root, "aml:A:r:0",
            expected_identity=service.ExpectedIdentity(root, "session", "A"))
        self.assertEqual(event.body, "first write original")
        conn = database.connect_readonly()
        self.assertEqual(conn.execute("SELECT count(*) FROM events").fetchone()[0], 0)
        conn.close()
        self.assertEqual(self.sources(), before)
        self.assertEqual(self.add("r", "A", "s", [self.text(1001)])[0], 200)
        self.assertEqual(self.plans[-1][0], [aml._embedding_document(event.metadata["title"], event.metadata["tags"], event.body)])
        self.assertEqual(self.sources(), before)

    def test_batch_limit_change_after_plan_is_rechecked_and_never_false_200(self):
        # A snapshot can become stale: execution, not plan, has final authority.
        original = aml._model_provider
        def factory(role, user_id=""):
            provider = original(role, user_id)
            if role == "embedding":
                real_plan = provider.plan
                def plan(texts):
                    result = real_plan(texts)
                    provider.max_request_batches = 1
                    return result
                provider.plan = plan
            return provider
        with patch.object(aml, "_model_provider", side_effect=factory):
            self.assertEqual(self.add("r", "A", "s", [self.text(11)])[0], 503)
        self.assertEqual(len(self.sources()), 1)
        self.assertEqual(self.fake.requests, [])
        self.assertEqual(self.add("r", "A", "s", ["changed"])[0], 200)

    def test_add_sensitive_54_combinations_preserve_rejection_and_partial_prefix(self):
        from tests.test_aml_sensitive_search import PATTERNS
        import json
        observations = []
        for number, secret in enumerate(PATTERNS):
            for message_index in range(3):
                for position in range(3):
                    user = "matrix-%d-%d-%d" % (number, message_index, position)
                    value = (secret + " suffix", "prefix " + secret + " suffix", "prefix " + secret)[position]
                    texts = ["benign first", "benign second", "benign third"]
                    texts[message_index] = value
                    before = len(self.sources())
                    sends = len(self.fake.requests)
                    status, body = self.add("r", user, "s", texts)
                    self.assertEqual(status, 503)
                    self.assertNotIn(secret, json.dumps(body))
                    snapshot = self.sources()
                    self.assertEqual(len(snapshot) - before, message_index)
                    self.assertEqual(self.add("r", user, "s", texts)[0], 503)
                    self.assertEqual(self.sources(), snapshot)
                    self.assertEqual(len(self.fake.requests), sends)
                    for path in snapshot:
                        self.assertNotIn(secret, Path(path).read_text())
                    observations.append((number, message_index, position, status))
        self.assertEqual(len(observations), 54)
        self.assertEqual(self.add("r", "matrix-0-1-1", "s", ["benign first", "corrected", "benign third"])[0], 200)
        self.assertEqual(self.add("r", "safe-first", "s", ["original"])[0], 200)
        sends = len(self.fake.requests)
        self.assertEqual(self.add("r", "safe-first", "s", [PATTERNS[0]])[0], 503)
        self.assertEqual(len(self.fake.requests), sends)

    def test_cache_removed_after_plan_fails_closed_with_original_source_preserved(self):
        import shutil
        body = self.text(11)
        self.assertEqual(self.add("r", "A", "s", [body])[0], 200)
        source = self.sources()
        checkpoint_root = self.providers[-1]._checkpoint.root
        checkpoint_backup = Path(self.temp.name) / "valid-checkpoint-backup"
        shutil.copytree(checkpoint_root, checkpoint_backup)
        budget_before = self.budget.snapshot()
        original = aml._model_provider
        def factory(role, user_id=""):
            provider = original(role, user_id)
            if role == "embedding":
                provider.max_request_batches = 1
                real_plan = provider.plan
                def plan(texts):
                    result = real_plan(texts)
                    self.assertEqual(result["required_batches"], 0)
                    shutil.rmtree(provider._checkpoint.root)
                    return result
                provider.plan = plan
            return provider
        sends = len(self.fake.requests)
        with patch.object(aml, "_model_provider", side_effect=factory):
            self.assertEqual(self.add("r", "A", "s", ["changed"])[0], 503)
        self.assertEqual(len(self.fake.requests), sends)
        self.assertEqual(self.sources(), source)
        # Durable SETTLED claims prohibit repayment when valid raw data is lost.
        self.assertEqual(self.add("r", "A", "s", ["changed"])[0], 503)
        self.assertEqual(len(self.fake.requests), sends)
        self.assertEqual(self.budget.snapshot(), budget_before)
        self.assertEqual(self.sources(), source)
        # Restoring the exact valid checkpoint recovers without new paid work.
        shutil.copytree(checkpoint_backup, checkpoint_root, dirs_exist_ok=True)
        self.assertEqual(self.add("r", "A", "s", ["changed"])[0], 200)
        self.assertEqual(len(self.fake.requests), sends)
        self.assertEqual(self.budget.snapshot(), budget_before)
        self.assertEqual(self.sources(), source)
