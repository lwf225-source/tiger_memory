"""Synthetic spending/checkpoint acceptance. No external sockets or credentials."""
from dataclasses import replace
from contextlib import closing
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import signal
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from tiger_memory.embedding_budget import BudgetPolicy, PersistentEmbeddingBudget
from tiger_memory.providers import ProviderUnavailable
from tiger_memory.remote_embeddings import DashScopeEmbeddingProvider
from tiger_memory.embedding_checkpoint import digest


def policy(**overrides):
    return replace(BudgetPolicy(endpoint="http://127.0.0.1:9/embeddings", region="test-only",
        model="text-embedding-v4", dimensions=64, token_limit_per_row=8192,
        price_nano_cny_per_token=500, max_cost_nano_cny=1_000_000_000,
        max_requests=1000, evidence_sha256=hashlib.sha256(b"synthetic supplier contract only").hexdigest(),
        test_only=True), **overrides)


def synthetic_claim(rows, label="synthetic"):
    return {"namespace": digest("budget-unit"), "fingerprint": digest("config"),
            "raw_keys": [digest([label, i]) for i in range(rows)]}


def raw_vector(text, dimensions):
    return [float(len(text)), float(sum(map(ord, text)) % 17 + 1)] + [0.0] * (dimensions - 2)


def evidence(name, value):
    destination = os.environ.get("TIGER_EMBEDDING_TEST_EVIDENCE")
    if destination:
        with (Path(destination) / (name + ".json")).open("x") as stream:
            json.dump(value, stream, indent=2)


class FakeResponse:
    status = 200
    def __init__(self, payload):
        self.payload = payload
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def read(self, _limit):
        return json.dumps(self.payload).encode()


class FakeTransport:
    def __init__(self, fail_batch=None, mutate=None, wire_file=None):
        self.requests = []
        self.fail_batch = fail_batch
        self.mutate = mutate
        self.wire_file = wire_file

    def open(self, request, timeout):
        payload = json.loads(request.data)
        self.requests.append(payload)
        if self.wire_file:
            with Path(self.wire_file).open("a") as stream:
                stream.write(str(len(payload["input"])) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        if len(self.requests) == self.fail_batch:
            raise TimeoutError("SYNTHETIC_NOT_A_REAL_KEY")
        response = {"model": "text-embedding-v4", "usage": {"total_tokens": len(payload["input"])},
                    "data": [{"index": i, "embedding": raw_vector(text, payload["dimensions"])}
                             for i, text in reversed(list(enumerate(payload["input"])))]}
        if self.mutate:
            self.mutate(response)
        return FakeResponse(response)


def provider(root, budget, user="synthetic-user", transport=None, **kwargs):
    p = DashScopeEmbeddingProvider(budget.policy.endpoint, "SYNTHETIC_NOT_A_REAL_KEY",
        dimensions=budget.policy.dimensions, cache_dir=Path(root) / user,
        allow_test_loopback=True, budget=budget, checkpoint_namespace=user, **kwargs)
    p._opener = transport or FakeTransport()
    return p


def process_worker(root, configured, ledger_id, start, pipe, phase="compete", user="user"):
    # Independent process + independent provider/cache; no HTTP/network involved.
    with patch("socket.socket", side_effect=AssertionError("NO_NETWORK")):
        budget = PersistentEmbeddingBudget(Path(root) / "budget", configured, ledger_id=ledger_id)
        fake = FakeTransport(wire_file=Path(root) / (user + ".wire"))
        p = provider(root, budget, user=user, transport=fake)
        if phase == "reserved":
            original = budget.reserve
            def reserve(*args):
                result = original(*args)
                pipe.send("reserved")
                signal.pause()
                return result
            budget.reserve = reserve
        elif phase == "sent":
            original = fake.open
            def opening(*args, **kwargs):
                response = original(*args, **kwargs)
                pipe.send("sent")
                signal.pause()
                return response
            fake.open = opening
        elif phase == "checkpoint":
            def settle(*args):
                pipe.send("checkpoint")
                signal.pause()
            budget.settle = settle
        if start:
            start.wait(10)
        try:
            p.embed(["synthetic one"])
            pipe.send({"status": "ok", "requests": len(fake.requests)})
        except ProviderUnavailable as exc:
            pipe.send({"status": str(exc), "requests": len(fake.requests)})
        finally:
            pipe.close()


class BudgetCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.policy = policy()
        self.budget = PersistentEmbeddingBudget.initialize(self.root / "budget", self.policy)
        network = patch("socket.socket", side_effect=AssertionError("NO_NETWORK"))
        network.start()
        self.addCleanup(network.stop)

    def reopen(self, policy_override=None):
        return PersistentEmbeddingBudget(self.root / "budget", policy_override or self.policy,
                                         ledger_id=self.budget.ledger_id)

    def test_cross_request_settlement_cache_hits_and_stop(self):
        p = provider(self.root, self.budget)
        expected = p.embed(["one"])
        self.assertEqual(p.last_attempt["state"], "SETTLED")
        self.assertEqual(self.reopen().snapshot()["occupied_nano_cny"], 500)
        self.budget.stop()
        replay = provider(self.root, self.reopen())
        self.assertEqual(replay.embed(["one"]), expected)
        self.assertEqual(replay._opener.requests, [])
        with self.assertRaisesRegex(ProviderUnavailable, "budget-stopped"):
            replay.embed(["two"])
        self.assertEqual(replay._opener.requests, [])

    def test_unknown_reservation_not_refunded_or_reset_on_reopen(self):
        attempt = self.budget.reserve(10, "0" * 64, synthetic_claim(10))
        self.assertEqual(self.reopen().snapshot()["occupied_nano_cny"], 8192 * 10 * 500)
        for invalid in (None, -1, True, 81921, 1.0):
            with self.assertRaises(ProviderUnavailable):
                self.budget.settle(attempt, invalid)
        self.assertEqual(self.budget.snapshot()["unknown"], 1)
        self.budget.settle(attempt, 10)
        with self.assertRaisesRegex(ProviderUnavailable, "already-settled"):
            self.budget.settle(attempt, 0)
        self.assertEqual(self.budget.snapshot()["occupied_nano_cny"], 5000)

    def test_external_receipt_binds_identity_and_never_initializes_missing_ledger(self):
        path = self.root / "receipt.json"
        path.write_text(json.dumps(self.budget.receipt()))
        path.chmod(0o600)
        with self.assertRaisesRegex(ProviderUnavailable, "test-policy-refused"):
            PersistentEmbeddingBudget.from_receipt(path)
        opened = PersistentEmbeddingBudget.from_receipt(path, allow_test_policy=True)
        self.assertEqual(opened.ledger_id, self.budget.ledger_id)
        internal = self.budget.root / "receipt.json"
        internal.write_bytes(path.read_bytes())
        internal.chmod(0o600)
        with self.assertRaisesRegex(ProviderUnavailable, "receipt-must-be-external"):
            PersistentEmbeddingBudget.from_receipt(internal, allow_test_policy=True)
        (self.budget.root / "ledger.sqlite3").unlink()
        (self.budget.root / "identity.json").unlink()
        with self.assertRaises(ProviderUnavailable):
            PersistentEmbeddingBudget.from_receipt(path, allow_test_policy=True)
        self.assertFalse((self.budget.root / "ledger.sqlite3").exists())

    def test_fail_closed_missing_corrupt_identity_policy_and_unwritable(self):
        with self.assertRaises(ProviderUnavailable):
            self.reopen(replace(self.policy, max_cost_nano_cny=self.policy.max_cost_nano_cny + 1))
        with self.assertRaises(ProviderUnavailable):
            PersistentEmbeddingBudget.initialize(self.root / "budget", self.policy)
        db = self.root / "budget/ledger.sqlite3"
        original = db.read_bytes()
        db.write_bytes(b"corrupted database")
        with self.assertRaises(ProviderUnavailable):
            self.reopen()
        db.write_bytes(original)
        db.chmod(0o400)
        with self.assertRaises(ProviderUnavailable):
            self.budget.reserve(1, "0" * 64, synthetic_claim(1))
        db.chmod(0o600)
        db.unlink()
        with self.assertRaises(ProviderUnavailable):
            self.reopen()
        (self.root / "budget/identity.json").unlink()
        with self.assertRaises(ProviderUnavailable):
            self.reopen()
        with self.assertRaises(ProviderUnavailable):
            PersistentEmbeddingBudget.initialize(self.root / "budget", self.policy)
        self.assertFalse(db.exists())

    def test_corrupt_row_and_marker_cannot_enable_send(self):
        self.budget.reserve(1, "1" * 64, synthetic_claim(1))
        with closing(sqlite3.connect(self.root / "budget/ledger.sqlite3")) as conn:
            conn.execute("UPDATE attempts SET reserved_tokens=0")
            conn.commit()
        with self.assertRaises(ProviderUnavailable):
            self.budget.reserve(1, "2" * 64, synthetic_claim(1))

    def test_two_processes_compete_for_one_reservation(self):
        configured = replace(self.policy, max_cost_nano_cny=8192 * 500, max_requests=1)
        other = self.root / "competition"
        other.mkdir()
        budget = PersistentEmbeddingBudget.initialize(other / "budget", configured)
        ctx = multiprocessing.get_context("spawn")
        start = ctx.Event()
        processes, readers = [], []
        for user in ("alpha", "beta"):
            reader, writer = ctx.Pipe(duplex=False)
            proc = ctx.Process(target=process_worker, args=(other, configured, budget.ledger_id,
                                                           start, writer, "compete", user))
            proc.start()
            writer.close()
            processes.append(proc)
            readers.append(reader)
        start.set()
        result = []
        for reader, proc in zip(readers, processes):
            self.assertTrue(reader.poll(15))
            result.append(reader.recv())
            proc.join(15)
            self.assertEqual(proc.exitcode, 0)
            reader.close()
        self.assertEqual(sum(x["requests"] for x in result), 1)
        self.assertEqual(sorted(x["status"] for x in result), ["embedding-budget-exhausted", "ok"])
        self.assertEqual(budget.snapshot()["requests"], 1)
        evidence("competing-processes", {"workers": result, "ledger": budget.snapshot()})

    def test_cost_reservation_blocks_other_process_and_survives_kill(self):
        configured = replace(self.policy, max_cost_nano_cny=8192 * 500)
        root = self.root / "cost-competition"
        root.mkdir()
        budget = PersistentEmbeddingBudget.initialize(root / "budget", configured)
        ctx = multiprocessing.get_context("spawn")
        reader, writer = ctx.Pipe(duplex=False)
        proc = ctx.Process(target=process_worker, args=(root, configured, budget.ledger_id,
                                                       None, writer, "reserved", "first"))
        proc.start()
        writer.close()
        try:
            self.assertTrue(reader.poll(15))
            self.assertEqual(reader.recv(), "reserved")
            second = provider(root, budget, user="second")
            with self.assertRaisesRegex(ProviderUnavailable, "budget-exhausted"):
                second.embed(["different request"])
            self.assertEqual(second._opener.requests, [])
        finally:
            os.kill(proc.pid, signal.SIGKILL)
            proc.join(15)
            reader.close()
        reopened = PersistentEmbeddingBudget(root / "budget", configured, ledger_id=budget.ledger_id)
        with self.assertRaisesRegex(ProviderUnavailable, "budget-exhausted"):
            provider(root, reopened, user="third").embed(["after restart"])
        evidence("cost-competition-kill", {"child_exit": proc.exitcode, "ledger": reopened.snapshot(),
                                            "second_sends": len(second._opener.requests)})

    def test_sigkill_before_send_after_send_and_after_checkpoint(self):
        ctx = multiprocessing.get_context("spawn")
        for phase in ("reserved", "sent", "checkpoint"):
            with self.subTest(phase=phase):
                root = self.root / phase
                root.mkdir()
                budget = PersistentEmbeddingBudget.initialize(root / "budget", self.policy)
                reader, writer = ctx.Pipe(duplex=False)
                proc = ctx.Process(target=process_worker, args=(root, self.policy, budget.ledger_id,
                                                               None, writer, phase, "user"))
                proc.start()
                writer.close()
                self.assertTrue(reader.poll(15))
                self.assertEqual(reader.recv(), phase)
                os.kill(proc.pid, signal.SIGKILL)
                proc.join(15)
                self.assertEqual(proc.exitcode, -signal.SIGKILL)
                reader.close()
                reopened = PersistentEmbeddingBudget(root / "budget", self.policy, ledger_id=budget.ledger_id)
                self.assertEqual(reopened.snapshot()["unknown"], 1)

                self.assertEqual(reopened.snapshot()["occupied_nano_cny"], 8192 * 500)
                wire = root / "user.wire"
                self.assertEqual(len(wire.read_text().splitlines()) if wire.exists() else 0,
                                 0 if phase == "reserved" else 1)
                replay = provider(root, reopened, user="user")
                if phase == "checkpoint":
                    replay.embed(["synthetic one"])
                else:
                    with self.assertRaisesRegex(ProviderUnavailable, "work-blocked"):
                        replay.embed(["synthetic one"])
                self.assertEqual(len(replay._opener.requests), 0)
                self.assertEqual(reopened.snapshot()["unknown"], 1)
                evidence("kill-" + phase, {"phase": phase, "exit_code": proc.exitcode,
                    "wire_calls_before_kill": len(wire.read_text().splitlines()) if wire.exists() else 0,
                    "replay_sends": len(replay._opener.requests), "ledger_after_replay": reopened.snapshot()})

    def test_same_namespace_concurrent_publish_sends_once(self):
        ctx = multiprocessing.get_context("spawn")
        start = ctx.Event()
        processes, readers = [], []
        for _ in range(2):
            reader, writer = ctx.Pipe(duplex=False)
            proc = ctx.Process(target=process_worker, args=(self.root, self.policy, self.budget.ledger_id,
                                                           start, writer, "compete", "same-user"))
            proc.start()
            writer.close()
            processes.append(proc)
            readers.append(reader)
        start.set()
        result = []
        for reader, proc in zip(readers, processes):
            self.assertTrue(reader.poll(15))
            result.append(reader.recv())
            proc.join(15)
            self.assertEqual(proc.exitcode, 0)
            reader.close()
        self.assertEqual([x["status"] for x in result], ["ok", "ok"])
        self.assertEqual(sum(x["requests"] for x in result), 1)
        self.assertEqual(self.budget.snapshot()["requests"], 1)
        evidence("concurrent-checkpoint", {"workers": result, "ledger": self.budget.snapshot()})

    def test_partial_second_batch_failure_reuses_ten_durable_raw_vectors(self):
        texts = ["unique-%02d" % i for i in range(11)]
        first = provider(self.root, self.budget, transport=FakeTransport(fail_batch=2))
        with self.assertRaisesRegex(ProviderUnavailable, "network-failed"):
            first.embed(texts)
        self.assertEqual([len(x["input"]) for x in first._opener.requests], [10, 1])
        second = provider(self.root, self.reopen())
        with self.assertRaisesRegex(ProviderUnavailable, "work-blocked"):
            second.embed(texts + [texts[0]])
        self.assertEqual(second._opener.requests, [])
        result = second.embed(texts[:10] + [texts[0]])
        self.assertEqual(result[0], result[-1])
        self.assertEqual(self.budget.snapshot()["unknown"], 1)
        self.assertEqual(self.budget.snapshot()["occupied_nano_cny"], (8192 + 10) * 500)
        self.assertEqual(self.budget.snapshot()["requests"], 2)
        evidence("partial-batch-recovery", {"initial_batches": [len(x["input"]) for x in first._opener.requests],
            "replay_batches": [], "missing_claim_blocked": True,
            "ledger": self.budget.snapshot(), "repeated_input_output_equal": result[0] == result[-1]})

    def test_valid_checkpoints_with_untrusted_usage_keep_unknown_cost(self):
        for name, usage in (("missing", None), ("negative", {"total_tokens": -1}),
                            ("boolean", {"total_tokens": True}),
                            ("disagree", {"total_tokens": 2, "prompt_tokens": 1}),
                            ("oversize", {"total_tokens": 8193})):
            with self.subTest(name=name):
                fake = FakeTransport(mutate=lambda obj, value=usage: obj.update(usage=value))
                p = provider(self.root, self.budget, user=name, transport=fake)
                with self.assertRaisesRegex(ProviderUnavailable, "usage-untrusted"):
                    p.embed(["known durable vector"])
                replay = provider(self.root, self.reopen(), user=name)
                replay.embed(["known durable vector"])
                self.assertEqual(replay._opener.requests, [])
        self.assertEqual(self.budget.snapshot()["unknown"], 5)
        self.assertEqual(self.budget.snapshot()["occupied_nano_cny"], 5 * 8192 * 500)
        self.assertTrue(self.budget.snapshot()["stopped"])
        evidence("untrusted-usage", self.budget.snapshot())

    def test_deadline_after_reserve_never_sends_or_refunds(self):
        now = [100.0]
        original = self.budget.reserve
        def reserve(*args):
            attempt = original(*args)
            now[0] += 6
            return attempt
        with patch("tiger_memory.remote_embeddings.monotonic", side_effect=lambda: now[0]):
            p = provider(self.root, self.budget, max_elapsed_seconds=5)
            with patch.object(self.budget, "reserve", side_effect=reserve):
                with self.assertRaisesRegex(ProviderUnavailable, "elapsed-budget-exhausted"):
                    p.embed(["deadline test"])
        self.assertEqual(p._opener.requests, [])
        self.assertEqual(self.budget.snapshot()["unknown"], 1)
        evidence("deadline-after-reserve", {"sends": 0, "ledger": self.budget.snapshot()})

    def test_invalid_responses_never_settle_or_publish(self):
        mutations = [lambda r: r.pop("model"), lambda r: r.update(model="wrong"),
            lambda r: r.update(data=[]), lambda r: r["data"][0].update(index=True),
            lambda r: r["data"][0].update(embedding=[1]),
            lambda r: r["data"][0].update(embedding=[float("nan")] * 64)]
        for i, mutate in enumerate(mutations):
            p = provider(self.root, self.budget, user="bad-%s" % i, transport=FakeTransport(mutate=mutate))
            with self.assertRaises(ProviderUnavailable):
                p.embed(["bad response test"])
            self.assertFalse(list(p._checkpoint.root.glob("raw-*.json")))
        self.assertEqual(self.budget.snapshot()["unknown"], len(mutations))

    def test_raw_chunk_order_fsum_and_float_hex_match_original_arithmetic(self):
        texts = ["中🙂a" * 7000, "x" * 8193, "tail", "中🙂a" * 7000]
        p = provider(self.root, self.budget)
        result = p.embed(texts)
        for text, actual in zip(texts, result):
            raw = [raw_vector(part, 64) for part in p._chunks(text)]
            mean = [math.fsum(v[i] / len(raw) for v in raw) for i in range(64)]
            scale = max(abs(v) for v in mean)
            scaled = [v / scale for v in mean]
            norm = math.hypot(*scaled)
            expected = [v / norm for v in scaled]
            self.assertEqual([v.hex() for v in actual], [v.hex() for v in expected])
        replay = provider(self.root, self.reopen())
        self.assertEqual(replay.embed(texts), result)
        self.assertEqual(replay._opener.requests, [])

    def test_checkpoint_corruption_half_write_and_write_failure(self):
        p = provider(self.root, self.budget)
        p.embed(["first"])
        target = next(p._checkpoint.root.glob("pooled-*.json"))
        target.write_text("{half-written")
        replay = provider(self.root, self.reopen())
        with self.assertRaisesRegex(ProviderUnavailable, "checkpoint-corrupt"):
            replay.embed(["first"])
        self.assertEqual(replay._opener.requests, [])
        q = provider(self.root, self.budget, user="write-failure")
        with patch("tiger_memory.embedding_checkpoint.os.replace", side_effect=OSError("SYNTHETIC")):
            with self.assertRaisesRegex(ProviderUnavailable, "checkpoint-write-failed"):
                q.embed(["second"])
        self.assertEqual(self.budget.snapshot()["unknown"], 1)
        self.assertEqual(len(q._opener.requests), 1)

    def test_namespaces_and_checksums_and_no_plaintext(self):
        for user in ("user-a", "user-b"):
            p = provider(self.root, self.budget, user=user)
            p.embed(["private-canary"])
            self.assertEqual(len(p._opener.requests), 1)
        for file in self.root.rglob("*.json"):
            text = file.read_text()
            for canary in ("private-canary", "SYNTHETIC_NOT_A_REAL_KEY", "user-a", "user-b"):
                self.assertNotIn(canary, text)
        self.assertEqual(self.budget.snapshot()["requests"], 2)

    def test_default_thousand_boundary_and_unicode_with_fake_transport(self):
        configured = replace(self.policy, max_cost_nano_cny=50_000_000_000)
        root = self.root / "boundary"
        root.mkdir()
        budget = PersistentEmbeddingBudget.initialize(root / "budget", configured)
        p = provider(root, budget)
        texts = ["unique-%s" % i for i in range(1000)]
        self.assertEqual(len(p.embed(texts)), 1000)
        self.assertEqual(len(p._opener.requests), 100)
        q = provider(root, budget, user="new-user")
        with self.assertRaisesRegex(ProviderUnavailable, "request-budget-exhausted"):
            q.embed(texts + ["one-more"])
        self.assertEqual(q._opener.requests, [])
        self.assertEqual(len(q._chunks("中🙂a" * 2048)), 2)
        evidence("chunk-boundary", {"chunks_1000_requests": len(p._opener.requests),
            "chunks_1001_requests": len(q._opener.requests), "ledger": budget.snapshot(),
            "official_input_envelope": "UNKNOWN"})

    def test_plan_is_read_only_uses_actual_cache_and_rechecks_after_changes(self):
        p = provider(self.root, self.budget)
        before = self.budget.snapshot()
        planned = p.plan(["first", "first", "second"])
        self.assertEqual((planned["input_count"], planned["unique_inputs"], planned["uncached_chunks"]), (3, 2, 2))
        self.assertFalse(p._checkpoint.root.exists())
        self.assertEqual(self.budget.snapshot(), before)
        self.assertEqual(p._opener.requests, [])
        p.embed(["first"])
        files = {str(f): hashlib.sha256(f.read_bytes()).hexdigest()
                 for f in p._checkpoint.root.rglob("*") if f.is_file()}
        replay = provider(self.root, self.reopen())
        planned = replay.plan(["first", "second"])
        self.assertEqual((planned["pooled_cache_hits"], planned["uncached_chunks"]), (1, 1))
        self.assertEqual(files, {str(f): hashlib.sha256(f.read_bytes()).hexdigest()
                                for f in p._checkpoint.root.rglob("*") if f.is_file()})
        # A snapshot is not a reservation. Deletion makes embed re-evaluate;
        # it cannot silently trust a formerly warm plan.
        for f in p._checkpoint.root.glob("*.json"):
            f.unlink()
        with self.assertRaisesRegex(ProviderUnavailable, "work-blocked"):
            replay.embed(["first", "second"])
        self.assertEqual(replay._opener.requests, [])
        evidence("readonly-plan", {"cold": p.plan(["third"]), "snapshot_atomic": False,
                                    "cache_change_rechecked": True})

    def test_plan_1001_cold_documents_and_partial_checkpoints(self):
        texts = ["unique-plan-%s" % i for i in range(1000)]
        p = provider(self.root, self.budget)
        self.assertTrue(p.plan(texts)["within_request_batch_limit"])
        self.assertFalse(p.plan(texts + ["query"])["within_request_batch_limit"])
        p.embed([texts[0]])
        planned = p.plan(texts + ["query"])
        # Existing instance spent one of its 100 batches; a new HTTP request
        # starts its per-request counter anew, without resetting shared money.
        self.assertFalse(planned["within_request_batch_limit"])
        q = provider(self.root, self.reopen())
        self.assertTrue(q.plan(texts + ["query"])["within_request_batch_limit"])
        failed = provider(self.root, self.budget, user="partial-plan", transport=FakeTransport(fail_batch=2))
        with self.assertRaises(ProviderUnavailable):
            failed.embed(texts[:11])
        snapshot = provider(self.root, self.reopen(), user="partial-plan").plan(texts[:11])
        self.assertEqual((snapshot["reusable_raw_chunks"], snapshot["uncached_chunks"], snapshot["required_batches"]),
                         (10, 1, 1))
        evidence("plan-cold-and-checkpoint", snapshot)

    def test_true_remote_without_budget_and_policy_mismatch_never_send(self):
        p = DashScopeEmbeddingProvider("https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings",
                                       "SYNTHETIC_NOT_A_REAL_KEY")
        p._opener = FakeTransport()
        with self.assertRaisesRegex(ProviderUnavailable, "budget-required"):
            p.embed(["must not leave"])
        self.assertEqual(p._opener.requests, [])
        with self.assertRaisesRegex(ProviderUnavailable, "provider-binding"):
            DashScopeEmbeddingProvider(self.policy.endpoint, "SYNTHETIC", dimensions=128,
                                       allow_test_loopback=True, budget=self.budget)

    def test_endpoint_drift_cannot_bypass_budget_via_test_loopback_flag(self):
        p = DashScopeEmbeddingProvider(self.policy.endpoint, "SYNTHETIC",
                                       allow_test_loopback=True)
        p._opener = FakeTransport()
        p.endpoint = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
        with self.assertRaisesRegex(ProviderUnavailable, "configuration-drift"):
            p.embed(["must not leave"])
        self.assertEqual(p._opener.requests, [])
        q = provider(self.root, self.budget)
        q.dimensions = 128
        with self.assertRaisesRegex(ProviderUnavailable, "provider-binding"):
            q.embed(["dimension drift"])
        self.assertEqual(self.budget.snapshot()["requests"], 0)


if __name__ == "__main__":
    unittest.main()
