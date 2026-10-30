"""Paid API calls are never used: every HTTP request targets a local stub."""
import json
import hashlib
import math
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from tiger_memory.providers import ProviderUnavailable, _digest
from tiger_memory.remote_embeddings import DashScopeEmbeddingProvider


class ProviderIdentityTests(unittest.TestCase):
    """Constructor metadata/cache identity checks; no sockets or API calls."""

    def test_dmx_description_is_accurate_without_changing_cache_fingerprint(self):
        key = "synthetic-key-only"
        for endpoint, name in (
            ("https://www.dmxapi.cn/v1/embeddings", "dmx-openai-compatible"),
            ("https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings",
             "dashscope-openai-compatible"),
        ):
            with self.subTest(endpoint=endpoint):
                provider = DashScopeEmbeddingProvider(endpoint, key)
                legacy_fingerprint = _digest({
                    "provider": "dashscope-openai-compatible", "schema": 2,
                    "endpoint": endpoint, "credential_hash": hashlib.sha256(key.encode()).hexdigest(),
                    "model": "text-embedding-v4", "dimensions": 1024,
                    "chunk_bytes": 8192, "aggregation": "ordered-fsum-mean-l2-v1",
                    "region": "unbudgeted-test",
                })
                description = provider.describe()
                self.assertEqual(description["provider"], name)
                self.assertEqual(description["fingerprint"], legacy_fingerprint)
                self.assertEqual(description["request_batches"], 0)
                self.assertNotIn(key, json.dumps(description))

    def test_endpoints_and_credentials_keep_separate_cache_identities(self):
        dmx = DashScopeEmbeddingProvider("https://www.dmxapi.cn/v1/embeddings", "synthetic-key")
        aliyun = DashScopeEmbeddingProvider(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings", "synthetic-key")
        other_key = DashScopeEmbeddingProvider("https://www.dmxapi.cn/v1/embeddings", "other-synthetic-key")
        self.assertEqual(len({p.describe()["fingerprint"] for p in (dmx, aliyun, other_key)}), 3)


class RemoteEmbeddingsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.requests = []
        self.respond = self.success
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.requests.append((self.path, payload, self.headers.get("Authorization")))
                status, headers, result = owner.respond(payload)
                self.send_response(status)
                for key, value in headers.items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(result if isinstance(result, bytes) else json.dumps(result).encode())

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop)
        self.endpoint = "http://127.0.0.1:%s/embeddings" % self.server.server_port

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    @staticmethod
    def vector(text, dimensions):
        return [float(len(text)), float(sum(map(ord, text)) % 17 + 1)] + [0.0] * (dimensions - 2)

    def success(self, payload):
        return 200, {}, {"data": [
            {"index": index, "embedding": self.vector(text, payload["dimensions"])}
            for index, text in reversed(list(enumerate(payload["input"])))
        ]}

    def provider(self, **kwargs):
        args = dict(endpoint=self.endpoint, api_key="secret-test-key", dimensions=64,
                    cache_dir=Path(self.temp.name).resolve(), allow_test_loopback=True)
        args.update(kwargs)
        return DashScopeEmbeddingProvider(**args)

    def test_reorders_indices_batches_deduplicates_and_normalizes(self):
        texts = ["sample-%s" % index for index in range(23)]
        provider = self.provider()
        result = provider.embed(texts + [texts[0]])
        self.assertEqual([len(item[1]["input"]) for item in self.requests], [10, 10, 3])
        self.assertEqual(result[0], result[-1])
        for text, vector in zip(texts, result):
            original = self.vector(text, 64)
            norm = math.hypot(*original)
            self.assertAlmostEqual(vector[0], original[0] / norm)
            self.assertAlmostEqual(math.hypot(*vector), 1.0)
        self.assertEqual(self.requests[0][2], "Bearer secret-test-key")
        self.assertEqual(self.requests[0][1]["model"], "text-embedding-v4")
        self.assertEqual(self.requests[0][1]["encoding_format"], "float")
        self.assertEqual(provider.describe()["request_batches"], 3)

    def test_long_unicode_text_preserved_and_mean_aggregated(self):
        text = "中🙂a" * 5000
        vector = self.provider().embed([text])[0]
        chunks = [part for _, payload, _ in self.requests for part in payload["input"]]
        self.assertEqual("".join(chunks), text)
        self.assertTrue(all(len(part.encode("utf-8")) <= 8192 for part in chunks))
        originals = [self.vector(part, 64) for part in chunks]
        mean = [sum(row[axis] for row in originals) / len(originals) for axis in range(64)]
        self.assertAlmostEqual(vector[0], mean[0] / math.hypot(*mean))

    def test_persistent_cache_contains_no_plaintext_and_is_isolated(self):
        text = "private-input-value"
        first = self.provider().embed([text])
        self.assertEqual(first, self.provider().embed([text]))
        self.assertEqual(len(self.requests), 1)
        for path in Path(self.temp.name).rglob("*.json"):
            contents = path.read_text()
            self.assertNotIn(text, contents)
            self.assertNotIn("secret-test-key", contents)
            self.assertEqual(set(json.loads(contents)), {"schema", "fingerprint", "namespace",
                "dimensions", "kind", "input_sha256", "values", "checksum"})
        self.provider(dimensions=128).embed([text])
        self.provider(api_key="another-key").embed([text])
        self.provider(endpoint=self.endpoint + "/other").embed([text])
        self.assertEqual(len(self.requests), 4)
        description = json.dumps(self.provider().describe())
        self.assertNotIn("secret-test-key", description)
        self.assertNotIn(self.endpoint, description)

    def test_empty_sequence_and_invalid_inputs_do_not_request(self):
        provider = self.provider()
        self.assertEqual(provider.embed([]), [])
        for invalid in ["text", [""], [" \t"], [None], [42], None, ["\ud800"]]:
            with self.subTest(invalid=repr(invalid)), self.assertRaises(ProviderUnavailable):
                provider.embed(invalid)
        self.assertFalse(self.requests)

    def test_rejects_invalid_output(self):
        good = {"index": 0, "embedding": [1.0] * 64}
        variants = [
            {}, {"data": []}, {"data": [None]},
            {"model": "wrong-model", "data": [good]},
            {"data": [dict(good, index=True)]}, {"data": [dict(good, index=1)]},
            {"data": [dict(good, embedding=[1.0])]},
            {"data": [dict(good, embedding=[float("nan")] * 64)]},
            {"data": [dict(good, embedding=[float("inf")] * 64)]},
            {"data": [dict(good, embedding=[True] * 64)]},
            {"data": [dict(good, embedding=["1"] * 64)]},
            {"data": [dict(good, embedding=[0.0] * 64)]}, b"not-json-secret",
        ]
        for payload in variants:
            with self.subTest(payload=repr(payload)):
                self.respond = lambda request, result=payload: (200, {}, result)
                with self.assertRaises(ProviderUnavailable) as caught:
                    self.provider().embed(["private-input"])
                self.assertNotIn("secret", str(caught.exception))
                self.assertNotIn("private-input", str(caught.exception))
        self.respond = lambda request: (200, {}, {"data": [good, good]})
        with self.assertRaisesRegex(ProviderUnavailable, "invalid-embedding-index"):
            self.provider().embed(["one", "two"])

    def test_http_failure_and_redirect_are_redacted_and_never_followed(self):
        for status in (302, 307, 401, 429, 500):
            self.requests.clear()
            self.respond = lambda request, code=status: (
                code, {"Location": self.endpoint + "/should-not-follow"},
                {"error": "private-input secret-test-key"})
            provider = self.provider()
            with self.assertRaisesRegex(ProviderUnavailable, "^embedding-http-%s$" % status):
                provider.embed(["private-input"])
            self.assertEqual(len(self.requests), 1)
            self.assertEqual(provider.describe()["status"], "unavailable")

    def test_environment_proxies_are_not_used(self):
        with patch.dict("os.environ", {"http_proxy": "http://127.0.0.1:1",
                                      "HTTP_PROXY": "http://127.0.0.1:1", "no_proxy": ""}):
            self.provider().embed(["sample"])
        self.assertEqual(len(self.requests), 1)

    def test_budget_is_shared_across_embed_calls_and_cache_hits_are_free(self):
        provider = self.provider(max_request_batches=1)
        provider.embed(["sample"])
        provider.embed(["sample"])
        with self.assertRaisesRegex(ProviderUnavailable, "embedding-request-budget-exhausted"):
            provider.embed(["different"])
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(provider.describe()["request_batches"], 1)

    def test_network_failure_is_redacted(self):
        provider = self.provider()
        with patch.object(provider._opener, "open", side_effect=TimeoutError("secret-test-key")):
            with self.assertRaisesRegex(ProviderUnavailable, "^embedding-network-failed$"):
                provider.embed(["private-input"])

    def test_over_budget_fails_before_any_paid_work(self):
        provider = self.provider(max_request_batches=1)
        with self.assertRaisesRegex(ProviderUnavailable, "embedding-request-budget-exhausted"):
            provider.embed(["item-%s" % index for index in range(11)])
        self.assertFalse(self.requests)
        self.assertEqual(provider.request_batches, 0)

    def test_elapsed_budget_default_and_description_are_explicit(self):
        now = [100.0]
        with patch("tiger_memory.remote_embeddings.monotonic", side_effect=lambda: now[0]):
            provider = self.provider()
            now[0] += 5
            report = provider.describe()
        self.assertEqual(report["elapsed_seconds"], 5)
        self.assertEqual(report["settings"]["max_elapsed_seconds"], 1200)
        self.assertEqual(report["settings"]["elapsed_budget_policy"],
                         "batch-boundary-checks-not-hard-deadline")
        self.assertFalse(self.requests)

    def test_elapsed_boundary_blocks_uncached_work_but_allows_cache(self):
        now = [100.0]
        with patch("tiger_memory.remote_embeddings.monotonic", side_effect=lambda: now[0]):
            provider = self.provider(max_elapsed_seconds=5)
            expected = provider.embed(["cached"])
            now[0] += 5
            self.assertEqual(provider.embed(["cached"]), expected)
            self.assertEqual(provider.embed([]), [])
            with self.assertRaisesRegex(ProviderUnavailable, "embedding-elapsed-budget-exhausted"):
                provider.embed(["uncached"])
            self.assertEqual(provider.describe()["fallback_reason"], "embedding-elapsed-budget-exhausted")
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(provider.request_batches, 1)

    def test_elapsed_budget_is_shared_and_reduces_socket_timeout(self):
        now = [100.0]
        with patch("tiger_memory.remote_embeddings.monotonic", side_effect=lambda: now[0]):
            provider = self.provider(timeout=10, max_elapsed_seconds=25)
            with patch.object(provider._opener, "open", wraps=provider._opener.open) as opening:
                provider.embed(["first"])
                self.assertEqual(opening.call_args.kwargs["timeout"], 10)
                now[0] += 22
                provider.embed(["second"])
                self.assertEqual(opening.call_args.kwargs["timeout"], 3)
                now[0] += 3
                with self.assertRaisesRegex(ProviderUnavailable, "embedding-elapsed-budget-exhausted"):
                    provider.embed(["third"])
                self.assertEqual(opening.call_count, 2)

    def test_elapsed_overshoot_after_response_stops_next_batch(self):
        now = [100.0]
        def response(payload):
            now[0] += 6
            return self.success(payload)
        self.respond = response
        with patch("tiger_memory.remote_embeddings.monotonic", side_effect=lambda: now[0]):
            provider = self.provider(max_elapsed_seconds=5)
            with self.assertRaisesRegex(ProviderUnavailable, "embedding-elapsed-budget-exhausted"):
                provider.embed(["row-%s" % index for index in range(11)])
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(provider.request_batches, 1)
        self.assertEqual(len(list(Path(self.temp.name).rglob("raw-*.json"))), 10)
        self.assertFalse(list(Path(self.temp.name).rglob("pooled-*.json")))

    def test_constructor_rejects_unsafe_or_invalid_configuration(self):
        for endpoint in ["http://example.com/embeddings", self.endpoint,
                         "https://user:secret@example.com/embeddings",
                         "https://example.com/embeddings?key=secret", "ftp://example.com/x",
                         "https://lewen.woa.com/", "https://lewen.woa.com./", "https://example.com:bad/x"]:
            with self.subTest(endpoint=endpoint), self.assertRaises(ProviderUnavailable):
                self.provider(endpoint=endpoint, allow_test_loopback=False)
        for overrides in [{"dimensions": 65}, {"dimensions": True}, {"timeout": 0},
                          {"timeout": 61}, {"timeout": float("nan")}, {"timeout": True},
                          {"api_key": ""}, {"api_key": "key\n"}, {"max_request_batches": 0},
                          {"max_elapsed_seconds": 0}, {"max_elapsed_seconds": -1},
                          {"max_elapsed_seconds": True}, {"max_elapsed_seconds": "5"},
                          {"max_elapsed_seconds": float("nan")}, {"max_elapsed_seconds": float("inf")}]:
            with self.subTest(overrides=overrides), self.assertRaises(ProviderUnavailable):
                self.provider(**overrides)
        self.assertFalse(self.requests)


if __name__ == "__main__":
    unittest.main()
