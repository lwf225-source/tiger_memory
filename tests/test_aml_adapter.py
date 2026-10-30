"""End-to-end AML adapter tests: a simulated platform talks real HTTP to the
adapter, which maps Add/Search onto the PLM service layer."""

from __future__ import annotations

import http.client
import json
import os
import threading
import unittest
from unittest import mock

from tiger_memory import aml_adapter

from tests.common import MemoryTestCase


class AMLServerCase(MemoryTestCase):
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

    # -- raw HTTP helpers (simulated platform) ---------------------------
    def request(self, method, path, payload=None, raw_body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            if raw_body is not None:
                body = raw_body
            elif payload is not None:
                body = json.dumps(payload)
            else:
                body = None
            conn.request(method, path, body=body, headers=headers or {})
            response = conn.getresponse()
            data = response.read()
            parsed = json.loads(data.decode("utf-8")) if data else None
            return response.status, parsed
        finally:
            conn.close()

    def add(self, request_id, user_id, session_id, texts, base_ms=1704067200000):
        payload = {
            "request_id": request_id,
            "user_id": user_id,
            "session_id": session_id,
            "messages": [
                {
                    "role": "user" if index % 2 == 0 else "assistant",
                    "timestamp": base_ms + index * 60000,
                    "content": text,
                }
                for index, text in enumerate(texts)
            ],
        }
        return self.request("POST", "/add", payload)

    def search(self, query, user_id, top_k=100, **extra):
        payload = {"query": query, "user_id": user_id, "top_k": top_k}
        payload.update(extra)
        return self.request("POST", "/search", payload)


class TestHTTPTransport(unittest.TestCase):
    def test_pending_connection_burst_is_served_after_accept_resumes(self):
        server = aml_adapter.make_server("127.0.0.1", 0)
        connections = []
        thread = None
        try:
            # Deliberately pause accept(), as can happen during scheduling
            # under load. This isolates TCP queue capacity from Add/storage.
            for _ in range(16):
                conn = http.client.HTTPConnection(
                    "127.0.0.1", server.server_address[1], timeout=2)
                connections.append(conn)
                conn.request("GET", "/health")
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            for conn in connections:
                response = conn.getresponse()
                self.assertEqual(response.status, 200)
                self.assertEqual(json.loads(response.read())["status"], "ok")
        finally:
            for conn in connections:
                conn.close()
            if thread is not None:
                server.shutdown()
                thread.join(timeout=5)
            server.server_close()


class TestContract(AMLServerCase):
    def test_health_is_unauthenticated(self):
        status, body = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ok")

    def test_unknown_path_is_404(self):
        status, _ = self.request("POST", "/status", {"x": 1})
        self.assertEqual(status, 404)
        status, _ = self.request("GET", "/add")
        self.assertEqual(status, 404)

    def test_wrong_method_is_405(self):
        status, _ = self.request("PUT", "/add", {"x": 1})
        self.assertEqual(status, 405)

    def test_malformed_json_is_400(self):
        status, body = self.request("POST", "/add", raw_body="{not json")
        self.assertEqual(status, 400)
        self.assertIn("detail", body)

    def test_add_echoes_ids_exactly(self):
        status, body = self.add("req-1", "user-1", "session-1", ["hello world"])
        self.assertEqual(status, 200)
        self.assertEqual(body, {
            "success": True,
            "request_id": "req-1",
            "user_id": "user-1",
            "session_id": "session-1",
        })

    def test_add_validation_matrix(self):
        valid = {
            "request_id": "r", "user_id": "u", "session_id": "s",
            "messages": [{"role": "user", "content": "text", "timestamp": 1704067200000}],
        }
        bad_payloads = [
            {},
            {"user_id": "u", "session_id": "s", "messages": valid["messages"]},
            dict(valid, request_id=""),
            dict(valid, request_id="  "),
            dict(valid, user_id=123),
            dict(valid, session_id=None),
            dict(valid, messages=[]),
            dict(valid, messages={}),
            dict(valid, messages=[1]),
            dict(valid, messages=[{"content": "text"}]),
            dict(valid, messages=[{"role": "", "content": "text"}]),
            dict(valid, messages=[{"role": "user"}]),
            dict(valid, messages=[{"role": "user", "content": "  "}]),
            dict(valid, messages=[{"role": "user", "content": "text", "timestamp": "1704067200000"}]),
            dict(valid, messages=[{"role": "user", "content": "text", "timestamp": True}]),
            dict(valid, messages=[{"role": "user", "content": "text", "timestamp": 1704067200000.7}]),
            "not-a-dict",
        ]
        for payload in bad_payloads:
            status, body = self.request("POST", "/add", payload)
            self.assertEqual(status, 422, "payload should be rejected: %r" % (payload,))
            self.assertIn("detail", body)
        # Lossless float timestamps and undeclared extra fields are tolerated.
        tolerant = dict(valid, request_id="r-tolerant", metadata={"ignored": True},
                        messages=[{"role": "user", "content": "text", "timestamp": 1704067200000.0,
                                   "name": "ignored"}])
        status, body = self.request("POST", "/add", tolerant)
        self.assertEqual(status, 200)
        self.assertTrue(body["success"])

    def test_search_validation_matrix(self):
        base = {"query": "q", "user_id": "u", "top_k": 100}
        bad_payloads = [
            {},
            {"user_id": "u", "top_k": 100},
            dict(base, query=""),
            dict(base, user_id=""),
            dict(base, top_k=None),
            {k: v for k, v in base.items() if k != "top_k"},
            dict(base, top_k="100"),
            dict(base, top_k=100.0),
            dict(base, top_k=True),
            dict(base, top_k=-1),
            dict(base, options="A"),
            dict(base, options=[1, 2]),
        ]
        for payload in bad_payloads:
            status, _ = self.request("POST", "/search", payload)
            self.assertEqual(status, 422, "payload should be rejected: %r" % (payload,))
        # top_k above the platform maximum clamps instead of failing.
        status, body = self.search("q", "u", top_k=10 ** 6)
        self.assertEqual(status, 200)
        self.assertEqual(body, {"data": []})

    def test_optional_auth(self):
        os.environ["AML_API_KEY"] = "secret-token"
        try:
            status, _ = self.add("req-auth", "user-auth", "s", ["token text"])
            self.assertEqual(status, 401)
            for header in ("Bearer secret-token", "Token secret-token"):
                status, body = self.request(
                    "POST", "/add",
                    {"request_id": "req-auth", "user_id": "user-auth", "session_id": "s",
                     "messages": [{"role": "user", "content": "token text"}]},
                    headers={"Authorization": header})
                self.assertEqual(status, 200, header)
                self.assertTrue(body["success"])
            status, _ = self.request("POST", "/search",
                                     {"query": "token", "user_id": "user-auth", "top_k": 5},
                                     headers={"X-Api-Key": "secret-token"})
            self.assertEqual(status, 200)
            # Health stays public under auth.
            status, _ = self.request("GET", "/health")
            self.assertEqual(status, 200)
        finally:
            os.environ.pop("AML_API_KEY", None)


class TestAddSearchFlow(AMLServerCase):
    def test_add_then_search_returns_timestamped_evidence(self):
        status, _ = self.add(
            "req-1", "user-1", "session-1",
            ["My favorite tea is high-mountain oolong from Alishan."],
            base_ms=1704067200000,
        )
        self.assertEqual(status, 200)
        status, body = self.search("What tea does the user like?", "user-1")
        self.assertEqual(status, 200)
        self.assertEqual(len(body["data"]), 1)
        item = body["data"][0]
        self.assertTrue(item["id"])
        self.assertIn("high-mountain oolong", item["content"])
        self.assertIsInstance(item["score"], (int, float))
        # 1704067200000 ms == 2024-01-01T00:00:00Z
        self.assertTrue(item["created_at"].startswith("2024-01-01T00:00:00"), item["created_at"])

    def test_search_is_immediately_visible_after_add_200(self):
        # The Add 200 promise: no rebuild or delay before Search can see it.
        for index in range(3):
            status, _ = self.add("req-%d" % index, "user-imm", "s",
                                 ["immediacy token zqvx%d" % index])
            self.assertEqual(status, 200)
            status, body = self.search("zqvx%d" % index, "user-imm")
            self.assertEqual(status, 200)
            hits = [item for item in body["data"] if "zqvx%d" % index in item["content"]]
            self.assertEqual(len(hits), 1)

    def test_user_isolation(self):
        self.add("req-a", "user-a", "s-a", ["Alice keeps her spare key under the blue flowerpot."])
        self.add("req-b", "user-b", "s-b", ["The wifi password is mango-sunrise-42."])
        status, body = self.search("wifi password", "user-a")
        self.assertEqual(status, 200)
        self.assertEqual(body["data"], [])
        status, body = self.search("wifi password", "user-b")
        self.assertEqual(status, 200)
        self.assertEqual(len(body["data"]), 1)
        self.assertIn("mango-sunrise-42", body["data"][0]["content"])
        # Cross-contamination in the other direction is also impossible.
        status, body = self.search("spare key flowerpot", "user-b")
        self.assertEqual(body["data"], [])

    def test_retry_is_idempotent(self):
        texts = ["idempotency token qwplm-7 appears exactly once."]
        for _ in range(3):
            status, body = self.add("req-dup", "user-dup", "s", texts)
            self.assertEqual(status, 200)
            self.assertTrue(body["success"])
        status, body = self.search("qwplm-7", "user-dup")
        self.assertEqual(status, 200)
        matching = [item for item in body["data"] if "qwplm-7" in item["content"]]
        self.assertEqual(len(matching), 1)

    def test_same_request_id_different_payload_is_first_write_wins(self):
        self.add("req-fww", "user-fww", "s", ["original content token aaa-first"])
        self.add("req-fww", "user-fww", "s", ["replacement content token bbb-second"])
        status, body = self.search("aaa-first", "user-fww")
        self.assertEqual(len(body["data"]), 1)
        status, body = self.search("bbb-second", "user-fww")
        self.assertEqual(body["data"], [])

    def test_same_request_id_different_user_is_independent(self):
        self.add("req-shared", "user-x", "s", ["token ccc-userx"])
        self.add("req-shared", "user-y", "s", ["token ddd-usery"])
        _, body_x = self.search("ccc-userx", "user-x")
        _, body_y = self.search("ddd-usery", "user-y")
        self.assertEqual(len(body_x["data"]), 1)
        self.assertEqual(len(body_y["data"], ), 1)

    def test_top_k_limits_results(self):
        texts = ["top-k probe token eee-%02d" % index for index in range(12)]
        self.add("req-tk", "user-tk", "s", texts)
        status, body = self.search("top-k probe token", "user-tk", top_k=5)
        self.assertEqual(status, 200)
        self.assertLessEqual(len(body["data"]), 5)
        self.assertGreater(len(body["data"]), 0)
        status, body = self.search("top-k probe token", "user-tk", top_k=100)
        self.assertLessEqual(len(body["data"]), 100)
        status, body = self.search("top-k probe token", "user-tk", top_k=0)
        self.assertEqual(body, {"data": []})

    def test_options_contribute_to_recall_but_never_replace_query(self):
        self.add("req-opt", "user-opt", "s", ["She mentioned the passcode is delta-99."])
        # The query alone shares no terms with the memory.
        _, empty = self.search("xyzzy unrelated", "user-opt")
        self.assertEqual(empty["data"], [])
        # Choice-question options join the recall query.
        status, body = self.search("xyzzy unrelated", "user-opt", options=["A. delta-99", "B. gamma-11"])
        self.assertEqual(status, 200)
        self.assertEqual(len(body["data"]), 1)
        self.assertIn("delta-99", body["data"][0]["content"])

    def test_search_unknown_user_is_empty_200(self):
        status, body = self.search("anything", "user-never-seen")
        self.assertEqual(status, 200)
        self.assertEqual(body, {"data": []})

    def test_search_degrades_to_empty_200_on_internal_error(self):
        self.add("req-err", "user-err", "s", ["resilience token fff-res"])
        with mock.patch.object(aml_adapter.service, "context_data", side_effect=RuntimeError("boom")):
            status, body = self.search("fff-res", "user-err")
        self.assertEqual(status, 200)
        self.assertEqual(body, {"data": []})

    def test_concurrent_adds_from_platform_workers(self):
        errors = []
        start = threading.Barrier(8)

        def worker(index):
            try:
                start.wait(timeout=10)
                status, _ = self.add("req-c%d" % index, "user-c", "s-%d" % index,
                                     ["concurrency token ggg-%d" % index])
                if status != 200:
                    errors.append("status %d for worker %d" % (status, index))
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(repr(exc))

        threads = [threading.Thread(target=worker, args=(index,)) for index in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        _, body = self.search("concurrency token", "user-c")
        found = {item["content"] for item in body["data"]}
        for index in range(8):
            self.assertIn("concurrency token ggg-%d" % index, found)


if __name__ == "__main__":
    unittest.main()
