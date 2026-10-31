"""Synthetic loopback HTTP framing regressions; never call a model."""
import http.client
import json
import socket
from unittest.mock import patch

from tiger_memory import aml_adapter as aml
from tests.test_aml_adapter import AMLServerCase


class HTTPBodyLengthTests(AMLServerCase):
    def setUp(self):
        super().setUp()
        flags = patch.multiple(aml, AML_EMBEDDING="off", AML_RERANKER="off",
                               AML_QUERY_REWRITE="off", AML_FACT_PROJECTION="off")
        flags.start()
        self.addCleanup(flags.stop)
        self.payload = {"request_id": "synthetic-r", "user_id": "synthetic-u",
                        "session_id": "synthetic-s", "messages": [{"role": "user", "content": "violet"}]}
        body = json.dumps(self.payload, separators=(",", ":")).encode()
        self.assertLessEqual(len(body), 176)
        self.body = body + b" " * (176 - len(body))

    def raw(self, body, declared, *, shutdown=True, trailing=b""):
        sock = socket.create_connection(("127.0.0.1", self.port), timeout=5)
        self.addCleanup(sock.close)
        header = b"POST /add HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n"
        if declared is not None:
            header += b"Content-Length: " + str(declared).encode() + b"\r\n"
        sock.sendall(header + b"\r\n" + body + trailing)
        if shutdown:
            sock.shutdown(socket.SHUT_WR)
        response = http.client.HTTPResponse(sock)
        response.begin()
        return sock, response

    def test_complete_json_short_entity_400_closed_and_zero_business_calls(self):
        with patch.object(aml, "add", wraps=aml.add) as add, \
                patch.object(aml, "_model_provider", side_effect=AssertionError("provider forbidden")) as provider:
            sock, response = self.raw(self.body, 276)
            self.assertEqual(response.status, 400)
            self.assertEqual(response.getheader("Connection"), "close")
            self.assertIn("length", json.loads(response.read())["detail"]["reason"])
            self.assertEqual(sock.recv(1), b"")
            add.assert_not_called()
            provider.assert_not_called()
        self.assertFalse(self.base.exists())

    def test_partial_json_eof_zero_write_then_new_connection_recovers(self):
        with patch.object(aml, "add", wraps=aml.add) as add:
            _, response = self.raw(self.body[:30], 176)
            self.assertEqual(response.status, 400)
            response.read()
            add.assert_not_called()
        self.assertFalse(self.base.exists())
        status, body = self.request("POST", "/add", payload=self.payload)
        self.assertEqual(status, 200)
        self.assertTrue(body["success"])
        self.assertEqual(len(list(self.base.rglob("*.md"))), 1)

    def test_equal_entity_succeeds_and_does_not_consume_next_request(self):
        # Extra bytes are the next HTTP request, not extra bytes of this entity.
        tail = b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
        with socket.create_connection(("127.0.0.1", self.port), timeout=5) as sock:
            sock.sendall(b"POST /add HTTP/1.1\r\nHost: localhost\r\nContent-Length: 176\r\n\r\n" + self.body + tail)
            sock.shutdown(socket.SHUT_WR)
            chunks = []
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)
        raw = b"".join(chunks)
        self.assertEqual(raw.count(b"HTTP/1.1 200"), 2)
        self.assertIn(b'"success": true', raw)
        self.assertIn(b'"status": "ok"', raw)
        self.assertEqual(len(list(self.base.rglob("*.md"))), 1)

    def test_missing_invalid_and_over_limit_lengths_do_not_write(self):
        for length in (None, "not-an-integer", -1, 0, aml.AML_MAX_REQUEST_BYTES + 1):
            with self.subTest(length=length), patch.object(aml, "add") as add:
                _, response = self.raw(b"", length)
                self.assertEqual(response.status, 400)
                response.read()
                add.assert_not_called()
        self.assertFalse(self.base.exists())

    def test_equal_length_invalid_json_is_400_and_zero_write(self):
        with patch.object(aml, "add") as add:
            body = b'{"not-complete":'
            _, response = self.raw(body, len(body))
            self.assertEqual(response.status, 400)
            response.read()
            add.assert_not_called()
        self.assertFalse(self.base.exists())
