"""Offline privacy and no-wait checks; no real provider/model requests."""
import email.message
import hashlib
import http.client
import io
import json
import socket
import time
import unittest
from urllib.error import HTTPError

from tiger_memory.transport_diagnostics import (
    MAX_BODY_BYTES, http_error_metadata, sanitize_http_error_metadata,
)

SECRET = "sk-DO-NOT-EMIT-credential-body-header-user-source"


def error(body, status=400, headers=None):
    return HTTPError("https://invalid.example/private?token=" + SECRET,
                     status, SECRET, headers, io.BytesIO(body))


class TransportDiagnosticsTests(unittest.TestCase):
    def test_known_code_and_type_without_message(self):
        body = json.dumps({"error": {"code": "context_length_exceeded",
                           "type": "invalid_request_error", "message": SECRET,
                           "request": {"input": SECRET}}, "api_key": SECRET}).encode()
        result = http_error_metadata(error(body))
        self.assertEqual(result["error_code"], "context_length_exceeded")
        self.assertEqual(result["error_type"], "invalid_request_error")
        self.assertEqual(result["http_status"], 400)
        self.assertNotIn(SECRET, json.dumps(result))
        self.assertNotIn("message", result)

    def test_unknown_codes_not_copied_even_identifier_shaped(self):
        result = http_error_metadata(error(json.dumps({"error": {
            "code": SECRET, "type": "a_private_user_label", "message": SECRET}}).encode()))
        self.assertNotIn("error_code", result)
        self.assertNotIn("error_type", result)
        self.assertNotIn(SECRET, json.dumps(result))

    def test_nested_and_nonstring_code_ignored(self):
        for value in ({"code": SECRET}, [SECRET], 123, None, True):
            with self.subTest(value=type(value)):
                result = http_error_metadata(error(json.dumps({"error": {
                    "code": value, "type": value}}).encode()))
                self.assertNotIn("error_code", result)
                self.assertNotIn("error_type", result)
                self.assertNotIn(SECRET, json.dumps(result))

    def test_top_level_dashscope_code(self):
        result = http_error_metadata(error(json.dumps({"code": "InvalidParameter",
                                             "message": SECRET}).encode()))
        self.assertEqual(result["error_code"], "InvalidParameter")

    def test_oversize_never_parse_truncated_body(self):
        prefix = b'{"error":{"code":"InvalidParameter"},"message":"'
        result = http_error_metadata(error(prefix + b'x' * (MAX_BODY_BYTES * 2)))
        self.assertEqual(result["body_state"], "oversized")
        self.assertEqual(result["body_bytes_observed"], MAX_BODY_BYTES + 1)
        self.assertNotIn("error_code", result)

    def test_malformed_and_non_json_bounded(self):
        for body in (b'{', b'\xff', SECRET.encode()):
            result = http_error_metadata(error(body))
            self.assertEqual(result["body_state"], "incomplete_or_invalid_json")
            self.assertNotIn(SECRET, json.dumps(result))

    def test_deep_json_is_bounded_and_not_walked(self):
        result = http_error_metadata(error(b'[' * 2000 + b']' * 2000))
        self.assertIn(result['body_state'], ('incomplete_or_invalid_json', 'unstructured_json'))
        self.assertEqual(result['body_bytes_observed'], 4000)

    def test_unstructured_and_empty(self):
        for body in (b'[]', b'"private string"', b'42', b'null'):
            self.assertEqual(http_error_metadata(error(body))["body_state"], "unstructured_json")
        self.assertEqual(http_error_metadata(error(b''))["body_state"], "empty")

    def test_identifier_hashed_never_raw(self):
        headers = email.message.Message()
        headers['x-request-id'] = SECRET
        headers['Authorization'] = SECRET
        result = http_error_metadata(error(b'{}', headers=headers))
        self.assertEqual(result['request_id_sha256'], hashlib.sha256(SECRET.encode()).hexdigest())
        self.assertNotIn(SECRET, json.dumps(result))
        self.assertNotIn('Authorization', result)

    def test_invalid_identifier_and_unknown_headers_ignored(self):
        headers = email.message.Message()
        headers['x-request-id'] = 'a' * 129
        headers['X-Arbitrary'] = SECRET
        result = http_error_metadata(error(b'{}', headers=headers))
        self.assertNotIn('request_id_sha256', result)
        self.assertNotIn(SECRET, json.dumps(result))

    def test_body_identifier_is_hashed(self):
        result = http_error_metadata(error(json.dumps({'request_id': SECRET}).encode()))
        self.assertEqual(result['request_id_sha256'], hashlib.sha256(SECRET.encode()).hexdigest())

    def test_revalidation_strips_arbitrary_attached_data(self):
        result = sanitize_http_error_metadata({
            'schema_version': 1, 'http_status': 400, 'message': SECRET,
            'error_code': SECRET, 'error_type': SECRET, 'body_state': SECRET,
            'request_id_sha256': SECRET, 'body_bytes_observed': 999999,
            'headers': {'Authorization': SECRET}})
        self.assertEqual(result, {'schema_version': 1, 'http_status': 400})
        for value in (None, [], {'schema_version': True, 'http_status': 400},
                      {'schema_version': 1, 'http_status': True}):
            self.assertIsNone(sanitize_http_error_metadata(value))

    def test_unknown_stream_is_never_read(self):
        class UnknownStream:
            def read(self, *a):
                raise AssertionError('must not read unknown stream')
        exc = error(b'{}')
        exc.fp = UnknownStream()
        self.assertEqual(http_error_metadata(exc)['body_state'], 'unavailable')

    def test_exception_chain_attached_metadata_is_revalidated(self):
        from tiger_memory.safe_diagnostics import exception_record
        from tiger_memory.providers import ProviderUnavailable
        exc = ProviderUnavailable('embedding-http-400')
        exc.http_error_metadata = {'schema_version': 1, 'http_status': 400,
                                  'error_code': 'InvalidParameter', 'message': SECRET,
                                  'request_id_sha256': SECRET}
        public = RuntimeError(SECRET); public.__cause__ = exc
        record = exception_record(public, 'search')
        encoded = json.dumps(record)
        self.assertNotIn(SECRET, encoded)
        self.assertEqual(record['exceptions'][1]['transport'], {
            'schema_version': 1, 'http_status': 400, 'error_code': 'InvalidParameter'})

    def test_socket_partial_body_does_not_wait_and_restores_timeout(self):
        client, server = socket.socketpair()
        try:
            client.settimeout(None)
            server.sendall(b'HTTP/1.1 400 Bad Request\r\nContent-Length: 1000\r\n\r\n{')
            response = http.client.HTTPResponse(client); response.begin()
            exc = HTTPError('https://invalid.example', 400, SECRET, response.headers, response)
            start = time.monotonic(); result = http_error_metadata(exc)
            self.assertLess(time.monotonic() - start, 0.2)
            self.assertIsNone(client.gettimeout())
            self.assertEqual(result['body_state'], 'incomplete_or_invalid_json')
            response.close()
        finally:
            client.close(); server.close()

    def test_socket_no_available_body_does_not_wait(self):
        client, server = socket.socketpair()
        try:
            client.settimeout(17.0)
            server.sendall(b'HTTP/1.1 400 Bad Request\r\nContent-Length: 1000\r\n\r\n')
            response = http.client.HTTPResponse(client); response.begin()
            exc = HTTPError('https://invalid.example', 400, SECRET, response.headers, response)
            start = time.monotonic(); result = http_error_metadata(exc)
            self.assertLess(time.monotonic() - start, 0.2)
            self.assertEqual(client.gettimeout(), 17.0)
            self.assertIn(result['body_state'], ('empty', 'unavailable'))
            response.close()
        finally:
            client.close(); server.close()

    def test_socket_chunked_partial_body_does_not_wait(self):
        client, server = socket.socketpair()
        try:
            server.sendall(b'HTTP/1.1 400 Bad Request\r\nTransfer-Encoding: chunked\r\n\r\n7\r\n{"error')
            response = http.client.HTTPResponse(client); response.begin()
            exc = HTTPError('https://invalid.example', 400, SECRET, response.headers, response)
            start = time.monotonic(); result = http_error_metadata(exc)
            self.assertLess(time.monotonic() - start, 0.2)
            self.assertIsNone(client.gettimeout())
            self.assertNotIn(SECRET, json.dumps(result))
            response.close()
        finally:
            client.close(); server.close()


if __name__ == '__main__':
    unittest.main()
