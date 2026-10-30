"""Client checks use the in-process candidate and localhost embedding fixture."""
import contextlib
import io
import json
import unittest
from unittest.mock import patch

from scripts import aml_smoke
from tests import test_aml_open_source as fixture


class SmokeTests(unittest.TestCase):
    def test_plan_does_not_read_keys_or_network(self):
        with patch.object(aml_smoke.os, "environ", {}), patch.object(aml_smoke.urllib.request, "build_opener", side_effect=AssertionError), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(aml_smoke.main([]), 0)
        self.assertFalse(json.loads(output.getvalue())["executed"])

    def test_url_boundaries(self):
        for url in ["http://example.com", "https://lewen.woa.com", "https://x.lewen.woa.com./aml",
                    "https://user@example.com", "https://example.com/?key=secret", "https://example.com/#x",
                    "https://example.com:bad", "https://example.com\\@other.com", "https://%6cewen.woa.com"]:
            with self.subTest(url=url), self.assertRaises(ValueError):
                aml_smoke.validate_url(url)
        self.assertEqual(aml_smoke.validate_url("https://example.com/aml/"), "https://example.com/aml")

    def test_candidate_contract_isolation_replay_and_four_embedding_batches(self):
        server = fixture.OpenSourceProfileTests()
        server.setUp()
        self.addCleanup(server.doCleanups)
        self.addCleanup(server.tearDown)
        report = aml_smoke.run("http://127.0.0.1:" + str(server.port), "fixture-service-key")
        self.assertTrue(report["passed"], report)
        self.assertEqual(len(report["results"]), 12)
        self.assertEqual(len(server.remote.requests), 4)
        self.assertNotIn("fixture-service-key", json.dumps(report))
        self.assertNotIn("violet boxes", json.dumps(report))

    def test_redirect_is_never_followed(self):
        self.assertIsNone(aml_smoke.NoRedirect().redirect_request(None, None, 302, "", {}, "https://example.com"))

    def test_provider_failure_stops_without_retry_or_body_leak(self):
        server = fixture.OpenSourceProfileTests()
        server.setUp()
        self.addCleanup(server.doCleanups)
        self.addCleanup(server.tearDown)
        server.remote.respond = lambda p: (429, {}, {"private": "do-not-copy-provider-body"})
        report = aml_smoke.run("http://127.0.0.1:" + str(server.port), "fixture-service-key")
        self.assertFalse(report["passed"])
        self.assertEqual(report["results"][-1]["check"], "add_a")
        self.assertEqual(report["results"][-1]["http_status"], 503)
        self.assertEqual(len(server.remote.requests), 1)
        self.assertNotIn("do-not-copy-provider-body", json.dumps(report))
