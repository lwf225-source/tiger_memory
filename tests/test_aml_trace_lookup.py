"""Storage association/transport invariants, not official-answer acceptance."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
import uuid
from unittest.mock import patch

from tiger_memory import aml_adapter as aml, aml_search_trace as tracing
from tests.test_aml_adapter import AMLServerCase
from tests.loopback import LoopbackSocket, loopback_resolve

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/resolve_aml_search_trace.py"
spec = importlib.util.spec_from_file_location("lookup_resolver_test", SCRIPT)
resolver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resolver)


class LookupTokenTests(unittest.TestCase):
    def test_only_storage_uuid_domains_are_linkable(self):
        key = str(uuid.uuid4())
        token = tracing.storage_lookup_token("a" * 32, "project", key)
        self.assertEqual(len(token), 64)
        self.assertNotEqual(token, tracing.storage_lookup_token("b" * 32, "project", key))
        self.assertNotEqual(token, tracing.storage_lookup_token("a" * 32, "source", key))
        for invalid in ("private-question", "private-user", key.upper(), key.replace("-", ""),
                        str(uuid.UUID(int=0)), None, 10):
            self.assertIsNone(tracing.storage_lookup_token("a" * 32, "source", invalid))
        for domain in ("query", "user", "content"):
            with self.assertRaises(ValueError):
                tracing.storage_lookup_token("a" * 32, domain, key)
        for optin in (1, None, "yes"):
            with self.assertRaises(ValueError):
                tracing.SearchTrace(binding_sha256="a" * 64, storage_links=optin)

    def test_disabled_path_does_not_resolve_storage_or_change_old_schema(self):
        trace = tracing.SearchTrace(binding_sha256="a" * 64)
        with patch("tiger_memory.model.project_id_for_root", side_effect=AssertionError("must not run")):
            with trace.capture() as event:
                tracing.note(storage_root=Path("/not-read"))
                trace.result(event, {"data": [{"id": str(uuid.uuid4()), "content": "private-body"}]})
        self.assertNotIn("storage_lookup", trace.snapshot())
        self.assertNotIn("expected_project_token", event)
        self.assertNotIn("source_token", event["results"][0])
        self.assertNotIn("diagnostic_incomplete", event)

    def test_lookup_error_is_contained_and_non_uuid_results_are_explicit(self):
        trace = tracing.SearchTrace(binding_sha256="a" * 64, storage_links=True)
        with patch("tiger_memory.model.project_id_for_root", side_effect=RuntimeError("PRIVATE")):
            with trace.capture() as event:
                tracing.note(storage_root=Path("/not-read"))
                trace.result(event, {"data": [{"id": "virtual-hit", "content": "private-body"}]})
        self.assertTrue(event["diagnostic_incomplete"])
        self.assertEqual(event["unlinked_result_ids"], 1)
        self.assertNotIn("PRIVATE", json.dumps(trace.snapshot()))


class LookupHTTPTests(AMLServerCase):
    def test_real_wire_payload_and_post_exit_catalog_resolution(self):
        trace = tracing.SearchTrace(binding_sha256="a" * 64, storage_links=True)
        headers = {"Authorization": "Bearer synthetic-local-key"}
        with patch("socket.socket", LoopbackSocket), \
                patch("socket.getaddrinfo", loopback_resolve), \
                patch.dict(os.environ, {"AML_API_KEY": "synthetic-local-key"}), \
                patch.multiple(aml, AML_EMBEDDING="off", AML_FACT_PROJECTION="off",
                               AML_QUERY_REWRITE="off", AML_STRICT_RETRIEVAL=False,
                               AML_RECENCY_BOOST=False), \
                patch.object(aml, "_model_provider", return_value=None):
            for user, body in (("private-user-a", "PrivateAlpha approved count is eight."),
                               ("private-user-b", "PrivateBeta approved count is nine.")):
                code, result = self.request("POST", "/add", {
                    "user_id": user, "session_id": "private-session", "request_id": "private-request",
                    "messages": [{"role": "user", "content": body}]}, headers=headers)
                self.assertEqual(code, 200)
                self.assertTrue(result["success"])
            responses = []
            for user, query in (("private-user-a", "PrivateAlpha count"),
                                ("private-user-b", "PrivateBeta count")):
                payload = {"user_id": user, "query": query, "top_k": 100}
                with patch.object(aml, "AML_SEARCH_TRACE", None):
                    disabled = self.request("POST", "/search", payload, headers=headers)
                with patch.object(aml, "AML_SEARCH_TRACE", trace):
                    enabled = self.request("POST", "/search", payload, headers=headers)
                    self.assertEqual(self.request("GET", "/status")[0], 401)
                self.assertEqual(enabled, disabled)
                self.assertEqual(enabled[0], 200)
                self.assertEqual(len(enabled[1]["data"]), 1)
                responses.append(enabled[1]["data"])
            snapshot = trace.snapshot()
            events = snapshot["events"]
            self.assertNotEqual(events[0]["retrieved_project_token"], events[1]["retrieved_project_token"])
            for event in events:
                self.assertEqual(event["expected_project_token"], event["retrieved_project_token"])
                self.assertEqual(event["unlinked_result_ids"], 0)

        # A separate process has no recorder or private HMAC key. Only the
        # exported public storage salt and original temporary catalog survive.
        snapshot_path = Path(self.temp.name) / "snapshot.json"
        snapshot_path.write_text(json.dumps(snapshot))
        catalog = self.base / "v2/catalog.sqlite3"
        before = hashlib.sha256(catalog.read_bytes()).hexdigest()
        trace.clear()
        del trace
        child = subprocess.run([sys.executable, "-I", "-B", str(SCRIPT),
                                "--snapshot", str(snapshot_path), "--catalog", str(catalog)],
                               capture_output=True, text=True, timeout=30)
        self.assertEqual(child.returncode, 0, child.stderr)
        resolved = json.loads(child.stdout)
        self.assertEqual(hashlib.sha256(catalog.read_bytes()).hexdigest(), before)
        self.assertEqual(len(resolved["events"]), 2)
        for item, returned in zip(resolved["events"], responses):
            self.assertEqual(item["status"], "project_resolved")
            self.assertEqual(item["routing_evidence"], "retrieved")
            self.assertEqual(item["retained_event_count"], 1)
            self.assertEqual(item["sources"][0]["status"], "event_resolved")
            self.assertEqual(item["sources"][0]["body_utf8_bytes"], len(returned[0]["content"].encode()))
        for secret in ("private-user", "private-session", "private-request", "PrivateAlpha", "PrivateBeta"):
            self.assertNotIn(secret, json.dumps(snapshot) + child.stdout)
        for returned in responses:
            self.assertNotIn(returned[0]["id"], json.dumps(snapshot) + child.stdout)

        # A source token from another project cannot resolve in this project.
        altered = json.loads(snapshot_path.read_text())
        altered["events"][0]["results"][0]["source_token"] = altered["events"][1]["results"][0]["source_token"]
        wrong = resolver.resolve(altered, catalog)
        self.assertEqual(wrong["events"][0]["sources"][0]["status"], "event_not_found")
        altered["events"][0]["retrieved_project_token"] = altered["events"][1]["retrieved_project_token"]
        self.assertEqual(resolver.resolve(altered, catalog)["events"][0]["status"],
                         "project_identity_mismatch")
        with self.assertRaises(ValueError):
            resolver.resolve({"events": []}, catalog)
        altered = json.loads(snapshot_path.read_text())
        altered["events"][0]["expected_project_token"] = "PRIVATE-INJECTED-FIELD"
        with self.assertRaises(ValueError):
            resolver.resolve(altered, catalog)
        altered = json.loads(snapshot_path.read_text())
        altered["events"][0]["expected_project_token"] = None
        altered["events"][0]["retrieved_project_token"] = None
        self.assertEqual(resolver.resolve(altered, catalog)["events"][0]["status"], "no_project_token")


if __name__ == "__main__":
    unittest.main()
