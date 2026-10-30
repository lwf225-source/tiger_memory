"""Local adapter/HTTP invariants, not answer-quality or official F1 tests."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tiger_memory import aml_adapter as aml
from tiger_memory.aml_search_trace import SearchTrace
from tests.test_aml_adapter import AMLServerCase


class SearchObservationTests(unittest.TestCase):
    def setUp(self):
        self.trace = SearchTrace(binding_sha256="a" * 64, capacity=4)
        self.patches = [
            patch.object(aml, "AML_SEARCH_TRACE", self.trace),
            patch.object(aml, "_OUTCOMES", {}),
            patch.object(aml, "_storage_identity_preflight", return_value=Path(__file__).parent),
            patch.object(aml, "_model_provider", return_value=None),
            patch.object(aml, "_query_provider_singleton", return_value=None),
            patch.multiple(aml, AML_STRICT_RETRIEVAL=False,
                           AML_FACT_PROJECTION="off", AML_ABSTAIN_THRESHOLD=0,
                           AML_TEMPORAL_INTENT=False, AML_QUERY_REWRITE="off"),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def request(self, **kw):
        return {"query": "private-question", "user_id": "private-user",
                "top_k": 2, **kw}

    def events(self):
        return self.trace.snapshot()["events"]

    def test_early_returns_are_counted_once_without_retrieval(self):
        with patch.object(aml.service, "context_data") as context:
            self.assertEqual(aml.search(self.request(top_k=0)), {"data": []})
            with patch.object(aml, "_storage_identity_preflight",
                              return_value=Path(__file__) / "missing"):
                self.assertEqual(aml.search(self.request()), {"data": []})
            context.assert_not_called()
        self.assertEqual(aml._OUTCOMES, {"search_empty": 2,
                         "search_empty_top_k_zero": 1, "search_empty_storage_missing": 1})
        self.assertEqual([e["reason"] for e in self.events()], ["top_k_zero", "storage_missing"])
        self.assertTrue(all(e["adapter_completed"] and e["result_count"] == 0
                            for e in self.events()))

    def test_trace_preserves_payload_order_and_filters_diagnostics(self):
        rows = [SimpleNamespace(ref_id="secret-id-2", body="乙-body", score=.4, valid_from="2020"),
                SimpleNamespace(ref_id="secret-id-1", body="甲-body", score=.8, valid_from="2021")]

        def context(*args, **kw):
            kw["diagnostics"]["source_coverage"] = {
                "added_sources": 3, "omitted_sources": 7,
                "budget_bytes": 32768, "unknown_text": "DO-NOT-STORE",
                "added_bytes": "DO-NOT-STORE",
            }
            return {}, rows

        with patch.object(aml.service, "context_data", side_effect=context):
            enabled = aml.search(self.request(options=["private-option"]))
            with patch.object(aml, "AML_SEARCH_TRACE", None):
                disabled = aml.search(self.request(options=["private-option"]))
            aml.search(self.request(options=["private-option"]))
        self.assertEqual(enabled, disabled)
        first, second = self.events()
        self.assertEqual(first["results"], second["results"])
        self.assertEqual([r["id_digest"] for r in first["results"]],
                         [self.trace.digest("source-id", r.ref_id) for r in rows])
        self.assertEqual(first["total_content_utf8_bytes"],
                         sum(len(r.body.encode()) for r in rows))
        self.assertEqual(first["coverage"], {"added_sources": 3, "omitted_sources": 7,
                                            "budget_bytes": 32768})
        serialized = json.dumps(self.trace.snapshot(), ensure_ascii=False)
        for value in ("private-question", "private-user", "private-option",
                      "secret-id", "乙-body", "甲-body", "DO-NOT-STORE"):
            self.assertNotIn(value, serialized)

    def test_empty_abstention_failure_and_validation_are_distinct(self):
        with patch.object(aml.service, "context_data", return_value=({}, [])):
            self.assertEqual(aml.search(self.request()), {"data": []})
            with patch.object(aml, "AML_ABSTAIN_THRESHOLD", .5):
                self.assertEqual(aml.search(self.request()),
                                 {"data": [], "plm_abstained": True, "plm_abstain": {}})
        with patch.object(aml.service, "context_data", side_effect=RuntimeError("SECRET-ERROR")):
            self.assertEqual(aml.search(self.request()), {"data": []})
            with patch.object(aml, "AML_STRICT_RETRIEVAL", True):
                with self.assertRaises(aml.RetrievalUnavailable):
                    aml.search(self.request())
        self.assertEqual([e["reason"] for e in self.events()],
                         ["retrieved_empty", "abstained", "legacy_failure_empty",
                          "retrieval_unavailable"])
        self.assertEqual(aml._OUTCOMES, {"search_empty": 2,
                                        "search_empty_abstained": 1, "search_failure": 2})
        with self.assertRaises(aml.ContractError):
            aml.search({"query": "DO-NOT-STORE"})
        last = self.events()[-1]
        self.assertEqual(last["reason"], "invalid_request")
        self.assertNotIn("query_digest", last)
        self.assertNotIn("SECRET-ERROR", json.dumps(self.trace.snapshot()))

    def test_rejected_sensitive_query_is_not_fingerprinted(self):
        with patch.multiple(aml, AML_STRICT_RETRIEVAL=True, AML_EMBEDDING="remote-v4"):
            with self.assertRaises(aml.SensitiveSearchRejected):
                aml.search(self.request(query="password=SYNTHETIC-ONLY"))
        event = self.events()[-1]
        self.assertEqual(event["reason"], "sensitive_rejected")
        self.assertNotIn("query_digest", event)

    def test_embedding_failure_is_bound_to_request_without_freeform_text(self):
        def unavailable(*args, **kwargs):
            kwargs["diagnostics"]["embedding"] = {
                "status": "unavailable", "fallback_reason": current[0]}
            return {}, []

        current = ["embedding-budget-unavailable"]
        with patch.object(aml.service, "context_data", side_effect=unavailable), \
                patch.object(aml, "AML_STRICT_RETRIEVAL", True):
            for failure in ("embedding-budget-unavailable", "SECRET-PROVIDER-PAYLOAD"):
                current[0] = failure
                with self.assertRaises(aml.RetrievalUnavailable):
                    aml.search(self.request())
        first, second = self.events()
        self.assertEqual(first["embedding_failure"], "embedding-budget-unavailable")
        self.assertEqual(second["embedding_failure"], "unclassified")
        self.assertEqual(first["query_digest"], second["query_digest"])
        self.assertNotIn("SECRET-PROVIDER-PAYLOAD", json.dumps(self.trace.snapshot()))

    def test_concurrent_requests_do_not_mix_and_ring_is_bounded(self):
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda i: aml.search(self.request(user_id=str(i), top_k=0)), range(30)))
        snapshot = self.trace.snapshot()
        self.assertEqual(snapshot["started"], 30)
        self.assertEqual(snapshot["evicted"], 26)
        self.assertEqual(len(snapshot["events"]), 4)
        self.assertEqual(len({e["sequence"] for e in snapshot["events"]}), 4)
        self.assertEqual(len({e["user_digest"] for e in snapshot["events"]}), 4)
        self.assertEqual(aml._OUTCOMES["search_empty_top_k_zero"], 30)
        snapshot["events"][0]["reason"] = "tampered"
        self.assertNotEqual(self.events()[0]["reason"], "tampered")
        self.trace.clear()
        self.assertEqual(self.events(), [])

    def test_diagnostics_failure_cannot_replace_success(self):
        with patch.object(self.trace, "result", side_effect=RuntimeError("SECRET")):
            self.assertEqual(aml.search(self.request(top_k=0)), {"data": []})
        self.assertTrue(self.events()[0]["diagnostic_incomplete"])
        self.assertTrue(self.events()[0]["adapter_completed"])
        self.assertNotIn("SECRET", json.dumps(self.trace.snapshot()))

    def test_cancellation_and_identity_conflict_are_not_converted_to_empty(self):
        for error, label in ((aml.RequestCancelled(), "cancelled"),
                             (aml.StorageIdentityConflict(), "identity_conflict")):
            with patch.object(aml, "_storage_identity_preflight", side_effect=error):
                with self.assertRaises(type(error)) as caught:
                    aml.search(self.request())
            self.assertIs(caught.exception, error)
            self.assertEqual(self.events()[-1]["reason"], label)
            self.assertFalse(self.events()[-1]["adapter_completed"])
        self.assertNotIn("private-path", json.dumps(self.trace.snapshot()))

    def test_keys_are_isolated_and_config_is_validated(self):
        other = SearchTrace(binding_sha256="b" * 64)
        self.assertNotEqual(self.trace.digest("query", "x"), other.digest("query", "x"))
        self.assertNotEqual(self.trace.digest("query", "x"), self.trace.digest("user", "x"))
        for capacity in (0, 129, True):
            with self.assertRaises(ValueError):
                SearchTrace(binding_sha256="a" * 64, capacity=capacity)
        with self.assertRaises(ValueError):
            SearchTrace(binding_sha256="not-a-manifest-hash")

    def test_runtime_config_digest_detects_changes_without_exporting_config(self):
        for revision in ("private-config-a", "private-config-b"):
            with patch.object(aml, "AML_RUNTIME_CONFIG", {"revision": revision}):
                aml.search(self.request(top_k=0))
        events = self.events()
        self.assertNotEqual(events[0]["runtime_config_sha256"],
                            events[1]["runtime_config_sha256"])
        self.assertNotIn("private-config", json.dumps(self.trace.snapshot()))


class SearchTraceHTTPTests(AMLServerCase):
    def test_status_requires_auth_and_search_wire_body_is_unchanged(self):
        trace = SearchTrace(binding_sha256="b" * 64)
        with patch.object(aml, "AML_SEARCH_TRACE", trace), \
             patch.dict(os.environ, {"AML_API_KEY": "local-test-only"}):
            headers = {"Authorization": "Bearer local-test-only"}
            self.assertEqual(self.request("GET", "/status")[0], 401)
            self.assertEqual(self.request("POST", "/search",
                             {"query": "local-query", "user_id": "missing", "top_k": 0},
                             headers=headers), (200, {"data": []}))
            status, body = self.request("GET", "/status", headers=headers)
            self.assertEqual(status, 200)
            self.assertEqual(body["search_trace"]["events"][0]["reason"], "top_k_zero")
            with patch.object(aml, "AML_SEARCH_TRACE", None):
                self.assertNotIn("search_trace", self.request("GET", "/status", headers=headers)[1])
        with patch.object(aml, "AML_SEARCH_TRACE", trace), \
             patch.dict(os.environ, {"AML_API_KEY": ""}):
            self.assertEqual(self.request("GET", "/status")[0], 401)
