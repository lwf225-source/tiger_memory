"""Failure-focused tests for the offline structural inspector, not retrieval."""
import copy
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location(
    "aml_diagnostic_inspector", Path(__file__).resolve().parents[1] /
    "scripts/inspect_aml_diagnostic.py")
INSPECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INSPECTOR)


def event():
    version = {"source_token": "b" * 64, "content_hmac": "c" * 64}
    return {"schema": "aml-full-diagnostic-v1", "request_id": "synthetic",
            "instance_id": "synthetic", "sequence": 1, "binding_sha256": "a" * 64,
            "key_id": "synthetic", "key_check": "d" * 64, "operation": "search",
            "http": True, "incomplete": [], "observation_complete": True,
            "replay_complete": False, "official_chain_complete": False,
            "stages": [
                {"stage": "eligible", "read_view_id": "synthetic-view",
                 "member_count": 1, "manifest_complete": True, "members": [version]},
                {"stage": "adapter_output", "result_count": 1, "results": [
                    {"rank": 0, "content_utf8_bytes": 10, "segments": [
                        {"start": 0, "end": 10, "read_view_id": "synthetic-view",
                         "kind": "event", **version}]}]},
                {"stage": "serialization"},
                {"stage": "transport", "phase": "write_started"},
                {"stage": "transport", "phase": "write_returned_not_client_receipt"}]}


class InspectorTests(unittest.TestCase):
    def test_structure_does_not_claim_authentication_or_quality(self):
        result = INSPECTOR.inspect([event()])
        self.assertTrue(result["structurally_valid"])
        self.assertFalse(result["authentication_verified"])
        self.assertFalse(result["replay_complete"])
        self.assertFalse(result["official_chain_complete"])

    def test_rejects_mixed_keys_including_reused_label(self):
        first, second = event(), event()
        second.update(sequence=2, request_id="second", key_check="e" * 64)
        result = INSPECTOR.inspect([first, second])
        self.assertFalse(result["comparable"])
        self.assertIsNone(result["version_links"]["write_to_output"])

    def test_rejects_missing_view_and_wrong_source_version(self):
        for change in ({"read_view_id": "absent"}, {"content_hmac": "f" * 64}):
            row = event()
            row["stages"][1]["results"][0]["segments"][0].update(change)
            self.assertIn("source_view_version_mismatch", INSPECTOR.inspect([row])["errors"])

    def test_rejects_missing_range_and_overlapping_ranges(self):
        for start, end in ((1, 10), (0, 9), (0, 11), (5, 2)):
            row = event()
            row["stages"][1]["results"][0]["segments"][0].update(start=start, end=end)
            self.assertFalse(INSPECTOR.inspect([row])["structurally_valid"])

    def test_rejects_dropped_export_and_duplicate_events(self):
        row = event()
        self.assertFalse(INSPECTOR.inspect([{"events": [row],
                                            "export_complete": False}])["structurally_valid"])
        self.assertIn("duplicate_request_event",
                      INSPECTOR.inspect([row, copy.deepcopy(row)])["errors"])

    def test_rejects_incomplete_manifest_terminal_and_official_claim(self):
        cases = []
        row = event()
        row["stages"][0]["manifest_complete"] = False
        cases.append(row)
        row = event()
        row["stages"].pop()
        cases.append(row)
        row = event()
        row["official_chain_complete"] = True
        cases.append(row)
        for row in cases:
            self.assertFalse(INSPECTOR.inspect([row])["structurally_valid"])


if __name__ == "__main__":
    unittest.main()
