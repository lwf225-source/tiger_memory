"""Candidate AML source-derived session-view tests.

Views are retrieval-only: the scored window is never stored as a separate
memory and Search still returns a single original event body.
"""
from __future__ import annotations

import uuid
import unittest
from collections import defaultdict
import json

from tiger_memory import aml_adapter, search as search_module, service
from tiger_memory.database import connect_readonly

from tests.common import MemoryTestCase


class AMLSessionViewTests(MemoryTestCase):
    def setUp(self):
        super().setUp()
        self.saved_switch = aml_adapter.AML_SESSION_VIEWS
        self.saved_rrf_switch = aml_adapter.AML_SESSION_RRF
        self.saved_singletons = dict(aml_adapter._MODEL_SINGLETONS)
        aml_adapter.AML_SESSION_VIEWS = False
        aml_adapter.AML_SESSION_RRF = False
        aml_adapter._MODEL_SINGLETONS.clear()
        self.user_id = "session-view-" + uuid.uuid4().hex
        self.session_id = "platform-session-private-" + uuid.uuid4().hex

    def tearDown(self):
        aml_adapter.AML_SESSION_VIEWS = self.saved_switch
        aml_adapter.AML_SESSION_RRF = self.saved_rrf_switch
        aml_adapter._MODEL_SINGLETONS.clear()
        aml_adapter._MODEL_SINGLETONS.update(self.saved_singletons)
        super().tearDown()

    def _add_conversation(self, session_views=False, session_rrf=False):
        saved = aml_adapter.AML_SESSION_VIEWS
        saved_rrf = aml_adapter.AML_SESSION_RRF
        aml_adapter.AML_SESSION_VIEWS = session_views
        aml_adapter.AML_SESSION_RRF = session_rrf
        try:
            response = aml_adapter.add({
                "request_id": "session-view-r1", "user_id": self.user_id,
                "session_id": self.session_id,
                "messages": [
                    {"role": "user", "timestamp": 1704067200000,
                     "content": "The Atlas migration asks for a cobalt database choice."},
                    {"role": "assistant", "timestamp": 1704067260000,
                     "content": "The final choice is Postgres."},
                    {"role": "user", "timestamp": 1704067320000,
                     "content": "Please record the decision."},
                ],
            })
        finally:
            aml_adapter.AML_SESSION_VIEWS = saved
            aml_adapter.AML_SESSION_RRF = saved_rrf
        self.assertTrue(response["success"])

    def _context(self, enabled):
        diagnostics = {}
        _project, results = service.context_data(
            aml_adapter.user_storage_dir(self.user_id), "Atlas cobalt",
            limit=10, scope="session", scope_id=self.user_id,
            adjacent=0, session_views=enabled, diagnostics=diagnostics,
        )
        return diagnostics, results

    def test_view_promotes_original_reply_using_previous_turn_context(self):
        self._add_conversation(session_views=True)
        _off_info, without_views = self._context(False)
        self.assertNotIn("The final choice is Postgres.", [item.body for item in without_views])

        info, with_views = self._context(True)
        reply = next(item for item in with_views if item.body == "The final choice is Postgres.")
        self.assertIn("session-view-substring", reply.reasons)
        self.assertNotIn("Atlas migration", reply.body)
        self.assertEqual(info["session_views"]["status"], "ready")
        self.assertEqual(info["session_views"]["radius"], 1)

    def test_session_identifier_is_not_written_to_event_metadata(self):
        self._add_conversation(session_views=True)
        project, _results = service.context_data(
            aml_adapter.user_storage_dir(self.user_id), "", scope="session", scope_id=self.user_id,
        )
        conn = connect_readonly()
        try:
            rows = conn.execute(
                "SELECT tags_json FROM events WHERE project_id=? AND scope='session' AND scope_id=?",
                (project["project_id"], self.user_id),
            ).fetchall()
        finally:
            conn.close()
        self.assertGreaterEqual(len(rows), 3)
        rendered = "\n".join(row["tags_json"] for row in rows)
        self.assertNotIn(self.session_id, rendered)
        self.assertIn("aml-session:", rendered)

    def test_default_adapter_path_keeps_session_view_channel_off(self):
        self._add_conversation()
        self.assertFalse(aml_adapter.AML_SESSION_VIEWS)
        body = aml_adapter.search({"query": "Atlas cobalt", "user_id": self.user_id, "top_k": 10})
        self.assertEqual(set(body), {"data"})

    def test_session_rrf_writes_only_the_one_way_session_tag(self):
        self._add_conversation(session_rrf=True)
        project, _results = service.context_data(
            aml_adapter.user_storage_dir(self.user_id), "", scope="session", scope_id=self.user_id,
        )
        conn = connect_readonly()
        try:
            rows = conn.execute(
                "SELECT tags_json FROM events WHERE project_id=? AND scope='session' AND scope_id=?",
                (project["project_id"], self.user_id),
            ).fetchall()
        finally:
            conn.close()
        rendered = "\n".join(row["tags_json"] for row in rows)
        self.assertNotIn(self.session_id, rendered)
        self.assertIn("aml-session:", rendered)

    def test_session_rrf_uses_aggregate_session_rank_for_tagged_events_only(self):
        key_a = ("event", "a")
        key_b = ("event", "b")
        key_c = ("event", "c")
        key_fact = ("fact", "f")
        records = {
            key_a: {"tags_json": json.dumps(["aml-session:one"])},
            key_b: {"tags_json": json.dumps(["aml-session:one"])},
            key_c: {"tags_json": json.dumps(["aml-session:two"])},
            key_fact: {"tags_json": json.dumps(["aml-session:one"])},
        }
        ranked = {key_a: 0.02, key_b: 0.01, key_c: 0.025, key_fact: 0.5}
        reasons = defaultdict(list)
        stats = search_module._session_rrf_adjust(ranked, records, reasons)
        self.assertEqual(stats, {"sessions": 2, "adjusted": 3})
        self.assertAlmostEqual(ranked[key_a], 0.02 + 1.0 / (search_module.SESSION_RRF_K + 1.0))
        self.assertAlmostEqual(ranked[key_b], 0.01 + 1.0 / (search_module.SESSION_RRF_K + 1.0))
        self.assertAlmostEqual(ranked[key_c], 0.025 + 1.0 / (search_module.SESSION_RRF_K + 2.0))
        self.assertEqual(ranked[key_fact], 0.5)
        self.assertEqual(reasons[key_a], ["session-rrf:1"])
        self.assertEqual(reasons[key_c], ["session-rrf:2"])


class AMLSessionRRFDefaultTests(MemoryTestCase):
    """The 1.3.1 frozen default writes only a one-way session digest."""

    def test_default_rrf_tags_events_without_persisting_the_platform_session_id(self):
        self.assertTrue(aml_adapter.AML_SESSION_RRF)
        self.assertEqual(search_module.SESSION_RRF_K, 30.0)
        user_id = "default-session-rrf-" + uuid.uuid4().hex
        session_id = "platform-session-private-" + uuid.uuid4().hex
        response = aml_adapter.add({
            "request_id": "default-rrf-r1", "user_id": user_id, "session_id": session_id,
            "messages": [{"role": "user", "timestamp": 1704067200000,
                          "content": "The project selected cobalt storage."}],
        })
        self.assertTrue(response["success"])
        project, _results = service.context_data(
            aml_adapter.user_storage_dir(user_id), "", scope="session", scope_id=user_id,
        )
        conn = connect_readonly()
        try:
            rows = conn.execute(
                "SELECT tags_json FROM events WHERE project_id=? AND scope='session' AND scope_id=?",
                (project["project_id"], user_id),
            ).fetchall()
        finally:
            conn.close()
        rendered = "\n".join(row["tags_json"] for row in rows)
        self.assertNotIn(session_id, rendered)
        self.assertIn("aml-session:", rendered)
        # The digest remains metadata for RRF grouping only. It is absent from
        # FTS and the persisted lexical vector, so it cannot perturb relevance.
        conn = connect_readonly()
        try:
            fts_tags = conn.execute(
                "SELECT tags FROM memory_fts WHERE project_id=? AND record_type='event'",
                (project["project_id"],),
            ).fetchall()
        finally:
            conn.close()
        self.assertTrue(fts_tags)
        self.assertTrue(all("aml-session:" not in row["tags"] for row in fts_tags))
        old_threshold = aml_adapter.AML_ABSTAIN_THRESHOLD
        try:
            aml_adapter.AML_ABSTAIN_THRESHOLD = 0.05
            response = aml_adapter.search({"query": "cobalt storage", "user_id": user_id, "top_k": 10})
        finally:
            aml_adapter.AML_ABSTAIN_THRESHOLD = old_threshold
        self.assertTrue(response["data"])
        self.assertNotIn("plm_abstained", response)


if __name__ == "__main__":
    unittest.main()
