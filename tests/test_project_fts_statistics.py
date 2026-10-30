"""Regressions for the experimental, opt-in project FTS statistics API."""
import hashlib
import importlib
import sqlite3
from contextlib import contextmanager
from unittest.mock import patch

from tests.common import MemoryTestCase
from tiger_memory import database, service
from tiger_memory.model import PLMError


search_module = importlib.import_module("tiger_memory.search")
REAL_CONNECT = sqlite3.connect


class ProjectFTSStatisticsTests(MemoryTestCase):
    def setUp(self):
        super().setUp()
        for body in (
            "cobalt cobalt cobalt padding", "saffron saffron saffron pad",
            "cobalt padding padding plain", "saffron padding padding pad",
        ):
            event = service.write_event(
                self.project, "message", body, recorded_at="2026-01-01T00:00:00Z")
        self.project_id = event.project_id
        self.conn = database.connect()
        self.addCleanup(self.conn.close)

    @contextmanager
    def patched(self, target, name, *args, **kwargs):
        patcher = patch.object(target, name, *args, **kwargs)
        self.addCleanup(patcher.stop)
        with patcher as value:
            yield value

    def source_hashes(self):
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in self.base.rglob("*.md")}

    def results(self, **kwargs):
        return [item.as_dict() for item in search_module.search(
            self.conn, self.project_id, "cobalt saffron", limit=100,
            profile="lexical", **kwargs)]

    def project_rows(self, **kwargs):
        diagnostics = {}
        rows = search_module._project_fts_rows(
            self.conn, self.project_id, '"cobalt" OR "saffron"', "1", [],
            diagnostics, kwargs.get("rows", 100000), kwargs.get("bytes", 64 * 1024 * 1024))
        return rows, diagnostics["fts_statistics"]["channels"][0]

    def global_rows(self):
        return self.conn.execute(
            "SELECT record_type,ref_id,bm25(memory_fts,0,0,0,8,4,1) AS rank "
            "FROM memory_fts WHERE memory_fts MATCH ? AND project_id=? ORDER BY rank,ref_id",
            ('"cobalt" OR "saffron"', self.project_id)).fetchall()

    def rank_bits(self, rows):
        return [(r["record_type"], r["ref_id"], r["rank"].hex()) for r in rows]

    def test_isolation_matches_a_only_raw_scores_and_default_does_not_copy(self):
        expected_raw = self.rank_bits(self.global_rows())
        expected_results = self.results()
        original_sources = self.source_hashes()
        other = self.project.parent / "other-project"
        other.mkdir()
        (other / "pyproject.toml").write_text("[project]\nname='other'\n", encoding="utf-8")
        for _ in range(12):
            service.write_event(other, "message", "cobalt cobalt cobalt padding")
        # This fixture must actually perturb global statistics, not merely add rows.
        self.assertNotEqual(expected_raw, self.rank_bits(self.global_rows()))
        with self.patched(search_module, "_project_fts_rows",
                          wraps=search_module._project_fts_rows) as scoped:
            default = self.results()
            self.assertEqual(default, self.results(fts_statistics_scope="global"))
            scoped.assert_not_called()
        rows, info = self.project_rows()
        self.assertEqual(expected_raw, self.rank_bits(rows))
        self.assertEqual(info["copied_rows"], 4)
        self.assertEqual(expected_results, self.results(fts_statistics_scope="project"))
        _, via_service = service.context_data(
            self.project, "cobalt saffron", limit=100, fts_statistics_scope="project")
        self.assertEqual(expected_results, [r.as_dict() for r in via_service])
        current = self.source_hashes()
        self.assertTrue(all(current.get(p) == h for p, h in original_sources.items()))
        self.assertFalse(self.conn.in_transaction)

    def test_limits_fail_closed_and_faults_preserve_caller_writes(self):
        before = self.source_hashes()
        _, cost = self.project_rows()
        self.project_rows(rows=cost["copied_rows"], bytes=cost["copied_utf8_bytes"])
        for limits in ({"rows": cost["copied_rows"] - 1},
                       {"bytes": cost["copied_utf8_bytes"] - 1}):
            with self.patched(search_module.sqlite3, "connect",
                              side_effect=AssertionError("allocated before preflight")):
                with self.assertRaisesRegex(PLMError, "scoped_stats_limit_exceeded"):
                    self.project_rows(**limits)
            self.assertFalse(self.conn.in_transaction)
        with self.assertRaisesRegex(ValueError, "invalid FTS statistics scope"):
            self.results(fts_statistics_scope="unknown")
        for profile in ("legacy", "passages"):
            with self.assertRaisesRegex(ValueError, "require lexical"):
                search_module.search(self.conn, self.project_id, "cobalt", profile=profile,
                                     fts_statistics_scope="project")
        for limits in ({"fts_statistics_max_rows": 0}, {"fts_statistics_max_bytes": True}):
            with self.assertRaisesRegex(ValueError, "invalid FTS statistics limits"):
                self.results(fts_statistics_scope="project", **limits)

        # A real uncommitted caller write must survive every scoped-read failure.
        self.conn.execute("BEGIN")
        self.conn.execute("INSERT INTO metadata(key,value) VALUES(?,?)",
                          ("project-fts-caller", "uncommitted"))
        self.project_rows()
        self.assertTrue(self.conn.in_transaction)
        for stage in ("create", "insert", "match", "subset"):
            with self.subTest(stage=stage):
                opened = []

                class FaultConnection(sqlite3.Connection):
                    def __init__(local, *args, **kwargs):
                        super().__init__(*args, **kwargs)
                        local.closed_seen = False
                        opened.append(local)

                    def execute(local, sql, *args, **kwargs):
                        if stage == "create" and sql.startswith("CREATE VIRTUAL"):
                            raise sqlite3.OperationalError("injected create failure")
                        if stage == "match" and "bm25(" in sql:
                            raise sqlite3.OperationalError("injected MATCH failure")
                        if stage == "subset" and "bm25(" in sql:
                            return super().execute(
                                "SELECT record_type,ref_id,0 AS rank FROM memory_fts WHERE 0")
                        return super().execute(sql, *args, **kwargs)

                    def executemany(local, sql, *args, **kwargs):
                        if stage == "insert":
                            raise sqlite3.OperationalError("injected insert failure")
                        return super().executemany(sql, *args, **kwargs)

                    def close(local):
                        local.closed_seen = True
                        return super().close()

                def connect(*args, **kwargs):
                    self.assertEqual(args[0], ":memory:")
                    return REAL_CONNECT(*args, factory=FaultConnection, **kwargs)

                traces = []
                self.conn.set_trace_callback(traces.append)
                try:
                    with self.patched(search_module.sqlite3, "connect", connect):
                        error = ("scoped_stats_match_mismatch" if stage == "subset"
                                 else "scoped_stats_unavailable")
                        with self.assertRaisesRegex(PLMError, error):
                            self.results(fts_statistics_scope="project")
                finally:
                    self.conn.set_trace_callback(None)
                self.assertEqual(len(opened), 1)
                self.assertTrue(opened[0].closed_seen)
                self.assertFalse(any("bm25(" in sql for sql in traces))
                self.assertTrue(self.conn.in_transaction)
                self.assertEqual(self.conn.execute(
                    "SELECT value FROM metadata WHERE key='project-fts-caller'").fetchone()[0],
                    "uncommitted")
                observer = database.connect_readonly()
                try:
                    self.assertIsNone(observer.execute(
                        "SELECT value FROM metadata WHERE key='project-fts-caller'").fetchone())
                finally:
                    observer.close()
        self.conn.rollback()
        self.assertEqual(before, self.source_hashes())

    def test_rewrite_and_embedding_run_outside_helper_snapshot(self):
        test = self
        calls = []

        class Rewrite:
            def describe(self):
                return {"provider": "test-rewrite", "remote": False}

            def rewrite(self, query):
                calls.append(("rewrite", test.conn.in_transaction))
                test.assertFalse(test.conn.in_transaction)
                return {"expansions": ["saffron padding"], "hyde": ""}

        class Embedding:
            def describe(self):
                return {"provider": "test-embedding", "remote": False}

            def embed(self, texts):
                calls.append(("embed", test.conn.in_transaction))
                test.assertFalse(test.conn.in_transaction)
                return [[1.0, 0.0] for _ in texts]

        before = self.source_hashes()
        info = {}
        project = self.results(fts_statistics_scope="project", diagnostics=info,
                               query_provider=Rewrite(), embedding_provider=Embedding())
        self.assertEqual(len(info["fts_statistics"]["channels"]), 2)
        self.assertEqual(calls[0], ("rewrite", False))
        self.assertTrue(any(kind == "embed" for kind, _ in calls))
        self.assertTrue(all(not active for _, active in calls))
        self.assertEqual(project, self.results(query_provider=Rewrite(), embedding_provider=Embedding()))
        self.assertFalse(self.conn.in_transaction)
        self.assertEqual(before, self.source_hashes())
