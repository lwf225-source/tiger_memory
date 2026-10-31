"""Immutable-source recovery and rebuild guards; all stores are synthetic."""
from contextlib import closing
import hashlib
from unittest.mock import patch

from tests.common import MemoryTestCase
from tiger_memory import database, service
from tiger_memory.events import parse_event, serialize_event
from tiger_memory.reliability import SourceRecoveryError
from tiger_memory.search import search


class SourceRecoveryTests(MemoryTestCase):
    def _hashes(self):
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in self.base.rglob("*.md")}

    def _orphan(self, key="opaque:a:b"):
        atomic_write = service.atomic_write

        def fail_after_source(path, text):
            atomic_write(path, text)
            if path.suffix == ".md" and "events" in path.parts:
                raise OSError("synthetic failure before database commit")

        with patch.object(service, "atomic_write", fail_after_source):
            with self.assertRaises(OSError):
                service.write_event(self.project, "original", "cobalt original",
                                    idempotency_key=key)
        return parse_event(next(self.base.rglob("*.md")).resolve())

    def test_unique_orphan_retry_preserves_original_identity_and_bytes(self):
        original = self._orphan()
        before = self._hashes()
        recovered = service.write_event(self.project, "changed", "amber retry",
                                        idempotency_key="opaque:a:b")
        self.assertEqual(original, recovered)
        self.assertEqual(before, self._hashes())
        report = service.rebuild_index()
        self.assertTrue(report["swapped"], report)
        with closing(database.connect_readonly()) as conn, conn:
            row = conn.execute("SELECT event_id,body FROM events").fetchone()
            self.assertEqual(tuple(row), (original.event_id, original.body))
        self.assertEqual(before, self._hashes())

    def test_recovery_failure_keeps_one_source_for_later_retry(self):
        original = self._orphan()
        before = self._hashes()
        with patch.object(service, "_index_validated_write", side_effect=OSError("retry index failure")):
            with self.assertRaises(OSError):
                service.write_event(self.project, "changed", "retry", idempotency_key="opaque:a:b")
        self.assertEqual(before, self._hashes())
        recovered = service.write_event(self.project, "changed", "retry", idempotency_key="opaque:a:b")
        self.assertEqual(recovered.event_id, original.event_id)
        self.assertEqual(before, self._hashes())

    def test_duplicate_sources_refuse_add_and_rebuild_without_changing_sources(self):
        first = service.write_event(self.project, "one", "cobalt", idempotency_key="same")
        second = service.write_event(self.project, "two", "amber", idempotency_key="different")
        # Construct a pre-existing ambiguous store without invoking recovery.
        second.metadata["idempotency_key"] = "same"
        second.path.write_text(serialize_event(second.metadata, second.body), encoding="utf-8")
        with closing(database.connect()) as conn, conn:
            conn.execute("DELETE FROM events WHERE event_id=?", (first.event_id,))
            conn.commit()
        before = self._hashes()
        with self.assertRaisesRegex(SourceRecoveryError, "SOURCE_RECOVERY_AMBIGUOUS"):
            service.write_event(self.project, "retry", "retry", idempotency_key="same")
        result = service.rebuild_index()
        self.assertFalse(result["swapped"])
        self.assertTrue(result["errors"])
        self.assertEqual(before, self._hashes())

    def test_valid_profiles_and_fact_projection_rebuild_keep_typed_references(self):
        event = service.write_event(self.project, "source", "cobalt source")
        service.write_fact(self.project, "engine", "cobalt", source_event_id=event.event_id)
        service.write_event(self.project, "derived", "cobalt derived",
                            extra={"source_event_id": event.event_id})
        before = self._hashes()
        for profile in ("legacy", "lexical", "passages"):
            with self.subTest(profile=profile):
                with closing(database.connect_readonly()) as conn, conn:
                    self.assertTrue(search(conn, event.project_id, "cobalt", profile=profile))
                result = service.rebuild_index()
                self.assertTrue(result["swapped"], result)
                self.assertEqual(result["errors"], [])
        self.assertEqual(before, self._hashes())
