from __future__ import annotations

import io
import json
import os
import uuid
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from tiger_memory.cli import build_parser, main
from tiger_memory.database import connect
from tiger_memory.events import atomic_write, serialize_event
from tiger_memory.model import PLMError, sha256_bytes, v2_root
from tiger_memory.passages import locate_passage, validate_passage
from tiger_memory.service import (
    context_data, evidence_data, forget, render_context, supersede_fact, write_event, write_fact,
)
from tests.common import MemoryTestCase


class PassageServiceTests(MemoryTestCase):
    def event(self, **kwargs):
        return write_event(self.project, "局部证据", "User: 仅限本地。\nAssistant: 尚未真机验收，不得发布。", **kwargs)

    def read(self, event, **kwargs):
        arguments = {"body_sha256": sha256_bytes(event.body.encode("utf-8")), "start": 3, "end": 12,
                     "before_chars": 2, "after_chars": 3, "max_chars": 18}
        arguments.update(kwargs)
        return evidence_data(self.project, event.event_id, **arguments)

    def test_exact_expansion_reads_actual_parent_without_usage_or_mutation(self):
        event = self.event()
        raw_before = event.path.read_bytes()
        with patch("tiger_memory.service.record_usage") as usage:
            span = self.read(event)
        usage.assert_not_called()
        self.assertEqual(span["text"], event.body[1:15])
        self.assertEqual(span["parent_id"], event.event_id)
        self.assertEqual(span, validate_passage(event.body, span, event.event_id))
        self.assertEqual(event.path.read_bytes(), raw_before)

    def test_default_context_profile_is_unchanged_and_passages_is_opt_in(self):
        event = self.event()
        _, baseline = context_data(self.project, "不得发布")
        _, selected = context_data(self.project, "不得发布", retrieval_profile="passages")
        self.assertEqual(baseline[0].source_event_id, event.event_id)
        self.assertEqual(baseline[0].evidence_spans, [])
        self.assertTrue(selected[0].evidence_spans)
        for span in selected[0].evidence_spans:
            validate_passage(event.body, span, event.event_id)
        with patch("tiger_memory.service.context_data", return_value=({"project_id": event.project_id}, [])) as context:
            render_context(self.project, "query", retrieval_profile="passages")
        self.assertEqual(context.call_args.args[-1], "passages")
        with self.assertRaises(PLMError):
            context_data(self.project, "", retrieval_profile="unknown")

    def test_project_and_private_scope_are_checked_before_any_file_read(self):
        event = self.event(scope="session", scope_id="private-1")
        digest = sha256_bytes(event.body.encode("utf-8"))
        other = self.project.parent / "other-project"
        other.mkdir()
        with patch("tiger_memory.service._read_evidence_event") as reader:
            for cwd, scope, scope_id in ((self.project, "project", ""), (self.project, "session", ""),
                                        (self.project, "session", "private-2"), (other, "session", "private-1")):
                with self.subTest(scope=scope, scope_id=scope_id), self.assertRaises(PLMError):
                    evidence_data(cwd, event.event_id, digest, 0, 4, scope=scope, scope_id=scope_id)
        reader.assert_not_called()
        span = self.read(event, scope="session", scope_id="private-1")
        self.assertEqual(span["parent_id"], event.event_id)

    def test_retracted_parent_or_expired_source_chain_cannot_be_expanded(self):
        event = self.event()
        forget(self.project, event.event_id)
        with patch("tiger_memory.service._read_evidence_event") as reader, self.assertRaises(PLMError):
            self.read(event)
        reader.assert_not_called()
        old = write_fact(self.project, "build.mode", "local", valid_from="2026-01-01T00:00:00Z")
        derived = write_event(self.project, "来源摘要", "这个流程只针对本地构建，不适用于公开环境。",
                              extra={"source_event_id": old.event_id})
        supersede_fact(self.project, old.metadata["fact_id"], "checked", valid_from="2099-01-01T00:00:00Z")
        self.read(derived)
        with patch("tiger_memory.service.utc_now", return_value="2100-01-01T00:00:00.000000Z"):
            with patch("tiger_memory.service._read_evidence_event") as reader, self.assertRaises(PLMError):
                self.read(derived)
        reader.assert_not_called()

    def test_fact_uses_assertion_event_identity_and_preserves_conditions(self):
        fact = write_fact(self.project, "build.mode", "local only", conditions={"network": "forbidden"})
        span = self.read(fact, start=0, end=len(fact.body), before_chars=0, after_chars=0, max_chars=200)
        self.assertIn("Conditions:", span["text"])
        with self.assertRaises(PLMError):
            evidence_data(self.project, fact.metadata["fact_id"], span["body_sha256"], 0, 5)

    def test_legacy_display_path_is_not_used_for_evidence_read(self):
        event = self.event(source_legacy_path=str(self.project / "nonexistent-v1.md"))
        self.assertEqual(self.read(event)["parent_id"], event.event_id)

    def test_stale_hash_offsets_or_unknown_identity_fail_without_body_in_errors(self):
        event = self.event()
        for changes in ({"body_sha256": "0" * 64}, {"start": -1}, {"end": len(event.body) + 1},
                        {"start": 4, "end": 4}, {"max_chars": 2}, {"body_sha256": "sentinel-sensitive-invalid-hash"}):
            with self.subTest(changes=changes), self.assertRaises(PLMError) as caught:
                self.read(event, **changes)
            self.assertEqual(str(caught.exception), "evidence unavailable: request or provenance validation failed")
        with self.assertRaises(PLMError):
            evidence_data(self.project, str(uuid.uuid4()), "0" * 64, 0, 4)

    def test_tampered_source_body_or_identity_is_rejected(self):
        event = self.event()
        atomic_write(event.path, serialize_event(event.metadata, "changed source body"))
        with self.assertRaises(PLMError):
            self.read(event)
        changed = dict(event.metadata, id=str(uuid.uuid4()))
        atomic_write(event.path, serialize_event(changed, event.body))
        with self.assertRaises(PLMError):
            self.read(event)
        changed = dict(event.metadata, recorded_at=123)
        atomic_write(event.path, serialize_event(changed, event.body))
        with self.assertRaises(PLMError):
            self.read(event)
        atomic_write(event.path, "---\nsecret-looking-field: sentinel-not-for-errors\n---\n")
        with self.assertRaises(PLMError) as caught:
            self.read(event)
        self.assertNotIn("sentinel", str(caught.exception))
        self.assertNotIn(str(event.path), str(caught.exception))

    def test_symlink_final_intermediate_and_permissive_file_are_rejected(self):
        event = self.event()
        saved = event.path.with_suffix(".saved")
        event.path.rename(saved)
        event.path.symlink_to(saved)
        with self.assertRaises(PLMError):
            self.read(event)
        event.path.unlink()
        saved.rename(event.path)
        os.chmod(event.path, 0o644)
        with self.assertRaises(PLMError):
            self.read(event)
        os.chmod(event.path, 0o600)
        parent = event.path.parent
        moved = parent.with_name(parent.name + "-saved")
        parent.rename(moved)
        parent.symlink_to(moved, target_is_directory=True)
        with self.assertRaises(PLMError):
            self.read(event)

    def test_index_path_traversal_and_outside_path_are_rejected(self):
        event = self.event()
        unsafe = [str(v2_root() / "projects" / ".." / ".." / "outside.md"), str(self.project / "outside.md")]
        for path in unsafe:
            conn = connect()
            conn.execute("UPDATE events SET source_path=? WHERE event_id=?", (path, event.event_id))
            conn.commit()
            conn.close()
            with self.subTest(path=path), self.assertRaises(PLMError):
                self.read(event)

    def test_cli_evidence_json_and_failure_are_read_only_and_profile_is_forwarded(self):
        event = self.event()
        digest = sha256_bytes(event.body.encode("utf-8"))
        args = ["evidence", "--cwd", str(self.project), "--event-id", event.event_id,
                "--body-sha256", digest, "--start", "0", "--end", "8", "--before-chars", "0", "--after-chars", "0"]
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors), patch("tiger_memory.service.record_usage") as usage:
            self.assertEqual(main(args), 0)
        usage.assert_not_called()
        self.assertEqual(json.loads(output.getvalue()), locate_passage(event.body, event.event_id, 0, 8))
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            self.assertEqual(main(args + ["--max-chars", "1"]), 2)
        self.assertEqual(output.getvalue(), "")
        self.assertNotIn(event.body, errors.getvalue())
        self.assertEqual(build_parser().parse_args(["context"]).retrieval_profile, "lexical")
        with patch("tiger_memory.cli.context_data", return_value=({"project_id": event.project_id}, [])) as context:
            with redirect_stdout(io.StringIO()), patch("tiger_memory.cli.record_usage"):
                self.assertEqual(main(["context", "--cwd", str(self.project), "--query", "local", "--json", "--retrieval-profile", "passages"]), 0)
        self.assertEqual(context.call_args.args[-1], "passages")
