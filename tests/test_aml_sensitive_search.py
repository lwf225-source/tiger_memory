"""Synthetic strict remote query rejection, including actual options joins."""
import contextlib
import io
import json
import os
from unittest.mock import patch

from tiger_memory import aml_adapter as aml
from tests.test_aml_adapter import AMLServerCase

PATTERNS = (
    "-----BEGIN PRIVATE KEY-----SYNTHETIC-----END PRIVATE KEY-----",
    "sk-" + "SYNTHETICx" * 3,
    "ghp_" + "SYNTHETICx" * 3,
    "AKIA" + "0" * 16,
    "eyJ" + "a" * 12 + "." + "b" * 12 + "." + "c" * 8,
    "password=SYNTHETIC_ONLY",
)

class StrictSearchSensitiveTests(AMLServerCase):
    def setUp(self):
        super().setUp()
        self.flags = patch.multiple(aml, AML_STRICT_RETRIEVAL=True, AML_EMBEDDING="remote-v4",
            AML_RERANKER="off", AML_QUERY_REWRITE="off", AML_FACT_PROJECTION="off",
            AML_OPTIONS_AS_QUERY_TERMS=True)
        self.flags.start()
        self.addCleanup(self.flags.stop)

    def test_six_patterns_positions_and_early_returns_zero_provider_cache_or_source(self):
        stream = io.StringIO()
        with patch.object(aml, "_model_provider", side_effect=AssertionError("provider forbidden")) as provider, \
             patch.object(aml, "_storage_identity_preflight", side_effect=AssertionError("storage forbidden")) as storage, \
             contextlib.redirect_stderr(stream):
            for token in PATTERNS:
                for query in (token + " suffix", "prefix " + token + " suffix", "prefix " + token):
                    for top_k in (0, 2):
                        with self.subTest(pattern=PATTERNS.index(token), top_k=top_k):
                            status, body = self.search(query, "unknown-synthetic-user", top_k)
                            self.assertEqual((status, body), (503, {"detail": {"reason": "sensitive query rejected"}}))
                            self.assertNotIn(token, json.dumps(body))
            provider.assert_not_called()
            storage.assert_not_called()
        self.assertEqual(stream.getvalue(), "")
        self.assertFalse(self.base.exists())

    def test_options_only_and_join_boundary(self):
        # Whitespace between '=' and value is detected only after the actual join.
        with patch.object(aml, "_model_provider") as provider:
            for query, options in (("normal", [PATTERNS[-1]]), ("password=", ["SYNTHETIC_ONLY"])):
                self.assertEqual(self.search(query, "unknown", 0, options=options)[0], 503)
            provider.assert_not_called()
        self.assertFalse(self.base.exists())

    def test_auth_and_schema_have_priority(self):
        with patch.dict(os.environ, {"AML_API_KEY": "synthetic-auth-only"}), patch.object(aml, "assert_no_secrets") as detector:
            self.assertEqual(self.search(PATTERNS[-1], "u")[0], 401)
            detector.assert_not_called()
        with patch.object(aml, "assert_no_secrets") as detector:
            self.assertEqual(self.search(PATTERNS[-1], "u", -1)[0], 422)
            detector.assert_not_called()

    def test_disabled_options_and_legacy_are_unchanged(self):
        with patch.object(aml, "AML_OPTIONS_AS_QUERY_TERMS", False):
            self.assertEqual(self.search("ordinary", "unknown", 0, options=[PATTERNS[-1]]), (200, {"data": []}))
        with patch.object(aml, "AML_STRICT_RETRIEVAL", False), patch.object(aml, "assert_no_secrets") as detector:
            self.assertEqual(self.search(PATTERNS[-1], "unknown", 0), (200, {"data": []}))
            detector.assert_not_called()

    def test_normal_actual_query_is_preserved(self):
        root = aml.ensure_user_storage("u")
        seen = []
        def context(*args, **kwargs):
            seen.append(args[1])
            kwargs["diagnostics"]["embedding"] = {"status": "ready"}
            return {}, []
        with patch.object(aml.service, "context_data", side_effect=context), patch.object(aml, "_model_provider"):
            self.assertEqual(self.search("普通 query", "u", 2, options=["one", "二"]), (200, {"data": []}))
        self.assertEqual(seen, ["普通 query\none\n二"])
