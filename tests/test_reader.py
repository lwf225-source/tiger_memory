import hashlib
import json
import os
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from tiger_memory.model import PLMError
from tiger_memory.reader import LocalCausalReader
from tests.common import MemoryTestCase


class ReaderTests(MemoryTestCase):
    def artifacts(self):
        folder = self.project / "model"
        folder.mkdir()
        payloads = {"config.json": '{"model_type":"qwen3"}', "tokenizer.json": "{}", "tokenizer_config.json": "{}", "model.safetensors": "fixture not real weights"}
        hashes = {}
        for name, value in payloads.items():
            (folder / name).write_text(value)
            hashes[name] = hashlib.sha256(value.encode()).hexdigest()
        (folder / "download-manifest.json").write_text(json.dumps({"model_id": "test", "revision": "r1", "files": hashes}))
        return LocalCausalReader(folder, "test", "r1")

    def test_constructor_does_not_import_model_runtime(self):
        with patch("importlib.import_module", side_effect=AssertionError("no runtime import")):
            model = LocalCausalReader(self.project / "absent", "test", "r1")
            with self.assertRaises(PLMError):
                model.prepare()

    def test_artifacts_bound_to_identity_and_bytes(self):
        model = self.artifacts()
        self.assertEqual(model._validate_artifacts()["revision"], "r1")
        model.revision = "different"
        with self.assertRaises(PLMError):
            model._validate_artifacts()
        model.revision = "r1"
        (model.path / "config.json").write_text("changed")
        with self.assertRaises(PLMError):
            model._validate_artifacts()

    def test_links_code_and_unbound_artifacts_rejected(self):
        model = self.artifacts()
        path = model.path / "remote.py"
        path.write_text("raise RuntimeError('must not run')")
        with self.assertRaises(PLMError):
            model._validate_artifacts()
        path.unlink()
        path = model.path / "extra.json"
        path.write_text("{}")
        with self.assertRaises(PLMError):
            model._validate_artifacts()
        path.unlink()
        path.symlink_to(model.path / "config.json")
        with self.assertRaises(PLMError):
            model._validate_artifacts()

    def test_stat_identity_rejects_mid_run_change(self):
        model = self.artifacts()
        model._signature = model._stat_signature()
        model._assert_unchanged()
        (model.path / "config.json").write_text("changed")
        with self.assertRaises(PLMError):
            model._assert_unchanged()

    def test_limits(self):
        for kwargs in ({"seed": -1}, {"max_input_tokens": 1}, {"max_generation_seconds": 301}, {"output_mode": "auto"}):
            with self.assertRaises(PLMError):
                LocalCausalReader(self.project, "test", "r1", **kwargs)

    def prepared_fake(self):
        model = self.artifacts()
        model._manifest = {"fixture": True}
        model._signature = model._stat_signature()
        model._model = object()
        model._tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: [1, 2, 3])
        return model

    def test_generation_is_greedy_no_thinking_and_reports_tokens(self):
        model = self.prepared_fake()
        seen = {}
        def template(messages, **kwargs):
            seen["template"] = kwargs
            return [1, 2, 3]
        model._tokenizer.apply_chat_template = template
        runtime, sample, mx, mlx = (ModuleType(name) for name in ("mlx_lm", "mlx_lm.sample_utils", "mlx.core", "mlx"))
        def stream(*args, **kwargs):
            seen["generation"] = kwargs
            yield SimpleNamespace(text="{}", generation_tokens=2)
        runtime.stream_generate = stream
        sample.make_sampler = lambda **kwargs: seen.setdefault("sampler", kwargs)
        mx.random = SimpleNamespace(seed=lambda value: seen.setdefault("seed", value))
        mx.clear_cache = lambda: seen.setdefault("cleared", True)
        mlx.core = mx
        with patch.dict(sys.modules, {"mlx_lm": runtime, "mlx_lm.sample_utils": sample, "mlx": mlx, "mlx.core": mx}):
            result = model.generate([{"role": "user", "content": "fixture"}], max_new_tokens=2)
        self.assertEqual(seen["template"], {"tokenize": True, "add_generation_prompt": True, "enable_thinking": False})
        self.assertEqual(seen["sampler"], {"temp": 0})
        self.assertEqual(seen["seed"], 0)
        self.assertTrue(seen["cleared"])
        self.assertEqual(result["input_tokens"], 3)
        self.assertEqual(result["output_tokens"], 2)
        self.assertEqual(result["finish_reason"], "length")

    def test_extra_reader_fields_and_invalid_role_rejected(self):
        model = self.prepared_fake()
        for messages in ([{"role": "user", "content": "q", "answer": "gold"}],
                         [{"role": "tool", "content": "q"}], [], "question"):
            with self.assertRaises(PLMError):
                model.generate(messages)

    def test_same_size_content_change_with_restored_mtime_rejected(self):
        model = self.artifacts()
        model._signature = model._stat_signature()
        path = model.path / "config.json"
        stat = path.stat()
        path.write_text(path.read_text().replace("qwen3", "qwen4"))
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        with self.assertRaises(PLMError): model._assert_unchanged()

    def test_mode_change_after_binding_rejected(self):
        model = self.prepared_fake()
        model._manifest = {"decoding": {"output_mode": "free"}}
        model.output_mode = "json_schema"
        with self.assertRaises(PLMError): model.describe()

    def test_constrained_generation_uses_mask_and_never_repairs(self):
        from tests.test_structured_output import FakeMX, Logits, Vector, byte_vocabulary
        from tiger_memory.structured_output import grammar_manifest
        model = self.prepared_fake()
        model.output_mode = "json_schema"
        model._vocabulary = byte_vocabulary()
        model._manifest = {"decoding": {"output_mode": "json_schema"}, "structured_output": grammar_manifest()}
        runtime, sample, mx, mlx = (ModuleType(name) for name in ("mlx_lm", "mlx_lm.sample_utils", "mlx.core", "mlx"))
        for name in ("array", "isfinite", "any", "full"):
            setattr(mx, name, getattr(FakeMX, name))
        mx.random = SimpleNamespace(seed=lambda value: None)
        mx.clear_cache = lambda: None
        mlx.core = mx
        sample.make_sampler = lambda **kwargs: kwargs
        output = '{"answer":"","abstained":true,"citations":[]}'
        def stream(*args, **kwargs):
            processor = kwargs["logits_processors"][0]
            sequence = [999]
            for byte in output.encode():
                processor(Vector(sequence), Logits(258))
                sequence.append(byte)
            processor(Vector(sequence), Logits(258))
            yield SimpleNamespace(text=output, generation_tokens=len(output) + 1)
        runtime.stream_generate = stream
        messages = [{"role": "system", "content": "fixed"}, {"role": "user", "content": json.dumps({
            "question": "q", "question_date": "", "sources": []})}]
        modules = {"mlx_lm": runtime, "mlx_lm.sample_utils": sample, "mlx": mlx, "mlx.core": mx}
        with patch.dict(sys.modules, modules):
            result = model.generate(messages)
            self.assertEqual(result["text"], output)
            self.assertTrue(result["structured_output"]["grammar_valid"])
            self.assertTrue(result["structured_output"]["empty_sources_forced_abstention"])
            # A faulty backend producing a wrapper is not repaired or accepted.
            output = "```json\n{}\n```"
            with self.assertRaises(PLMError): model.generate(messages)

    def test_constrained_truncation_remains_length_failure(self):
        from tests.test_structured_output import byte_vocabulary
        from tiger_memory.structured_output import grammar_manifest
        model = self.prepared_fake()
        model.output_mode = "json_schema"
        model._vocabulary = byte_vocabulary()
        model._manifest = {"decoding": {"output_mode": "json_schema"}, "structured_output": grammar_manifest()}
        runtime, sample, mx, mlx = (ModuleType(name) for name in ("mlx_lm", "mlx_lm.sample_utils", "mlx.core", "mlx"))
        runtime.stream_generate = lambda *a, **kw: iter([SimpleNamespace(text="{", generation_tokens=1)])
        # The production generator supports close(); preserve that contract in the fake.
        def stream(*args, **kwargs): yield SimpleNamespace(text="{", generation_tokens=1)
        runtime.stream_generate = stream
        sample.make_sampler = lambda **kwargs: kwargs
        mx.random = SimpleNamespace(seed=lambda value: None)
        mx.clear_cache = lambda: None
        mlx.core = mx
        messages = [{"role": "system", "content": "fixed"}, {"role": "user", "content": json.dumps({
            "question": "q", "question_date": "", "sources": []})}]
        with patch.dict(sys.modules, {"mlx_lm": runtime, "mlx_lm.sample_utils": sample, "mlx": mlx, "mlx.core": mx}):
            result = model.generate(messages, max_new_tokens=1)
        self.assertEqual(result["finish_reason"], "length")
        self.assertFalse(result["structured_output"]["grammar_valid"])
        self.assertEqual(result["text"], "{")
