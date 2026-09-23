from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path


class MemoryTestCase(unittest.TestCase):
    def confirmation(self, key, value):
        import json
        import hashlib
        import uuid
        from tiger_memory.evidence import fact_claim
        claim = fact_claim(key, value)
        path = self.project / ("confirmation-" + str(uuid.uuid4()) + ".json")
        raw = json.dumps({"schema_version": 1, "receipt_type": "user_confirmation", "asserted_claim": claim, "decision": "confirmed"}, ensure_ascii=False)
        path.write_text(raw, encoding="utf-8")
        return {"source_path": str(path.resolve()), "source_hash": hashlib.sha256(raw.encode()).hexdigest(), "asserted_claim": claim}

    def output_evidence(self, command="npm run verify"):
        import hashlib
        path = self.project / "tool-output.txt"
        path.write_text("All checks passed", encoding="utf-8")
        return {"command": command, "output_path": str(path.resolve()), "output_hash": hashlib.sha256(path.read_bytes()).hexdigest()}

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name) / "memory"
        self.project = Path(self.temp.name) / "workspace" / "中文项目"
        self.project.mkdir(parents=True)
        (self.project / "pyproject.toml").write_text("[project]\nname='fixture'\nversion='0'\n", encoding="utf-8")
        self.previous = os.environ.get("TIGER_MEMORY_DIR")
        os.environ["TIGER_MEMORY_DIR"] = str(self.base)

    def tearDown(self):
        if self.previous is None:
            os.environ.pop("TIGER_MEMORY_DIR", None)
        else:
            os.environ["TIGER_MEMORY_DIR"] = self.previous
        self.temp.cleanup()


def legacy_slug(root: Path) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(root.resolve()).strip(os.sep)) or "root"


def write_legacy(base: Path, root: Path, title: str, content: str, bucket: str = "") -> Path:
    folder = bucket or legacy_slug(root)
    notes = base / "projects" / folder / "notes"
    notes.mkdir(parents=True, exist_ok=True)
    path = notes / ("2026-01-01-" + str(abs(hash((str(root), title)))) + ".md")
    path.write_text(
        "\n".join([
            "---", "title: " + title, "time: 2026-01-01 08:00:00 +0800",
            "project_root: " + str(root), "tags: 测试,项目", "---", "", content, "",
        ]),
        encoding="utf-8",
    )
    return path
