import base64
import csv
import hashlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock
import zipfile

from tiger_memory import __version__, build_backend as backend


@unittest.skipUnless((backend.ROOT / "pyproject.toml").is_file(), "packaging checks require the source manifest")
class PackagingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="plm-package-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "source"
        package = self.root / "tiger_memory"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text('__version__ = "' + __version__ + '"\n')
        (package / "build_backend.py").write_bytes(Path(backend.__file__).read_bytes())
        (package / "cli.py").write_text("def main():\n    return 0\n")
        (self.root / "pyproject.toml").write_bytes((backend.ROOT / "pyproject.toml").read_bytes())
        (self.root / "README.md").write_text("Runtime fixture\n")
        (self.root / "THIRD_PARTY.md").write_text("Public attribution fixture\n")
        for name in backend.SDIST_PUBLIC_FILES:
            if name.startswith(("scripts/", "examples/", "tests/", "docs/", "skill/", "evaluation/", "repro/")):
                path = self.root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("PUBLIC_FIXTURE\n")
        for name in ("output/private.txt", ".codegraph/codegraph.db", "memory/fact.md", "models/weights.bin", "tiger_memory/__pycache__/cache.pyc"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("PRIVATE_SENTINEL")
        patcher = mock.patch.object(backend, "ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.dist = Path(self.temporary.name) / "dist"

    def test_wheel_record_metadata_and_allowlist(self):
        filename = backend.build_wheel(self.dist)
        with zipfile.ZipFile(self.dist / filename) as archive:
            names = archive.namelist()
            self.assertTrue(all(name.startswith(("tiger_memory/", "tiger_memory-")) for name in names))
            self.assertEqual({name for name in names if name.endswith(".py") and "/_bundle/" not in name}, {
                "tiger_memory/__init__.py", "tiger_memory/build_backend.py", "tiger_memory/cli.py"})
            self.assertFalse(any("__pycache__" in name for name in names))
            for name in names:
                self.assertNotIn(b"PRIVATE_SENTINEL", archive.read(name))
            prefix = backend._identity() + ".dist-info/"
            metadata = archive.read(prefix + "METADATA").decode()
            self.assertIn("Version: " + __version__, metadata)
            self.assertIn("Requires-Python: >=3.9", metadata)
            self.assertNotIn("Requires-Dist:", metadata)
            self.assertNotIn("License:", metadata)
            self.assertIn(b"tiger_memory = tiger_memory.cli:main", archive.read(prefix + "entry_points.txt"))
            self.assertEqual(archive.read(prefix + "THIRD_PARTY.md"), b"Public attribution fixture\n")
            records = list(csv.reader(io.StringIO(archive.read(prefix + "RECORD").decode())))
            self.assertEqual({row[0] for row in records}, set(names))
            for name, digest, size in records:
                if name.endswith("/RECORD"):
                    self.assertEqual((digest, size), ("", ""))
                    continue
                data = archive.read(name)
                expected = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
                self.assertEqual(digest, "sha256=" + expected)
                self.assertEqual(int(size), len(data))

    def test_prepared_metadata_is_checked(self):
        prepared = self.dist / backend.prepare_metadata_for_build_wheel(self.dist)
        backend.build_wheel(self.dist, metadata_directory=prepared)
        (prepared / "METADATA").write_text("different metadata")
        with self.assertRaisesRegex(ValueError, "prepared metadata"):
            backend.build_wheel(self.dist, metadata_directory=prepared)

    def test_sdist_rebuilds_identical_wheel_without_repository(self):
        wheel = self.dist / backend.build_wheel(self.dist)
        expected = wheel.read_bytes()
        sdist = self.dist / backend.build_sdist(self.dist)
        expanded = Path(self.temporary.name) / "expanded"
        expanded.mkdir()
        with tarfile.open(sdist) as archive:
            root_prefix = backend._identity() + "/"
            names = set(archive.getnames())
            for name in backend.SDIST_PUBLIC_FILES:
                if (self.root / name).is_file():
                    self.assertIn(root_prefix + name, names)
            for member in archive:
                self.assertTrue(member.isfile())
                path = expanded / member.name
                self.assertTrue(path.resolve().is_relative_to(expanded.resolve()))
                self.assertNotIn("..", Path(member.name).parts)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.extractfile(member).read())
        clean_source = expanded / backend._identity()
        rebuilt = Path(self.temporary.name) / "rebuilt"
        result = subprocess.run([sys.executable, "-c", "from tiger_memory.build_backend import build_wheel; import sys; build_wheel(sys.argv[1])", str(rebuilt)],
                                cwd=clean_source, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((rebuilt / wheel.name).read_bytes(), expected)

    def test_supplied_license_text_is_preserved_without_inventing_a_claim(self):
        license_text = b"User-selected license fixture\n"
        (self.root / "LICENSE").write_bytes(license_text)
        with zipfile.ZipFile(self.dist / backend.build_wheel(self.dist)) as archive:
            prefix = backend._identity() + ".dist-info/"
            self.assertEqual(archive.read(prefix + "LICENSE"), license_text)
            self.assertNotIn(b"License:", archive.read(prefix + "METADATA"))

    def test_sdist_requires_referenced_verification_assets(self):
        (self.root / "examples/code_memory/cases.json").unlink()
        with self.assertRaisesRegex(ValueError, "missing public source distribution file"):
            backend.build_sdist(self.dist)

    def test_version_mismatch_and_symlink_are_rejected(self):
        path = self.root / "pyproject.toml"
        original = path.read_text()
        path.write_text(original.replace(__version__, "0.0.0"))
        with self.assertRaisesRegex(ValueError, "version differs"):
            backend.build_wheel(self.dist)
        path.write_text(original)
        (self.root / "tiger_memory" / "external.py").symlink_to(self.root / "output" / "private.txt")
        with self.assertRaisesRegex(ValueError, "symlinks"):
            backend.build_wheel(self.dist)

    def test_manifest_changes_cannot_silently_misstate_dependencies_or_entrypoint(self):
        path = self.root / "pyproject.toml"
        original = path.read_text()
        path.write_text(original.replace("dependencies = []", 'dependencies = ["unexpected-dependency"]'))
        with self.assertRaisesRegex(ValueError, "dependencies"):
            backend.build_wheel(self.dist)
        path.write_text(original.replace("tiger_memory.cli:main", "tiger_memory.cli:another"))
        with self.assertRaisesRegex(ValueError, "entrypoint"):
            backend.build_wheel(self.dist)


@unittest.skipUnless((backend.ROOT / "pyproject.toml").is_file(), "packaging checks require the source manifest")
@unittest.skipIf(os.environ.get("PLM_INSTALL_VALIDATION") == "1", "isolated build integration is covered by the outer install verification")
class IsolatedInstallTests(unittest.TestCase):
    def test_offline_install_and_standalone_memory_without_host_config(self):
        from scripts.verify_install import verify
        result = verify()
        self.assertGreaterEqual(result["passed"], 8)


if __name__ == "__main__":
    unittest.main()
