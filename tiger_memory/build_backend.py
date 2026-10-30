"""Dependency-free PEP 517 backend for PLM's pure-Python runtime.

Only explicitly selected source files enter distributions. Indexes, memories,
model weights, user output, and host-specific configuration are never included.
The wheel and source archive include the same offline verification resources.
"""

import ast
import base64
import csv
import gzip
import hashlib
import io
from pathlib import Path
import re
import tarfile
import zipfile

from . import __version__


ROOT = Path(__file__).resolve().parent.parent
NOTICES = ("THIRD_PARTY.md", "LICENSE", "LICENSE.md", "LICENSE.txt", "NOTICE")
SDIST_PUBLIC_FILES = (
    "pyproject.toml", "README.md", "docs/CODEGRAPH.md", "docs/AML_SUBMISSION.md",
    "docs/OPEN_SOURCE_PREP.md", "repro/aml-open-source.env.example", *NOTICES,
    "repro/model-requirements.lock", "repro/linux-cpu-requirements.lock",
    "repro/bge-small-en-v1.5.json",
    "scripts/tiger_memory.py", "scripts/verify_install.py", "scripts/verify_codegraph.py",
    "scripts/evaluate_code_memory.py", "examples/code_memory/README.md",
    "examples/code_memory/before.py", "examples/code_memory/after.py",
    "examples/code_memory/cases.json", "tests/__init__.py", "tests/common.py", "tests/test_packaging.py",
    "tests/test_codegraph.py", "tests/test_code_memory_evaluation.py",
    "skill/SKILL.md", "skill/agents/openai.yaml",
    "skill/legacy/memory_context_v1.py", "skill/legacy/memory_write_v1.py",
    "skill/references/task-workflow.md",
    "scripts/aml_preflight.py", "scripts/aml_smoke.py", "scripts/audit_batch3.py",
    "scripts/prepare_longmemeval.py", "scripts/prepare_qa_audit.py",
    "scripts/verify_release.py", "scripts/inspect_aml_diagnostic.py",
    "scripts/resolve_aml_search_trace.py", "scripts/demo_active_memory_task.py",
    "scripts/verify_active_memory.py", "docs/ACTIVE_MEMORY.md",
    "docs/AML_SEARCH_DIAGNOSTICS.md", "docs/EXPERIMENTAL_PROJECT_FTS.md",
    "docs/GITHUB_SYNC_20261030.md",
    "scripts/aml_reference_README.md", "scripts/aml_reference.env.example",
    "scripts/aml_reference_manifest.json", "scripts/aml_candidate_decision.json",
)

# Narrow public roots only: never sweep evaluation reports, model assets,
# local worktrees or user-created scripts into a distribution.
PUBLIC_TREES = {
    "examples/active_memory": {".json", ".py", ".md"},
    "tests": {".py"},
    "evaluation/synthetic-v1": {".json", ".py", ".md"},
    "evaluation/project-docs-v1": {".json", ".py", ".md"},
    "evaluation/long-context-v1": {".json", ".py", ".md"},
}


def _source_bytes(name):
    path = ROOT / name
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != ROOT and ROOT in parent.parents):
        raise ValueError("source must not contain symlinks")
    return path.read_bytes() if path.is_file() else None


def _project():
    # The manifest uses string literals for these fields; this narrow parser
    # avoids a TOML dependency on Python 3.9 and permits fully offline builds.
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    section = re.search(r"(?ms)^\[project\]\s*\n(.*?)(?=^\[|\Z)", text)
    if section is None:
        raise ValueError("missing project metadata")
    values = {}
    for key in ("name", "version", "description", "requires-python"):
        match = re.search(r"(?m)^" + re.escape(key) + r"\s*=\s*(.+)$", section.group(1))
        if match is None:
            raise ValueError("missing project metadata: " + key)
        value = ast.literal_eval(match.group(1))
        if not isinstance(value, str) or "\n" in value or "\r" in value:
            raise ValueError("invalid project metadata: " + key)
        values[key] = value
    if values["version"] != __version__:
        raise ValueError("pyproject version differs from package version")
    if values["name"] != "tiger_memory":
        raise ValueError("unexpected distribution name")
    dependencies = re.search(r"(?m)^dependencies\s*=\s*(.+)$", section.group(1))
    if dependencies is None or ast.literal_eval(dependencies.group(1)) != []:
        raise ValueError("this runtime backend requires dependencies = []")
    scripts = re.search(r"(?ms)^\[project.scripts\]\s*\n(.*?)(?=^\[|\Z)", text)
    entrypoint = re.search(r"(?m)^tiger_memory\s*=\s*(.+)$", scripts.group(1)) if scripts else None
    if entrypoint is None or ast.literal_eval(entrypoint.group(1)) != "tiger_memory.cli:main":
        raise ValueError("unexpected tiger_memory console entrypoint")
    return values


def _identity():
    project = _project()
    return project["name"].replace("-", "_") + "-" + project["version"]


def _metadata_files():
    project = _project()
    metadata = "\n".join([
        "Metadata-Version: 2.1", "Name: " + project["name"],
        "Version: " + project["version"], "Summary: " + project["description"],
        "Requires-Python: " + project["requires-python"],
        "Description-Content-Type: text/markdown", "", "",
    ])
    readme = ROOT / "README.md"
    if readme.is_file() and not readme.is_symlink():
        metadata += readme.read_text(encoding="utf-8")
    files = {
        "METADATA": metadata.encode("utf-8"),
        "WHEEL": b"Wheel-Version: 1.0\nGenerator: plm-build-backend\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        "entry_points.txt": b"[console_scripts]\ntiger_memory = tiger_memory.cli:main\n",
        "top_level.txt": b"tiger_memory\n",
    }
    # Preserve attribution and license text when supplied, without inferring a
    # license declaration when the project has not selected one.
    for name in NOTICES:
        content = _source_bytes(name)
        if content is not None:
            files[name] = content
    return files


def _package_files():
    package = ROOT / "tiger_memory"
    if package.is_symlink():
        raise ValueError("package directory must not be a symlink")
    for path in sorted(package.rglob("*.py")):
        relative = path.relative_to(ROOT)
        if "__pycache__" in relative.parts or "_bundle" in relative.parts:
            continue
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != ROOT and ROOT in parent.parents):
            raise ValueError("source must not contain symlinks")
        yield relative.as_posix(), path.read_bytes()


def _support_files():
    names = set(SDIST_PUBLIC_FILES)
    for folder, extensions in PUBLIC_TREES.items():
        base = ROOT / folder
        if base.is_symlink():
            raise ValueError("source must not contain symlinks")
        for path in base.rglob("*"):
            if "__pycache__" not in path.parts and path.suffix in extensions:
                names.add(path.relative_to(ROOT).as_posix())
    files = {}
    for name in sorted(names):
        content = _source_bytes(name)
        if content is None and name not in NOTICES:
            raise ValueError("missing public source distribution file: " + name)
        if content is not None:
            files[name] = content
    return files


def get_requires_for_build_wheel(config_settings=None):
    return []


def get_requires_for_build_sdist(config_settings=None):
    return []


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    name = _identity() + ".dist-info"
    target = Path(metadata_directory) / name
    target.mkdir(parents=True, exist_ok=True)
    for filename, content in _metadata_files().items():
        (target / filename).write_bytes(content)
    return name


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    identity = _identity()
    metadata = _metadata_files()
    if metadata_directory is not None:
        # Wheel metadata must match the metadata previously given to the frontend.
        supplied = Path(metadata_directory)
        for filename, content in metadata.items():
            if (supplied / filename).read_bytes() != content:
                raise ValueError("prepared metadata no longer matches source")
    files = dict(_package_files())
    files.update({"tiger_memory/_bundle/" + name: content
                  for name, content in _support_files().items()})
    prefix = identity + ".dist-info/"
    files.update({prefix + name: content for name, content in metadata.items()})
    records = io.StringIO(newline="")
    writer = csv.writer(records, lineterminator="\n")
    for name, content in sorted(files.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode("ascii")
        writer.writerow([name, "sha256=" + digest, str(len(content))])
    writer.writerow([prefix + "RECORD", "", ""])
    files[prefix + "RECORD"] = records.getvalue().encode("utf-8")
    filename = identity + "-py3-none-any.whl"
    Path(wheel_directory).mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(Path(wheel_directory) / filename, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content)
    return filename


def build_sdist(sdist_directory, config_settings=None):
    identity = _identity()
    files = dict(_package_files())
    files.update(_support_files())
    files["PKG-INFO"] = _metadata_files()["METADATA"]
    filename = identity + ".tar.gz"
    Path(sdist_directory).mkdir(parents=True, exist_ok=True)
    with (Path(sdist_directory) / filename).open("wb") as destination:
        with gzip.GzipFile(filename="", mode="wb", fileobj=destination, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for name, content in sorted(files.items()):
                    info = tarfile.TarInfo(identity + "/" + name)
                    info.size = len(content)
                    info.mode = 0o644
                    info.mtime = 0
                    archive.addfile(info, io.BytesIO(content))
    return filename
