#!/usr/bin/env python3
"""Build and install the runtime offline in a disposable, host-independent venv.

Use --with-codegraph to also exercise an already installed CodeGraph CLI. Nothing
is installed globally; user memories, Codex configuration and indexes are unused.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tiger_memory import __version__, build_backend


def verify(with_codegraph=False):
    started = time.monotonic()
    checks = []
    executables = {name: shutil.which(name) for name in ("git", "codegraph", "node", "dirname", "readlink")}
    if not executables["git"]:
        raise RuntimeError("git is required for installation acceptance")
    if with_codegraph and not executables["codegraph"]:
        raise RuntimeError("--with-codegraph requires an already installed CodeGraph CLI")
    with tempfile.TemporaryDirectory(prefix="plm-install-acceptance-") as temporary:
        base = Path(temporary).resolve()
        home = base / "clean-home"
        home.mkdir()
        work = base / "unrelated-project"
        work.mkdir()
        toolbin = base / "tools"
        toolbin.mkdir()
        # Only Git is initially on PATH; any host CodeGraph CLI is inaccessible.
        (toolbin / "git").symlink_to(executables["git"])
        env = {
            "HOME": str(home), "PATH": str(toolbin), "LANG": "en_US.UTF-8",
            "PYTHONNOUSERSITE": "1", "PIP_CONFIG_FILE": os.devnull,
            "PIP_NO_INDEX": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
            "TIGER_MEMORY_DIR": str(base / "memory"),
            "CODEX_HOME": str(home / ".codex"),
            "CODEGRAPH_NO_DAEMON": "1", "DO_NOT_TRACK": "1",
            "CODEGRAPH_NO_DOWNLOAD": "1", "CODEGRAPH_NO_UPDATE_CHECK": "1",
        }
        # Required to start executables on Windows, if this verifier is ported.
        for key in ("SYSTEMROOT", "WINDIR"):
            if key in os.environ:
                env[key] = os.environ[key]

        def run(args, expected=0, timeout=120):
            process = subprocess.run([str(arg) for arg in args], cwd=work, env=env,
                                     capture_output=True, text=True, timeout=timeout)
            if process.returncode != expected:
                raise AssertionError({"command": [str(arg) for arg in args],
                                      "returncode": process.returncode,
                                      "stdout": process.stdout[-4000:], "stderr": process.stderr[-4000:]})
            return process.stdout

        archive = base / build_backend.build_sdist(str(base))
        environment = base / "venv"
        run([sys.executable, "-m", "venv", str(environment)])
        binary = environment / ("Scripts" if os.name == "nt" else "bin")
        python = binary / ("python.exe" if os.name == "nt" else "python")
        plm = binary / ("tiger_memory.exe" if os.name == "nt" else "tiger_memory")
        env["PATH"] = str(binary) + os.pathsep + str(toolbin)
        # Default PEP 517 build isolation stays enabled; requires=[] needs no index.
        run([python, "-m", "pip", "install", "--no-index", "--no-deps", "--no-cache-dir", archive])
        checks.append("offline-sdist-pep517-install")
        details = json.loads(run([python, "-c", "import json, pathlib, importlib.metadata as m, tiger_memory as p; "
                                 "d=m.distribution('tiger_memory'); "
                                 "print(json.dumps({'module':str(pathlib.Path(p.__file__).resolve()),'version':d.version,'requires':d.requires}))"]))
        assert Path(details["module"]).is_relative_to(environment), details
        assert details["version"] == __version__ and not details["requires"], details
        checks.append("installed-site-packages-version-and-zero-dependencies")
        run([python, "-c", "import importlib, pkgutil, tiger_memory; "
             "[importlib.import_module(m.name) for m in pkgutil.iter_modules(tiger_memory.__path__, tiger_memory.__name__+'.')]"])
        checks.append("all-runtime-modules-import")
        candidate = json.loads(run([python, "-m", "tiger_memory.aml_open_source", "--check-config"], expected=2))
        assert candidate["configured"] is False and candidate["remote_probe_performed"] is False, candidate
        assert candidate["config"]["embedding"]["model"] == "text-embedding-v4", candidate
        run([python, "-c", "from pathlib import Path; import tiger_memory; "
             "p=Path(tiger_memory.__file__).parent/'_bundle'; "
             "assert (p/'docs/OPEN_SOURCE_PREP.md').is_file(); "
             "assert (p/'repro/aml-open-source.env.example').is_file()"])
        checks.append("open-source-entrypoint-and-resources-without-api-call")
        assert "context" in run([plm, "--help"])
        assert "--with-code" in run([plm, "context", "--help"])
        assert "init" in run([plm, "code", "--help"])
        checks.append("standalone-console-entrypoints")
        run(["git", "init", "-q"])
        run(["git", "config", "user.email", "fixture@local"])
        run(["git", "config", "user.name", "fixture"])
        (work / "app.py").write_text("def helper():\n    return 7\n\ndef entry():\n    return helper()\n", encoding="utf-8")
        (work / ".gitignore").write_text(".codegraph/\n", encoding="utf-8")
        run(["git", "add", "."])
        run(["git", "commit", "-qm", "isolated-fixture"])
        run([plm, "write", "--cwd", work, "--title", "entry decision", "--content", "Keep entry compatible for existing callers."])
        arguments = [plm, "context", "--cwd", work, "--query", "entry", "--json"]
        historical = json.loads(run(arguments))
        assert any("Keep entry compatible" in result["body"] for result in historical["results"]), historical
        assert "code_evidence" not in historical
        checks.append("isolated-memory-write-and-recall")
        status = json.loads(run([plm, "code", "status", "--cwd", work]))
        assert status["status"] != "ready" and status["cli_available"] is False, status
        combined = json.loads(run(arguments + ["--with-code"]))
        assert combined["code_evidence"]["status"] != "ready", combined
        assert combined["results"] == historical["results"], combined
        checks.append("missing-codegraph-preserves-memory-results")
        assert not (work / ".codegraph").exists()
        checks.append("no-implicit-index")
        if with_codegraph:
            # CodeGraph's bundled shell launcher resolves its directory using
            # these standard utilities; do not inherit the user's whole PATH.
            for name in ("codegraph", "node", "dirname", "readlink"):
                if executables[name]:
                    (toolbin / name).symlink_to(executables[name])
            initialized = json.loads(run([plm, "code", "init", "--cwd", work, "--timeout", "180"], timeout=200))
            assert initialized["status"] == "ready", initialized
            combined = json.loads(run(arguments + ["--with-code"]))
            assert combined["code_evidence"]["status"] == "ready", combined
            assert "app.py" in combined["code_evidence"]["text"] and "helper" in combined["code_evidence"]["text"]
            assert combined["results"] == historical["results"], combined
            checks.append("installed-runtime-real-codegraph-combined-evidence")
        assert not (home / ".codex").exists()
        checks.append("no-codex-configuration")
    return {"passed": len(checks), "checks": checks, "version": __version__,
            "with_codegraph": with_codegraph, "elapsed_s": round(time.monotonic() - started, 2)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-codegraph", action="store_true")
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments.with_codegraph), ensure_ascii=False, indent=2))
