"""Read-only AML store topology checks; never inspect service credentials."""
from __future__ import annotations

import argparse
import json
import os
import re
import stat
import subprocess
from pathlib import Path

MARKER = ".tiger-aml-open-source.json"
PROFILE = "aml-open-source-v4-1"
GIT_TIMEOUT_SECONDS = 5
_GIT_TOPOLOGY_ENV = (
    "GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_CEILING_DIRECTORIES",
    "GIT_DISCOVERY_ACROSS_FILESYSTEM", "GIT_NAMESPACE",
)
HINTS = {
    "store_absolute_required": "Choose a dedicated absolute store path outside any Git checkout.",
    "store_canonical_required": "Use the canonical path without symlink components or '..'.",
    "store_daily_overlap": "Choose a separate directory outside the daily project-memory tree and its ancestors.",
    "store_not_directory": "Choose a directory path whose existing ancestors are directories.",
    "store_git_environment": "Remove Git topology environment overrides from the service environment.",
    "store_inside_git": "Move the AML store outside all Git worktrees and bare repositories; keep source and data separate.",
    "store_git_probe_unavailable": "Install Git for local topology checks; the service has not started.",
    "store_git_probe_timeout": "Resolve the local Git probe timeout before starting the service.",
    "store_git_probe_failed": "Resolve the local Git probe failure; it is not treated as a non-Git directory.",
    "store_marker_conflict": "Refusing nonempty or differently configured store; choose a fresh dedicated directory or the matching original configuration.",
    "store_io_error": "Check directory access and marker permissions; no store files were changed.",
}


class StorePreflightError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code + ": " + HINTS[code])


def _reject(code):
    raise StorePreflightError(code)


def _outside_git(ancestor, environ):
    if any(environ.get(name) for name in _GIT_TOPOLOGY_ENV):
        _reject("store_git_environment")
    for directory in (ancestor, *ancestor.parents):
        try:
            (directory / ".git").lstat()
        except FileNotFoundError:
            continue
        # Includes worktree .git files, invalid markers, and broken symlinks:
        # none is a supported stable non-Git storage topology.
        _reject("store_inside_git")
    # Match the service's Git configuration context; ignoring core.worktree
    # overrides here could approve a path the request-time guard rejects.
    # rev-parse is local; stdout/stderr and environment values never leave this
    # checker. Tests supply a clean environment and disable user Git config.
    probe_env = dict(environ)
    probe_env["LC_ALL"] = "C"
    try:
        result = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(ancestor), "rev-parse",
             "--is-inside-work-tree", "--is-bare-repository"],
            capture_output=True, text=True, timeout=GIT_TIMEOUT_SECONDS,
            env=probe_env, check=False,
        )
    except FileNotFoundError:
        _reject("store_git_probe_unavailable")
    except subprocess.TimeoutExpired:
        _reject("store_git_probe_timeout")
    except (OSError, UnicodeError):
        _reject("store_git_probe_failed")
    if result.returncode == 0:
        if "true" in result.stdout.splitlines():
            _reject("store_inside_git")
        # Unexpected success (e.g. an administrative Git directory) is not
        # positive evidence of a safe non-repository location.
        _reject("store_git_probe_failed")
    # Apple Git may prepend a temporary-directory warning. Require exactly the
    # non-repository fatal diagnostic, rather than mistaking that warning for
    # a topology failure or accepting every exit 128 as a safe directory.
    fatal = [line for line in result.stderr.splitlines() if line.startswith("fatal:")]
    if result.returncode != 128 or len(fatal) != 1 or not fatal[0].startswith("fatal: not a git repository"):
        _reject("store_git_probe_failed")


def validate_store(raw, expected_identity=None, *, environ=None):
    """Return a canonical Path after read-only checks, or a sanitized error.

    This checks startup topology, not writability, free space, descendant user
    catalogs, dynamic topology races, remote availability or budget readiness.
    expected_identity binds an existing marker to the launcher's config hash.
    """
    env = os.environ if environ is None else environ
    try:
        if not isinstance(raw, str) or not raw or "\0" in raw:
            _reject("store_absolute_required")
        root = Path(raw).expanduser()
        if not root.is_absolute():
            _reject("store_absolute_required")
        for component in (root, *root.parents):
            if component.is_symlink():
                _reject("store_canonical_required")
        resolved = root.resolve()
        if root != resolved:
            _reject("store_canonical_required")
        daily = (Path.home() / ".codex" / "project-memory").resolve()
        if root == daily or root in daily.parents or daily in root.parents:
            _reject("store_daily_overlap")
        ancestor = root
        while True:
            try:
                mode = ancestor.stat().st_mode
                break
            except FileNotFoundError:
                ancestor = ancestor.parent
            except NotADirectoryError:
                _reject("store_not_directory")
        if not stat.S_ISDIR(mode):
            _reject("store_not_directory")
        _outside_git(ancestor, env)
        marker = root / MARKER
        if root.exists() and any(root.iterdir()):
            if marker.is_symlink() or not marker.is_file() or marker.stat().st_size > 4096:
                _reject("store_marker_conflict")
            try:
                identity = json.loads(marker.read_text(encoding="utf-8"))
            except (ValueError, UnicodeError):
                _reject("store_marker_conflict")
            valid = (isinstance(identity, dict) and set(identity) == {"profile", "config_sha256"}
                     and identity["profile"] == PROFILE
                     and isinstance(identity["config_sha256"], str)
                     and re.fullmatch(r"[0-9a-f]{64}", identity["config_sha256"]))
            if not valid or (expected_identity is not None and identity != expected_identity):
                _reject("store_marker_conflict")
        return root
    except StorePreflightError:
        raise
    except (OSError, RuntimeError, ValueError):
        _reject("store_io_error")


def inspect_store(raw, expected_identity=None, *, environ=None):
    try:
        validate_store(raw, expected_identity, environ=environ)
        issues = []
    except StorePreflightError as exc:
        issues = [exc.code]
    return {"passed": not issues, "issues": issues,
            "hints": [HINTS[code] for code in issues],
            "configuration_binding_checked": expected_identity is not None,
            "writability": "not_tested", "network_probe_performed": False,
            "scope": "read-only-startup-store-topology"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True, help="absolute dedicated non-Git store path")
    args = parser.parse_args(argv)
    report = inspect_store(args.store)
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
