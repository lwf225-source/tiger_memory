"""Bounded synthetic-demo audit guard; this is not an OS sandbox."""
import builtins
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import unquote, urlsplit


class DemoGuardError(RuntimeError):
    pass


def install_guard(root, runtime, child_scripts=(), deny_fixture_reads=False):
    root, runtime = Path(root).resolve(), Path(runtime).resolve()
    children = {str(Path(path).resolve()) for path in child_scripts}
    fixture_root = runtime / "examples" / "active_memory" / "fixtures"
    private = Path.home() / ".codex" / "project-memory"
    state = {"handler": False}

    def inside(path, boundary=root):
        if isinstance(path, int):
            return True
        return Path(os.fsdecode(path)).resolve(strict=False).is_relative_to(boundary)

    def audit(event, args):
        if state["handler"] and (event == "import" or event == "open" or event.startswith("socket.")
                or event in {"os.listdir", "os.scandir", "subprocess.Popen", "sqlite3.connect"}):
            raise DemoGuardError("handler I/O or import denied")
        if event.startswith("socket."):
            raise DemoGuardError("network denied")
        if event == "import" and str(args[0]).split(".")[0] in {
                "torch", "transformers", "sentence_transformers", "openai", "anthropic"}:
            raise DemoGuardError("provider import denied")
        if event == "open":
            path, mode, flags = args
            if not isinstance(path, int):
                if inside(path, private) or (deny_fixture_reads and inside(path, fixture_root)):
                    raise DemoGuardError("private or fixture read denied")
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
                isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)))
            if writing and not inside(path) and os.fsdecode(path) != os.devnull:
                raise DemoGuardError("outside write denied")
        if event == "sqlite3.connect":
            path = os.fsdecode(args[0])
            if path.startswith("file:"):
                uri = urlsplit(path)
                if (uri.netloc not in {"", "localhost"} or uri.fragment or not uri.path.startswith("/")
                        or uri.query not in {"", "mode=ro", "mode=rw", "mode=rwc"}
                        or any(ord(c) < 32 for c in path)
                        or any(c == "%" and (len(uri.path[i+1:i+3]) != 2
                            or any(d not in "0123456789abcdefABCDEF" for d in uri.path[i+1:i+3]))
                            for i, c in enumerate(uri.path))):
                    raise DemoGuardError("database URI denied")
                path = unquote(uri.path, encoding="utf-8", errors="strict")
            if "\x00" in path or not inside(path):
                raise DemoGuardError("outside database denied")
        if event in {"os.mkdir", "os.remove", "os.rmdir", "os.chmod", "os.utime", "os.truncate"}:
            if not inside(args[0]):
                raise DemoGuardError("outside mutation denied")
        if event in {"os.rename", "os.link", "os.symlink"}:
            paths = args[1:2] if event == "os.symlink" else args[:2]
            if not all(inside(path) for path in paths):
                raise DemoGuardError("outside link or rename denied")
        if event == "subprocess.Popen":
            _exe, argv, _cwd, _env = args
            if not isinstance(argv, (list, tuple)):
                raise DemoGuardError("shell denied")
            argv = [str(item) for item in argv]
            git = (len(argv) == 5 and Path(argv[0]).name == "git" and argv[1] == "-C"
                   and inside(argv[2]) and argv[3:] == ["rev-parse", "--show-toplevel"])
            child = (len(argv) >= 2 and Path(argv[0]).resolve() == Path(sys.executable).resolve()
                     and argv[1] in children)
            if not (git or child):
                raise DemoGuardError("subprocess denied")
    sys.addaudithook(audit)
    return state


@contextmanager
def handler_guard(state):
    original = builtins.__import__
    def reject_import(*args, **kwargs):
        raise DemoGuardError("handler import denied")
    state["handler"] = True
    builtins.__import__ = reject_import
    try:
        yield
    finally:
        builtins.__import__ = original
        state["handler"] = False
