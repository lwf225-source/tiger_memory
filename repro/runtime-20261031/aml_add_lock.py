"""Per-user admission for cooperating AML remote Add on local POSIX storage.

Lock order: admission -> service maintenance SH -> project EX -> SQLite.
Embedding starts after release. The lock is non-reentrant; service and
checkpoint code must never acquire it. Neither flock nor these path observations
fence administrators or non-cooperating old writers. Never unlink the lock file.
"""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import stat
from time import monotonic


class AddLockUnavailable(RuntimeError):
    def __init__(self):
        super().__init__("add admission unavailable")


@contextmanager
def add_admission_lock(root: Path):
    """Serialize source writes for one user without quota or expiry."""
    descriptor = None
    started = monotonic()
    try:
        root = Path(root)
        if not root.is_absolute() or root.resolve() != root or any(p.is_symlink() for p in (root, *root.parents)):
            raise AddLockUnavailable()
        try:
            flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
            descriptor = os.open(str(root / ".aml-add.lock"), flags, 0o600)
            os.set_inheritable(descriptor, False)
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
                raise AddLockUnavailable()
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            acquired = monotonic()
        except (OSError, ValueError):
            raise AddLockUnavailable() from None
        yield {"acquired": True, "wait_seconds": max(0.0, acquired - started),
               "started_monotonic": started, "acquired_monotonic": acquired}
    finally:
        if descriptor is not None:
            os.close(descriptor)
