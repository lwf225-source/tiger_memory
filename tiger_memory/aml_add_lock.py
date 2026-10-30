"""Per-user admission for cooperating AML remote Add on local POSIX storage.

Lock order: admission -> service maintenance SH -> project EX -> SQLite.
Embedding starts after release. The lock is non-reentrant; service, budget and
checkpoint code must never acquire it. Neither flock nor these path observations
fence administrators or non-cooperating old writers. Never unlink the lock file.
"""
from contextlib import contextmanager
from . import request_budget
import fcntl
import math
import os
from pathlib import Path
import stat
from time import monotonic, sleep


class AddBusy(RuntimeError):
    """Local admission wait expired; no financial reservation is implied."""
    def __init__(self, started, finished):
        super().__init__("add temporarily busy")
        self.elapsed_seconds = max(0.0, finished - started)


class AddLockUnavailable(RuntimeError):
    def __init__(self):
        super().__init__("add admission unavailable")


@contextmanager
def add_admission_lock(root: Path, timeout: float = 5.0):
    """Bound waiting, not filesystem calls or the protected source-write phase.

    Zero means one immediate nonblocking attempt. A positive timeout is checked
    again after acquisition before protected work, including late scheduling.
    Each invocation independently opens a private regular file and closes its
    non-inheritable descriptor. O_NONBLOCK avoids FIFO-open waiting; it does not
    make arbitrary filesystem operations bounded.
    """
    try:
        valid_timeout = (not isinstance(timeout, bool) and isinstance(timeout, (int, float))
                         and math.isfinite(timeout) and timeout >= 0)
    except OverflowError:
        valid_timeout = False
    if not valid_timeout:
        raise AddLockUnavailable()
    descriptor = None
    request_budget.check("admission-open")
    started = monotonic()
    deadline = started + timeout
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
            while True:
                request_budget.check("admission-wait")
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    now = monotonic()
                    if timeout == 0 or now >= deadline:
                        raise AddBusy(started, now) from None
                    request_budget.wait(min(.02, max(0.0, deadline - now)), "admission-wait")
            request_budget.check("admission-acquired")
            acquired = monotonic()
            if timeout > 0 and acquired >= deadline:
                raise AddBusy(started, acquired)
        except (OSError, ValueError):
            raise AddLockUnavailable() from None
        yield {"acquired": True, "wait_seconds": max(0.0, acquired - started),
               "started_monotonic": started, "acquired_monotonic": acquired}
    finally:
        if descriptor is not None:
            os.close(descriptor)
