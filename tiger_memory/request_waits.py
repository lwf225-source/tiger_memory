"""Cooperative local waits. Kernel stalls are not made interruptible here."""
import fcntl
import os
import signal
import sqlite3
import subprocess

from . import request_budget as rb


def flock(fd, operation):
    if rb.current_request_budget() is None:
        return fcntl.flock(fd, operation)
    while True:
        rb.check("file-lock")
        try:
            fcntl.flock(fd, operation | fcntl.LOCK_NB)
            rb.check("file-lock-acquired")
            return
        except BlockingIOError:
            rb.wait(.02, "file-lock")


def git_root(cwd):
    """One controlled process group; expired Git never becomes identity fallback."""
    rb.check("git")
    child = subprocess.Popen(["git", "-C", str(cwd), "rev-parse", "--show-toplevel"],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             text=True, start_new_session=True)
    completed = False
    try:
        while True:
            try:
                stdout, _ = child.communicate(timeout=rb.timeout(.05, "git"))
                rb.check("git-complete")
                completed = True
                if child.returncode:
                    raise subprocess.CalledProcessError(child.returncode, child.args)
                return stdout
            except subprocess.TimeoutExpired:
                rb.check("git")
    finally:
        # Kill descendants even if the direct child exited but retained pipes
        # are still open in its process group. No arbitrary escaped-child claim.
        if not completed:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.communicate()


class DeadlineConnection(sqlite3.Connection):
    """Catalog only. Not the financial ledger's reserve/settle connection.

    Recompute busy wait before each statement; progress interrupts long SQL.
    An admitted source publish/commit disables interruption until its safe
    boundary completes. No budget preserves the standard Connection path.
    """
    def _before(self):
        rb.check("catalog-sql")
        milliseconds = max(0, int(rb.timeout(30.0, "catalog-busy") * 1000))
        super().execute("PRAGMA busy_timeout=%d" % milliseconds)
        self.set_progress_handler(self._progress, 1000)

    def _progress(self):
        try:
            rb.check("catalog-progress")
            return 0
        except rb.RequestCancelled:
            return 1

    def _call(self, method, *args, **kwargs):
        self._before()
        try:
            return method(*args, **kwargs)
        except sqlite3.OperationalError:
            rb.check("catalog-sql")
            raise

    def execute(self, *args, **kwargs):
        return self._call(super().execute, *args, **kwargs)

    def executemany(self, *args, **kwargs):
        return self._call(super().executemany, *args, **kwargs)

    def executescript(self, *args, **kwargs):
        return self._call(super().executescript, *args, **kwargs)

    def commit(self):
        return self._call(super().commit)

    def rollback(self):
        # Rollback remains possible after cancellation/progress interruption.
        self.set_progress_handler(None, 0)
        return super().rollback()
