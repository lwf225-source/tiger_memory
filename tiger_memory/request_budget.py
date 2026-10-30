"""Optional cooperative request deadline. Never a hard OS/remote cancellation.

No ambient budget means ordinary CLI behavior. A safe boundary permits only
the already admitted atomic publication/commit to finish; cancellation is still
recorded and the caller must check before admitting another unit of work.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import math
import threading
from time import monotonic

from .providers import ProviderUnavailable


class RequestCancelled(ProviderUnavailable):
    def __init__(self):
        super().__init__("request temporarily unavailable")


_CURRENT = ContextVar("aml_request_budget", default=None)
_FINISHING = ContextVar("aml_request_finishing", default=False)


class RequestBudget:
    def __init__(self, seconds, generation=0):
        if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                or not math.isfinite(seconds) or seconds <= 0):
            raise ValueError("invalid request duration")
        self.started = monotonic()
        self.deadline = self.started + seconds
        self.generation = generation
        self.phase = "read"
        self.reason = None
        self._lock = threading.Lock()
        self.cancelled = threading.Event()

    def cancel(self, reason):
        if reason not in {"deadline", "disconnect", "shutdown"}:
            raise ValueError("invalid cancellation reason")
        with self._lock:
            if self.reason is None:
                self.reason = reason
            self.cancelled.set()

    def check(self, phase="business"):
        self.phase = phase
        if _FINISHING.get():
            return
        if monotonic() >= self.deadline:
            self.cancel("deadline")
        if self.cancelled.is_set():
            raise RequestCancelled()

    def remaining(self, phase="business"):
        self.check(phase)
        return max(0.0, self.deadline - monotonic())


def current_request_budget():
    return _CURRENT.get()


def check(phase="business"):
    budget = current_request_budget()
    if budget is not None:
        budget.check(phase)


def timeout(cap, phase="wait"):
    budget = current_request_budget()
    if budget is None or _FINISHING.get():
        return cap
    return min(cap, budget.remaining(phase))


def finishing():
    return _FINISHING.get()


@contextmanager
def request_scope(budget):
    token = _CURRENT.set(budget)
    try:
        yield budget
    finally:
        _CURRENT.reset(token)


@contextmanager
def safe_boundary(phase="publish"):
    check(phase)
    token = _FINISHING.set(True)
    try:
        yield
    finally:
        _FINISHING.reset(token)


def wait(seconds, phase="wait"):
    budget = current_request_budget()
    if budget is None or finishing():
        from time import sleep
        sleep(seconds)
    else:
        budget.cancelled.wait(timeout(seconds, phase))
        budget.check(phase)
