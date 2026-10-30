"""Opt-in HTTP lifecycle for synthetic deadline validation.

The watchdog owns a socket object and one request generation, never a numeric
fd. It shuts down blocked I/O, but business cancellation is cooperative only.
The handler retires and joins it before any next keepalive generation begins.
"""
import socket
import threading
from time import monotonic

from .request_budget import RequestBudget, RequestCancelled, request_scope


class SocketWatchdog:
    def __init__(self, connection, budget):
        self.connection = connection
        self.budget = budget
        self.stage = "read"
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.retired = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def set_stage(self, stage):
        with self.lock:
            self.budget.check(stage)
            self.stage = stage

    def _run(self):
        if self.done.wait(max(0, self.budget.deadline - monotonic())):
            return
        with self.lock:
            if self.retired:
                return
            self.budget.cancel("deadline")
            if self.stage in {"read", "write"}:
                try:
                    self.connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    def close(self):
        with self.lock:
            self.retired = True
            self.done.set()
        # Never join while holding the lock needed by the watchdog.
        self.thread.join()


class RequestLifecycle:
    def handle_one_request(self):
        seconds = getattr(self.server, "request_timeout", None)
        if seconds is None:
            return super().handle_one_request()
        self._generation = getattr(self, "_generation", 0) + 1
        budget = RequestBudget(seconds, self._generation)
        self._request_budget = budget
        self._response_started = False
        self._business_completed = False
        self._response_delivered = False
        watchdog = SocketWatchdog(self.connection, budget)
        self._watchdog = watchdog
        watchdog.start()
        try:
            with request_scope(budget):
                try:
                    super().handle_one_request()
                except RequestCancelled:
                    self.close_connection = True
                    # No new write grace period after D. Business expiry may
                    # deliver one immediate sanitized 503; partial output closes.
                    if watchdog.stage == "business" and not self._response_started:
                        self._response_started = True
                        body = b'{"detail":{"reason":"request temporarily unavailable"}}'
                        response = (b"HTTP/1.1 503 Service Unavailable\r\nConnection: close\r\n"
                                    b"Content-Type: application/json\r\nContent-Length: "
                                    + str(len(body)).encode() + b"\r\n\r\n" + body)
                        try:
                            self.connection.setblocking(False)
                            self.connection.send(response)
                        except OSError:
                            pass
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError):
                    budget.cancel("disconnect")
                    self.close_connection = True
        finally:
            watchdog.close()
            if budget.cancelled.is_set():
                self.close_connection = True
            observer = getattr(self.server, "request_observer", None)
            if observer is not None:
                observer({"generation": budget.generation, "started": budget.started,
                          "deadline": budget.deadline, "finished": monotonic(),
                          "reason": budget.reason, "stage": watchdog.stage,
                          "business_completed": self._business_completed,
                          "response_delivered": self._response_delivered,
                          "watchdog_alive": watchdog.thread.is_alive()})
            self._watchdog = None

    def _io_stage(self, phase):
        watchdog = getattr(self, "_watchdog", None)
        if watchdog is not None:
            watchdog.set_stage(phase)

    def end_headers(self):
        self._io_stage("write")
        self._response_started = True
        return super().end_headers()
