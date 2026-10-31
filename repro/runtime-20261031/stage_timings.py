"""Bounded, process-local, privacy-safe inclusive stage timings.

Only fixed code-owned stage names are accepted; no paths, identities, payloads,
SQL, exceptions or credentials are retained. Counters reset at process start.
Nested stages overlap and MUST NOT be summed as a request's wall-clock time.
Histogram buckets are disjoint upper bounds in milliseconds, last is infinity.
Timing is diagnostic only: instrumentation failures never alter business flow.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import threading
import time

STAGES = (
    'add_total', 'search_total', 'add_admission_wait', 'add_records',
    'maintenance_lock_wait', 'project_lock_wait', 'source_recovery',
    'source_publish', 'source_file_fsync', 'source_directory_fsync',
    'write_event', 'write_transaction', 'index_write', 'db_connect',
    'db_begin_wait', 'db_commit', 'db_busy_backoff',
    'embedding_total', 'embedding_checkpoint_wait', 'embedding_prepare',
    'embedding_batch', 'embedding_http', 'embedding_backoff',
    'embedding_cache_publish', 'lexical_prepare', 'write_lock_held',
    'identity_validation', 'embedding_mode_wait', 'embedding_input_wait', 'embedding_input_held',
)
OPERATION_STAGES = ('embedding_mode_wait', 'embedding_total', 'embedding_checkpoint_wait', 'embedding_input_wait',
                    'embedding_input_held', 'embedding_prepare', 'embedding_http',
                    'embedding_cache_publish', 'source_recovery', 'write_lock_held')
STAGES += tuple(op + '_' + name for op in ('add', 'search') for name in OPERATION_STAGES)
_OPERATION = ContextVar('fixed_stage_operation', default=None)
BOUNDS_MS = (1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 30000, 60000)
_LOCK = threading.Lock()
_DATA = {name: {'count': 0, 'errors': 0, 'active': 0, 'total_ns': 0,
                'max_ns': 0, 'buckets': [0] * (len(BOUNDS_MS) + 1)}
         for name in STAGES}


class phase:
    """Inclusive elapsed interval; exceptions from the business body propagate."""
    def __init__(self, name):
        self.name = name
        self.started = None
        self.active = False
        op = _OPERATION.get()
        self.scoped = phase(op + '_' + name) if op in ('add', 'search') and name in OPERATION_STAGES else None

    def __enter__(self):
        if self.scoped is not None:
            self.scoped.__enter__()
        try:
            if self.name not in _DATA:
                return self
            self.started = time.perf_counter_ns()
            with _LOCK:
                _DATA[self.name]['active'] += 1
                self.active = True
        except Exception:
            self.started = None
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.scoped is not None:
            self.scoped.__exit__(exc_type, exc_value, traceback)
        try:
            try:
                ended = time.perf_counter_ns()
            except Exception:
                ended = None
            with _LOCK:
                if self.active:
                    _DATA[self.name]['active'] -= 1
                    self.active = False
                if self.started is None or ended is None:
                    return False
                elapsed = max(0, ended - self.started)
                row = _DATA[self.name]
                row['count'] += 1
                row['errors'] += int(exc_type is not None)
                row['total_ns'] += elapsed
                row['max_ns'] = max(row['max_ns'], elapsed)
                index = next((i for i, bound in enumerate(BOUNDS_MS)
                              if elapsed <= bound * 1000000), len(BOUNDS_MS))
                row['buckets'][index] += 1
        except Exception:
            # In particular do not replace an exception already unwinding.
            pass
        return False


def timed(name):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            token = _OPERATION.set(name[:-6]) if name in ('add_total', 'search_total') else None
            try:
                with phase(name):
                    return function(*args, **kwargs)
            finally:
                if token is not None:
                    _OPERATION.reset(token)
        return wrapped
    return decorate


@contextmanager
def timed_acquire(context, name):
    """Measure only entry/wait, retaining the original context's exit semantics."""
    import sys
    with phase(name):
        value = context.__enter__()
    try:
        yield value
    except BaseException:
        if not context.__exit__(*sys.exc_info()):
            raise
    else:
        context.__exit__(None, None, None)


def timing_status():
    try:
        with _LOCK:
            stages = {
                name: {'count': r['count'], 'errors': r['errors'],
                       'active': r['active'],
                       'total_ms': round(r['total_ns'] / 1000000, 3),
                       'max_ms': round(r['max_ns'] / 1000000, 3),
                       'histogram': list(r['buckets'])}
                for name, r in _DATA.items()
            }
        return {'schema_version': 1, 'scope': 'process',
                'semantics': 'inclusive-nested-not-additive',
                'bucket_upper_ms': list(BOUNDS_MS) + [None], 'stages': stages}
    except Exception:
        return {'schema_version': 1, 'available': False}
