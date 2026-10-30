"""Strict, checksummed numeric checkpoints for remote embedding batches.

Chunks are stored raw, before pooling or normalization. A failed write is an
error, never a successful but secretly non-durable embedding. No source text or
credentials are stored. One user namespace lock covers lookup through publish.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from time import monotonic, sleep

from .providers import ProviderUnavailable
from . import request_budget as request_limits
from .security import ensure_private_dir


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


class VectorCheckpoint:
    def __init__(self, root, fingerprint, namespace, dimensions):
        root = Path(root)
        if root.is_symlink() or any(p.is_symlink() for p in root.parents):
            raise ProviderUnavailable("embedding-checkpoint-unsafe-path")
        self.root = root.resolve() / "checkpoints-v1" / digest(namespace) / fingerprint
        self.context = {"schema": 1, "fingerprint": fingerprint, "namespace": digest(namespace),
                        "dimensions": dimensions}
        self.dimensions = dimensions

    def _safe(self):
        if any(p.is_symlink() for p in (self.root, *self.root.parents)):
            raise ProviderUnavailable("embedding-checkpoint-unsafe-path")

    @contextmanager
    def locked(self, timeout=5.0):
        fd = None
        end = monotonic() + max(0.0, timeout)
        try:
            request_limits.check("embedding-checkpoint-open")
            self._safe()
            ensure_private_dir(self.root)
            fd = os.open(str(self.root / ".lock"), os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            while True:
                request_limits.check("embedding-checkpoint-lock")
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if monotonic() >= end:
                        raise ProviderUnavailable("embedding-checkpoint-busy") from None
                    request_limits.wait(min(.02, max(0., end - monotonic())), "embedding-checkpoint-lock")
            request_limits.check("embedding-checkpoint-acquired")
            if timeout > 0 and monotonic() >= end:
                raise ProviderUnavailable("embedding-checkpoint-busy")
            self._safe()
            yield
        except ProviderUnavailable:
            raise
        except (OSError, ValueError, RuntimeError):
            raise ProviderUnavailable("embedding-checkpoint-unavailable") from None
        finally:
            if fd is not None:
                os.close(fd)

    def _path(self, kind, key):
        if kind not in {"raw", "pooled"} or len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ProviderUnavailable("embedding-checkpoint-key")
        return self.root / (kind + "-" + key + ".json")

    def get(self, kind, key):
        path = self._path(kind, key)
        try:
            self._safe()
            if path.is_symlink():
                raise ValueError()
            if not path.exists():
                return None
            if not path.is_file() or path.stat().st_nlink != 1 or path.stat().st_size > 256 * 1024:
                raise ValueError()
            obj = json.loads(path.read_bytes())
            checksum = obj.pop("checksum")
            expected = {**self.context, "kind": kind, "input_sha256": key, "values": obj["values"]}
            if obj != expected or checksum != digest(obj):
                raise ValueError()
            values = obj["values"]
            if (not isinstance(values, list) or len(values) != self.dimensions
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)):
                raise ValueError()
            if kind == "pooled" and not any(values):
                raise ValueError()
            return [float(v) for v in values]
        except ProviderUnavailable:
            raise
        except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError):
            # No transparent re-send on corrupted data. Operator review can
            # restore validated data later. Missing claimed data cannot be re-sent.
            raise ProviderUnavailable("embedding-checkpoint-corrupt") from None

    def put(self, kind, key, values):
        path = self._path(kind, key)
        temp = None
        try:
            self._safe()
            values = [float(v) for v in values]
            if len(values) != self.dimensions or not all(math.isfinite(v) for v in values):
                raise ValueError()
            old = self.get(kind, key)
            if old is not None:
                if [v.hex() for v in old] != [v.hex() for v in values]:
                    raise ProviderUnavailable("embedding-checkpoint-conflict")
                return
            obj = {**self.context, "kind": kind, "input_sha256": key, "values": values}
            obj["checksum"] = digest(obj)
            fd, temp = tempfile.mkstemp(prefix=".pending-", dir=str(self.root))
            with os.fdopen(fd, "w") as stream:
                os.fchmod(stream.fileno(), 0o600)
                json.dump(obj, stream, sort_keys=True, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, path)
            fd = os.open(str(self.root), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            if [v.hex() for v in self.get(kind, key)] != [v.hex() for v in values]:
                raise ValueError()
        except ProviderUnavailable:
            raise
        except (OSError, ValueError, TypeError, RuntimeError):
            raise ProviderUnavailable("embedding-checkpoint-write-failed") from None
        finally:
            if temp is not None:
                try:
                    Path(temp).unlink(missing_ok=True)
                except OSError:
                    pass
