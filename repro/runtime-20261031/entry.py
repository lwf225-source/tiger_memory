"""Portable entry for the source-verified 2026-10-31 service snapshot."""
from pathlib import Path
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent


def setup():
    # Verify every published source before importing the isolated runtime.
    manifest = json.loads((HERE / "MANIFEST.json").read_text())
    for name, expected in manifest["files"].items():
        path = HERE / name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError("release-source-drift")
    sys.path.insert(0, str(HERE / "base"))
    sys.path.insert(0, str(HERE))
    import runtime_loader
    aml, remote, config = runtime_loader.install()
    import signal
    signal.signal(signal.SIGUSR1, remote.enable_checkpoint_parallel)
    import tiger_memory
    if Path(tiger_memory.__file__).resolve().parent != HERE / "base" / "tiger_memory":
        raise RuntimeError("unexpected-runtime-package")
    import safe_ingest
    safe_ingest.install(aml)
    for operation in ("add", "search"):
        original = getattr(aml, operation)
        def normalized(request, _operation=operation, _original=original):
            return _original(aml.normalize_text_request(request, _operation))
        setattr(aml, operation, normalized)
    import safe_diagnostics
    safe_diagnostics.install(aml)
    return aml, remote, config


if __name__ == "__main__":
    _, _, config = setup()
    raise SystemExit(config.main())
