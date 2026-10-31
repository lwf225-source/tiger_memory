"""Synthetic regression against the bundled runtime; no provider requests."""
from pathlib import Path
import json, os, sys, unittest
P = Path(__file__).resolve().parent
sys.path[:0] = [str(P), str(P / "base")]
import runtime_loader
runtime_loader.install()
runtime_loader.load("tiger_memory.safe_diagnostics", "safe_diagnostics.py")
from tiger_memory.remote_embeddings import enable_checkpoint_parallel
enable_checkpoint_parallel()
names = ["test_embedding_parallel", "test_transport_diagnostics", "test_source_capacity", "test_recovery_filter", "test_lock_optimization", "test_incremental_recovery", "test_stage_timings", "test_sqlite_optimization", "test_performance", "tests.test_expected_identity", "tests.test_source_recovery", "tests.test_aml_adapter", "tests.test_aml_http_body", "tests.test_aml_store_preflight"]
if len(sys.argv) > 1:
    names = sys.argv[1:]
result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(names))
print(json.dumps({"tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped), "passed": result.wasSuccessful()}))
raise SystemExit(not result.wasSuccessful())
