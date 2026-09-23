"""Regression checks for the public delivery contract (no live user store)."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from tiger_memory import cli, ops


class ReleaseDeliveryTests(unittest.TestCase):
    def test_verify_exit_status_matches_report(self):
        for passed, expected in ((False, 1), (True, 0)):
            with self.subTest(passed=passed), mock.patch.object(ops, 'doctor'), \
                    mock.patch.object(cli, 'verify_acceptance', return_value={'passed': passed}), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(cli.main(['verify']), expected)
                self.assertEqual(json.loads(output.getvalue())['passed'], passed)

    def test_empty_test_directory_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'tests').mkdir()
            self.assertFalse(ops.run_bundle_tests(root)['passed'])
            # A test-named file containing no test cases must fail too.
            (root / 'tests/test_empty.py').write_text('pass\n')
            result = ops.run_bundle_tests(root)
            self.assertFalse(result['passed'])
            self.assertEqual(result['tests_run'], 0)

    def test_verifier_uses_active_interpreter_and_isolated_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'tests').mkdir()
            (root / 'tests/test_ok.py').write_text('pass\n')
            result = subprocess.CompletedProcess([], 0, '', 'Ran 1 test in 0.001s\n\nOK\n')
            with mock.patch.dict('os.environ', {'OPENAI_API_KEY': 'test-only-sentinel'}), \
                    mock.patch.object(ops.subprocess, 'run', return_value=result) as run:
                self.assertTrue(ops.run_bundle_tests(root)['passed'])
                self.assertEqual(run.call_args[0][0][0], sys.executable)
                environment = run.call_args.kwargs['env']
                self.assertNotIn('OPENAI_API_KEY', environment)
                self.assertTrue(environment['TIGER_MEMORY_DIR'].startswith(environment['HOME']))

    def test_wheel_fingerprint_tracks_imported_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / 'tiger_memory'
            root = package / '_bundle'
            (root / 'skill').mkdir(parents=True)
            (root / 'skill/SKILL.md').write_text('public skill')
            (package / 'ops.py').write_text('first')
            with mock.patch.object(ops, '__file__', str(package / 'ops.py')):
                self.assertEqual(ops.repository_root(), root.resolve())
                before = ops.code_fingerprint()
                (package / 'ops.py').write_text('second')
                self.assertNotEqual(ops.code_fingerprint(), before)


if __name__ == '__main__':
    unittest.main()
