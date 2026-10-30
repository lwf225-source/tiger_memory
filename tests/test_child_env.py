"""Installation-specific child lookup and artifact-binding regression tests."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tests import child_env as launcher


class ChildEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='plm-child-env-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def test_explicit_paths_replace_polluted_parent_and_disable_user_site(self):
        with patch.dict(os.environ, {'PYTHONPATH': str(self.root / 'host-decoy')}):
            bound = launcher.artifact_binding()
            env = launcher.child_env({'PATH': os.defpath, 'HOME': str(self.root),
                'PYTHONPATH': '/wrong', 'PYTHONUSERBASE': '/wrong', 'PYTHONHOME': '/wrong'})
        self.assertEqual(env['PYTHONPATH'].split(os.pathsep), list(dict.fromkeys(
            [bound['runtime_parent'], bound['test_root']])))
        self.assertEqual(env['PYTHONNOUSERSITE'], '1')
        self.assertNotIn('PYTHONUSERBASE', env)
        self.assertNotIn('PYTHONHOME', env)

    def test_actual_worker_ignores_cwd_decoy_and_writes_separate_provenance(self):
        decoy = self.root / 'tiger_memory'
        decoy.mkdir()
        marker = self.root / 'wrong-import'
        (decoy / '__init__.py').write_text('from pathlib import Path\nPath(' + repr(str(marker)) + ').touch()\n')
        with patch.dict(os.environ, {'PLM_TEST_CHILD_RECORD_DIR': str(self.root / 'records')}):
            child = launcher.child_popen([sys.executable, '-B', '-c',
                'import tiger_memory.service'], cwd=self.root,
                env={'HOME': str(self.root), 'PATH': os.defpath, 'PYTHONPATH': str(self.root)},
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = child.communicate(timeout=15)
        self.assertEqual((child.returncode, stdout, stderr), (0, b'', b''))
        self.assertFalse(marker.exists())
        record = json.loads(child.provenance_path.read_text())
        self.assertTrue(record['validated'])
        self.assertIn('tiger_memory/service.py', record['loaded'])
        self.assertEqual(record['runtime_parent'], launcher.artifact_binding()['runtime_parent'])

    def test_external_artifact_binding_rejects_wrong_parent_root(self):
        binding = launcher.artifact_binding()
        binding['runtime_parent'] = str(self.root / 'wrong')
        contract = self.root / 'expected.json'
        contract.write_text(json.dumps(binding))
        with patch.dict(os.environ, {'PLM_TEST_ARTIFACT_BINDING': str(contract)}):
            with self.assertRaisesRegex(RuntimeError, 'binding mismatch'):
                launcher.artifact_binding()

    def _damaged_child(self, mode):
        original = launcher.artifact_binding()
        runtime = self.root / 'lib'
        tests = self.root / 'bundle'
        shutil.copytree(Path(original['runtime_parent']) / 'tiger_memory', runtime / 'tiger_memory',
                        ignore=shutil.ignore_patterns('_bundle', '__pycache__'))
        (tests / 'tests').mkdir(parents=True)
        (tests / 'tests/__init__.py').write_text('')
        binding = {'runtime_parent': str(runtime), 'test_root': str(tests),
                   'runtime_hashes': launcher._hashes(runtime / 'tiger_memory'),
                   'test_hashes': launcher._hashes(tests / 'tests')}
        victim = runtime / 'tiger_memory/aml_add_lock.py'
        if mode == 'missing': victim.unlink()
        else: victim.write_text(victim.read_text() + '\n# changed artifact\n')
        marker = self.root / 'executed'
        contract = self.root / 'launch.json'
        contract.write_text(json.dumps({'binding': binding, 'argv': ['-c',
            'from pathlib import Path; Path(' + repr(str(marker)) + ').touch()'],
            'record': str(self.root / 'record.json')}))
        process = subprocess.run([sys.executable, '-B', str(Path(launcher.__file__).resolve()),
            '--child', str(contract)], cwd=self.root,
            env={'HOME': str(self.root), 'PATH': os.defpath, 'PYTHONNOUSERSITE': '1'},
            capture_output=True, timeout=15)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn(b'runtime artifact missing or changed', process.stderr)
        self.assertFalse(marker.exists())

    def test_missing_runtime_module_fails_before_worker(self):
        self._damaged_child('missing')

    def test_changed_runtime_bytes_fail_before_worker(self):
        self._damaged_child('changed')
