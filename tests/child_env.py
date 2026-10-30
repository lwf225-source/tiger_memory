"""Controlled Python children for source, skill/lib, and installed test layouts.

Only test infrastructure: no inherited PYTHONPATH or user site. Each real worker
validates its bound bytes and records provenance separately from stdout/stderr.
An acceptance runner can supply PLM_TEST_ARTIFACT_BINDING (JSON file containing
runtime_parent, test_root, runtime_hashes and test_hashes) for an external bind.
"""
from pathlib import Path
import atexit
import hashlib
import json
import os
import runpy
import subprocess
import sys
import tempfile
import uuid


def _hashes(folder):
    return {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob('*.py'))
            if '__pycache__' not in p.parts and '_bundle' not in p.parts}


def artifact_binding():
    import tiger_memory
    runtime = Path(tiger_memory.__file__).resolve().parent.parent
    tests = Path(__file__).resolve().parent.parent
    binding = {'runtime_parent': str(runtime), 'test_root': str(tests),
               'runtime_hashes': _hashes(runtime / 'tiger_memory'),
               'test_hashes': _hashes(tests / 'tests')}
    expected = os.environ.get('PLM_TEST_ARTIFACT_BINDING')
    if expected and json.loads(Path(expected).read_text()) != binding:
        raise RuntimeError('test artifact binding mismatch')
    return binding


def child_env(environment, binding=None):
    """Preserve explicit business env, replace interpreter lookup configuration."""
    binding = binding or artifact_binding()
    env = dict(environment)
    env.pop('PYTHONHOME', None)
    env.pop('PYTHONUSERBASE', None)
    paths = list(dict.fromkeys([binding['runtime_parent'], binding['test_root']]))
    env.update(PYTHONPATH=os.pathsep.join(paths), PYTHONNOUSERSITE='1',
               PYTHONDONTWRITEBYTECODE='1')
    env.setdefault('GIT_CONFIG_NOSYSTEM', '1')
    env.setdefault('GIT_CONFIG_GLOBAL', os.devnull)
    return env


def child_popen(argv, *, env, **kwargs):
    """Popen-compatible worker handle; only existing -B -c/-m calls are accepted."""
    argv = [str(part) for part in argv]
    if argv[:2] != [sys.executable, '-B'] or len(argv) < 4 or argv[2] not in ('-c', '-m'):
        raise ValueError('expected active interpreter with -B and -c or -m')
    binding = artifact_binding()
    record_root = os.environ.get('PLM_TEST_CHILD_RECORD_DIR')
    if record_root:
        directory = Path(record_root).resolve()
        directory.mkdir(parents=True, exist_ok=True)
    else:
        temporary = tempfile.TemporaryDirectory(prefix='plm-child-provenance-')
        atexit.register(temporary.cleanup)
        directory = Path(temporary.name).resolve()
    token = uuid.uuid4().hex
    contract = directory / (token + '.launch.json')
    record = directory / (token + '.provenance.json')
    payload = {'binding': binding, 'argv': argv[2:], 'record': str(record)}
    contract.write_text(json.dumps(payload), encoding='utf-8')
    contract.chmod(0o600)
    process = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()),
                                '--child', str(contract)], env=child_env(env, binding), **kwargs)
    process.provenance_path = record
    return process


def _launch(contract_path):
    payload = json.loads(Path(contract_path).read_text())
    binding = payload['binding']
    runtime = Path(binding['runtime_parent']).resolve()
    tests = Path(binding['test_root']).resolve()
    if _hashes(runtime / 'tiger_memory') != binding['runtime_hashes'] or not binding['runtime_hashes']:
        raise RuntimeError('runtime artifact missing or changed')
    if _hashes(tests / 'tests') != binding['test_hashes'] or not binding['test_hashes']:
        raise RuntimeError('test artifact missing or changed')
    if runtime != tests and (tests / 'tiger_memory').exists():
        raise RuntimeError('unexpected runtime shadow in test artifact')
    # Script execution avoids cwd as implicit sys.path[0]. Keep only explicit
    # artifact roots before standard interpreter paths; no workspace fallback.
    roots = list(dict.fromkeys([str(runtime), str(tests)]))
    script_dir = str(Path(__file__).resolve().parent)
    cwd = str(Path.cwd().resolve())
    sys.path[:] = roots + [p for p in sys.path if p and p not in roots and p not in (script_dir, cwd)]
    record_path = Path(payload['record'])
    record = {'pid': os.getpid(), 'executable': sys.executable, 'cwd': os.getcwd(),
              'runtime_parent': str(runtime), 'test_root': str(tests),
              'runtime_hashes': binding['runtime_hashes'],
              'test_hashes': binding['test_hashes'], 'loaded': {},
              'worker': {'kind': payload['argv'][0],
                         'sha256' if payload['argv'][0] == '-c' else 'module':
                         hashlib.sha256(payload['argv'][1].encode()).hexdigest()
                         if payload['argv'][0] == '-c' else payload['argv'][1]},
              'validated': True}

    def save():
        tmp = record_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(record, sort_keys=True), encoding='utf-8')
        tmp.chmod(0o600)
        os.replace(tmp, record_path)

    def audit(event, args):
        if event != 'exec' or args[0].co_name != '<module>':
            return
        filename = args[0].co_filename
        if filename.startswith('<'):
            return
        path = Path(filename).resolve()
        for root, prefix, hashes in ((runtime / 'tiger_memory', 'tiger_memory', binding['runtime_hashes']),
                                     (tests / 'tests', 'tests', binding['test_hashes'])):
            try:
                rel = path.relative_to(root).as_posix()
            except ValueError:
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if hashes.get(rel) != digest:
                raise RuntimeError('loaded artifact hash mismatch')
            record['loaded'][prefix + '/' + rel] = {'origin': str(path), 'sha256': digest}
            save()
            break

    save()
    sys.addaudithook(audit)
    import tiger_memory
    if Path(tiger_memory.__file__).resolve() != runtime / 'tiger_memory/__init__.py':
        raise RuntimeError('runtime import origin mismatch')
    import tests as test_package
    if Path(test_package.__file__).resolve() != tests / 'tests/__init__.py':
        raise RuntimeError('test import origin mismatch')
    mode, worker, *arguments = payload['argv']
    if mode == '-c':
        sys.argv = ['-c', *arguments]
        exec(compile(worker, '<string>', 'exec'), {'__name__': '__main__', '__builtins__': __builtins__})
    else:
        sys.argv = [worker, *arguments]
        runpy.run_module(worker, run_name='__main__', alter_sys=True)


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] != '--child':
        raise SystemExit('test worker launcher requires a contract')
    _launch(sys.argv[2])
