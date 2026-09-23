#!/usr/bin/env python3
"""Offline release regression: real wheel, sdist, verify and skill installs.

Only temporary stores/skills are used. No API credentials or models are needed.
Use --output DIRECTORY to retain command evidence (synthetic data only).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tiger_memory import build_backend


def verify(output=None):
    records = []
    checks = []
    started = time.monotonic()
    if output:
        output = Path(output).resolve()
        output.mkdir(parents=True, exist_ok=True)

    def save():
        if output:
            (output / 'release-commands.json').write_text(json.dumps(records, indent=2), encoding='utf-8')

    with tempfile.TemporaryDirectory(prefix='plm-release-check-') as directory:
        base = Path(directory).resolve()
        work = base / 'unrelated-project'
        work.mkdir()
        (work / 'pyproject.toml').write_text('[project]\nname="fixture"\nversion="0"\n')
        env = {key: os.environ[key] for key in ('PATH', 'LANG', 'SYSTEMROOT', 'TMPDIR') if key in os.environ}
        env.update({'HOME': str(base / 'home'), 'USERPROFILE': str(base / 'home'),
                    'TIGER_MEMORY_DIR': str(base / 'memory'),
                    'PLM_SKILL_BACKUP_DIR': str(base / 'backups'),
                    'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1',
                    'PIP_CONFIG_FILE': os.devnull, 'PIP_NO_INDEX': '1',
                    'PIP_DISABLE_PIP_VERSION_CHECK': '1'})

        def run(label, args, expected=0, cwd=work):
            result = subprocess.run([str(a) for a in args], cwd=cwd, env=env,
                                    capture_output=True, text=True, timeout=300)
            records.append({'label': label, 'argv': [str(a) for a in args],
                            'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
            save()
            if result.returncode != expected:
                raise AssertionError(label + ': ' + str(records[-1]))
            print(label + ': completed', flush=True)
            return result.stdout

        wheel = base / build_backend.build_wheel(base)
        archive = base / build_backend.build_sdist(base)
        with zipfile.ZipFile(wheel) as bundle:
            assert not any('memory_eval.py' in n or '__pycache__' in n for n in bundle.namelist())
            assert 'tiger_memory/_bundle/tests/test_release_delivery.py' in bundle.namelist()
        extracted = base / 'extracted'
        with tarfile.open(archive) as bundle:
            for member in bundle:
                path = extracted / member.name
                assert member.isfile() and path.resolve().is_relative_to(extracted.resolve())
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(bundle.extractfile(member).read())
        source = extracted / build_backend._identity()
        rebuilt = base / 'rebuilt'
        run('sdist-rebuild-wheel', [sys.executable, '-c',
            'from tiger_memory.build_backend import build_wheel; import sys; build_wheel(sys.argv[1])', rebuilt], cwd=source)
        assert (rebuilt / wheel.name).read_bytes() == wheel.read_bytes()
        checks.append('sdist-rebuilds-byte-identical-wheel')
        run('create-venv', [sys.executable, '-m', 'venv', base / 'venv'])
        binary = base / 'venv' / ('Scripts' if os.name == 'nt' else 'bin')
        python = binary / ('python.exe' if os.name == 'nt' else 'python')
        plm = binary / ('tiger_memory.exe' if os.name == 'nt' else 'tiger_memory')
        env['PATH'] = str(binary) + os.pathsep + env.get('PATH', '')
        run('install-wheel', [python, '-m', 'pip', 'install', '--no-index', '--no-cache-dir', wheel])
        checks.append('offline-wheel-install')
        fresh = json.loads(run('wheel-verify-empty', [plm, 'verify'], expected=1))
        assert fresh['tests']['passed'] and fresh['tests']['tests_run'] > 400
        assert fresh['passed'] is False and fresh['retrieval']['status'] == 'not_measured'
        checks.append('wheel-verify-runs-real-tests-and-fails-unmeasured-migration')
        # One explicit legacy fixture exercises the success exit status. This is
        # a migration smoke check, never a retrieval-quality benchmark.
        notes = base / 'memory/projects/fixture/notes'
        notes.mkdir(parents=True)
        (notes / 'note.md').write_text('---\ntitle: zircon archive policy\ntime: 2026-01-01 08:00:00 +0800\nproject_root: ' + str(work) + '\ntags: fixture\n---\n\nzircon archive policy uses violet boxes.\n')
        run('migrate-fixture', [plm, 'migrate'])
        migrated = json.loads(run('wheel-verify-migrated', [plm, 'verify']))
        assert migrated['passed'] and migrated['retrieval']['queries'] == 1
        checks.append('wheel-verify-migrated-success-exit')
        for kind, command in (('wheel', [plm]), ('sdist-source', [python, source / 'scripts/tiger_memory.py'])):
            target = base / ('skill-' + kind)
            result = json.loads(run(kind + '-skill-install', command + ['install', '--target', target]))
            assert result['target'] == str(target)
            assert (target / 'tests/test_release_delivery.py').is_file()
            assert (target / 'references/task-workflow.md').is_file()
            assert not (target / 'scripts/memory_eval.py').exists()
            run(kind + '-skill-doctor', [python, target / 'scripts/tiger_memory.py', 'doctor'])
            checks.append(kind + '-skill-install-with-validation')
        assert not (base / 'home/.codex/skills').exists()
        checks.append('no-global-skill-install')
        artifacts = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (wheel, archive)}
    result = {'passed': True, 'checks': checks, 'artifact_sha256': artifacts,
              'elapsed_s': round(time.monotonic() - started, 2),
              'scope': 'local offline delivery; migration fixture is not answer-quality evidence'}
    if output:
        (output / 'release-result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    print(json.dumps(verify(parser.parse_args().output), indent=2))
