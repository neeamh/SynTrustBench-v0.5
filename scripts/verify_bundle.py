"""Check that this folder can run SynTrustBench v0.5: Python, platform, pinned packages, imports, data hashes,
and the preregistration. Run from the bundle root:  python scripts/verify_bundle.py  [--quiet]

Exits non-zero on anything that would invalidate a confirmatory run. Platform and an unfrozen preregistration
are warnings here (smoke runs are fine); the notebook itself refuses a full run off Linux x86_64.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import platform
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CDC_MIRROR_SHA256 = '19f367e3e3350768f0c144c5d73ee5b355f67a57eaaa86ca7bd8aec594d8b1d0'
QUIET = '--quiet' in sys.argv
problems: list[str] = []
warnings: list[str] = []


def say(message: str) -> None:
    if not QUIET:
        print(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def check_python_and_platform() -> None:
    version = platform.python_version()
    say(f'python   {version}  ({sys.executable})')
    if not version.startswith('3.12.'):
        problems.append(f'Python {version}; the preregistered environment is Python 3.12 (run ./setup.sh)')
    system, machine = platform.system(), platform.machine()
    say(f'platform {system} {machine}')
    if (system, machine) != ('Linux', 'x86_64'):
        warnings.append(f'{system} {machine}: fine for smoke runs; the confirmatory run needs Linux x86_64')


def check_pins() -> None:
    pins = {}
    for line in (ROOT / 'requirements-stb-v0.5.txt').read_text().splitlines():
        match = re.match(r'^\s*([A-Za-z0-9_.\-]+)==([^\s#]+)', line)
        if match:
            pins[match.group(1)] = match.group(2)
    wrong = []
    for name, wanted in pins.items():
        try:
            installed = importlib.metadata.version(name).split('+', 1)[0]
        except importlib.metadata.PackageNotFoundError:
            installed = None
        if installed != wanted:
            wrong.append(f'{name} {installed} (pinned {wanted})')
    say(f'packages {len(pins) - len(wrong)}/{len(pins)} pinned versions installed')
    if wrong:
        problems.append('package versions differ from the pins: ' + '; '.join(wrong))


def check_imports() -> None:
    sys.path.insert(0, str(ROOT))
    for module in ('syntrustbench', 'syntrustbench_campaign.evaluation', 'stb.stages', 'stb.analysis',
                   'stb.calibrate', 'sdv', 'sdmetrics', 'snsynth', 'torch'):
        try:
            importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001 - report every failing import
            problems.append(f'cannot import {module}: {type(exc).__name__}: {exc}')
    say('imports  checked')


def check_data() -> None:
    for name in ('cdc_frozen', 'uci_frozen'):
        folder = ROOT / 'data' / name
        manifest = json.loads((folder / 'split_manifest.json').read_text())
        expected = dict(manifest.get('split_hashes', {}))
        if 'patient_audit_sha256' in manifest:
            expected['patient_role_audit'] = manifest['patient_audit_sha256']
        for stem, digest in expected.items():
            path = folder / f'{stem}.csv'
            if not path.is_file():
                problems.append(f'missing {path.relative_to(ROOT)}')
            elif sha256(path) != digest:
                problems.append(f'{path.relative_to(ROOT)} does not match its recorded hash')
        say(f'data     {name}: {len(expected)} files checked against split_manifest.json')
    mirror = ROOT / 'data' / 'cdc_brfss2015_mirror.csv'
    if mirror.is_file() and sha256(mirror) != CDC_MIRROR_SHA256:
        problems.append('data/cdc_brfss2015_mirror.csv does not match the recorded CDC hash')


def check_prereg() -> None:
    prereg = ROOT / 'PREREGISTRATION_v0.5.md'
    first = prereg.read_text().splitlines()[0]
    say(f'prereg   sha256 {sha256(prereg)}')
    if 'DRAFT' in first.upper():
        warnings.append('the preregistration is still marked DRAFT; freeze it before the confirmatory run')


def main() -> int:
    for check in (check_python_and_platform, check_pins, check_imports, check_data, check_prereg):
        try:
            check()
        except Exception as exc:  # noqa: BLE001 - one failing check should not hide the others
            problems.append(f'{check.__name__} failed: {type(exc).__name__}: {exc}')
    for message in warnings:
        print(f'WARNING  {message}')
    for message in problems:
        print(f'PROBLEM  {message}')
    if problems:
        print('Bundle check FAILED')
        return 1
    say('Bundle check passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
