"""Results on disk: one JSON file per job (the one-writer rule), plus a run manifest.

Every job writes exactly one file, atomically, under ``<root>/<stage>/<job_id>.json``. Nothing ever appends
to a shared file, so any number of workers can run at once and an interrupted run resumes by skipping jobs
whose file already exists. Analysis reads the files back; it never needs a database.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
from typing import Iterator

from syntrustbench_campaign.core import clean_json

from syntrustbench_campaign.environment import PINS as _LIBRARY_PINS

# Every package pinned by the library (environment.PINS) plus STB's own runtime dependency.
PINNED = tuple(name for name, _ in _LIBRARY_PINS) + ('threadpoolctl',)


def _safe(job_id: str) -> str:
    return job_id.replace('/', '__')


@dataclass(frozen=True)
class ResultStore:
    root: Path

    def path(self, stage: str, job_id: str) -> Path:
        return Path(self.root) / stage / f'{_safe(job_id)}.json'

    def done(self, stage: str, job_id: str) -> bool:
        return self.path(stage, job_id).is_file()

    def write(self, stage: str, job_id: str, payload: dict) -> Path:
        path = self.path(stage, job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f'.{os.getpid()}.tmp')
        tmp.write_text(json.dumps(clean_json(payload), indent=1, allow_nan=False))
        os.replace(tmp, path)
        return path

    def read(self, stage: str, job_id: str) -> dict:
        return json.loads(self.path(stage, job_id).read_text())

    def iter(self, stage: str) -> Iterator[dict]:
        folder = Path(self.root) / stage
        for path in sorted(folder.glob('*.json')) if folder.is_dir() else []:
            yield json.loads(path.read_text())

    def count(self, stage: str) -> int:
        folder = Path(self.root) / stage
        return len(list(folder.glob('*.json'))) if folder.is_dir() else 0


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hashes(code_root: Path) -> dict:
    files = sorted(list((code_root / 'stb').glob('*.py')) + list((code_root / 'syntrustbench_campaign').glob('*.py'))
                   + list((code_root / 'stb_cdc').glob('*.py')))
    return {str(p.relative_to(code_root)): file_sha256(p) for p in files}


def write_manifest(store: ResultStore, plan, code_root: Path, prereg: Path) -> dict:
    """Record what produced these results. Refuses to mix platforms or code versions in one results folder."""
    versions = {}
    for name in PINNED:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    manifest = {'plan': plan.name, 'plan_fingerprint': plan.fingerprint(),
                'platform': {'system': platform.system(), 'machine': platform.machine(),
                             'python': platform.python_version(), 'cpu_count': os.cpu_count()},
                'packages': versions, 'code': code_hashes(code_root),
                'preregistration_sha256': file_sha256(prereg) if Path(prereg).is_file() else None}
    path = Path(store.root) / 'manifest.json'
    if path.is_file():
        old = json.loads(path.read_text())
        for key in ('plan_fingerprint', 'platform', 'packages', 'code', 'preregistration_sha256'):
            if old.get(key) != manifest[key]:
                raise RuntimeError(f'Results folder was produced with a different {key}; use a new folder. '
                                   f'(One platform and one code version per results folder.)')
        return old
    manifest['created'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1))
    return manifest
