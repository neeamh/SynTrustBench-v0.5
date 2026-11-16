"""How far a run has got: finished jobs per stage and failures, read from the results folders.

    python scripts/progress.py            # every results folder under runs/
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGES = ('nulls', 'bases', 'calibration', 'main', 'generators', 'lod', 'a6', 'validation', 'table_lod',
          'dcr_nulls', 'dcr', 'records')


def main() -> None:
    runs = ROOT / 'runs'
    folders = sorted(p for p in runs.glob('*/*') if (p / 'manifest.json').is_file()) if runs.is_dir() else []
    if not folders:
        print('No results folders yet under runs/.')
        return
    for folder in folders:
        print(f'== {folder.relative_to(ROOT)}')
        for stage in STAGES:
            done = len(list((folder / stage).rglob('*.json'))) if (folder / stage).is_dir() else 0
            if done:
                print(f'   {stage:12s} {done:6d} finished')
        failures = list((folder / 'failures').glob('*')) if (folder / 'failures').is_dir() else []
        print(f'   failures     {len(failures):6d}' + ('   <- see ' + str((folder / 'failures').relative_to(ROOT))
                                                       if failures else ''))


if __name__ == '__main__':
    main()
