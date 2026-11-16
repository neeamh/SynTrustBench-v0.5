"""Dataset definitions for STB v0.5: the only module that knows how UCI and CDC differ.

Both datasets enter through the library's frozen-split loader. STB's own copies of the splits live in
``stb_code/data/<name>_frozen``. Null runs are drawn with STB's own seed salt ('stb-v0.5'), so they never
coincide with the gate paper's runs (which used 'v0.4.1').
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from syntrustbench_campaign.config import ExperimentConfig, from_mapping, seed_for
from syntrustbench_campaign.core import file_hash, save_frame, write_json
from syntrustbench_campaign.data import stratified_indices

from .operators import DatasetSpec

STB_NULL_SALT = 'stb-v0.5'
STB_CODE = Path(__file__).resolve().parent.parent

# Value that breaks one clinical range rule (constraint_violation flaw).
VIOLATIONS = {'uci': ('time_in_hospital', 15), 'cdc': ('BMI', 99)}


@dataclass(frozen=True)
class Dataset:
    name: str                 # 'uci' or 'cdc'
    cfg: ExperimentConfig     # library configuration (schema, evaluators, generators)
    spec: DatasetSpec         # what the flaw operators need
    frozen_dir: Path

    @property
    def label(self) -> str:
        return {'uci': 'UCI Diabetes 130-US (earliest encounter)',
                'cdc': 'CDC Diabetes Health Indicators (BRFSS 2015)'}[self.name]


def _with_frozen(cfg: ExperimentConfig, frozen_dir: Path, threads: int) -> ExperimentConfig:
    data = replace(cfg.data, frozen_dir=str(frozen_dir), frozen_manifest=str(frozen_dir / 'split_manifest.json'))
    calibration = replace(cfg.calibration, null_seed_salt=STB_NULL_SALT)
    execution = replace(cfg.execution, threads_per_worker=threads, dry_run=False)
    return replace(cfg, data=data, calibration=calibration, execution=execution)


def load_dataset(name: str, *, scale: float = 1.0, work: Path | None = None, threads: int = 4) -> Dataset:
    """Load a dataset definition. ``scale`` < 1 subsamples every split and uses the library's minimum of
    20 bootstrap resamples (smoke runs only; the preregistered run uses 200)."""
    frozen = STB_CODE / 'data' / f'{name}_frozen'
    if not (frozen / 'split_manifest.json').is_file():
        raise FileNotFoundError(f'Frozen splits missing: {frozen}')
    if scale < 1:
        target = 'Diabetes_binary' if name == 'cdc' else 'readmitted_30d'
        frozen = scaled_copy(frozen, Path(work or STB_CODE / 'runs' / 'smoke') / f'{name}_frozen_x{scale}', scale, target)
    if name == 'uci':
        base = from_mapping(json.loads((STB_CODE / 'configs' / 'uci_gate_configuration.json').read_text()))
        cfg = _with_frozen(base, frozen, threads)
    elif name == 'cdc':
        sys.path.insert(0, str(STB_CODE / 'stb_cdc'))
        from cdc_adapter import cdc_config
        cfg = _with_frozen(cdc_config(frozen, output_root=str(STB_CODE / 'runs' / 'unused'), checkout_dir='.'),
                           frozen, threads)
    else:
        raise ValueError(f'Unknown dataset {name!r}')
    if scale < 1:
        cfg = replace(cfg, bootstrap_iterations=20)
    return Dataset(name, cfg, DatasetSpec.from_config(cfg, VIOLATIONS[name]), frozen)


def scaled_copy(src: Path, dst: Path, scale: float, target: str) -> Path:
    """Stratified subsample of every frozen split, with its own manifest (smoke runs only)."""
    if (dst / 'split_manifest.json').is_file():
        return dst
    dst.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(src / 'patient_role_audit.csv')
    paths, rows = {}, []
    for role in ('real_train', 'calibration_pool', 'real_test'):
        frame = pd.read_csv(src / f'{role}.csv', keep_default_na=False, na_values=[''], dtype=str)
        keep = np.sort(stratified_indices(frame[target].astype(int), max(50, int(round(scale * len(frame)))),
                                          seed_for('stb-smoke-scale', role, scale)))
        ids = audit[audit.role == role].sort_values('analytic_row').patient_id_sha256.to_numpy()
        paths[role] = save_frame(frame.iloc[keep], dst / f'{role}.csv')
        rows += [{'role': role, 'analytic_row': i, 'patient_id_sha256': ids[k]} for i, k in enumerate(keep)]
    audit_path = save_frame(pd.DataFrame(rows), dst / 'patient_role_audit.csv')
    write_json(dst / 'split_manifest.json', {'split_hashes': {k: file_hash(v) for k, v in paths.items()},
               'patient_audit': 'patient_role_audit.csv', 'patient_audit_sha256': file_hash(audit_path),
               'scaled_from': str(src), 'scale': scale, 'smoke_only': True})
    return dst
