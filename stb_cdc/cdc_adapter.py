"""CDC Diabetes Health Indicators adapter for the SynTrustBench campaign.

New file. The UCI code path is untouched: CDC enters the campaign through the
library's existing frozen-split loader (CohortBuilder._frozen), with a CDC
configuration that declares the schema, public DP domains, flaw targets and
range constraints.

Data: CDC Diabetes Health Indicators, UCI ML Repository id 891 (CC BY 4.0),
derived from the BRFSS 2015 survey. 253,680 respondents, 21 features, binary
outcome Diabetes_binary. There is no patient identifier: each row is one
survey respondent and is treated as a distinct person. About 14% of rows are
exact duplicates of another row because most columns are yes/no answers; they
are different respondents, not repeated records, and are kept.

Usage (from stb_code/):
    python stb_cdc/cdc_adapter.py build --raw /path/to/cdc.csv --out data/cdc_frozen
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd

STB_CODE = Path(__file__).resolve().parent.parent
if str(STB_CODE) not in sys.path:
    sys.path.insert(0, str(STB_CODE))  # STB snapshot of syntrustbench_campaign

from syntrustbench_campaign.config import (  # noqa: E402
    CalibrationConfig, DataConfig, ExecutionConfig, ExperimentConfig, GeneratorConfig, InjectionConfig,
    JudgeConfig)
from syntrustbench_campaign.core import file_hash, save_frame, write_json  # noqa: E402
from syntrustbench_campaign.data import split_three  # noqa: E402

DATASET_NAME = 'CDC Diabetes Health Indicators (BRFSS 2015; UCI id 891)'
TARGET = 'Diabetes_binary'
CONTINUOUS = ('BMI', 'MentHlth', 'PhysHlth')
BINARY = ('HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke', 'HeartDiseaseorAttack',
          'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare',
          'NoDocbcCost', 'DiffWalk', 'Sex')
ORDINAL = {'GenHlth': 5, 'Age': 13, 'Education': 6, 'Income': 8}  # codes 1..k (BRFSS codebook)
CATEGORICAL = BINARY + tuple(ORDINAL)
RAW_FEATURES = set(CONTINUOUS) | set(CATEGORICAL)
EXPECTED_ROWS = 253_680

# Public domains from the BRFSS codebook, not from the data.
RANGES = {'BMI': (12, 98), 'MentHlth': (0, 30), 'PhysHlth': (0, 30)}
CONSTRAINTS = tuple(f'{c} >= {lo}' for c, (lo, _) in RANGES.items()) + \
              tuple(f'{c} <= {hi}' for c, (_, hi) in RANGES.items())
DP_DOMAINS = (('BMI', 12, 60, 2, 98), ('MentHlth', 0, 30, 1, 30), ('PhysHlth', 0, 30, 1, 30))
DP_CATEGORIES = tuple((c, ('0', '1')) for c in BINARY) + \
                tuple((c, tuple(str(i) for i in range(1, k + 1))) for c, k in ORDINAL.items()) + \
                ((TARGET, ('0', '1')),)

# Flaw targets. PROVISIONAL until the STB preregistration freezes them.
# - Subgroup: respondents aged 80+ (Age code 13; 6.8% of rows, 18.5% diabetic), the closest
#   analogue of UCI's minority-subgroup suppression (CDC has no race variable).
# - Shuffle: BMI, a strong predictor (UCI used number_inpatient).
# - Missingness: the three count-valued columns.
# - Copy doses: chance exact matches are ~9.3% of honest rows on CDC, so copying needs about
#   0.5% of the table to be detectable; the grid brackets that limit.
CDC_INJECTIONS = InjectionConfig(
    doses=(('exact_copy', (.0005, .002, .005, .01, .05)),
           ('subgroup_suppression', (.10, .50, .90)),
           ('feature_shuffle', (.10, .50, .90)),
           ('mcar', (.05, .15, .30))),
    subgroup_column='Age', subgroup_value='13',
    shuffle_feature='BMI', missing_features=CONTINUOUS)


@dataclass(frozen=True)
class CDCExperimentConfig(ExperimentConfig):
    """ExperimentConfig with CDC benchmark metadata. Extra fields enter the fingerprint."""
    dataset_name: str = DATASET_NAME
    dp_domain_source: str = 'BRFSS 2015 codebook domains and analyst-declared caps; no training statistics'
    clinical_constraints: tuple[str, ...] = CONSTRAINTS
    rows_may_coincide: bool = True  # distinct respondents share exact rows; see overlap_policy.py
    # Legacy v0.3 fixed-threshold report: its privacy rule fails any exact training match, and ~9% of
    # honest CDC rows match by chance, so it fails every honest table. It is ~75% of profile cost.
    run_official_v03: bool = False

    def benchmark(self, *, smoke: bool = False) -> dict:
        value = super().benchmark(smoke=smoke)
        value['dataset']['name'] = self.dataset_name
        value['clinical_constraints'] = list(self.clinical_constraints)
        return value


def cdc_config(frozen_dir: str | Path, *, output_root: str, checkout_dir: str, smoke: bool = False,
               dry_run: bool = False, workers: int = 1, threads: int = 4, stages=None,
               allow_cuda: bool = False) -> CDCExperimentConfig:
    frozen_dir = Path(frozen_dir).resolve()
    data = DataConfig(raw_csv='', frozen_dir=str(frozen_dir), frozen_manifest=str(frozen_dir / 'split_manifest.json'),
                      continuous=CONTINUOUS, categorical=CATEGORICAL, target=TARGET,
                      protected=('Sex', 'Age', 'Income'))
    generators = GeneratorConfig(seeds=(17, 29, 43, 59, 71), allow_cuda=allow_cuda,
                                 dp_domains=DP_DOMAINS, dp_categories=DP_CATEGORIES)
    execution = ExecutionConfig(output_root=output_root, checkout_dir=str(Path(checkout_dir).resolve()),
                                dry_run=dry_run, workers=workers, threads_per_worker=threads,
                                smoke_mode=smoke, **({'stages': tuple(stages)} if stages else {}))
    return CDCExperimentConfig(data=data, generators=generators, injections=CDC_INJECTIONS,
                               calibration=CalibrationConfig(repetitions=100, fit_runs=50),
                               judges=JudgeConfig(), execution=execution)


# --------------------------------------------------------------------------- raw data
def load_raw(path: str | Path, allow_row_count: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """Read the UCI/Kaggle CSV; return analytic frame and a per-person key."""
    raw = pd.read_csv(path)
    ids = raw.pop('ID') if 'ID' in raw.columns else pd.Series(np.arange(len(raw)), name='row')
    missing, extra = (RAW_FEATURES | {TARGET}) - set(raw), set(raw) - RAW_FEATURES - {TARGET}
    if missing or extra:
        raise ValueError(f'Unexpected CDC schema. Missing {sorted(missing)}; extra {sorted(extra)}.')
    if raw.isna().any().any():
        raise ValueError('CDC source has missing values; expected none.')
    values = raw.to_numpy(dtype=float)
    if not np.all(values == np.round(values)):
        raise ValueError('CDC source has non-integer codes.')
    if len(raw) != EXPECTED_ROWS and not allow_row_count:
        raise ValueError(f'Expected {EXPECTED_ROWS} rows, found {len(raw)}.')
    frame = raw.astype(int)
    for c, (lo, hi) in RANGES.items():
        if frame[c].min() < lo or frame[c].max() > hi:
            raise ValueError(f'{c} outside public range {lo}-{hi}.')
    for c in BINARY + (TARGET,):
        if not set(frame[c].unique()) <= {0, 1}:
            raise ValueError(f'{c} is not 0/1.')
    for c, k in ORDINAL.items():
        if not set(frame[c].unique()) <= set(range(1, k + 1)):
            raise ValueError(f'{c} outside codes 1..{k}.')
    return frame, ids


def analytic(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[list(CONTINUOUS)].astype(int).copy()
    for c in CATEGORICAL:
        out[c] = frame[c].astype(int).astype(str)
    out[TARGET] = frame[TARGET].astype(int)
    return out[list(CONTINUOUS + CATEGORICAL + (TARGET,))]


def build_frozen(raw_csv: str | Path, out_dir: str | Path, allow_row_count: bool = False) -> dict:
    """Write real_train / calibration_pool / real_test in the frozen-split format."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame, ids = load_raw(raw_csv, allow_row_count)
    cfg = cdc_config(out, output_root=str(out / '_unused'), checkout_dir='.')
    data = analytic(frame)
    data['_patient_hash'] = [hashlib.sha256(f'cdc891:{i}'.encode()).hexdigest() for i in ids]
    if not data['_patient_hash'].is_unique:
        raise ValueError('Respondent keys are not unique.')
    parts = dict(zip(('real_train', 'calibration_pool', 'real_test'), split_three(data, cfg, cfg.data.split_seed)))
    paths, audit_rows, summary = {}, [], []
    for name, part in parts.items():
        paths[name] = save_frame(part[list(cfg.data.columns)], out / f'{name}.csv')
        audit_rows += [{'role': name, 'analytic_row': i, 'patient_id_sha256': h}
                       for i, h in enumerate(part['_patient_hash'])]
        summary.append({'role': name, 'rows': len(part), 'prevalence': float(part[TARGET].mean())})
    audit = save_frame(pd.DataFrame(audit_rows), out / 'patient_role_audit.csv')
    keys = pd.util.hash_pandas_object(data[list(cfg.data.columns)], index=False)
    manifest = {
        'dataset': DATASET_NAME, 'source_csv_sha256': file_hash(raw_csv), 'rows': len(data),
        'split_hashes': {k: file_hash(p) for k, p in paths.items()},
        'patient_audit': 'patient_role_audit.csv', 'patient_audit_sha256': file_hash(audit),
        'split': 'stratified 40/40/20 (library split_three, split_seed %d)' % cfg.data.split_seed,
        'person_key': 'source ID column if present, else row position; one row = one respondent',
        'exact_duplicate_rows_share': float(keys.duplicated(keep=False).mean()),
        'roles': summary}
    write_json(out / 'split_manifest.json', manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build', help='write frozen CDC splits')
    b.add_argument('--raw', required=True)
    b.add_argument('--out', required=True)
    b.add_argument('--allow-row-count', action='store_true', help='accept a row count other than 253,680')
    args = parser.parse_args()
    if args.command == 'build':
        manifest = build_frozen(args.raw, args.out, args.allow_row_count)
        print({k: manifest[k] for k in ('rows', 'exact_duplicate_rows_share', 'roles')})


if __name__ == '__main__':
    main()
