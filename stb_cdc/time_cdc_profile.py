"""Time one honest CDC profile, piece by piece, and extrapolate campaign cost.

Runs the same evaluations a campaign 'profile' job runs (official v0.3, the five
v0.4 dimensions, SDMetrics Quality Report) on the honest positive-control table,
plus optional generator fits. Not paper evidence: this is for sizing jobs.

Usage (from stb_code/, with the pinned venv):
    python stb_cdc/time_cdc_profile.py --frozen data/cdc_frozen --scale 0.1 --work runs/cdc_timing_0.1
    python stb_cdc/time_cdc_profile.py --frozen data/cdc_frozen --scale 1.0 --work runs/cdc_timing_1.0 \
        --generators GaussianCopula,CTGAN --epochs 5
"""
from __future__ import annotations

import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--frozen', required=True, help='frozen CDC splits from cdc_adapter.py build')
parser.add_argument('--scale', type=float, default=0.1, help='fraction of every split to keep (1.0 = full)')
parser.add_argument('--work', required=True)
parser.add_argument('--threads', type=int, default=4, help='threads per job, as on the server')
parser.add_argument('--generators', default='GaussianCopula', help='comma list; empty string for none')
parser.add_argument('--epochs', type=int, default=5, help='CTGAN/TVAE epochs for timing (per-epoch cost is reported)')
parser.add_argument('--skip-official', action='store_true')
parser.add_argument('--skip-robustness', action='store_true')
args = parser.parse_args()
for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'LOKY_MAX_CPU_COUNT'):
    os.environ[var] = str(args.threads)

import copy  # noqa: E402
from dataclasses import replace  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
from pathlib import Path  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402

import pandas as pd  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdc_adapter import TARGET, cdc_config  # noqa: E402
from syntrustbench_campaign.config import seed_for  # noqa: E402
from syntrustbench_campaign.core import file_hash, save_frame, write_json  # noqa: E402
from syntrustbench_campaign.data import CohortBuilder, RoleAssigner, read_analytic, stratified_indices  # noqa: E402
from syntrustbench_campaign.evaluation import (  # noqa: E402
    EquityV04, OfficialProfiler, PrivacyV04, RobustnessV04, build_context)
from syntrustbench_campaign.judges import quality_report  # noqa: E402
from syntrustbench_campaign.reference import ReferenceFidelity, ReferenceUtility  # noqa: E402

timings: dict[str, float] = {}


def timed(label):
    class _T:
        def __enter__(self):
            self.t = time.perf_counter(); print(f'[{time.strftime("%H:%M:%S")}] {label} ...', flush=True)
        def __exit__(self, *exc):
            timings[label] = round(time.perf_counter() - self.t, 2)
            print(f'    {label}: {timings[label]:.1f} s', flush=True)
    return _T()


def scaled_frozen(src: Path, dst: Path, scale: float) -> Path:
    if scale >= 1:
        return src
    dst.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(src / 'patient_role_audit.csv')
    paths, rows = {}, []
    for name in ('real_train', 'calibration_pool', 'real_test'):
        frame = pd.read_csv(src / f'{name}.csv', keep_default_na=False, na_values=[''], dtype=str)
        keep = stratified_indices(frame[TARGET].astype(int), max(50, int(round(scale * len(frame)))),
                                  seed_for('timing_scale', name, scale))
        keep.sort()
        ids = audit[audit.role == name].sort_values('analytic_row').patient_id_sha256.to_numpy()
        paths[name] = save_frame(frame.iloc[keep], dst / f'{name}.csv')
        rows += [{'role': name, 'analytic_row': i, 'patient_id_sha256': ids[k]} for i, k in enumerate(keep)]
    audit_path = save_frame(pd.DataFrame(rows), dst / 'patient_role_audit.csv')
    write_json(dst / 'split_manifest.json', {'split_hashes': {k: file_hash(p) for k, p in paths.items()},
               'patient_audit': 'patient_role_audit.csv', 'patient_audit_sha256': file_hash(audit_path),
               'scaled_from': str(src), 'scale': scale})
    return dst


def main() -> None:
    work = Path(args.work).resolve()
    for sub in ('cohort', 'roles', 'context', 'official_v03'):
        (work / sub).mkdir(parents=True, exist_ok=True)
    frozen = scaled_frozen(Path(args.frozen).resolve(), work / 'frozen_scaled', args.scale)
    checkout = os.environ.get('STB_FROZEN_CHECKOUT', '.')
    cfg = cdc_config(frozen, output_root=str(work / 'unused_runs'), checkout_dir=checkout, threads=args.threads)
    with threadpool_limits(limits=args.threads):
        with timed('prepare: cohort'):
            cohort = CohortBuilder(cfg).build(work / 'cohort')
        with timed('prepare: roles'):
            roles = RoleAssigner(cfg).main(cohort, work / 'roles')
        print('    role sizes:', roles['role_counts'])
        with timed('context (real-vs-real floor + real baselines)'):
            context = build_context(cfg, roles, work / 'context')
        candidate = read_analytic(roles['paths']['positive'], cfg)
        if not args.skip_official:
            with timed('profile: official v0.3'):
                OfficialProfiler(cfg).evaluate(roles, candidate, work / 'official_v03')
        with open(context['context_path'], 'rb') as handle:
            ctx = pickle.load(handle)
        with timed('profile: v04 fidelity'):
            f = ReferenceFidelity(ctx).evaluate(candidate).to_dict()
        state = copy.deepcopy(ctx.rng.bit_generator.state)
        with timed('profile: v04 utility'):
            u = ReferenceUtility(ctx).evaluate(candidate).to_dict()
        members, test = [read_analytic(roles['paths'][k], cfg) for k in ('train', 'test')]
        with timed('profile: v04 privacy'):
            p = PrivacyV04(cfg, members, test, seed_for('timing', 'privacy'), False,
                           getattr(ctx, 'privacy_real_cache', None)).evaluate(candidate).to_dict()
        with timed('profile: v04 equity'):
            e = EquityV04(cfg, ctx, seed_for('timing', 'equity')).evaluate(candidate).to_dict()
        if not args.skip_robustness:
            with timed('profile: v04 robustness'):
                RobustnessV04(cfg, ctx, u, state, seed_for('timing', 'robustness')).evaluate(candidate)
        reference = read_analytic(roles['paths']['reference'], cfg)
        with timed('profile: SDMetrics quality report'):
            q = quality_report(reference, candidate, cfg)
        train = read_analytic(roles['paths']['train'], cfg)
        for name in [g for g in args.generators.split(',') if g]:
            from syntrustbench_campaign.generators import generator
            gcfg = replace(cfg, generators=replace(cfg.generators, epochs=args.epochs))
            model = generator(name, gcfg, 17)
            with timed(f'generator: {name} fit'):
                model.fit(train)
            with timed(f'generator: {name} sample {len(train)} rows'):
                model.sample(len(train), 17)

    profile = sum(v for k, v in timings.items() if k.startswith('profile:'))
    summary = {
        'scale': args.scale, 'threads': args.threads, 'role_counts': roles['role_counts'],
        'platform': platform.platform(), 'cpu_count': os.cpu_count(),
        'timings_seconds': timings, 'one_profile_seconds': round(profile, 1),
        'honest_table_statuses': {'fidelity': f['status'], 'utility': u['status'], 'privacy': p['status'],
                                  'equity': e['status'], 'sdmetrics_quality': q['score']},
        'exact_match': {k: p['metrics'][k]['estimate'] for k in
                        ('real_real_coincidence_rate', 'exact_training_duplicate_rate', 'exact_match_binomial_pvalue')},
        'epochs_used_for_deep_generators': args.epochs,
    }
    for n_profiles in (200, 1200):
        summary[f'cpu_hours_for_{n_profiles}_profiles'] = round(n_profiles * profile / 3600, 1)
    write_json(work / 'timing_summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
