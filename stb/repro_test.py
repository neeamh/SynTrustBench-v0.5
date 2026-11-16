"""Reproduction test for the v0.5 evaluator (two forms, plus isolation).

  A. Signals that use no randomness must equal the v0.4.2 code path exactly (regression test of the signal
     code; v0.4.2 path = fidelity then utility on one shared stream, privacy seeded by seed_for(key)).
  B. Signals that use randomness must agree with the v0.4.2 path within their seed-to-seed spread.
  C. Isolation: scoring one dimension alone gives exactly the values it gets inside a full v0.5 profile.

A signal is classed as random if it changes across >= 2 different table seeds under v0.5; otherwise it is
deterministic and form A applies. A stored-value comparison against the gate paper's saved profiles can be
added with --stored PROFILE.json --context PKL --roles ROLES.json (read-only), with explicit permission.

Usage (from stb_code/):  python -m stb.repro_test --frozen data/cdc_frozen --scale 0.1 --work runs/repro
"""
from __future__ import annotations

import argparse
import platform
import copy
import json
import math
from pathlib import Path
import pickle
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'stb_cdc'))

from cdc_adapter import cdc_config  # noqa: E402
from syntrustbench_campaign.config import GATE_DIMENSIONS, seed_for  # noqa: E402
from syntrustbench_campaign.data import CohortBuilder, RoleAssigner, read_analytic  # noqa: E402
from syntrustbench_campaign.evaluation import PrivacyV04, build_context, reference_signals  # noqa: E402
from syntrustbench_campaign.reference import ReferenceFidelity, ReferenceUtility  # noqa: E402

from stb import dimensions  # noqa: E402
from stb.operators import DatasetSpec, FlawContext, plant  # noqa: E402


def flat(signals: dict) -> dict:
    return {(d, m): v.get('estimate') for d in GATE_DIMENSIONS for m, v in signals.get(d, {}).items()}


def same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return (math.isnan(a) and math.isnan(b)) or a == b


def v042_path(base, table, key) -> dict:
    ctx = copy.deepcopy(base.context)
    f = ReferenceFidelity(ctx).evaluate(table).to_dict()
    u = ReferenceUtility(ctx).evaluate(table).to_dict()
    sig = reference_signals(f, u, ctx)
    p = PrivacyV04(base.cfg, base.members, base.test, seed_for(key, 'privacy'), False,
                   getattr(ctx, 'privacy_real_cache', None)).evaluate(table).to_dict()
    sig['privacy'] = p['signals']
    return sig


def scaled_frozen(src: Path, dst: Path, scale: float, target: str) -> Path:
    import pandas as pd
    from syntrustbench_campaign.core import file_hash, save_frame, write_json
    from syntrustbench_campaign.data import stratified_indices
    if scale >= 1:
        return src
    dst.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(src / 'patient_role_audit.csv')
    paths, rows = {}, []
    for name in ('real_train', 'calibration_pool', 'real_test'):
        frame = pd.read_csv(src / f'{name}.csv', keep_default_na=False, na_values=[''], dtype=str)
        keep = np.sort(stratified_indices(frame[target].astype(int), max(50, int(round(scale * len(frame)))),
                                          seed_for('repro_scale', name, scale)))
        ids = audit[audit.role == name].sort_values('analytic_row').patient_id_sha256.to_numpy()
        paths[name] = save_frame(frame.iloc[keep], dst / f'{name}.csv')
        rows += [{'role': name, 'analytic_row': i, 'patient_id_sha256': ids[k]} for i, k in enumerate(keep)]
    audit_path = save_frame(pd.DataFrame(rows), dst / 'patient_role_audit.csv')
    write_json(dst / 'split_manifest.json', {'split_hashes': {k: file_hash(v) for k, v in paths.items()},
               'patient_audit': 'patient_role_audit.csv', 'patient_audit_sha256': file_hash(audit_path)})
    return dst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--frozen', required=True)
    ap.add_argument('--scale', type=float, default=0.1)
    ap.add_argument('--work', required=True)
    ap.add_argument('--reseeds', type=int, default=20)
    args = ap.parse_args()
    work = Path(args.work).resolve()
    for sub in ('cohort', 'roles', 'context'):
        (work / sub).mkdir(parents=True, exist_ok=True)
    cfg0 = cdc_config(args.frozen, output_root=str(work / 'unused'), checkout_dir='.')
    frozen = scaled_frozen(Path(args.frozen).resolve(), work / 'frozen', args.scale, cfg0.data.target)
    cfg = cdc_config(frozen, output_root=str(work / 'unused'), checkout_dir='.')
    cohort = CohortBuilder(cfg).build(work / 'cohort')
    roles = RoleAssigner(cfg).null(cohort, 'repro', work / 'roles')
    context = build_context(cfg, roles, work / 'context')
    base = dimensions.Base.load(cfg, roles, context['context_path'])
    honest = read_analytic(roles['paths']['positive'], cfg)
    spec = DatasetSpec.from_config(cfg, violation=('BMI', 99))
    planted, _ = plant('copy_random', honest, '0.005', FlawContext(spec, base.members, base.test), seed=5)
    report, ok = {'rows': roles['role_counts']}, True
    for label, table in (('honest', honest), ('copy_0.5pct', planted)):
        key = f'repro/{label}'
        full = dimensions.evaluate(base, table, key)
        # C. isolation: each gating dimension alone == inside the full profile.
        isolation = {}
        for dim in GATE_DIMENSIONS:
            alone = dimensions.evaluate(base, table, key, dims=(dim,))
            isolation[dim] = all(same(flat(alone['signals'])[k], v) for k, v in flat(full['signals']).items()
                                 if k[0] == dim)
        # Spread across table seeds (v0.5), to classify signals as deterministic or random.
        draws = [flat(dimensions.evaluate(base, table, f'{key}/reseed{i}')['signals']) for i in range(args.reseeds)]
        old = flat(v042_path(base, table, key))
        new = flat(full['signals'])
        rows = []
        for k in new:
            values = np.array([d[k] for d in draws if d[k] is not None], dtype=float)
            random = len(values) > 1 and np.ptp(values) > 0
            if not random:
                passed = same(old[k], new[k])
                rows.append({'signal': '/'.join(k), 'form': 'A exact', 'v042': old[k], 'v05': new[k], 'pass': passed})
            else:
                mu, sd = float(values.mean()), float(values.std(ddof=1))
                z = abs(old[k] - mu) / sd if sd > 0 and old[k] is not None else float('inf')
                passed = z <= 4
                rows.append({'signal': '/'.join(k), 'form': 'B noise', 'v042': old[k], 'v05_mean': mu, 'v05_sd': sd,
                             'z': z, 'pass': passed})
            ok &= passed
        ok &= all(isolation.values())
        report[label] = {'isolation': isolation, 'signals': rows}
    (work / 'repro_report.json').write_text(json.dumps(report, indent=2, default=float))
    for label in ('honest', 'copy_0.5pct'):
        r = report[label]
        print(f"\n== {label}: isolation {r['isolation']}")
        for row in r['signals']:
            extra = f"z={row['z']:.2f}" if row['form'].startswith('B') else ''
            print(f"  {'PASS' if row['pass'] else 'FAIL'}  {row['form']:8s} {row['signal']:62s} {extra}")
    print('\nREPRODUCTION TEST', 'PASSED' if ok else 'FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
