"""Compare the v0.5 evaluator against signals stored by the gate paper's run (read-only inputs).

Form A: signals with no randomness must equal the stored values exactly.
Form B: signals with randomness must lie within the v0.5 seed-to-seed spread (|z| <= 4).
Usage: python -m stb.stored_compare --dir runs/gate_readonly_null51 --config CONFIGURATION.json --key null/51
"""
from __future__ import annotations

import argparse
import platform
import json
import math
from pathlib import Path

import numpy as np

from syntrustbench_campaign.config import GATE_DIMENSIONS, from_mapping
from syntrustbench_campaign.data import read_analytic

from stb import dimensions


def flat(signals):
    return {(d, m): v.get('estimate') for d in GATE_DIMENSIONS for m, v in signals.get(d, {}).items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--key', required=True)
    ap.add_argument('--reseeds', type=int, default=20)
    a = ap.parse_args()
    d = Path(a.dir)
    cfg = from_mapping(json.loads(Path(a.config).read_text()))
    roles = json.loads((d / 'roles_result.json').read_text())
    roles['paths'] = {k: str(d / f'roles_{k}.csv') for k in roles['paths']}
    base = dimensions.Base.load(cfg, roles, str(d / 'reference_context.pkl'))
    table = read_analytic(roles['paths']['positive'], cfg)
    stored = flat(json.loads((d / 'profile.json').read_text())['v04']['signals'])
    new = flat(dimensions.evaluate(base, table, a.key)['signals'])
    draws = [flat(dimensions.evaluate(base, table, f'{a.key}/reseed{i}', dims=('utility', 'privacy'))['signals'])
             for i in range(a.reseeds)]
    ok = True
    for k in new:
        vals = np.array([x[k] for x in draws if k in x and x[k] is not None], float)
        random = len(vals) > 1 and np.ptp(vals) > 0
        if not random:
            same = (stored[k] is None and new[k] is None) or (stored[k] is not None and new[k] is not None and
                    (stored[k] == new[k] or (math.isnan(stored[k]) and math.isnan(new[k]))))
            print(f"{'PASS' if same else 'FAIL'}  A exact  {'/'.join(k):62s} stored={stored[k]!r} v05={new[k]!r}")
        else:
            mu, sd = vals.mean(), vals.std(ddof=1)
            z = abs(stored[k] - mu) / sd
            same = z <= 4
            print(f"{'PASS' if same else 'FAIL'}  B noise  {'/'.join(k):62s} z={z:.2f} (stored {stored[k]:.5f}, v05 {mu:.5f} +/- {sd:.5f})")
        ok &= bool(same)
    print('platform:', platform.platform(), platform.machine())
    print('Form A is exact only on the platform that produced the stored values (HistGradientBoosting and\n'
          'lbfgs differ across CPU architectures); run this on that machine.')
    print('STORED-VALUE COMPARISON', 'PASSED' if ok else 'FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
