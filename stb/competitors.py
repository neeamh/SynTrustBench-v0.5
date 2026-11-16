"""Other benchmarks' statistics, scored on exactly the tables STB scores (head-to-head, preregistration §7).

Each competitor statistic is calibrated the same way STB is: as a one-signal family with the full-conformal
rule on the same fit runs, at the same alpha. A competitor "fails" a table when its oriented statistic ranks
above the r-th largest fit-run value. Native pass/fail rules are recorded only where the tool publishes one.

SDMetrics' distance metrics are slow on large tables; they run on a fixed, seeded subsample of
COMPETITOR_SUBSAMPLE rows (SDMetrics' own subsampling option), recorded with every score. Because STB sees the
full table, the subsampled DCR metrics are SECONDARY comparators (disclosed). The PRIMARY privacy comparator
for H1 is NewRowSynthesis, which runs on the full table, and the primary fidelity comparator is the Quality
Report (full table). Preregistration §7.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from syntrustbench_campaign.calibration import conformal_decision
from syntrustbench_campaign.config import seed_for

COMPETITOR_SUBSAMPLE = 2000


@dataclass(frozen=True)
class Statistic:
    name: str
    benchmark: str
    dimension: str            # the STB dimension it is compared with
    higher_is_worse: bool
    compute: Callable         # (base, table, dataset) -> float
    native_threshold: float | None = None   # None: the tool publishes no pass/fail rule
    note: str = ''
    role: str = 'secondary'                 # 'primary' comparators carry the H1 claim (§7)
    subsample: int | None = None            # rows the tool saw, if it did not see the whole table


def _metadata(ds) -> dict:
    return {'columns': {c: {'sdtype': 'numerical' if c in ds.cfg.data.continuous else 'categorical'}
                        for c in ds.cfg.data.columns}}


def _seeded(fn):
    def wrapped(base, table, ds):
        np.random.seed(seed_for('stb-v0.5', 'competitor', fn.__name__, len(table)) % (2 ** 32))
        return float(fn(base, table, ds))
    wrapped.__name__ = fn.__name__
    return wrapped


@_seeded
def _sdm_quality(base, table, ds):
    from syntrustbench_campaign.judges import quality_report
    return quality_report(base.context.reference, table, ds.cfg)['score']


@_seeded
def _sdm_dcr_baseline(base, table, ds):
    from sdmetrics.single_table import DCRBaselineProtection
    return DCRBaselineProtection.compute(base.members, table, _metadata(ds), None,
                                         num_rows_subsample=COMPETITOR_SUBSAMPLE)


@_seeded
def _sdm_dcr_overfitting(base, table, ds):
    from sdmetrics.single_table import DCROverfittingProtection
    return DCROverfittingProtection.compute(base.members, table, base.test, _metadata(ds), None,
                                            num_rows_subsample=COMPETITOR_SUBSAMPLE)


def new_row_synthesis(real, synthetic, metadata, tolerance=0.01, chunk=2000) -> float:
    """SDMetrics 0.33.0 NewRowSynthesis, vectorized. Same rule: a synthetic row is matched when some real row has
    equal categorical values (NA matches NA) and, for every numerical column, |real - syn| <= |tolerance * syn|
    (NA matches NA); the score is the share of unmatched synthetic rows. SDMetrics loops one pandas query per
    synthetic row, which took over 20 minutes per full-size CDC table; this takes seconds. Equality with the
    library is checked in ``self_test`` and must hold exactly."""
    cols = metadata['columns']
    num = [c for c in real.columns if c in cols and cols[c]['sdtype'] == 'numerical']
    cat = [c for c in real.columns if c in cols and cols[c]['sdtype'] != 'numerical']

    def keys(df):
        k = df[cat].astype(object)
        return k.where(k.notna(), '\x00NA').astype(str)

    r = pd.concat([keys(real).reset_index(drop=True),
                   real[num].reset_index(drop=True).astype(float).add_prefix('\x01r_')], axis=1)
    s = pd.concat([keys(synthetic).reset_index(drop=True),
                   synthetic[num].reset_index(drop=True).astype(float).add_prefix('\x01s_')], axis=1)
    on = cat
    if not cat:
        r['\x00k'] = 0
        s['\x00k'] = 0
        on = ['\x00k']
    s['\x01sid'] = np.arange(len(s))
    r = r.drop_duplicates()
    matched = np.zeros(len(s), dtype=bool)
    for start in range(0, len(s), chunk):
        m = s.iloc[start:start + chunk].merge(r, on=on, how='inner')
        if m.empty:
            continue
        ok = np.ones(len(m), dtype=bool)
        for c in num:
            sv, rv = m['\x01s_' + c].to_numpy(float), m['\x01r_' + c].to_numpy(float)
            with np.errstate(invalid='ignore'):
                ok &= np.where(np.isnan(sv), np.isnan(rv), np.abs(rv - sv) <= np.abs(tolerance * sv))
        matched[m['\x01sid'].to_numpy()[ok]] = True
    return float(1 - matched.mean()) if len(s) else float('nan')


@_seeded
def _sdm_new_row_synthesis(base, table, ds):
    return new_row_synthesis(base.members, table, _metadata(ds))


def self_test(trials: int = 6) -> None:
    """The vectorized NewRowSynthesis must equal SDMetrics' own implementation exactly (copies, NAs, values above
    100 where the 1% tolerance admits neighbours)."""
    import warnings
    from sdmetrics.single_table import NewRowSynthesis
    rng = np.random.default_rng(3)
    meta = {'columns': {'g': {'sdtype': 'categorical'}, 'h': {'sdtype': 'categorical'}, 'k': {'sdtype': 'categorical'},
                        'x': {'sdtype': 'numerical'}, 'y': {'sdtype': 'numerical'}}}

    def make(n, wide):
        df = pd.DataFrame({'g': rng.choice(['a', 'b', 'c', None], n, p=[.4, .3, .25, .05]), 'h': rng.integers(0, 2, n),
                           'k': rng.integers(1, 4, n).astype(str), 'x': rng.integers(0, 140 if wide else 40, n).astype(float),
                           'y': rng.integers(1, 15, n).astype(float)})
        df.loc[rng.random(n) < .03, 'x'] = np.nan
        return df
    for t in range(trials):
        real, syn = make(3000, t % 2 == 0), make(800, t % 2 == 0)
        syn.iloc[:100] = real.iloc[rng.choice(3000, 100)].to_numpy()
        syn.loc[100:150, 'x'] = syn.loc[100:150, 'x'] + 1
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            expected = NewRowSynthesis.compute(real, syn, meta)
        got = new_row_synthesis(real, syn, meta)
        assert abs(expected - got) < 1e-12, (t, expected, got)


if __name__ == '__main__':
    self_test()
    print('competitors self-test passed: vectorized NewRowSynthesis equals SDMetrics 0.33.0')


STATISTICS = {s.name: s for s in (
    Statistic('sdmetrics_quality', 'SDMetrics', 'fidelity', False, _sdm_quality,
              note='Quality Report overall score (column shapes + pair trends); higher is better.', role='primary'),
    Statistic('sdmetrics_dcr_baseline', 'SDMetrics', 'privacy', False, _sdm_dcr_baseline,
              note='DCRBaselineProtection; higher means synthetic rows are farther from training rows.',
              subsample=COMPETITOR_SUBSAMPLE),
    Statistic('sdmetrics_dcr_overfitting', 'SDMetrics', 'privacy', False, _sdm_dcr_overfitting,
              note='DCROverfittingProtection; higher means less closer-to-train-than-to-holdout.',
              subsample=COMPETITOR_SUBSAMPLE),
    Statistic('sdmetrics_new_row_synthesis', 'SDMetrics', 'privacy', False, _sdm_new_row_synthesis,
              note='NewRowSynthesis; share of synthetic rows not matching a training row (full table; SDMetrics '
                   '0.33.0 rule, vectorized and verified identical, see self_test).',
              role='primary'),
    # SynthEval is not included in v0.5 (preregistration §3): it was not installed and its metric list could not
    # be frozen before the preregistration was. It can only be added by a dated amendment before scoring.
)}


def score_all(names, base, table, ds) -> dict:
    """Scores for the named statistics; a statistic that errors is recorded as unavailable, never dropped."""
    out = {}
    for name in names:
        stat = STATISTICS[name]
        try:
            value = stat.compute(base, table, ds)
            out[name] = {'value': value, 'available': bool(np.isfinite(value))}
        except Exception as exc:  # noqa: BLE001 - recorded per table
            out[name] = {'value': None, 'available': False, 'error': f'{type(exc).__name__}: {exc}'}
    return out


def calibrate(fit_scores: dict) -> dict:
    """``fit_scores``: {statistic: [value on each fit run]} -> per-statistic oriented fit vectors."""
    model = {}
    for name, values in fit_scores.items():
        stat = STATISTICS[name]
        values = [v for v in values if v is not None and np.isfinite(v)]
        direction = 1 if stat.higher_is_worse else -1
        model[name] = {'direction': direction, 'fit_harm': [direction * v for v in values], 'n_fit': len(values),
                       'dimension': stat.dimension}
    return model


def decide(model: dict, name: str, value, alpha: float, floor: float = 1e-8) -> str:
    """Full-conformal one-signal decision: 'Fail', 'Pass' or 'NotEvaluated'."""
    if value is None or not np.isfinite(value) or name not in model:
        return 'NotEvaluated'
    m = model[name]
    fit = np.asarray(m['fit_harm'], dtype=float)[:, None]
    result = conformal_decision(fit, np.array([m['direction'] * value]), alpha, floor)
    if not result['finite_sample_valid']:
        return 'NotEvaluated'
    return 'Fail' if result['candidate_max'] > result['threshold'] else 'Pass'
