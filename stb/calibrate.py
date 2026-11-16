"""STB v0.5 calibration: refit the full-conformal maxT rule on STB's own null runs.

STB never reuses the gate paper's frozen model. On each dataset it fits on its own n_fit = 100 honest
null runs, chosen by run id before any scoring. The decision rule is the library's (cited, not re-derived):
pool the candidate with the n fit runs, standardize every oriented signal by the pooled median and SD, and
fail a gating dimension when its own maximum exceeds the r-th largest fit-run family maximum,
r = floor(alpha (n + 1)). With n = 100 and alpha = 0.05, r = 5, so every Fail has evidence e <= 5/101.
"""
from __future__ import annotations

import numpy as np

from syntrustbench_campaign.calibration import (CalibratedEvaluator, NullCalibrator, family_of,
                                                rejection_rank)
from syntrustbench_campaign.config import CalibrationConfig, DIAGNOSTIC_DIMENSIONS, DIMENSIONS, GATE_DIMENSIONS

from .confidence import evidence_strength

STB_N_FIT = 100
ALPHAS = (0.025, 0.05, 0.10)
PRIMARY_ALPHA = 0.05


class _ExplicitAssignment(NullCalibrator):
    """The library calibrator with STB's prespecified fit ids instead of a random assignment."""

    def __init__(self, cfg: CalibrationConfig, fit_ids, other_ids):
        super().__init__(cfg)
        self._ids = (list(fit_ids), list(other_ids))

    def assignment(self, smoke: bool = False):
        return self._ids


def fit_calibration(null_signals: dict, fit_ids, other_ids=(), alphas=ALPHAS) -> dict:
    """``null_signals``: {run_id: signals dict as saved by the v0.5 evaluator}. Only fit_ids are read."""
    fit_ids = list(fit_ids)
    if len(set(fit_ids)) != len(fit_ids):
        raise ValueError('Duplicate fit ids.')
    if set(fit_ids) & set(other_ids):
        raise ValueError('Fit runs and bases must be disjoint.')
    cfg = CalibrationConfig(repetitions=len(fit_ids) + max(1, len(other_ids)), fit_runs=len(fit_ids),
                            target_rates=tuple(alphas), primary_rate=PRIMARY_ALPHA)
    profiles = {i: {'signals': {d: null_signals[i].get(d, {}) for d in DIMENSIONS}} for i in fit_ids}
    model = _ExplicitAssignment(cfg, fit_ids, other_ids).fit(profiles)
    model['stb'] = {'version': '0.5', 'n_fit': len(fit_ids),
                    'rejection_rank': {str(a): rejection_rank(len(fit_ids), a) for a in alphas},
                    'evidence_resolution': 1 / (len(fit_ids) + 1)}
    return model


def judge(model: dict, signals: dict, *, alpha: float = PRIMARY_ALPHA, missing: dict | None = None,
          hard_fail: dict | None = None) -> dict:
    """Verdict and evidence strength per dimension from one table's saved signals.

    Gating dimensions are decided by the gate family; equity and robustness by the diagnostic family
    (reported, never part of the gate unless admitted under preregistration section 8)."""
    missing, hard_fail = missing or {}, hard_fail or {}
    evaluator = CalibratedEvaluator(model)
    out = {}
    for family, dims in (('gate', GATE_DIMENSIONS), ('diagnostic', DIAGNOSTIC_DIMENSIONS)):
        present = [d for d in dims if signals.get(d)]
        if not present:
            continue
        decision = evaluator.family_decision(family, signals, alpha)
        evidence = evidence_strength(model, signals, family)
        for dim in present:
            status, audit = evaluator.status(dim, signals[dim], alpha, missing=bool(missing.get(dim)),
                                             hard_fail=bool(hard_fail.get(dim)), all_signals=signals,
                                             decision=decision)
            out[dim] = {'verdict': status, 'evidence': evidence.get(dim), 'max_z': audit.get('max_z_pooled'),
                        'family': family}
    return out


def single_dimension_model(model: dict, dimension: str) -> dict:
    """STB restricted to one dimension as its own family (the matched head-to-head variant, §7)."""
    sub = dict(model)
    sub['columns'] = [dict(c, family='gate') for c in model['columns'] if c['dimension'] == dimension]
    sub['blocked_dimensions'] = [d for d in model.get('blocked_dimensions', []) if d == dimension]
    return sub


def verdict_of(result: dict, dimension: str) -> str:
    return result.get(dimension, {}).get('verdict', 'NotEvaluated')


def self_test() -> None:
    """Every Fail must carry e <= r/(n+1) = 5/101 at alpha = 0.05 with n_fit = 100."""
    rng = np.random.default_rng(7)
    layout = {'fidelity': ['f1', 'f2', 'f3'], 'utility': ['u1', 'u2'], 'privacy': ['p1', 'p2']}

    def draw(shift=None):
        shift = shift or {}
        return {d: {m: {'estimate': float(rng.normal() + shift.get(d, 0.0)), 'higher_is_worse': True}
                    for m in ms} for d, ms in layout.items()} | {'equity': {}, 'robustness': {}}

    nulls = {i: draw() for i in range(250)}
    model = fit_calibration(nulls, range(100), range(100, 250))
    assert model['stb']['rejection_rank']['0.05'] == 5 and len(model['fit_ids']) == 100
    bound = 5 / 101
    fails = honest_fails = 0
    for i in range(3000):
        shifted = i % 2 == 0
        signals = draw({'privacy': rng.uniform(0, 5)} if shifted else None)
        result = judge(model, signals)
        for dim, r in result.items():
            if r['verdict'] == 'Fail':
                fails += 1
                assert r['evidence']['e'] <= bound + 1e-12, (dim, r['evidence']['e'])
        if not shifted and any(r['verdict'] == 'Fail' for r in result.values()):
            honest_fails += 1
    assert fails > 100
    # The guarantee is marginal over the fit runs AND the candidate: refit on fresh nulls each trial.
    trials, honest_fails = 600, 0
    for t in range(trials):
        fresh = {i: draw() for i in range(100)}
        m = fit_calibration(fresh, range(100))
        honest_fails += any(r['verdict'] == 'Fail' for r in judge(m, draw()).values())
    rate = honest_fails / trials
    se = (bound * (1 - bound) / trials) ** 0.5
    assert rate <= bound + 3 * se, (rate, bound)

if __name__ == '__main__':
    self_test()
    print('calibration self-test passed: n_fit = 100, r = 5, every Fail has e <= 5/101')
