"""STB v0.5 verdict confidence (spec v1 + amendments v1.1).

Three statements, always per dimension and never merged into one number:
  * Fail / Conditional -> evidence strength: a full-conformal p-value against the calibration runs,
    computed with the verdict's own pooled standardization (amendment A1).
  * Pass -> limits of detection: fixed-sequence one-sided Clopper-Pearson over a descending dose grid
    (official); a probit curve in log10 dose (descriptive only).
  * Generator across seeds -> seed agreement (modal verdict, k/5) and ranking stability (Kendall tau).
Every number traces back to counts of null runs or planted-flaw runs; there is no learned confidence model.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
import math

import numpy as np
from scipy.optimize import minimize
from scipy.stats import beta, kendalltau, norm

PI = 0.95
GAMMA = 0.95
GATE_DIMENSIONS = ('fidelity', 'utility', 'privacy')

FLAW_PHRASES = {
    'copy_random': 'copying of training records',
    'copy_outlier': 'copying of rare (outlying) training records',
    'near_copy': 'near-copies of training records',
    'shuffle': 'a broken relationship between one predictor and the rest',
    'marginal_shift': 'a shift in the numeric columns',
    'constraint_violation': 'clinical-rule violations',
    'label_flip': 'flipped outcome labels',
    'outcome_decouple': 'an outcome cut loose from the predictors',
    'subgroup_degrade': 'a suppressed subgroup',
}


# --------------------------------------------------------------------------- evidence strength
def _standardize(matrix: np.ndarray, floor: float) -> np.ndarray:
    center = np.median(matrix, axis=0)
    scale = np.maximum(np.std(matrix, axis=0, ddof=1), floor)
    return (matrix - center) / scale


def _available(value) -> bool:
    return isinstance(value, (int, float, np.number)) and bool(np.isfinite(value))


def evidence_strength(model: dict, signals: dict, family: str = 'gate') -> dict:
    """Per-dimension conformal p-values for one candidate.

    Uses exactly the columns, orientation and pooled standardization of the frozen calibration model's
    family decision (``CalibratedEvaluator.family_decision``): the candidate is pooled with the n fit runs,
    every signal is standardized by the pooled median and SD, and for each dimension
        e_k = (1 + #{fit runs whose dimension-k maximum >= the candidate's}) / (n + 1).
    Valid (P(e_k <= u) <= u) when the candidate is exchangeable with the fit runs. Because a dimension
    fails when its maximum exceeds the r-th largest fit-run *family* maximum, every Fail has
    e_k <= r / (n + 1).
    """
    cols = [c for c in model['columns'] if c['family'] == family
            and _available(signals.get(c['dimension'], {}).get(c['metric'], {}).get('estimate'))]
    if not cols:
        return {}
    fit = np.array([c['fit_harm'] for c in cols], dtype=float).T
    candidate = np.array([c['direction'] * signals[c['dimension']][c['metric']]['estimate'] for c in cols])
    n = fit.shape[0]
    z = _standardize(np.vstack([fit, candidate]), model['scale_floor'])
    out = {}
    for dim in dict.fromkeys(c['dimension'] for c in cols):
        idx = [i for i, c in enumerate(cols) if c['dimension'] == dim]
        fit_max, cand_max = z[:n, idx].max(axis=1), float(z[n, idx].max())
        at_least = int(np.sum(fit_max >= cand_max))
        out[dim] = {'e': (1 + at_least) / (n + 1), 'at_least_as_extreme': at_least, 'n_fit': n,
                    'candidate_max_z': cand_max, 'resolution': 1 / (n + 1),
                    'columns': [c['metric'] for c in (cols[i] for i in idx)]}
    return out


def evidence_text(e: dict) -> str:
    k, n = e['at_least_as_extreme'], e['n_fit']
    if k == 0:
        return f'More extreme than all {n} honest calibration tables'
    return f'{k} of {n} honest calibration tables were at least as extreme'


# --------------------------------------------------------------------------- binomial tools
def clopper_pearson_lower(hits: int, trials: int, gamma: float = GAMMA) -> float:
    """One-sided lower confidence bound for a binomial proportion (exact)."""
    if trials <= 0:
        return 0.0
    if hits <= 0:
        return 0.0
    return float(beta.ppf(1 - gamma, hits, trials - hits + 1))


def wilson_interval(hits: int, trials: int, level: float = 0.95) -> tuple[float, float]:
    if trials <= 0:
        return (float('nan'), float('nan'))
    z = norm.ppf(1 - (1 - level) / 2)
    p = hits / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def misses_allowed(trials: int, pi: float = PI, gamma: float = GAMMA) -> int | None:
    """Largest number of misses at which R trials still certify pi; None if even 0 misses cannot."""
    allowed = None
    for misses in range(trials + 1):
        if clopper_pearson_lower(trials - misses, trials, gamma) >= pi:
            allowed = misses
        else:
            break
    return allowed


# --------------------------------------------------------------------------- limit of detection
def fixed_sequence_lod(doses, hits, trials, pi: float = PI, gamma: float = GAMMA) -> dict:
    """Official LoD (spec §4.6a). Doses are tested from high to low; stop at the first dose whose
    one-sided Clopper-Pearson lower bound falls below pi. Dose 0 is excluded (it is the false-alarm check).

    Guarantee: with probability >= gamma, every claim p(d) >= pi for the certified doses is true
    simultaneously, provided replicates within a dose are independent draws of the planting procedure.
    No monotonicity and no multiple-testing correction are needed (fixed-sequence testing).
    """
    rows = sorted(((float(d), int(x), int(r)) for d, x, r in zip(doses, hits, trials) if float(d) > 0),
                  key=lambda t: -t[0])
    steps, lod = [], None
    for d, x, r in rows:
        lb = clopper_pearson_lower(x, r, gamma)
        certified = lb >= pi
        steps.append({'dose': d, 'hits': x, 'trials': r, 'rate': x / r if r else None, 'cp_lower': lb,
                      'certified': certified})
        if not certified:
            break
        lod = d
    if lod is None and rows and misses_allowed(rows[0][2], pi, gamma) is None:
        status = 'insufficient_replicates'   # even a perfect record at the top dose could not certify pi
    elif lod is None:
        status = 'not_detectable_in_range'
    elif lod == rows[-1][0]:
        status = 'censored_at_lowest_dose'  # true LoD may be smaller than the grid allows
    else:
        status = 'certified'
    return {'lod': lod, 'status': status, 'steps': steps, 'pi': pi, 'gamma': gamma,
            'method': 'fixed-sequence one-sided Clopper-Pearson'}


def _probit_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    if len(np.unique(y)) < 2 or len(np.unique(x)) < 2:
        return None

    def nll(params):
        a, b = params
        p = np.clip(norm.cdf(a + b * x), 1e-12, 1 - 1e-12)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    start = np.array([0.0, 1.0])
    res = minimize(nll, start, method='Nelder-Mead', options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 5000})
    a, b = res.x
    if not res.success and res.status != 2:
        return None
    return float(a), float(b)


def probit_lod(doses, outcomes, pi: float = PI, n_boot: int = 2000, seed: int = 0) -> dict:
    """Descriptive LoD: probit of detection on log10(dose), with a replicate bootstrap interval.

    ``doses`` and ``outcomes`` are replicate-level arrays (dose > 0 only are used). The bootstrap resamples
    replicates within each dose. Quoted only for figures; the official LoD is fixed_sequence_lod.
    """
    d = np.asarray(doses, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    keep = d > 0
    d, y = d[keep], y[keep]
    x = np.log10(d)
    fit = _probit_fit(x, y)

    def lod_from(params):
        if params is None or params[1] <= 0:
            return None
        return float(10 ** ((norm.ppf(pi) - params[0]) / params[1]))

    point = lod_from(fit)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(d == level) for level in np.unique(d)]
    boots = []
    for _ in range(n_boot if point is not None else 0):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        value = lod_from(_probit_fit(x[idx], y[idx]))
        if value is not None:
            boots.append(value)
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if len(boots) >= 0.9 * n_boot else None
    return {'lod_fit': point, 'lod_fit_ci': ci, 'params': fit, 'bootstrap_kept': len(boots), 'n_boot': n_boot,
            'note': 'descriptive only; the official LoD is the fixed-sequence Clopper-Pearson dose'}


def false_alarm(hits_at_zero: int, trials_at_zero: int) -> dict:
    lo, hi = wilson_interval(hits_at_zero, trials_at_zero)
    return {'hits': hits_at_zero, 'trials': trials_at_zero,
            'rate': hits_at_zero / trials_at_zero if trials_at_zero else None, 'wilson95': (lo, hi)}


# --------------------------------------------------------------------------- seeds
_CONSERVATIVE = {'Fail': 0, 'NotEvaluated': 1, 'Conditional': 2, 'Pass': 3}


def seed_agreement(verdicts: list[str]) -> dict:
    counts = Counter(verdicts)
    top = max(counts.values())
    modal = sorted([v for v, c in counts.items() if c == top], key=lambda v: _CONSERVATIVE.get(v, 9))[0]
    agree, seeds = counts[modal], len(verdicts)
    tie = sum(c == top for c in counts.values()) > 1
    if agree == seeds:
        flag = 'stable'
    elif agree == seeds - 1 and not tie:
        flag = 'mostly stable'
    else:
        flag = 'seed-dependent'
    return {'modal_verdict': modal, 'agree': agree, 'seeds': seeds, 'flag': flag, 'counts': dict(counts)}


def ranking_stability(scores_by_seed: dict) -> dict:
    """Mean pairwise Kendall tau across seeds. ``scores_by_seed``: {seed: {generator: score}}."""
    seeds = sorted(scores_by_seed)
    taus = []
    for a, b in combinations(seeds, 2):
        common = sorted(set(scores_by_seed[a]) & set(scores_by_seed[b]))
        if len(common) < 3:
            continue
        tau = kendalltau([scores_by_seed[a][g] for g in common], [scores_by_seed[b][g] for g in common]).statistic
        if np.isfinite(tau):
            taus.append(float(tau))
    return {'mean_kendall_tau': float(np.mean(taus)) if taus else None, 'pairs': len(taus)}


# --------------------------------------------------------------------------- records and text
def _fmt_fraction(f: float) -> str:
    pct = 100 * f
    return f'{pct:.3g}%' if pct >= 0.01 else f'{pct:.2g}%'


def lod_phrase(item: dict) -> str:
    phrase = FLAW_PHRASES.get(item['flaw'], item['flaw'])
    if item.get('status') == 'insufficient_replicates':
        return f'No limit for {phrase}: too few replicates to certify'
    if item.get('lod_fraction') is None:
        return f'Cannot rule out {phrase} at any tested dose'
    rows = item.get('lod_rows')
    size = f"{_fmt_fraction(item['lod_fraction'])} of rows" + (f' ({rows:,} rows)' if rows is not None else '')
    text = f'Rules out {phrase} at >= {size}'
    if item.get('status') == 'censored_at_lowest_dose':
        text += ' (the smallest dose tested; smaller may also be caught)'
    return text


def verdict_text(record: dict) -> str:
    dim = record['dimension'].capitalize()
    parts = [f"**{dim}: {record['verdict']}**"]
    seed = record.get('seed_agreement')
    if seed:
        parts[0] += f" ({seed['agree']}/{seed['seeds']} seeds)"
    ev = record.get('evidence')
    if ev and record['verdict'] in ('Fail', 'Conditional', 'Pass'):
        parts.append(evidence_text(ev) + '.')
    if record['verdict'] == 'Pass':
        lods = record.get('lod_dataset', [])
        honest = [x for x in lods if x.get('matrix', 'honest') == 'honest']
        certified = [lod_phrase(x) for x in honest if x.get('lod_fraction') is not None]
        uncertified = [lod_phrase(x) for x in honest if x.get('lod_fraction') is None]
        if certified:
            parts.append('; '.join(certified) + f', each detected with >= {PI:.0%} probability ({GAMMA:.0%} confidence).')
        if uncertified:
            parts.append('; '.join(uncertified) + '.')
        for x in record.get('lod_table_specific', []):
            parts.append('For this table: ' + lod_phrase(x) + '.')
    if record['verdict'] == 'NotEvaluated':
        parts.append('Required evidence was unavailable; no confidence statement is made.')
    return ' '.join(parts)


def dimension_record(*, dataset: str, table_id: str, dimension: str, verdict: str, evidence: dict | None,
                     lod_dataset: list | None = None, lod_table_specific: list | None = None,
                     seed_agreement_result: dict | None = None) -> dict:
    record = {'dataset': dataset, 'table_id': table_id, 'dimension': dimension, 'verdict': verdict,
              'evidence': None if verdict == 'NotEvaluated' else evidence,
              'lod_dataset': (lod_dataset or []) if verdict == 'Pass' else [],
              'lod_table_specific': (lod_table_specific or []) if verdict == 'Pass' else [],
              'seed_agreement': seed_agreement_result}
    record['text'] = verdict_text(record)
    return record


# --------------------------------------------------------------------------- self test
def self_test() -> None:
    # Spec §4.4 table.
    assert misses_allowed(50) is None and abs(clopper_pearson_lower(50, 50) - 0.942) < 1e-3
    # Too few replicates is reported as such, never as 'not detectable'.
    assert fixed_sequence_lod([0.5, 1.0], [6, 6], [6, 6])['status'] == 'insufficient_replicates'
    assert fixed_sequence_lod([0.5, 1.0], [10, 20], [100, 100])['status'] == 'not_detectable_in_range'
    r = dimension_record(dataset='x', table_id='t', dimension='privacy', verdict='Pass',
                         evidence={'e': 1.0, 'at_least_as_extreme': 20, 'n_fit': 20},
                         lod_dataset=[{'flaw': 'copy_random', 'matrix': 'honest', 'lod_fraction': None,
                                       'status': 'insufficient_replicates'}])
    assert 'each detected' not in r['text'] and 'too few replicates' in r['text'], r['text']
    assert misses_allowed(60) == 0 and misses_allowed(100) == 1
    assert misses_allowed(150) == 2 and misses_allowed(200) == 4
    assert misses_allowed(50, pi=.90) == 1 and misses_allowed(100, pi=.90) == 4
    assert misses_allowed(150, pi=.90) == 8 and misses_allowed(200, pi=.90) == 12
    # Fixed sequence stops at the first failure, even if a lower dose would pass.
    out = fixed_sequence_lod([.25, .1, .05, .01], [150, 149, 120, 150], [150] * 4)
    assert out['lod'] == .1 and out['status'] == 'certified' and len(out['steps']) == 3
    assert fixed_sequence_lod([.25, .1], [100, 100], [150, 150])['status'] == 'not_detectable_in_range'
    assert fixed_sequence_lod([.25, .1], [150, 150], [150, 150])['status'] == 'censored_at_lowest_dose'
    # Simultaneous coverage under a true detection curve (Monte Carlo).
    rng = np.random.default_rng(3)
    grid = np.array([.25, .1, .05, .025, .01, .005])
    truth = norm.cdf(2.2 * (np.log10(grid) + 2.0))  # true p(d)
    false_claims = 0
    for _ in range(2000):
        hits = rng.binomial(150, truth)
        res = fixed_sequence_lod(grid, hits, [150] * len(grid))
        if res['lod'] is not None and np.any(truth[grid >= res['lod']] < PI):
            false_claims += 1
    assert false_claims / 2000 <= 1 - GAMMA + 0.01, false_claims
    # Probit recovers a known LoD.
    d = np.repeat(grid, 400)
    y = rng.random(d.size) < np.repeat(truth, 400)
    pr = probit_lod(d, y, n_boot=200)
    true_lod = 10 ** (norm.ppf(PI) / 2.2 - 2.0)
    assert abs(np.log10(pr['lod_fit']) - np.log10(true_lod)) < 0.05, (pr['lod_fit'], true_lod)
    # Evidence strength on a 100-run calibration (Fail => e <= 5/101 is tested in calibrate.py).
    fit = rng.normal(size=(100, 4))  # STB calibrates on 100 null runs
    model = {'scale_floor': 1e-8, 'columns': [
        {'family': 'gate', 'dimension': dim, 'metric': f'm{i}', 'direction': 1, 'fit_harm': fit[:, i].tolist()}
        for i, dim in enumerate(['fidelity', 'fidelity', 'utility', 'privacy'])]}
    sig = {'fidelity': {'m0': {'estimate': 0.0}, 'm1': {'estimate': 0.1}},
           'utility': {'m2': {'estimate': 0.0}}, 'privacy': {'m3': {'estimate': 25.0}}}
    ev = evidence_strength(model, sig)
    assert ev['privacy']['e'] == 1 / 101 and ev['fidelity']['e'] > 0.2
    assert seed_agreement(['Pass'] * 5)['flag'] == 'stable'
    assert seed_agreement(['Pass'] * 4 + ['Fail'])['flag'] == 'mostly stable'
    assert seed_agreement(['Pass'] * 3 + ['Fail'] * 2)['flag'] == 'seed-dependent'
    rec = dimension_record(dataset='cdc', table_id='t', dimension='privacy', verdict='Pass',
                           evidence=ev['fidelity'],
                           lod_dataset=[{'flaw': 'copy_random', 'matrix': 'honest', 'lod_fraction': .005,
                                         'lod_rows': 254, 'status': 'certified'},
                                        {'flaw': 'near_copy', 'matrix': 'honest', 'lod_fraction': None}],
                           seed_agreement_result=seed_agreement(['Pass'] * 5))
    assert 'Rules out copying of training records at >= 0.5% of rows (254 rows)' in rec['text']
    assert 'Cannot rule out near-copies' in rec['text']


if __name__ == '__main__':
    self_test()
    print('confidence self-test passed')
