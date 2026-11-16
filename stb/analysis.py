"""From result files to the paper's evidence. Pure functions of saved results; nothing is re-scored.

    outcomes_frame        one row per (replicate, dose) from the 'lod' stage
    lod_summary           official LoD (fixed-sequence Clopper-Pearson), probit curve, dose-0 false alarms
    validation_doses      the grid doses around each LoD that the fresh validation replicates use
    validation_summary    observed detection at those doses on fresh bases (§7.1)
    transfer_gap          honest-matrix LoD checked on generator matrices (§7.3)
    head_to_head          LoD per benchmark statistic at a matched false-alarm rule, and ratios (§7.6)
    null_false_alarms     honest non-acceptance on bases (H3)
    generator_table       verdicts, evidence and seed agreement for real generators (§6)
    dimension_records     the per-dimension JSON records and verdict text (§10)
    equity_admission      the preregistered admission rule (§8)
    a6_agreement          fast (target-only) vs full re-scoring on the A6 check replicates
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import beta

from syntrustbench_campaign.config import GATE_DIMENSIONS

from . import competitors
from .calibrate import judge, single_dimension_model, verdict_of
from .confidence import (dimension_record, false_alarm, fixed_sequence_lod, probit_lod, ranking_stability,
                         seed_agreement, wilson_interval)
from .operators import GRIDS, SINGLE_DOSE, TARGET_DIMENSION


# ------------------------------------------------------------------------------------------- outcomes
def outcomes_frame(store, stage: str = 'lod') -> pd.DataFrame:
    rows = []
    for job in store.iter(stage):
        for d in job.get('doses', []):
            rows.append({'matrix': job.get('matrix', 'table'), 'family': job['family'],
                         'target': TARGET_DIMENSION[job['family']], 'base_ref': job.get('base_ref'),
                         'purpose': job.get('purpose', 'table'), 'eligible': job.get('eligible', True),
                         'n_rows': job.get('n_rows'), 'generator': job.get('generator'), 'seed': job.get('seed'),
                         **{k: d[k] for k in ('dose', 'rows', 'fraction', 'verdict', 'detected',
                                              'detected_secondary', 'missing')},
                         'off_target_fail': ','.join(d['off_target_fail']),
                         'target_signals': d['target_signals'], 'competitors': d.get('competitors', {})})
    return pd.DataFrame(rows)


def _curve(frame: pd.DataFrame, outcome: str = 'detected') -> pd.DataFrame:
    """Hits and trials per dose (NotEvaluated counts as a miss: an unscorable table is not a detection)."""
    g = frame.groupby(['dose', 'fraction', 'rows'], as_index=False).agg(hits=(outcome, 'sum'), trials=(outcome, 'size'))
    return g.sort_values('fraction')


def _lod_from(frame: pd.DataFrame, pi: float, gamma: float, outcome: str = 'detected') -> dict:
    curve = _curve(frame, outcome)
    positive = curve[curve.fraction > 0]
    official = fixed_sequence_lod(positive.fraction, positive.hits, positive.trials, pi, gamma)
    lod_rows = None
    if official['lod'] is not None:
        lod_rows = int(positive.loc[positive.fraction == official['lod'], 'rows'].iloc[0])
    replicate = frame[frame.fraction > 0]
    fit = probit_lod(replicate.fraction, replicate[outcome].astype(float), pi, n_boot=2000)
    zero = curve[curve.fraction == 0]
    fa = false_alarm(int(zero.hits.sum()), int(zero.trials.sum())) if len(zero) else None
    return {'lod_fraction': official['lod'], 'lod_rows': lod_rows, 'status': official['status'],
            'steps': official['steps'], 'lod_fit': fit['lod_fit'], 'lod_fit_ci': fit['lod_fit_ci'],
            'false_alarm': fa, 'curve': curve.to_dict('records')}


def lod_summary(df: pd.DataFrame, pi: float = 0.95, gamma: float = 0.95, purpose: str = 'estimation') -> pd.DataFrame:
    out = []
    for (matrix, family), frame in df[df.purpose == purpose].groupby(['matrix', 'family']):
        bases = frame.base_ref.nunique()
        eligible = frame[frame.eligible]
        row = {'matrix': matrix, 'family': family, 'target': TARGET_DIMENSION[family],
               'bases': bases, 'eligible_bases': eligible.base_ref.nunique()}
        if row['eligible_bases'] == 0:
            out.append({**row, 'status': 'matrix not evaluable'})
            continue
        out.append({**row, **_lod_from(eligible, pi, gamma)})
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------- validation
def validation_doses(family: str, lod_fraction: float | None, n_rows: int) -> tuple[str, ...]:
    """The LoD dose and its grid neighbours (stand-ins for half and double the LoD, spec §7.1)."""
    from .operators import dose_fraction, grid_for
    grid = grid_for(family, n_rows)
    if lod_fraction is None:
        return grid[-2:]  # not detectable in range: re-check the two highest doses
    idx = min(range(len(grid)), key=lambda i: abs(dose_fraction(grid[i], n_rows) - lod_fraction))
    return tuple(grid[max(0, idx - 1): idx + 2])


def validation_summary(df: pd.DataFrame, lods: pd.DataFrame, pi: float = 0.95) -> pd.DataFrame:
    rows = []
    for _, lod in lods[lods.matrix == 'honest'].iterrows():
        v = df[(df.purpose == 'validation') & (df.matrix == 'honest') & (df.family == lod.family) & (df.fraction > 0)]
        for (dose, fraction), g in v.groupby(['dose', 'fraction']):
            hits, trials = int(g.detected.sum()), len(g)
            lo, hi = wilson_interval(hits, trials)
            at_lod = lod.get('lod_fraction') is not None and np.isclose(fraction, lod.lod_fraction)
            rows.append({'family': lod.family, 'dose': dose, 'fraction': fraction, 'hits': hits, 'trials': trials,
                         'rate': hits / trials, 'wilson95': (lo, hi), 'at_lod': at_lod,
                         'pass': (hi >= pi) if at_lod else None})
    return pd.DataFrame(rows)


def transfer_gap(df: pd.DataFrame, lods: pd.DataFrame) -> pd.DataFrame:
    rows = []
    honest = lods[lods.matrix == 'honest'].set_index('family')
    for _, row in lods[lods.matrix != 'honest'].iterrows():
        if row.family not in honest.index or honest.loc[row.family].get('lod_fraction') is None:
            continue
        lod = honest.loc[row.family, 'lod_fraction']
        g = df[(df.matrix == row.matrix) & (df.family == row.family) & (df.purpose == 'estimation') & df.eligible
               & np.isclose(df.fraction, lod)]
        if len(g):
            hits, trials = int(g.detected.sum()), len(g)
            rows.append({'matrix': row.matrix, 'family': row.family, 'honest_lod': lod, 'hits': hits,
                         'trials': trials, 'rate': hits / trials, 'wilson95': wilson_interval(hits, trials)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------- head-to-head
def head_to_head(df: pd.DataFrame, stb_model: dict, comp_model: dict, alpha: float = 0.05,
                 pi: float = 0.95, gamma: float = 0.95) -> pd.DataFrame:
    """LoD of every benchmark statistic on the honest matrix, each calibrated by the same conformal rule."""
    rows = []
    base = df[(df.matrix == 'honest') & (df.purpose == 'estimation')]
    for family, frame in base.groupby('family'):
        target = TARGET_DIMENSION[family]
        if target not in ('privacy', 'fidelity'):
            continue
        # Matched bases: where competitors were scored (on a budgeted subset of bases), every benchmark,
        # STB included, is evaluated on exactly those bases, so no LoD is computed from more replicates.
        positive = frame[frame.fraction > 0]
        scored = positive.competitors.map(lambda c: bool(c))
        if scored.any():
            frame = frame[frame.base_ref.isin(set(positive.base_ref[scored]))]
        roles = {'STB (gate, as shipped)': 'stb', f'STB ({target} only)': 'stb'}
        variants = {'STB (gate, as shipped)': frame.detected}
        single = single_dimension_model(stb_model, target)
        variants[f'STB ({target} only)'] = frame.target_signals.map(
            lambda s: verdict_of(judge(single, {target: s}), target) == 'Fail')
        present = set().union(*[set(c) for c in positive.competitors if isinstance(c, dict)]) if len(positive) else set()
        for name, stat in competitors.STATISTICS.items():
            if stat.dimension != target or name not in comp_model or name not in present:
                continue
            roles[f'{stat.benchmark}: {name}'] = stat.role + (f' (subsample {stat.subsample} rows)' if stat.subsample else '')
            variants[f'{stat.benchmark}: {name}'] = frame.competitors.map(
                lambda c, n=name: competitors.decide(comp_model, n, (c.get(n) or {}).get('value'), alpha) == 'Fail')
        stb_ref = None
        for label, detected in variants.items():
            res = _lod_from(frame.assign(detected=detected.astype(bool)), pi, gamma)
            if label.startswith(f'STB ({target}'):
                stb_ref = res['lod_fraction']
            rows.append({'family': family, 'benchmark': label, 'role': roles[label],
                         'bases': int(frame.base_ref.nunique()), 'lod_fraction': res['lod_fraction'],
                         'lod_rows': res['lod_rows'], 'status': res['status'], 'false_alarm': res['false_alarm']})
    table = pd.DataFrame(rows)
    if len(table):
        ref = table[table.benchmark.str.contains(' only)', regex=False)].set_index('family').lod_fraction
        table['ratio_vs_stb'] = [(r.lod_fraction / ref.get(r.family)) if r.lod_fraction and ref.get(r.family) else None
                                 for r in table.itertuples()]
    return table


# ------------------------------------------------------------------------------------------- nulls and generators
def null_false_alarms(store, model: dict, ids, alpha: float = 0.05,
                      roles: tuple[str, ...] = ('estimation', 'validation')) -> pd.DataFrame:
    """H3 (preregistration §9). Honest Fails on bases never used for fitting, per role and pooled ('all').

    The gate's pooled Fail count is tested against the nominal bound r/(n_fit + 1) with an exact one-sided binomial
    test at level 0.05 (H3 holds if it does not reject); each dimension gets an exact one-sided 95% upper bound only.
    NotEvaluated verdicts are counted separately and never as Fails. For a gate refitted after equity admission,
    pass roles=('validation',): the estimation bases were used to admit it.
    """
    from scipy.stats import binom
    n_fit = len(model['columns'][0]['fit_harm'])
    bound = int(np.floor(alpha * (n_fit + 1))) / (n_fit + 1)
    by_role = {}
    for role in roles:
        verdicts = []
        for run_id in getattr(ids, role):
            if not store.done('nulls', f'{run_id:03d}'):
                continue
            ev = store.read('nulls', f'{run_id:03d}')['evidence']
            verdicts.append(judge(model, ev['signals'], alpha=alpha, missing=ev['missing']))
        by_role[role] = verdicts
    if len(roles) > 1:
        by_role['all'] = [v for role in roles for v in by_role[role]]
    rows = []
    for role, verdicts in by_role.items():
        n = len(verdicts)
        for dim in GATE_DIMENSIONS + ('gate',):
            if dim == 'gate':
                fails = sum(any(v[d]['verdict'] == 'Fail' for d in GATE_DIMENSIONS) for v in verdicts)
                not_evaluated = sum(any(v[d]['verdict'] == 'NotEvaluated' for d in GATE_DIMENSIONS) for v in verdicts)
            else:
                fails = sum(v.get(dim, {}).get('verdict') == 'Fail' for v in verdicts)
                not_evaluated = sum(v.get(dim, {}).get('verdict') == 'NotEvaluated' for v in verdicts)
            upper = float(beta.ppf(0.95, fails + 1, n - fails)) if n and fails < n else 1.0
            pooled_gate = role == ('all' if len(roles) > 1 else roles[0]) and dim == 'gate'
            p_value = float(binom.sf(fails - 1, n, bound)) if (pooled_gate and n) else None
            rows.append({'bases': role, 'dimension': dim, 'fails': fails, 'not_evaluated': not_evaluated, 'runs': n,
                         'rate': fails / n if n else None, 'exact_upper95': upper, 'nominal_bound': bound,
                         'binomial_p': p_value,
                         'h3_holds': (p_value >= 0.05) if p_value is not None else None})
    return pd.DataFrame(rows)


def generator_table(store, model: dict, alpha: float = 0.05) -> pd.DataFrame:
    rows = []
    for job in store.iter('generators'):
        ev = job['evidence']
        res = judge(model, ev['signals'], alpha=alpha, missing=ev['missing'], hard_fail=ev['hard_fail'])
        for dim, r in res.items():
            ev_dim = r['evidence'] or {}
            rows.append({'generator': job['generator'], 'seed': job['seed'], 'dimension': dim, 'family': r['family'],
                         'verdict': r['verdict'], 'e': ev_dim.get('e'), 'at_least': ev_dim.get('at_least_as_extreme'),
                         'n_fit': ev_dim.get('n_fit'), 'max_z': r['max_z']})
    return pd.DataFrame(rows)


def seed_summary(gen: pd.DataFrame, unstable_below: float = 2 / 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Seed agreement per generator and dimension, and H4 ranking stability (preregistration §9): a dimension's
    ranking is unstable if the mean Kendall tau-b over seed pairs is below 2/3."""
    agreement = [{'generator': g, 'dimension': d, **seed_agreement(list(frame.sort_values('seed').verdict))}
                 for (g, d), frame in gen.groupby(['generator', 'dimension'])]
    stability = []
    for dim, frame in gen.groupby('dimension'):
        scores = {s: dict(zip(f.generator, -f.max_z)) for s, f in frame.groupby('seed')}  # higher = better
        result = ranking_stability(scores)
        tau = result['mean_kendall_tau']
        stability.append({'benchmark': 'STB', 'dimension': dim, **result,
                          'unstable': (tau < unstable_below) if tau is not None else None})
    return pd.DataFrame(agreement), pd.DataFrame(stability)



def competitor_ranking_stability(store, comp_model: dict, unstable_below: float = 2 / 3) -> pd.DataFrame:
    """H4 for SDMetrics: rank generators within each seed by each calibrated statistic (oriented so that larger is
    worse, as in calibration) and apply the same mean Kendall tau-b threshold as for STB."""
    by_stat = defaultdict(lambda: defaultdict(dict))
    for job in store.iter('generators'):
        for name, score in (job.get('competitors') or {}).items():
            if name in comp_model and score.get('available'):
                harm = comp_model[name]['direction'] * score['value']
                by_stat[name][job['seed']][job['generator']] = -harm   # higher = better
    rows = []
    for name, scores in by_stat.items():
        result = ranking_stability(scores)
        tau = result['mean_kendall_tau']
        rows.append({'benchmark': f'SDMetrics: {name}', 'dimension': comp_model[name]['dimension'], **result,
                     'unstable': (tau < unstable_below) if tau is not None else None})
    return pd.DataFrame(rows)

# ------------------------------------------------------------------------------------------- records
def dimension_records(dataset: str, gen: pd.DataFrame, lods: pd.DataFrame, table_lods: pd.DataFrame | None,
                      agreement: pd.DataFrame) -> list[dict]:
    records = []
    for row in gen[gen.dimension.isin(GATE_DIMENSIONS)].itertuples():
        # The confidence attached to a Pass is the honest-matrix LoD of each graded family. Generator-matrix LoDs
        # are transfer checks (reported separately), and single-dose families are detection rates, not limits.
        families = lods[(lods.target == row.dimension) & (lods.matrix == 'honest') & ~lods.family.isin(SINGLE_DOSE)]
        lod_dataset = [{'flaw': f.family, 'matrix': f.matrix, 'lod_fraction': f.get('lod_fraction'),
                        'lod_rows': f.get('lod_rows'), 'status': f.get('status'), 'lod_fit': f.get('lod_fit'),
                        'lod_fit_ci': f.get('lod_fit_ci')} for _, f in families.iterrows()]
        table_specific = []
        if table_lods is not None and len(table_lods) and row.dimension == 'privacy':
            mine = table_lods[(table_lods.generator == row.generator) & (table_lods.seed == row.seed)]
            table_specific = [{'flaw': f.family, 'lod_fraction': f.get('lod_fraction'), 'lod_rows': f.get('lod_rows'),
                               'status': f.get('status')} for _, f in mine.iterrows()]
        seeds = agreement[(agreement.generator == row.generator) & (agreement.dimension == row.dimension)]
        records.append(dimension_record(
            dataset=dataset, table_id=f'{row.generator}_seed{row.seed}', dimension=row.dimension, verdict=row.verdict,
            evidence={'e': row.e, 'at_least_as_extreme': int(row.at_least), 'n_fit': int(row.n_fit)}
                     if row.e is not None and row.at_least is not None else None,
            lod_dataset=lod_dataset, lod_table_specific=table_specific,
            seed_agreement_result=seeds.iloc[0].to_dict() if len(seeds) else None))
    return records


def table_lod_summary(store, pi: float = 0.95, gamma: float = 0.95) -> pd.DataFrame:
    df = outcomes_frame(store, 'table_lod')
    rows = []
    for (g, s, family), frame in df.groupby(['generator', 'seed', 'family']) if len(df) else []:
        rows.append({'generator': g, 'seed': s, 'family': family, **_lod_from(frame, pi, gamma)})
    return pd.DataFrame(rows)


def equity_admission(store, model: dict, lods: pd.DataFrame, ids, alpha: float = 0.05) -> dict:
    """Preregistration §8: (a) evaluable everywhere, (b) no excess honest non-acceptance, (c) an official LoD."""
    from syntrustbench_campaign.calibration import binomial_admission
    fit_ok = all(store.read('nulls', f'{i:03d}')['evidence']['signals'].get('equity') for i in ids.fit
                 if store.done('nulls', f'{i:03d}'))
    verdicts = [judge(model, store.read('nulls', f'{i:03d}')['evidence']['signals'], alpha=alpha).get('equity', {})
                .get('verdict', 'NotEvaluated') for i in ids.estimation if store.done('nulls', f'{i:03d}')]
    evaluable = fit_ok and all(v != 'NotEvaluated' for v in verdicts)
    fails = sum(v == 'Fail' for v in verdicts)
    pvalue, no_excess = binomial_admission(fails, len(verdicts), 0.05, 0.05) if verdicts else (None, False)
    lod = lods[(lods.matrix == 'honest') & (lods.family == 'subgroup_degrade')]
    has_lod = bool(len(lod)) and lod.iloc[0].get('lod_fraction') is not None
    return {'evaluable': evaluable, 'honest_fails': fails, 'bases': len(verdicts), 'binomial_p': pvalue,
            'no_excess': no_excess, 'official_lod': has_lod, 'admitted': bool(evaluable and no_excess and has_lod)}


# ------------------------------------------------------------------------------------------- A6 check
def a6_agreement(store) -> pd.DataFrame:
    """Amendment A6: does scoring only the target dimension (other gating dimensions copied from the base's
    dose 0) give the same TARGET verdict as a full re-scoring? This is what the LoDs depend on, and the A6
    criterion (>= 99% per family) applies to it. Off-target agreement is shown for information only: the fast
    path copies the other dimensions from the unflawed base, so it cannot see a flaw's real off-target effects
    (use ``h2_attribution`` for those)."""
    full = outcomes_frame(store, 'a6')
    if full.empty:
        return pd.DataFrame()
    fast = outcomes_frame(store, 'lod')
    fast = fast[(fast.matrix == 'honest') & (fast.purpose == 'estimation')]
    key = ['family', 'base_ref', 'dose']
    m = full[key + ['verdict', 'off_target_fail']].merge(fast[key + ['verdict', 'off_target_fail']], on=key,
                                                          suffixes=('_full', '_fast'))
    m['target_agree'] = m.verdict_full == m.verdict_fast
    m['off_target_agree'] = m.off_target_fail_full == m.off_target_fail_fast
    out = m.groupby('family').agg(tables=('target_agree', 'size'), target_agree=('target_agree', 'sum'),
                                  off_target_agree=('off_target_agree', 'sum')).reset_index()
    out['target_rate'] = out.target_agree / out.tables
    out['target_wilson95'] = [wilson_interval(int(a), int(t)) for a, t in zip(out.target_agree, out.tables)]
    out['a6_holds'] = out.target_rate >= 0.99
    return out


def h2_attribution(store) -> pd.DataFrame:
    """H2, descriptive: on the fully re-scored replicates (stage 'a6'), how often each gating dimension fails,
    per family and dose. The target column is the flaw's own dimension; the others show co-firing. Only full
    evaluations are used, because the fast path cannot observe off-target effects."""
    full = outcomes_frame(store, 'a6')
    if full.empty:
        return pd.DataFrame()
    full = full[full.fraction > 0]
    rows = []
    for (family, dose, fraction), g in full.groupby(['family', 'dose', 'fraction']):
        row = {'family': family, 'target': TARGET_DIMENSION[family], 'dose': dose, 'fraction': fraction,
               'tables': len(g), f'fails_target': float(g.detected.mean())}
        for dim in GATE_DIMENSIONS:
            if dim != TARGET_DIMENSION[family]:
                row[f'fails_{dim}'] = float(g.off_target_fail.str.split(',').map(lambda xs, d=dim: d in xs).mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(['family', 'fraction'])


# ------------------------------------------------------------------------------------------- DCR comparison
def dcr_comparison(store, comp_model: dict, stb_model: dict, budget_model: dict, alpha: float = 0.05) -> pd.DataFrame:
    """Supporting figure (§7): detection rates, with Wilson intervals, at the DCR doses on the DCR bases, for
    each DCR statistic, STB at DCR's row budget, and STB as shipped (full table, from the 'lod' stage).
    Every curve is calibrated by the same full-conformal rule on the same fit runs. Not an LoD."""
    rows = []
    lod = outcomes_frame(store)
    lod = lod[(lod.matrix == 'honest') & (lod.purpose == 'estimation')]
    for job in store.iter('dcr'):
        family, base = job['family'], job['base_ref']
        for d in job['doses']:
            full = lod[(lod.family == family) & (lod.base_ref == base) & (lod.dose == d['dose'])]
            entry = {'family': family, 'base_ref': base, 'dose': d['dose'],
                     'STB (as shipped, full table)': bool(full.detected.iloc[0]) if len(full) else None,
                     f"STB (privacy, {job['m']}-row budget)": verdict_of(
                         judge(budget_model, {'privacy': d['stb_budget_signals']}), 'privacy') == 'Fail'}
            for name, value in d['competitors'].items():
                entry[f'SDMetrics: {name}'] = competitors.decide(comp_model, name, (value or {}).get('value'), alpha) == 'Fail'
            rows.append(entry)
    if not rows:
        return pd.DataFrame()
    wide = pd.DataFrame(rows)
    long = wide.melt(id_vars=['family', 'base_ref', 'dose'], var_name='benchmark', value_name='detected').dropna()
    out = []
    for (family, benchmark, dose), g in long.groupby(['family', 'benchmark', 'dose']):
        hits, trials = int(g.detected.astype(bool).sum()), len(g)
        out.append({'family': family, 'benchmark': benchmark, 'dose': dose, 'fraction': float(dose), 'hits': hits,
                    'trials': trials, 'rate': hits / trials, 'wilson95': wilson_interval(hits, trials)})
    return pd.DataFrame(out).sort_values(['family', 'benchmark', 'fraction'])
