"""STB v0.5 per-dimension evaluator with independent random streams.

Each table gets one seed, ``table_seed = seed_for('stb-v0.5', table_key)``. A numpy SeedSequence spawns one
child stream per dimension (fidelity, utility, privacy, equity, robustness), so scoring a dimension alone
uses exactly the randomness it would use inside a full profile. Calibration nulls and planted tables go
through this same code, which is what guarantees they match.

The signal implementations are the library's, unchanged. Only the random-number plumbing differs from
v0.4.2 (where utility continued fidelity's stream).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
import pickle

import numpy as np
import pandas as pd

from syntrustbench_campaign.config import DIMENSIONS, GATE_DIMENSIONS, seed_for
from syntrustbench_campaign.core import Gate, available_number
from syntrustbench_campaign.data import read_analytic
from syntrustbench_campaign.evaluation import EquityV04, PrivacyV04, RobustnessV04, signal
from syntrustbench_campaign.reference import ReferenceFidelity, ReferenceUtility

RNG_POLICY = ('v0.5: one SeedSequence per table (seed_for("stb-v0.5", table_key)) spawning independent '
              'child streams for fidelity, utility, privacy, equity and robustness, in that order')


def table_streams(table_key: str) -> dict[str, np.random.SeedSequence]:
    children = np.random.SeedSequence(seed_for('stb-v0.5', table_key)).spawn(len(DIMENSIONS))
    return dict(zip(DIMENSIONS, children))


def _int_seed(stream: np.random.SeedSequence) -> int:
    return int(stream.generate_state(1, dtype=np.uint32)[0])


@dataclass
class Base:
    """One honest base: its roles, its cached real-data context, and (optionally) its dose-0 signals."""
    cfg: object
    roles: dict
    context: object                                     # ReferenceContext with privacy_real_cache attached
    members: pd.DataFrame = field(repr=False, default=None)
    test: pd.DataFrame = field(repr=False, default=None)
    dose0: dict | None = None                           # evidence of the unflawed base table

    @classmethod
    def load(cls, cfg, roles: dict, context_path: str) -> 'Base':
        with open(context_path, 'rb') as handle:        # only trusted, locally produced artifacts
            ctx = pickle.load(handle)
        members, test = [read_analytic(roles['paths'][k], cfg) for k in ('train', 'test')]
        return cls(cfg, roles, ctx, members, test)


def evaluate(base: Base, synthetic: pd.DataFrame, table_key: str, dims=GATE_DIMENSIONS, *,
             fill_from: dict | None = None, smoke: bool = False) -> dict:
    """Score ``dims`` on ``synthetic``. Dimensions not scored are copied from ``fill_from`` (usually the
    base's dose-0 evidence) and listed under 'filled_from_base'. Raw signals are always returned."""
    cfg, ctx = base.cfg, base.context
    streams = table_streams(table_key)
    dims = tuple(d for d in DIMENSIONS if d in set(dims))
    if 'robustness' in dims and 'utility' not in dims:
        dims = tuple(d for d in DIMENSIONS if d in set(dims) | {'utility'})
    results, signals = {}, {}

    f = u = None
    if 'fidelity' in dims:
        c = copy.copy(ctx); c.rng = np.random.default_rng(streams['fidelity'])
        f = ReferenceFidelity(c).evaluate(synthetic).to_dict()
    if 'utility' in dims:
        c = copy.copy(ctx); c.rng = np.random.default_rng(streams['utility'])
        u = ReferenceUtility(c).evaluate(synthetic).to_dict()
    if f is not None or u is not None:
        ref = {'fidelity': fidelity_signals(f, ctx) if f else None, 'utility': utility_signals(u, ctx) if u else None}
        for dim, value in (('fidelity', f), ('utility', u)):
            if value is None:
                continue
            signals[dim] = ref[dim]
            unavailable = [k for k, v in ref[dim].items() if v['status'] == 'NotEvaluated']
            if dim == 'fidelity':
                unavailable += [f'column:{col}' for col in cfg.data.continuous
                                if any(not pd.to_numeric(fr[col], errors='coerce').notna().any()
                                       for fr in (synthetic, ctx.reference, ctx.train))]
            if unavailable:
                value['status'] = Gate.aggregate([value['status'], 'NotEvaluated'])
            value['missing_required_evidence'] = unavailable
            results[dim] = value
    if 'privacy' in dims:
        p = PrivacyV04(cfg, base.members, base.test, _int_seed(streams['privacy']), smoke,
                       getattr(ctx, 'privacy_real_cache', None)).evaluate(synthetic).to_dict()
        signals['privacy'] = p.pop('signals'); results['privacy'] = p
    if 'equity' in dims:
        e = EquityV04(cfg, ctx, _int_seed(streams['equity'])).evaluate(synthetic).to_dict()
        signals['equity'] = e.pop('signals'); results['equity'] = e
    if 'robustness' in dims:
        start = np.random.default_rng(streams['utility']).bit_generator.state
        r = RobustnessV04(cfg, ctx, results['utility'], start, _int_seed(streams['robustness']),
                          smoke).evaluate(synthetic).to_dict()
        signals['robustness'] = r.pop('signals'); results['robustness'] = r

    filled = []
    for dim in DIMENSIONS:
        if dim not in signals and fill_from and dim in fill_from.get('signals', {}):
            signals[dim] = copy.deepcopy(fill_from['signals'][dim])
            results[dim] = copy.deepcopy(fill_from['dimensions'][dim])
            filled.append(dim)
    missing = {d: (results[d]['status'] == 'NotEvaluated' or bool(results[d].get('missing_required_evidence')))
               for d in results}
    hard_fail = {'utility': bool(synthetic[cfg.data.target].nunique() < 2) if 'utility' in dims else
                 bool((fill_from or {}).get('hard_fail', {}).get('utility', False)),
                 'robustness': bool(results.get('robustness', {}).get('verdict_flip', False))}
    return {'table_key': table_key, 'scored': list(dims), 'filled_from_base': filled, 'signals': signals,
            'dimensions': results, 'missing': missing, 'hard_fail': hard_fail, 'rng_policy': RNG_POLICY,
            'stb_version': '0.5'}


def fidelity_signals(f: dict, ctx) -> dict:
    """Same as the library's reference_signals, fidelity half."""
    fm = f['metrics']
    out = {k: signal(fm[k].get('estimate')) for k in (
        'mass_weighted_coverage_deficit', 'mean_normalized_wasserstein_excess',
        'mean_jensen_shannon_divergence_excess', 'association_matrix_error_excess',
        'clinical_constraint_violation_rate')}
    out['detection_auc_excess'] = signal(fm['detection_auc'].get('estimate') - ctx.floor['detection_auc'])
    return out


def utility_signals(u: dict, ctx) -> dict:
    """Same as the library's reference_signals, utility half."""
    um, out = u['metrics'], {}
    for probe in ctx.PROBES:
        m = um[f'synthetic_retention_{probe}']
        out[f'synthetic_retention_{probe}'] = signal(m['ci_high'] if m['status'] != 'NotEvaluated' else None,
                                                    higher=False,
                                                    reason='Retention upper CI unavailable or weak real-trained baseline.')
    for key in ('relative_calibration_in_the_large', 'relative_calibration_slope'):
        out[key + '_high'] = signal(um[key].get('estimate'))
        out[key + '_low'] = signal(um[key].get('estimate'), higher=False)
    return out
