"""Planted-flaw jobs: limits of detection, their validation, and the table-specific privacy limit (spec §4-5, §7).

One job = one replicate: a single base table, one flaw family, every dose in the family's grid (nested, one
seed). Only the flaw's target dimension is re-scored; the other dimensions keep the base's dose-0 evidence
(amendment A6). Every dose stores the verdicts, evidence strength and the target dimension's raw signals, so
LoDs, variants and an equity-admitted refit can all be recomputed from disk.
"""
from __future__ import annotations

from syntrustbench_campaign.config import GATE_DIMENSIONS, seed_for
from syntrustbench_campaign.data import read_analytic

from . import competitors, dimensions
from .bases import Workspace, load_main_base, load_null_base
from .calibrate import judge
from .operators import TARGET_DIMENSION, FlawContext, dose_fraction, dose_rows, grid_for, plant


def _model(ws: Workspace) -> dict:
    return ws.store.read('calibration', 'model')['model']


def _flaw_context(ds, base) -> FlawContext:
    cache = getattr(base.context, 'privacy_real_cache', None) or {}
    encoder = cache.get('encoder')
    return FlawContext(ds.spec, base.members, base.test, encode=encoder.transform if encoder else None)


def _score_doses(*, ws, ds, base, table0, dose0, family, doses, key_prefix, flaw_seed, model, alpha,
                 competitor_names=(), competitor_dose0=None, full_evaluation=False) -> list[dict]:
    target = TARGET_DIMENSION[family]
    full_dims = tuple(dict.fromkeys(GATE_DIMENSIONS + (target,)))
    ctx, n = _flaw_context(ds, base), len(table0)
    relevant = tuple(c for c in competitor_names if competitors.STATISTICS[c].dimension == target)
    rows = []
    for dose in doses:
        if dose_fraction(dose, n) == 0:
            evidence, audit = dose0, {'rows_changed': 0}
            comp = {c: (competitor_dose0 or {}).get(c) for c in relevant}
        else:
            table, audit = plant(family, table0, dose, ctx, flaw_seed)
            if full_evaluation:   # A6 check: recompute every gating dimension, copy nothing from the base
                evidence = dimensions.evaluate(base, table, f'{key_prefix}/{dose}', dims=full_dims)
            else:
                evidence = dimensions.evaluate(base, table, f'{key_prefix}/{dose}', dims=(target,), fill_from=dose0)
            comp = competitors.score_all(relevant, base, table, ds) if relevant else {}
        verdicts = judge(model, evidence['signals'], alpha=alpha, missing=evidence['missing'],
                         hard_fail=evidence['hard_fail'])
        target_verdict = verdicts.get(target, {}).get('verdict', 'NotEvaluated')
        rows.append({'dose': dose, 'rows': dose_rows(dose, n), 'fraction': dose_fraction(dose, n),
                     'rows_changed': audit.get('rows_changed'),
                     'verdict': target_verdict,
                     'detected': target_verdict == 'Fail',
                     'detected_secondary': target_verdict in ('Fail', 'Conditional'),
                     # meaningful only under full_evaluation: the fast path copies off-target dims from dose 0
                     'off_target_fail': sorted(d for d, v in verdicts.items()
                                               if d != target and d in GATE_DIMENSIONS and v['verdict'] == 'Fail'),
                     'verdicts': {d: v['verdict'] for d, v in verdicts.items()},
                     'evidence': {d: (v['evidence'] or {}).get('e') for d, v in verdicts.items()},
                     'target_signals': evidence['signals'].get(target, {}),
                     'missing': evidence['missing'].get(target, False),
                     'competitors': comp})
    return rows


def lod_job(ws: Workspace, *, matrix: str, family: str, base_ref, purpose: str, doses=None,
            competitor_names: tuple = (), alpha: float = 0.05, full_evaluation: bool = False) -> dict:
    """One replicate (worker job, stage 'lod'). matrix: 'honest' (base_ref = null run id) or a generator
    name (base_ref = 'seed/k', the k-th fresh sample of that fitted generator)."""
    ds, model = ws.dataset(), _model(ws)
    target = TARGET_DIMENSION[family]
    if matrix == 'honest':
        base, record = load_null_base(ws, int(base_ref), ds)
        table0 = read_analytic(record['roles']['paths']['positive'], ds.cfg)
        dose0, comp0 = base.dose0, record.get('competitors')
    else:
        seed, k = map(int, str(base_ref).split('/'))
        base = load_main_base(ws, ds)
        table0 = read_analytic(ws.store.read('generators', f'{matrix}/{seed}')['matrix_paths'][k], ds.cfg)
        dose0 = dimensions.evaluate(base, table0, f'{ds.name}/{matrix}/{base_ref}/dose0',
                                    dims=GATE_DIMENSIONS)
        comp0 = None
    eligible = True
    if matrix != 'honest':
        eligible = judge(model, dose0['signals'], alpha=alpha, missing=dose0['missing'])[target]['verdict'] == 'Pass'
    grid = ('0',) + grid_for(family, len(table0)) if doses is None else tuple(doses)
    payload = {'dataset': ds.name, 'matrix': matrix, 'family': family, 'target': target, 'base_ref': str(base_ref),
               'purpose': purpose, 'n_rows': len(table0), 'eligible': eligible, 'full_evaluation': full_evaluation}
    if not eligible:
        return {**payload, 'doses': [], 'note': f'Base does not pass {target} at dose 0; excluded (§4.3).'}
    flaw_seed = seed_for('stb-v0.5', 'flaw', ds.name, matrix, family, str(base_ref), purpose)
    payload['doses'] = _score_doses(ws=ws, ds=ds, base=base, table0=table0, dose0=dose0, family=family,
                                    doses=grid, key_prefix=f'{ds.name}/{matrix}/{family}/{base_ref}/{purpose}',
                                    flaw_seed=flaw_seed, model=model, alpha=alpha,
                                    competitor_names=competitor_names if matrix == 'honest' else (),
                                    competitor_dose0=comp0, full_evaluation=full_evaluation)
    return payload


def table_lod_job(ws: Workspace, *, generator: str, seed: int, family: str, replicate: int,
                  alpha: float = 0.05) -> dict:
    """Copies planted into the very table being judged (worker job, stage 'table_lod'; spec §5)."""
    ds, model = ws.dataset(), _model(ws)
    record = ws.store.read('generators', f'{generator}/{seed}')
    dose0 = record['evidence']
    verdict = judge(model, dose0['signals'], alpha=alpha, missing=dose0['missing'])['privacy']['verdict']
    payload = {'dataset': ds.name, 'generator': generator, 'seed': seed, 'family': family, 'replicate': replicate}
    if verdict != 'Pass':
        return {**payload, 'applicable': False, 'note': f'Table privacy verdict is {verdict}; not applicable.'}
    base = load_main_base(ws, ds)
    table0 = read_analytic(record['table_path'], ds.cfg)
    flaw_seed = seed_for('stb-v0.5', 'table-lod', ds.name, generator, seed, family, replicate)
    doses = ('0',) + grid_for(family, len(table0))
    rows = _score_doses(ws=ws, ds=ds, base=base, table0=table0, dose0=dose0, family=family, doses=doses,
                        key_prefix=f'{ds.name}/gen/{generator}/{seed}/{family}/r{replicate}',
                        flaw_seed=flaw_seed, model=model, alpha=alpha)
    return {**payload, 'applicable': True, 'n_rows': len(table0), 'doses': rows}
