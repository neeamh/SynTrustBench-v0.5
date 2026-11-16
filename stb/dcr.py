"""The DCR comparison (preregistration §7): a supporting, descriptive figure, not a certified LoD.

SDMetrics' DCR metrics are the field's standard privacy check. They are too slow to score on the full LoD grid,
so they are scored at a few doses (``plan.dcr_doses``) of copy_random and near_copy on the first
``plan.dcr_bases`` estimation bases, and reported as detection rates with Wilson intervals.

Fairness. SDMetrics' own ``num_rows_subsample`` option samples BOTH the synthetic table and the real training
table down to the same number of rows, so a copied row is only seen as a copy if its donor happens to be in the
real sample. To make the comparison like-for-like, STB is also scored on that budget: its privacy dimension is
computed on a seeded ``m``-row sample of the released table against an ``m``-row sample of the training role,
and calibrated on the same kind of samples of the fit runs (stage 'dcr_nulls'). STB's full-table verdicts at the
same doses and bases come from the 'lod' stage, so the figure shows three curves: DCR, STB at DCR's budget, and
STB as shipped.
"""
from __future__ import annotations

import numpy as np

from syntrustbench_campaign.config import seed_for
from syntrustbench_campaign.data import read_analytic

from . import competitors, dimensions
from .bases import Workspace, load_null_base
from .operators import plant
from .lod import _flaw_context


def _sample_rows(n: int, m: int, *parts) -> np.ndarray:
    rng = np.random.default_rng(seed_for('stb-v0.5', 'dcr-subsample', *map(str, parts)))
    return np.sort(rng.choice(n, size=min(m, n), replace=False))


def _budget_base(base: dimensions.Base, m: int, *parts) -> dimensions.Base:
    """A copy of the base whose training role is an m-row sample (what the subsampled DCR compares against).
    The real-data privacy cache is dropped, so the privacy floor is recomputed for this training sample."""
    import copy
    small = copy.copy(base)
    rows = _sample_rows(len(base.members), m, 'train', *parts)
    small.members = base.members.iloc[rows].reset_index(drop=True)
    small.context = copy.copy(base.context)
    if hasattr(small.context, 'privacy_real_cache'):
        small.context.privacy_real_cache = None
    return small


def stb_at_budget(base, table, m: int, key: str, *parts) -> dict:
    """STB's privacy signals with the same row budget as the subsampled DCR metrics."""
    small = _budget_base(base, m, *parts)
    sample = table.iloc[_sample_rows(len(table), m, 'release', *parts)].reset_index(drop=True)
    return dimensions.evaluate(small, sample, key, dims=('privacy',))


def dcr_null_job(ws: Workspace, *, run_id: int, m: int) -> dict:
    """STB privacy at DCR's budget on an honest null table (worker job, stage 'dcr_nulls')."""
    ds = ws.dataset()
    base, record = load_null_base(ws, run_id, ds)
    table = read_analytic(record['roles']['paths']['positive'], ds.cfg)
    ev = stb_at_budget(base, table, m, f'{ds.name}/dcr-null/{run_id}', 'null', run_id)
    return {'run_id': run_id, 'm': m, 'signals': ev['signals'], 'missing': ev['missing']}


def dcr_job(ws: Workspace, *, family: str, base_id: int, doses: tuple[str, ...], statistics: tuple[str, ...],
            m: int) -> dict:
    """One base, one family, the DCR doses (worker job, stage 'dcr'). The planted tables are identical to the
    'lod' stage's (same flaw seed), so STB's full-table verdicts can be joined on (family, base, dose)."""
    ds = ws.dataset()
    base, record = load_null_base(ws, base_id, ds)
    table0 = read_analytic(record['roles']['paths']['positive'], ds.cfg)
    ctx = _flaw_context(ds, base)
    flaw_seed = seed_for('stb-v0.5', 'flaw', ds.name, 'honest', family, str(base_id), 'estimation')
    rows = []
    for dose in doses:
        table, audit = plant(family, table0, dose, ctx, flaw_seed)
        ev = stb_at_budget(base, table, m, f'{ds.name}/dcr/{family}/{base_id}/{dose}', 'base', base_id)
        rows.append({'dose': dose, 'rows_changed': audit.get('rows_changed'),
                     'competitors': competitors.score_all(statistics, base, table, ds),
                     'stb_budget_signals': ev['signals'].get('privacy', {}),
                     'stb_budget_missing': bool(ev['missing'].get('privacy', False))})
    return {'family': family, 'base_ref': str(base_id), 'm': m, 'n_rows': len(table0), 'doses': rows}
