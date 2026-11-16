"""Turn the plan into job lists, one function per experiment stage. The notebook calls these in order."""
from __future__ import annotations

import hashlib
import json

from syntrustbench_campaign.config import GATE_DIMENSIONS

from . import competitors
from .bases import Workspace, dims_for_null, main_roles_job, null_role, null_run_job
from .calibrate import fit_calibration
from .generators import generator_job
from .dcr import dcr_job, dcr_null_job
from .lod import lod_job, table_lod_job
from .operators import SINGLE_DOSE, TARGET_DIMENSION
from .plan import Plan
from .runner import Job


def null_jobs(ws: Workspace, plan: Plan) -> list[Job]:
    jobs = []
    dcr_runs = set(plan.ids.fit) | set(plan.ids.estimation[:plan.dcr_bases])
    for run_id in plan.ids.all:
        role = null_role(run_id, plan.ids)
        # DCR is slow: scored only where it is needed (fit runs to calibrate it, DCR bases for dose 0).
        names = tuple(c for c in plan.competitors if c not in plan.dcr_competitors or run_id in dcr_runs)
        jobs.append(Job('nulls', f'{run_id:03d}', null_run_job,
                        dict(ws=ws, run_id=run_id, role=role, dims=dims_for_null(role, plan.robustness_on),
                             competitor_names=names)))
    return jobs


def dcr_null_jobs(ws: Workspace, plan: Plan) -> list[Job]:
    """STB privacy at DCR's row budget on the fit runs (to calibrate it) and the DCR bases (dose 0)."""
    ids = plan.ids.fit + plan.ids.estimation[:plan.dcr_bases]
    return [Job('dcr_nulls', f'{i:03d}', dcr_null_job, dict(ws=ws, run_id=i, m=plan.dcr_rows)) for i in ids]


def dcr_jobs(ws: Workspace, plan: Plan) -> list[Job]:
    return [Job('dcr', f'{family}/{b:03d}', dcr_job,
                dict(ws=ws, family=family, base_id=b, doses=plan.dcr_doses, statistics=plan.dcr_competitors,
                     m=plan.dcr_rows))
            for family in plan.dcr_families for b in plan.ids.estimation[:plan.dcr_bases]]


def calibrate_budget(ws: Workspace, plan: Plan) -> dict:
    """Calibration of STB-at-DCR's-budget (privacy only) on the subsampled fit runs."""
    records = {i: ws.store.read('dcr_nulls', f'{i:03d}') for i in plan.ids.fit}
    model = fit_calibration({i: {'privacy': r['signals'].get('privacy', {})} for i, r in records.items()},
                            plan.ids.fit)
    ws.store.write('calibration', 'stb_budget', {'model': model})
    return {'n_fit': model['stb']['n_fit'], 'blocked_dimensions': model['blocked_dimensions']}


def calibrate(ws: Workspace, plan: Plan) -> dict:
    """Fit STB's calibration on its own fit runs, and the competitors' on the same runs (not parallel)."""
    records = {i: ws.store.read('nulls', f'{i:03d}') for i in plan.ids.fit}
    model = fit_calibration({i: r['evidence']['signals'] for i, r in records.items()}, plan.ids.fit,
                            plan.ids.estimation + plan.ids.validation)
    digest = hashlib.sha256(json.dumps(model, sort_keys=True).encode()).hexdigest()
    ws.store.write('calibration', 'model', {'model': model, 'sha256': digest})
    comp = competitors.calibrate({name: [(r['competitors'].get(name) or {}).get('value') for r in records.values()]
                                  for name in plan.competitors})
    ws.store.write('calibration', 'competitors', {'model': comp})
    return {'sha256': digest, 'n_fit': model['stb']['n_fit'], 'rejection_rank': model['stb']['rejection_rank'],
            'blocked_dimensions': model['blocked_dimensions'], 'excluded_signals': model['excluded']}


def main_jobs(ws: Workspace) -> list[Job]:
    return [Job('main', 'roles', main_roles_job, dict(ws=ws))]


def generator_jobs(ws: Workspace, plan: Plan) -> list[Job]:
    dims = GATE_DIMENSIONS + ('equity',) + (('robustness',) if 'generator' in plan.robustness_on else ())
    return [Job('generators', f'{g}/{s}', generator_job,
                dict(ws=ws, generator=g, seed=s, epochs=plan.epochs,
                     extra_samples=plan.samples_per_generator_seed if g in plan.matrix_generators else 0,
                     dims=dims, competitor_names=plan.competitors, allow_smoke_only=not plan.evidence))
            for g in plan.generators for s in plan.seeds]


def honest_lod_jobs(ws: Workspace, plan: Plan, families=None) -> list[Job]:
    jobs = []
    for family in families or plan.honest_families:
        for i, base_id in enumerate(plan.estimation_bases(family)):
            scored = family in plan.competitor_families and i < plan.competitor_replicates
            jobs.append(Job('lod', f'honest/{family}/{base_id:03d}/estimation', lod_job,
                            dict(ws=ws, matrix='honest', family=family, base_ref=base_id, purpose='estimation',
                                 doses=('1.0',) if family in SINGLE_DOSE else None,
                                 competitor_names=plan.full_grid_competitors if scored else (), alpha=plan.alpha)))
    return jobs


def a6_jobs(ws: Workspace, plan: Plan) -> list[Job]:
    """Amendment A6 check: a fixed random 5% of honest estimation replicates per family, re-scored with a FULL
    evaluation (every gating dimension recomputed, nothing copied from the base). Same flaw seed and table keys
    as the 'lod' stage, so the planted tables and random streams are identical; only the A6 shortcut differs."""
    import math

    import numpy as np
    jobs = []
    for family in plan.honest_families:
        bases = plan.estimation_bases(family)
        k = max(1, math.ceil(plan.a6_fraction * len(bases)))
        seed = int(hashlib.sha256(f'stb-v0.5/a6/{family}'.encode()).hexdigest()[:8], 16)
        picks = sorted(np.random.default_rng(seed).choice(len(bases), size=k, replace=False))
        for i in picks:
            base_id = bases[i]
            jobs.append(Job('a6', f'honest/{family}/{base_id:03d}/estimation', lod_job,
                            dict(ws=ws, matrix='honest', family=family, base_ref=base_id, purpose='estimation',
                                 doses=('1.0',) if family in SINGLE_DOSE else None, alpha=plan.alpha,
                                 full_evaluation=True)))
    return jobs


def matrix_lod_jobs(ws: Workspace, plan: Plan) -> list[Job]:
    jobs = []
    for g in plan.matrix_generators:
        refs = [f'{s}/{k}' for s in plan.seeds for k in range(plan.samples_per_generator_seed)]
        for family in plan.generator_matrix_families:
            for ref in refs[:plan.replicates_for(family)]:
                jobs.append(Job('lod', f'{g}/{family}/{ref}/estimation', lod_job,
                                dict(ws=ws, matrix=g, family=family, base_ref=ref, purpose='estimation',
                                     alpha=plan.alpha)))
    return jobs


def table_lod_jobs(ws: Workspace, plan: Plan) -> list[Job]:
    return [Job('table_lod', f'{g}/{s}/{family}/r{r:03d}', table_lod_job,
                dict(ws=ws, generator=g, seed=s, family=family, replicate=r, alpha=plan.alpha))
            for g in plan.generators for s in plan.seeds for family in plan.table_specific_families
            for r in range(plan.table_specific_replicates)]


def validation_jobs(ws: Workspace, plan: Plan, doses_by_family: dict) -> list[Job]:
    """Fresh bases (validation ids) at the doses chosen from the estimation LoDs."""
    jobs = []
    for family, doses in doses_by_family.items():
        for base_id in plan.ids.validation[:plan.validation_replicates]:
            jobs.append(Job('lod', f'honest/{family}/{base_id:03d}/validation', lod_job,
                            dict(ws=ws, matrix='honest', family=family, base_ref=base_id, purpose='validation',
                                 doses=('0',) + tuple(doses), alpha=plan.alpha)))
    return jobs


def target_of(family: str) -> str:
    return TARGET_DIMENSION[family]
