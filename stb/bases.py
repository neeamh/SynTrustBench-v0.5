"""Honest null runs and the main roles.

A null run draws pseudo-roles (T, R, P, E) from the development data only, builds the real-data context
(real-vs-real floor, real-trained probe baselines, privacy nearest-neighbour cache) and scores its honest
table P. Fit runs feed the calibration; estimation and validation runs are the bases flaws are planted into.
The main roles use the real split, including the untouched held-out set E, and are used only for generator
tables.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from syntrustbench_campaign.config import DIMENSIONS, GATE_DIMENSIONS
from syntrustbench_campaign.data import CohortBuilder, RoleAssigner, read_analytic
from syntrustbench_campaign.evaluation import build_context

from . import competitors, dimensions
from .datasets import Dataset, load_dataset
from .store import ResultStore


@dataclass(frozen=True)
class Workspace:
    """Where one dataset's results live. Everything a worker needs is rebuilt from these three values."""
    dataset_name: str
    results_root: str
    scale: float = 1.0

    @property
    def root(self) -> Path:
        return Path(self.results_root) / self.dataset_name

    @property
    def store(self) -> ResultStore:
        return ResultStore(self.root)

    def dataset(self) -> Dataset:
        return load_dataset(self.dataset_name, scale=self.scale, work=Path(self.results_root) / '_scaled')

    def folder(self, *parts) -> Path:
        path = self.root.joinpath(*map(str, parts))
        path.mkdir(parents=True, exist_ok=True)
        return path


# ----------------------------------------------------------------------------------------------- cohort
def prepare_cohort(ws: Workspace) -> dict:
    """Load the frozen splits once and write the cohort manifest (idempotent)."""
    path = ws.root / 'cohort' / 'split_manifest.json'
    if not path.is_file():
        CohortBuilder(ws.dataset().cfg).build(ws.folder('cohort'))
    return json.loads(path.read_text())


def _cohort(ws: Workspace) -> dict:
    return json.loads((ws.root / 'cohort' / 'split_manifest.json').read_text())


# ----------------------------------------------------------------------------------------------- null runs
def null_role(run_id: int, ids) -> str:
    for name in ('fit', 'estimation', 'validation'):
        if run_id in getattr(ids, name):
            return name
    raise ValueError(f'Run {run_id} is not in the plan.')


def dims_for_null(role: str, robustness_on: tuple[str, ...]) -> tuple[str, ...]:
    dims = GATE_DIMENSIONS + ('equity',)
    return dims + ('robustness',) if role in robustness_on else dims


def null_run_job(ws: Workspace, run_id: int, role: str, dims: tuple[str, ...], competitor_names: tuple) -> dict:
    """Build one null run and score its honest table (worker job, stage 'nulls')."""
    ds = ws.dataset()
    folder = ws.folder('bases', f'null_{run_id:03d}')
    roles = RoleAssigner(ds.cfg).null(_cohort(ws), run_id, folder / 'roles')
    context = build_context(ds.cfg, roles, ws.folder('bases', f'null_{run_id:03d}', 'context'))
    base = dimensions.Base.load(ds.cfg, roles, context['context_path'])
    honest = read_analytic(roles['paths']['positive'], ds.cfg)
    evidence = dimensions.evaluate(base, honest, f'{ds.name}/null/{run_id}', dims=dims)
    scores = competitors.score_all(competitor_names, base, honest, ds)
    return {'run_id': run_id, 'role': role, 'roles': roles, 'context_path': context['context_path'],
            'evidence': evidence, 'competitors': scores}


def load_null_base(ws: Workspace, run_id: int, ds: Dataset | None = None) -> tuple[dimensions.Base, dict]:
    """Base object (with its dose-0 evidence attached) and the honest table's role paths."""
    record = ws.store.read('nulls', f'{run_id:03d}')
    ds = ds or ws.dataset()
    base = dimensions.Base.load(ds.cfg, record['roles'], record['context_path'])
    base.dose0 = record['evidence']
    return base, record


# ----------------------------------------------------------------------------------------------- main roles
def main_roles_job(ws: Workspace) -> dict:
    """The real split's roles and context (worker job, stage 'main'). Generator tables are judged here."""
    ds = ws.dataset()
    folder = ws.folder('main')
    roles = RoleAssigner(ds.cfg).main(_cohort(ws), folder / 'roles')
    context = build_context(ds.cfg, roles, ws.folder('main', 'context'))
    return {'roles': roles, 'context_path': context['context_path']}


def load_main_base(ws: Workspace, ds: Dataset | None = None) -> dimensions.Base:
    record = ws.store.read('main', 'roles')
    ds = ds or ws.dataset()
    return dimensions.Base.load(ds.cfg, record['roles'], record['context_path'])


def all_dimensions_present(signals: dict) -> dict:
    """Calibration needs every dimension key, even when a dimension was not scored."""
    return {d: signals.get(d, {}) for d in DIMENSIONS}
