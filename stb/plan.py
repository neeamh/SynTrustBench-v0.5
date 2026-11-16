"""The STB v0.5 experiment plan: every preregistered number in one frozen object.

``full_plan()`` matches PREREGISTRATION_v0.5.md. ``smoke_plan()`` keeps the same structure at toy sizes so
the whole notebook can be exercised in minutes; its results are never evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json

from .operators import GRIDS, TARGET_DIMENSION


@dataclass(frozen=True)
class RunIds:
    """Null-run ids, fixed before any scoring (preregistration section 5)."""
    fit: tuple[int, ...]
    estimation: tuple[int, ...]
    validation: tuple[int, ...]

    def __post_init__(self):
        sets = [set(self.fit), set(self.estimation), set(self.validation)]
        if any(a & b for i, a in enumerate(sets) for b in sets[i + 1:]):
            raise ValueError('Fit, estimation and validation ids must be disjoint.')

    @property
    def all(self) -> tuple[int, ...]:
        return self.fit + self.estimation + self.validation


@dataclass(frozen=True)
class Plan:
    name: str
    datasets: tuple[str, ...]
    ids: RunIds
    alpha: float = 0.05
    pi: float = 0.95
    gamma: float = 0.95
    replicates: dict = field(default_factory=lambda: {'privacy': 150, 'fidelity': 100, 'utility': 100,
                                                      'equity': 100})
    validation_replicates: int = 100
    table_specific_replicates: int = 100
    honest_families: tuple[str, ...] = ('copy_random', 'copy_outlier', 'near_copy', 'copy_test', 'copy_control',
                                        'shuffle',
                                        'marginal_shift', 'constraint_violation', 'label_flip',
                                        'outcome_decouple', 'subgroup_degrade')
    optional_families: tuple[str, ...] = ('noisy_copy',)          # run only if the budget allows
    generator_matrix_families: tuple[str, ...] = ('copy_random', 'copy_outlier', 'near_copy')
    table_specific_families: tuple[str, ...] = ('copy_random', 'copy_outlier')
    generators: tuple[str, ...] = ('GaussianCopula', 'CTGAN', 'TVAE', 'DPAIM')
    seeds: tuple[int, ...] = (17, 29, 43, 59, 71)
    matrix_generators: tuple[str, ...] = ('GaussianCopula', 'CTGAN')
    samples_per_generator_seed: int = 30                          # fresh samples -> matrix bases
    min_eligible_matrix_bases: int = 20
    robustness_on: tuple[str, ...] = ('fit', 'generator')         # diagnostic; expensive
    competitors: tuple[str, ...] = ('sdmetrics_quality', 'sdmetrics_dcr_baseline',
                                    'sdmetrics_dcr_overfitting', 'sdmetrics_new_row_synthesis')
    # Head-to-head budget (preregistration §7): competitors are scored on these families only, on the first
    # competitor_replicates estimation bases; every benchmark's head-to-head LoD uses exactly those bases.
    competitor_families: tuple[str, ...] = ('copy_random', 'near_copy', 'shuffle', 'marginal_shift')
    competitor_replicates: int = 100
    full_grid_competitors: tuple[str, ...] = ('sdmetrics_quality', 'sdmetrics_new_row_synthesis')  # certified LoDs
    # DCR comparison (supporting figure, detection rates only): few doses, few bases, STB also scored at DCR's
    # row budget. dcr_rows must equal competitors.COMPETITOR_SUBSAMPLE.
    dcr_competitors: tuple[str, ...] = ('sdmetrics_dcr_baseline', 'sdmetrics_dcr_overfitting')
    dcr_families: tuple[str, ...] = ('copy_random', 'near_copy')
    dcr_doses: tuple[str, ...] = ('0', '0.01', '0.05', '0.25')
    dcr_bases: int = 30
    dcr_rows: int = 2000
    a6_fraction: float = 0.05                                     # amendment A6 full-evaluation check
    epochs: int | None = None                                     # None = library default
    data_scale: float = 1.0                                       # < 1 only in smoke plans
    evidence: bool = True                                         # False in smoke plans

    def replicates_for(self, family: str) -> int:
        return self.replicates[TARGET_DIMENSION[family]]

    def estimation_bases(self, family: str) -> tuple[int, ...]:
        r = self.replicates_for(family)
        if r > len(self.ids.estimation):
            raise ValueError(f'{family}: {r} replicates but only {len(self.ids.estimation)} estimation bases.')
        return self.ids.estimation[:r]

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, default=str).encode()).hexdigest()


def full_plan() -> Plan:
    return Plan(name='stb-v0.5-full', datasets=('cdc', 'uci'),
                ids=RunIds(fit=tuple(range(0, 100)), estimation=tuple(range(100, 250)),
                           validation=tuple(range(250, 350))))


def smoke_plan(generators: tuple[str, ...] = ('GaussianCopula',)) -> Plan:
    """Same stages, toy sizes. n_fit = 20 gives r = floor(0.05 x 21) = 1, so Fail is still possible.
    ``generators=('SmokeResample',)`` swaps in a bootstrap stand-in where SDV is not installed."""
    return Plan(name='stb-v0.5-smoke', datasets=('cdc',),
                ids=RunIds(fit=tuple(range(0, 20)), estimation=tuple(range(100, 106)),
                           validation=tuple(range(250, 253))),
                replicates={'privacy': 6, 'fidelity': 4, 'utility': 4, 'equity': 4},
                validation_replicates=3, table_specific_replicates=3,
                honest_families=('copy_random', 'copy_outlier', 'near_copy', 'copy_test', 'copy_control', 'shuffle',
                                 'marginal_shift', 'constraint_violation', 'label_flip', 'outcome_decouple',
                                 'subgroup_degrade'),
                generators=generators, seeds=(17, 29), matrix_generators=generators,
                samples_per_generator_seed=3, min_eligible_matrix_bases=2, robustness_on=(), dcr_bases=3,
                competitors=('sdmetrics_quality', 'sdmetrics_dcr_baseline', 'sdmetrics_dcr_overfitting',
                             'sdmetrics_new_row_synthesis'),   # full list on purpose: the smoke run must exercise the DCR path
                epochs=1, data_scale=0.05, evidence=False)


def grids(plan: Plan) -> dict:
    return {f: GRIDS[f] for f in plan.honest_families + plan.optional_families}
