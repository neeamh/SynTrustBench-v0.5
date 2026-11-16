"""STB v0.5 flaw operators with dose axes (preregistration §6; spec v1 §4.2 with amendments A3-A5).

Every operator:
  * keeps the table's columns, dtypes and row count;
  * is nested within a replicate: one seed fixes a random row order (and donors), and a dose acts on a
    prefix of that order, so a higher dose contains every change of a lower dose;
  * returns (table, audit) where the audit records the rows changed and the realised dose.

A dose is a string: '3r' means 3 rows; '0.005' means a fraction of rows (or of subgroup members, or of IQR for
marginal_shift). ``dose_rows`` turns a dose into a row count for a given table size (at least 1 row, spec §8.3).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import pandas as pd

COPY_GRID = ('1r', '3r', '10r', '30r', '0.001', '0.0025', '0.005', '0.01', '0.02', '0.05', '0.1', '0.25')
GRIDS = {
    'copy_random': COPY_GRID,
    'copy_outlier': COPY_GRID,
    'near_copy': COPY_GRID,
    'noisy_copy': COPY_GRID,
    'copy_test': ('1.0',),
    'copy_control': ('1.0',),
    'shuffle': ('0.05', '0.1', '0.2', '0.3', '0.5', '0.8', '1.0'),
    'marginal_shift': ('0.05', '0.1', '0.2', '0.3', '0.5', '1.0'),
    'constraint_violation': ('0.001', '0.005', '0.01', '0.02', '0.05', '0.1'),
    'label_flip': ('0.02', '0.05', '0.1', '0.2', '0.3', '0.5'),
    'outcome_decouple': ('0.05', '0.1', '0.2', '0.4', '0.7', '1.0'),
    'subgroup_degrade': ('0.1', '0.25', '0.5', '0.75', '0.9', '1.0'),
}
TARGET_DIMENSION = {
    'copy_random': 'privacy', 'copy_outlier': 'privacy', 'near_copy': 'privacy', 'noisy_copy': 'privacy',
    'copy_test': 'privacy', 'copy_control': 'privacy',
    'shuffle': 'fidelity', 'marginal_shift': 'fidelity', 'constraint_violation': 'fidelity',
    'label_flip': 'utility', 'outcome_decouple': 'utility', 'subgroup_degrade': 'equity',
}
# Families scored at one dose only (no grid, no LoD curve; reported as a detection rate).
# copy_control: every released row is a training row (the gate paper's copy control; must fail).
# copy_test: every released row is a real held-out record (Ganev's case; predicted blind spot of all benchmarks).
SINGLE_DOSE = ('copy_test', 'copy_control')


def dose_rows(dose: str, n: int) -> int:
    if dose.endswith('r'):
        return min(n, int(dose[:-1]))
    value = float(dose)
    return 0 if value == 0 else min(n, max(1, int(math.ceil(value * n))))


def grid_for(family: str, n: int) -> tuple[str, ...]:
    """The family's grid for a table of n rows, ordered by size. Row-count doses are kept only when they
    are smaller than the smallest fractional dose, so the grid is strictly increasing in rows."""
    grid = GRIDS[family]
    fractional = [d for d in grid if not d.endswith('r')]
    floor = min((dose_rows(d, n) for d in fractional), default=n + 1)
    kept = [d for d in grid if not d.endswith('r') or dose_rows(d, n) < floor]
    out, seen = [], set()
    for d in sorted(kept, key=lambda d: (dose_rows(d, n), dose_fraction(d, n))):
        key = dose_rows(d, n) if family != 'marginal_shift' else d
        if key not in seen:
            seen.add(key); out.append(d)
    return tuple(out)


def dose_fraction(dose: str, n: int) -> float:
    """Comparable numeric dose (fraction of rows) for ordering and reporting."""
    return dose_rows(dose, n) / n if dose.endswith('r') else float(dose)


@dataclass
class DatasetSpec:
    """What the operators need to know about a dataset; built from an ExperimentConfig."""
    continuous: tuple[str, ...]
    categorical: tuple[str, ...]
    target: str
    ranges: dict[str, tuple[int, int]]          # public integer range per count column
    shuffle_feature: str
    subgroup: tuple[str, str]                   # (column, value)
    violation: tuple[str, float]                # (column, value that breaks a clinical rule)

    @classmethod
    def from_config(cls, cfg, violation: tuple[str, float]) -> 'DatasetSpec':
        ranges = {c: (lower, overflow) for c, lower, _cap, _w, overflow in cfg.generators.dp_domains}
        inj = cfg.injections
        return cls(tuple(cfg.data.continuous), tuple(cfg.data.categorical), cfg.data.target, ranges,
                   inj.shuffle_feature, (inj.subgroup_column, inj.subgroup_value), violation)

    @property
    def features(self) -> tuple[str, ...]:
        return self.continuous + self.categorical


@dataclass
class FlawContext:
    """Per-base real data the operators may draw on. ``train`` is the base's T, ``test`` its E."""
    spec: DatasetSpec
    train: pd.DataFrame
    test: pd.DataFrame | None = None
    encode: object = None                       # callable(frame) -> 2-D array (STB privacy encoding)
    outlier_rows: np.ndarray | None = field(default=None, repr=False)

    def outliers(self, k: int = 10, top: float = 0.05) -> np.ndarray:
        """Indices of T's top `top` share by distance to the k-th nearest neighbour within T (spec §4.2)."""
        if self.outlier_rows is None:
            from sklearn.neighbors import NearestNeighbors
            matrix = np.asarray(self.encode(self.train) if self.encode else _default_encode(self.train, self.spec),
                                dtype=float)
            nn = NearestNeighbors(n_neighbors=k + 1).fit(matrix)
            dist, _ = nn.kneighbors(matrix)
            score = dist[:, k]                   # column 0 is the row itself (or an identical row)
            cut = max(1, int(math.ceil(top * len(score))))
            self.outlier_rows = np.argsort(-score, kind='stable')[:cut]
        return self.outlier_rows


def _default_encode(frame: pd.DataFrame, spec: DatasetSpec) -> np.ndarray:
    num = frame[list(spec.continuous)].apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy(float)
    num = (num - num.mean(0)) / np.where(num.std(0) > 0, num.std(0), 1)
    cats = pd.get_dummies(frame[list(spec.categorical)].astype(str)).to_numpy(float)
    return np.hstack([num, cats])


def _audit(rows, n, **extra) -> dict:
    rows = np.asarray(rows, dtype=int)
    return {'rows_changed': int(len(rows)), 'realized_fraction': len(rows) / n if n else 0.0,
            'selected_rows': rows.tolist(), **extra}


def _nested_rotation(values: np.ndarray, order: np.ndarray, grid: tuple[str, ...], dose: str, n: int) -> np.ndarray:
    """Rotate values within disjoint blocks of `order` defined by the grid (v0.4.2 construction).
    Every block up to `dose` is rotated, so a higher dose contains every rotation of a lower one,
    and the column's marginal distribution is unchanged."""
    out = values.copy()
    levels = sorted(grid, key=lambda d: dose_fraction(d, n))
    previous = 0
    for level in levels:
        end = dose_rows(level, n)
        block = order[previous:end]
        if dose_fraction(level, n) <= dose_fraction(dose, n) and len(block) > 1:
            out[block] = np.roll(values[block], 1)
        previous = max(previous, end)
    return out


def plant(family: str, table: pd.DataFrame, dose: str, ctx: FlawContext, seed: int) -> tuple[pd.DataFrame, dict]:
    """Plant one flaw at one dose. Same (family, seed) across doses gives nested changes."""
    spec, n = ctx.spec, len(table)
    out = table.copy(deep=True).reset_index(drop=True)
    if dose_fraction(dose, n) == 0:
        return out, _audit([], n, family=family, dose=dose)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    k = dose_rows(dose, n)
    rows = order[:k]
    cols = list(out.columns)

    if family in ('copy_random', 'copy_outlier', 'near_copy', 'noisy_copy', 'copy_control'):
        pool = ctx.outliers() if family == 'copy_outlier' else np.arange(len(ctx.train))
        donors_all = rng.choice(pool, size=n, replace=len(pool) < n)
        donors = donors_all[:k]
        out.iloc[rows] = ctx.train.iloc[donors][cols].to_numpy()
        extra = {'donor_rows': donors.tolist(), 'donor_pool_size': int(len(pool))}
        if family == 'near_copy':
            choices = rng.integers(0, len(spec.features), size=n)
            signs = rng.choice([-1, 1], size=n)
            picks = rng.random(n)
            changed = []
            for pos, r in enumerate(rows):
                col = spec.features[choices[pos]]
                if col in spec.continuous:
                    lo, hi = spec.ranges.get(col, (-np.inf, np.inf))
                    value = float(out.at[r, col])
                    new = value + signs[pos]
                    if new < lo or new > hi:
                        new = value - signs[pos]
                    out.at[r, col] = type(out.at[r, col])(new) if not isinstance(out.at[r, col], str) else new
                else:
                    levels = sorted(set(ctx.train[col].dropna().astype(str)) - {str(out.at[r, col])})
                    if levels:
                        out.at[r, col] = levels[int(picks[pos] * len(levels))]
                changed.append(col)
            extra['changed_field'] = changed
        if family == 'noisy_copy':
            # Spec v1 literal: N(0, (0.05 IQR)^2) added to every numeric field (IQR from the base's T);
            # categories unchanged; values are NOT rounded, so copies are never exact on numeric data.
            noise = {}
            for col in spec.continuous:
                x = pd.to_numeric(ctx.train[col], errors='coerce')
                sd = 0.05 * float(x.quantile(.75) - x.quantile(.25))
                draws = rng.normal(0.0, 1.0, size=n)[:k] * sd
                out[col] = pd.to_numeric(out[col], errors='coerce').astype(float)
                out.loc[rows, col] = out.loc[rows, col].to_numpy() + draws
                noise[col] = sd
            extra['noise_sd'] = noise
        return out, _audit(rows, n, family=family, dose=dose, **extra)

    if family == 'copy_test':
        if ctx.test is None or len(ctx.test) < n:
            raise ValueError('copy_test needs a held-out table at least as large as the candidate.')
        picks = rng.choice(len(ctx.test), size=n, replace=False)
        return ctx.test.iloc[picks][cols].reset_index(drop=True), _audit(np.arange(n), n, family=family, dose=dose,
                                                                         note='released rows are real held-out records')

    if family == 'shuffle':
        col = spec.shuffle_feature
        out[col] = _nested_rotation(out[col].to_numpy(), order, GRIDS['shuffle'], dose, n)
        return out, _audit(order[:dose_rows(dose, n)], n, family=family, dose=dose, column=col)

    if family == 'outcome_decouple':
        col = spec.target
        out[col] = _nested_rotation(out[col].to_numpy(), order, GRIDS['outcome_decouple'], dose, n)
        return out, _audit(order[:dose_rows(dose, n)], n, family=family, dose=dose)

    if family == 'label_flip':
        out.loc[rows, spec.target] = 1 - out.loc[rows, spec.target].astype(int)
        return out, _audit(rows, n, family=family, dose=dose)

    if family == 'constraint_violation':
        col, value = spec.violation
        out.loc[rows, col] = value
        return out, _audit(rows, n, family=family, dose=dose, column=col, value=value)

    if family == 'marginal_shift':
        size = float(dose)
        for col in spec.continuous:
            x = pd.to_numeric(ctx.train[col], errors='coerce')
            iqr = max(float(x.quantile(.75) - x.quantile(.25)), 1.0)
            lo, hi = spec.ranges.get(col, (-np.inf, np.inf))
            values = pd.to_numeric(out[col], errors='coerce')
            shifted = np.clip(np.round(values + size * iqr), lo, hi)
            out[col] = shifted.where(values.notna(), values).astype(out[col].dtype)
        return out, {'rows_changed': n, 'realized_fraction': 1.0, 'family': family, 'dose': dose,
                     'shift_iqr_units': size, 'deterministic': True}

    if family == 'subgroup_degrade':
        col, value = spec.subgroup
        mask = out[col].astype(str).eq(str(value)).to_numpy()
        members, others = np.flatnonzero(mask), np.flatnonzero(~mask)
        if not len(members) or not len(others):
            raise ValueError('subgroup_degrade needs subgroup members and replacement donors.')
        member_order = rng.permutation(members)
        donors_all = rng.choice(others, size=len(members), replace=True)
        m = min(len(members), max(1, int(math.ceil(float(dose) * len(members)))))
        chosen = member_order[:m]
        out.iloc[chosen] = table.reset_index(drop=True).iloc[donors_all[:m]][cols].to_numpy()
        return out, _audit(chosen, n, family=family, dose=dose, subgroup_size=int(len(members)),
                           realized_subgroup_fraction=m / len(members))

    raise ValueError(f'Unknown flaw family {family!r}')


# --------------------------------------------------------------------------- self test
def _toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({'a': rng.integers(0, 30, n), 'b': rng.integers(1, 14, n),
                          'g': rng.choice(['x', 'y', 'z'], n, p=[.2, .5, .3]),
                          'h': rng.choice(['0', '1'], n), 'out': rng.integers(0, 2, n)})
    spec = DatasetSpec(('a', 'b'), ('g', 'h'), 'out', {'a': (0, 30), 'b': (1, 14)}, 'a', ('g', 'x'), ('b', 15))
    return spec, frame


def self_test() -> None:
    spec, base = _toy()
    _, train = _toy(seed=1)
    _, test = _toy(seed=2)
    ctx = FlawContext(spec, train, test)
    for family in GRIDS:
        previous = set()
        for dose in ('0',) + grid_for(family, len(base)):
            out, audit = plant(family, base, dose, ctx, seed=11)
            assert list(out.columns) == list(base.columns) and len(out) == len(base), family
            changed = set(map(tuple, np.argwhere(~(out.astype(str).to_numpy() == base.astype(str).to_numpy()))))
            if dose == '0':
                assert not changed, family
            if family not in ('marginal_shift', 'copy_test'):
                assert previous <= changed, (family, dose)
            previous = changed
    # Copies are exact training rows; near-copies differ from their donor in exactly one field.
    out, audit = plant('copy_random', base, '30r', ctx, 5)
    for r, d in zip(audit['selected_rows'], audit['donor_rows']):
        assert out.iloc[r].astype(str).tolist() == train.iloc[d].astype(str).tolist()
    out, audit = plant('near_copy', base, '30r', ctx, 5)
    for r, d in zip(audit['selected_rows'], audit['donor_rows']):
        diff = sum(a != b for a, b in zip(out.iloc[r].astype(str), train.iloc[d].astype(str)))
        assert diff == 1, diff
    out, audit = plant('copy_control', base, '1.0', ctx, 5)
    train_rows = set(map(tuple, train.astype(str).to_numpy()))
    assert all(tuple(r) in train_rows for r in out.astype(str).to_numpy())   # every released row is a training row
    out, audit = plant('noisy_copy', base, '30r', ctx, 5)
    for r, d in zip(audit['selected_rows'], audit['donor_rows']):
        assert out.iloc[r][list(spec.categorical)].astype(str).tolist() == train.iloc[d][list(spec.categorical)].astype(str).tolist()
    assert all(sd > 0 for sd in audit['noise_sd'].values())
    # Rotations keep marginals; decoupling keeps prevalence; outliers come from the top 5%.
    out, _ = plant('shuffle', base, '1.0', ctx, 3)
    assert sorted(out['a']) == sorted(base['a'])
    out, _ = plant('outcome_decouple', base, '1.0', ctx, 3)
    assert out['out'].sum() == base['out'].sum()
    assert len(ctx.outliers()) == 20
    out, audit = plant('copy_outlier', base, '10r', ctx, 3)
    assert set(audit['donor_rows']) <= set(ctx.outliers().tolist())
    assert dose_rows('0.001', 400) == 1 and dose_rows('3r', 400) == 3
    assert grid_for('copy_random', 11443)[:4] == ('1r', '3r', '10r', '0.001')
    assert grid_for('copy_random', 50736)[:5] == ('1r', '3r', '10r', '30r', '0.001')


if __name__ == '__main__':
    self_test()
    print('operators self-test passed')
