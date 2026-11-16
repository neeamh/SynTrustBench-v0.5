"""Real generators x seeds on the main roles (spec §6; preregistration §10).

Each (generator, seed) job fits once on the main training role T, samples a size-matched table that STB and the
competitors judge, and, for the matrix generators, draws fresh extra samples that serve as gc/ctgan bases for
planted privacy flaws (amendment A7). Generators come from the library unchanged.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from syntrustbench_campaign.config import seed_for
from syntrustbench_campaign.core import save_frame
from syntrustbench_campaign.data import read_analytic

from . import competitors, dimensions
from .bases import Workspace, load_main_base

SMOKE_ONLY_GENERATORS = ('SmokeResample',)


class _SmokeResample:
    """Smoke-test stand-in when SDV is unavailable: bootstrap rows of T. Never used for evidence."""

    def __init__(self, seed: int):
        self.seed, self.timing = seed, {}

    def fit(self, train: pd.DataFrame):
        self.train = train
        return self

    def sample(self, n: int, seed: int) -> pd.DataFrame:
        return self.train.sample(n=n, replace=True, random_state=seed % (2 ** 32)).reset_index(drop=True)


def _model(name: str, cfg, seed: int, epochs: int | None):
    if name in SMOKE_ONLY_GENERATORS:
        return _SmokeResample(seed)
    from syntrustbench_campaign.generators import generator
    if epochs is not None:
        cfg = replace(cfg, generators=replace(cfg.generators, epochs=epochs))
    return generator(name, cfg, seed)


def table_key(dataset: str, generator: str, seed: int) -> str:
    return f'{dataset}/gen/{generator}/{seed}'


def generator_job(ws: Workspace, generator: str, seed: int, *, epochs: int | None, extra_samples: int,
                  dims: tuple[str, ...], competitor_names: tuple, allow_smoke_only: bool) -> dict:
    """Fit, sample, judge (worker job, stage 'generators', id '<generator>/<seed>')."""
    if generator in SMOKE_ONLY_GENERATORS and not allow_smoke_only:
        raise ValueError(f'{generator} is a smoke-test stand-in and cannot produce evidence.')
    ds = ws.dataset()
    base = load_main_base(ws, ds)
    n = base.roles['n']
    folder = ws.folder('generator_tables', generator, seed)
    model = _model(generator, ds.cfg, seed, epochs).fit(base.members)

    def draw(label: str, sample_seed: int) -> str:
        frame = model.sample(n, sample_seed)
        path = save_frame(frame[list(ds.cfg.data.columns)], folder / f'{label}.csv')
        return str(path)

    main_path = draw('main', seed_for(generator, seed, 'sampling'))
    table = read_analytic(main_path, ds.cfg)
    evidence = dimensions.evaluate(base, table, table_key(ds.name, generator, seed), dims=dims)
    matrix_paths = [draw(f'matrix_{k:02d}', seed_for(generator, seed, 'matrix', k)) for k in range(extra_samples)]
    return {'generator': generator, 'seed': seed, 'n_rows': n, 'table_path': main_path,
            'matrix_paths': matrix_paths, 'evidence': evidence,
            'competitors': competitors.score_all(competitor_names, base, table, ds),
            'timing': getattr(model, 'timing', {}), 'smoke_only': generator in SMOKE_ONLY_GENERATORS}


def exact_copy_count(base: dimensions.Base, table: pd.DataFrame) -> dict:
    """A directly checkable fact (no score): synthetic rows identical to a training row, and the chance level."""
    columns = list(base.members.columns)
    train = set(map(tuple, base.members[columns].astype(str).to_numpy()))
    copies = int(np.sum([tuple(r) in train for r in table[columns].astype(str).to_numpy()]))
    chance = int(np.sum([tuple(r) in train for r in base.test[columns].astype(str).to_numpy()]))
    return {'synthetic_rows_in_train': copies, 'synthetic_rows': len(table),
            'heldout_rows_in_train': chance, 'heldout_rows': len(base.test)}
