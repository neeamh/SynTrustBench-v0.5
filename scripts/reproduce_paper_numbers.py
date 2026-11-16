#!/usr/bin/env python3
"""Recompute every number, table and the dose-response figure of the paper from the saved evidence.

The evidence (results/evidence/stb-v0.5-full_json.tar.gz) holds every scored table's raw signals and verdicts
from the confirmatory run on both datasets: null runs, calibration models, planted-flaw replicates, validation
bases, generator tables, table-specific replicates, the DCR comparison and the per-dimension records. No table of
patient data is included or needed. Run from the artifact root after ./setup.sh:

    .venv/bin/python scripts/reproduce_paper_numbers.py            # tables only (minutes)
    .venv/bin/python scripts/reproduce_paper_numbers.py --probit   # also the display-only probit fits (slower)
"""
import argparse
import os
import sys
import tarfile
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('STB_FROZEN_CHECKOUT', str(ROOT / 'vendor' / 'frozen_v030'))

import pandas as pd  # noqa: E402

from stb import analysis  # noqa: E402
from stb.bases import Workspace  # noqa: E402
from stb.plan import full_plan  # noqa: E402

EVIDENCE = ROOT / 'results' / 'evidence'
TABLES = ROOT / 'results' / 'tables'


def unpack() -> Path:
    target = EVIDENCE / 'stb-v0.5-full'
    if not (target / 'cdc' / 'manifest.json').exists():
        target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(EVIDENCE / 'stb-v0.5-full_json.tar.gz') as tar:
            tar.extractall(target)
    return target


def save(df, dataset: str, name: str) -> None:
    out = TABLES / dataset
    out.mkdir(parents=True, exist_ok=True)
    (pd.DataFrame([df]) if isinstance(df, dict) else df).to_csv(out / f'{name}.csv', index=False)
    print(f'  wrote results/tables/{dataset}/{name}.csv')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--probit', action='store_true', help='also compute the display-only probit LoD fits')
    parser.add_argument('--datasets', nargs='+', default=['cdc', 'uci'])
    args = parser.parse_args()
    if not args.probit:  # the probit curve is for display only (paper Section 3.4); official limits do not use it
        analysis.probit_lod = lambda *a, **k: {'lod_fit': None, 'lod_fit_ci': None}
    root = unpack()
    plan = full_plan()
    for ds in args.datasets:
        print(f'== {ds}')
        store = Workspace(ds, str(root), plan.data_scale).store
        model = store.read('calibration', 'model')['model']
        comp = store.read('calibration', 'competitors')['model']
        budget = store.read('calibration', 'stb_budget')['model']
        save(analysis.null_false_alarms(store, model, plan.ids, plan.alpha), ds, 'h3_false_alarms')
        gen = analysis.generator_table(store, model, plan.alpha)
        agreement, stability = analysis.seed_summary(gen)
        save(gen, ds, 'generator_verdicts')
        save(agreement, ds, 'seed_agreement')
        save(pd.concat([stability, analysis.competitor_ranking_stability(store, comp)], ignore_index=True),
             ds, 'h4_ranking_stability')
        outcomes = analysis.outcomes_frame(store)
        lods = analysis.lod_summary(outcomes, plan.pi, plan.gamma)
        save(lods.drop(columns=['curve', 'steps'], errors='ignore'), ds, 'limits_of_detection')
        save(analysis.validation_summary(outcomes, lods, plan.pi), ds, 'validation')
        save(analysis.transfer_gap(outcomes, lods), ds, 'transfer_gap')
        save(analysis.table_lod_summary(store, plan.pi, plan.gamma).drop(columns=['curve', 'steps'], errors='ignore'),
             ds, 'table_specific_limits')
        save(analysis.head_to_head(outcomes, model, comp, plan.alpha, plan.pi, plan.gamma), ds, 'h1_head_to_head')
        save(analysis.dcr_comparison(store, comp, model, budget, plan.alpha), ds, 'dcr_comparison')
        save(analysis.equity_admission(store, model, lods, plan.ids, plan.alpha), ds, 'equity_admission')
        save(analysis.a6_agreement(store), ds, 'a6_shortcut_agreement')
        save(analysis.h2_attribution(store), ds, 'h2_cofiring')


if __name__ == '__main__':
    main()
