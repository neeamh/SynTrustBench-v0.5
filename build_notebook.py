"""Build STB_v05_main_experiment.ipynb. Edit the cells here, not in the .ipynb, so diffs stay readable."""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
md("""# SynTrustBench v0.5: main experiment

**Question.** Which whole benchmark gives verdicts you can trust, and what does each verdict rule out?

Every verdict on a gating dimension (fidelity, utility, privacy) carries a confidence statement, always per
dimension, never merged into one number:

| Verdict | Confidence statement | Source |
|---|---|---|
| Fail / Conditional | **Evidence strength** e = (1 + honest fit runs at least as extreme) / 101 | the gate paper's calibration machinery, refitted on STB's own 100 nulls and cited |
| Pass | **Limit of detection**: the smallest dose of each flaw caught with >= 95% probability (95% confidence) | new |
| Pass (privacy) | **Table-specific limit**: copies planted into the very table being judged | new |
| Generator | **Seed agreement**: modal verdict and k/5 | new |
| (all of the above) | **Validation**: fresh bases, claimed vs observed, transfer gap, head-to-head ratios | new |

Everything here follows `PREREGISTRATION_v0.5.md`. The code lives in `stb/` (one small module per job); this
notebook only calls it, stage by stage. Each stage writes one result file per job, so any cell can be
re-run and will skip finished work.

**Where to run.** Smoke mode runs anywhere in minutes and produces no evidence. The full run must happen on the
SJU server only (classifier signals differ slightly between Apple ARM and Linux x86; one results folder = one
platform)."""),

md("## 0. Settings"),
code("""# Settings come from environment variables so the run can be launched headless with run.sh;
# the defaults below apply when the notebook is opened by hand.
import os
from pathlib import Path


def physical_cores() -> int:
    \"\"\"Physical cores (not hyper-threads), so WORKERS x THREADS never oversubscribes the machine.\"\"\"
    import platform as _platform
    import subprocess
    try:
        if _platform.system() == 'Darwin':
            return int(subprocess.check_output(['sysctl', '-n', 'hw.physicalcpu']).strip())
        cores, phys, core = set(), None, None
        for line in open('/proc/cpuinfo').read().splitlines() + ['']:
            if line.startswith('physical id'):
                phys = line.split(':', 1)[1].strip()
            elif line.startswith('core id'):
                core = line.split(':', 1)[1].strip()
            elif not line.strip():
                if core is not None:
                    cores.add((phys, core))
                phys, core = None, None
        if cores:
            return len(cores)
    except Exception:  # noqa: BLE001 - fall back to a conservative guess
        pass
    return max(1, (os.cpu_count() or 2) // 2)


os.environ.setdefault('STB_FROZEN_CHECKOUT', str(Path.cwd() / 'vendor' / 'frozen_v030'))
SMOKE = os.environ.get('STB_SMOKE', '1') == '1'       # 1: toy sizes, any machine, no evidence. 0: the preregistered run.
DATASET = os.environ.get('STB_DATASET', 'cdc')        # 'cdc' (primary evidence) or 'uci' (replication)
THREADS = int(os.environ.get('STB_THREADS', 2 if SMOKE else 4))                    # threads per job
WORKERS = int(os.environ.get('STB_WORKERS', 2 if SMOKE else max(1, physical_cores() // THREADS)))  # parallel jobs
SMOKE_GENERATORS = ('GaussianCopula', 'CTGAN', 'TVAE', 'DPAIM')   # ('SmokeResample',) where SDV is not installed
assert DATASET in ('cdc', 'uci'), DATASET
print(f'SMOKE={SMOKE} DATASET={DATASET} WORKERS={WORKERS} THREADS={THREADS} (physical cores: {physical_cores()})')"""),
code("""import platform
from pathlib import Path

import pandas as pd

from stb import analysis, stages
from stb.bases import Workspace, prepare_cohort
from stb.operators import SINGLE_DOSE
from stb.plan import full_plan, smoke_plan
from stb.runner import run_jobs
from stb.store import write_manifest

CODE_ROOT = Path.cwd()
PLAN = smoke_plan(SMOKE_GENERATORS) if SMOKE else full_plan()
RESULTS_ROOT = CODE_ROOT / 'runs' / PLAN.name
ws = Workspace(DATASET, str(RESULTS_ROOT), PLAN.data_scale)
store = ws.store

if not SMOKE and (platform.system(), platform.machine()) != ('Linux', 'x86_64'):
    raise RuntimeError('The preregistered run is scored on the SJU server (Linux x86_64) only.')

manifest = write_manifest(store, PLAN, CODE_ROOT, CODE_ROOT / 'PREREGISTRATION_v0.5.md')
pd.set_option('display.max_colwidth', 80)
print(f'{PLAN.name} | {DATASET} | results in {ws.root}')
print('platform:', manifest['platform'], '| preregistration sha256:', (manifest['preregistration_sha256'] or '')[:12])

def run(jobs, label):
    \"\"\"Run a stage; stop the notebook if any job failed (tracebacks are in <results>/failures/).\"\"\"
    print(f'{label}: {len(jobs)} jobs')
    summary = run_jobs(jobs, store, workers=WORKERS, threads=THREADS)
    if summary['failed']:
        raise RuntimeError(f"{summary['failed']} {label} failed; see {ws.root / 'failures'}")
    return summary"""),

md("""## 1. Data
The frozen 40/40/20 person-disjoint splits (STB's own hash-verified copies). The held-out test set E is read
only when real generators are judged in section 4; null runs never touch it."""),
code("""cohort = prepare_cohort(ws)
pd.DataFrame([{'role': k, 'sha256': v[:12]} for k, v in cohort['split_hashes'].items()])"""),

md("""## 2. Honest null runs
Each run draws fresh pseudo-roles from the development data (STB's own seed salt, so no run coincides with a
gate-paper run), builds the real-data context, and scores its honest table. Ids are fixed in advance:

* **0-99 fit**: the calibration, all five dimensions (robustness included);
* **100-249 estimation bases** and **250-349 validation bases**: tables that flaws are planted into, with
  their dose-0 evidence.

Competitor statistics are scored on every run too: fit runs calibrate them, bases give their dose-0 rates."""),
code("""run(stages.null_jobs(ws, PLAN), 'null runs')"""),

md("""## 3. Calibration and evidence strength
STB refits the full-conformal maxT rule on its **own** fit runs (never the gate paper's frozen model). With
n = 100 and alpha = 0.05, r = floor(0.05 x 101) = 5: a dimension fails when its own maximum standardized signal
exceeds the 5th-largest fit-run maximum, and every Fail carries evidence e <= 5/101. The competitors are
calibrated by the same rule on the same runs."""),
code("""calibration = stages.calibrate(ws, PLAN)
model = store.read('calibration', 'model')['model']
comp_model = store.read('calibration', 'competitors')['model']
calibration"""),
md("""**H3 (a design check, not a discovery).** Honest Fails on bases never used for fitting, with exact one-sided
95% upper bounds; NotEvaluated is counted separately. H3 holds if the exact one-sided binomial test of the gate's
pooled Fail count against 5/101 does not reject at level 0.05 (row bases = 'all', dimension = 'gate')."""),
code("""analysis.null_false_alarms(store, model, PLAN.ids, PLAN.alpha)"""),

md("""## 4. Real generators x seeds
Each generator is fitted on the real training role and judged against the untouched held-out set, five seeds
each. The matrix generators also draw fresh samples that become bases for planted privacy flaws in section 6."""),
code("""run(stages.main_jobs(ws), 'main roles')
run(stages.generator_jobs(ws, PLAN), 'generator fits')
gen = analysis.generator_table(store, model, PLAN.alpha)
agreement, stability = analysis.seed_summary(gen)
gen.pivot_table(index=['generator', 'seed'], columns='dimension', values='verdict', aggfunc='first')"""),
code("""display(agreement)
# H4: mean Kendall tau-b of generator rankings across seed pairs; unstable if below 2/3
pd.concat([stability, analysis.competitor_ranking_stability(store, comp_model)], ignore_index=True)"""),

md("""## 5. Limits of detection on honest tables (the primary matrix)
For every flaw family, one job plants the flaw at every dose of its grid into one estimation base (nested
doses, one seed), and re-scores only the flaw's target dimension. The official LoD is the lowest dose of an
unbroken run, from the top of the grid down, whose one-sided Clopper-Pearson lower bound is >= 95%. The probit
fit is descriptive only. Dose 0 gives the false-alarm rate."""),
code("""run(stages.honest_lod_jobs(ws, PLAN), 'honest-matrix replicates')
outcomes = analysis.outcomes_frame(store)
lods = analysis.lod_summary(outcomes, PLAN.pi, PLAN.gamma)
lods[['matrix', 'family', 'target', 'eligible_bases', 'lod_fraction', 'lod_rows', 'status', 'lod_fit', 'false_alarm']]"""),

md("""### 5b. Check of the target-only shortcut (amendment A6)
Section 5 re-scores only each flaw's target dimension and copies the other gating dimensions from the base's dose 0.
Because a candidate is pooled into every column's standardization, that shortcut could in principle change a
verdict. A fixed random 5% of the replicates of every family are re-scored in full (same planted tables, same
random streams), and the verdicts are compared. Report the agreement rate; if it is not essentially 100%, the
section 5 LoDs must be recomputed from full evaluations."""),
code("""run(stages.a6_jobs(ws, PLAN), 'A6 full-evaluation check')
analysis.a6_agreement(store)"""),
code("""analysis.h2_attribution(store)   # H2, descriptive: target and co-firing rates on fully re-scored replicates"""),

md("""## 6. Do the limits transfer to generator output?
The same privacy flaws planted into fresh Gaussian copula and CTGAN samples. A base is eligible only if it passes
privacy at dose 0, so a detection means the planted flaw changed the verdict. The **transfer gap** is the observed
detection rate, in these messier tables, at the honest-matrix LoD."""),
code("""run(stages.matrix_lod_jobs(ws, PLAN), 'generator-matrix replicates')
outcomes = analysis.outcomes_frame(store)
lods = analysis.lod_summary(outcomes, PLAN.pi, PLAN.gamma)
analysis.transfer_gap(outcomes, lods)"""),

md("""## 7. Validation on fresh bases (confidence in the confidence)
Fresh validation bases (ids 250-349), fresh flaw seeds, at each honest LoD and its grid neighbours. The claim
holds if the Wilson interval at the LoD reaches 95%."""),
code("""n_rows = int(outcomes.n_rows.iloc[0])
honest = lods[lods.matrix == 'honest'].set_index('family')
doses = {f: analysis.validation_doses(f, honest.loc[f].get('lod_fraction'), n_rows)
         for f in honest.index if f not in SINGLE_DOSE}
run(stages.validation_jobs(ws, PLAN, doses), 'validation replicates')
outcomes = analysis.outcomes_frame(store)
analysis.validation_summary(outcomes, lods, PLAN.pi)"""),

md("""## 8. Table-specific privacy limit
For every generator table that passes privacy: copies planted into that very table, privacy re-scored."""),
code("""run(stages.table_lod_jobs(ws, PLAN), 'table-specific replicates')
table_lods = analysis.table_lod_summary(store, PLAN.pi, PLAN.gamma)
table_lods"""),

md("""## 9. Head-to-head (H1)
Every benchmark statistic gets an LoD on the same planted tables and the same bases (the budgeted subset where
competitors were scored), each calibrated by the same conformal rule on the same fit runs. The H1 ratio compares
each competitor with STB's privacy-only variant (the matched setting); STB as shipped is shown too.

**Primary comparators** (they see the full table): NewRowSynthesis for privacy, the Quality Report for fidelity.
The DCR metrics see a 2,000-row subsample and are **secondary**, with that disclosed.
**Preregistered predictions:** NewRowSynthesis roughly ties STB on copy_random (it counts exact matches); STB's
advantage, if any, shows on near_copy and on CDC's chance-match correction; copy_test is missed by every
benchmark (section 5 detection rate), a shared blind spot, not a win."""),
code("""h2h = analysis.head_to_head(outcomes, model, comp_model, PLAN.alpha, PLAN.pi, PLAN.gamma)
h2h"""),

md("""### 9b. The DCR comparison (supporting figure, detection rates only)
SDMetrics' DCR metrics are the field's standard privacy check but too slow for the full grid. They are scored at
four doses of copy_random and near_copy on the first 30 estimation bases. DCR's own subsampling shrinks both the
released table and the training table to 2,000 rows, so STB is also scored at that budget (and calibrated on the
same kind of samples of the fit runs). Three curves per family: DCR, STB at DCR's budget, STB as shipped."""),
code("""run(stages.dcr_null_jobs(ws, PLAN), 'STB at DCR budget: null runs')
stages.calibrate_budget(ws, PLAN)
budget_model = store.read('calibration', 'stb_budget')['model']
run(stages.dcr_jobs(ws, PLAN), 'DCR comparison')
analysis.dcr_comparison(store, comp_model, model, budget_model, PLAN.alpha)"""),

md("""## 10. Equity admission (preregistered rule)
Equity is promoted to gating on this dataset only if it is evaluable everywhere, shows no excess honest
non-acceptance, and has an official LoD. If it is admitted, every verdict and LoD on this dataset is recomputed
from the saved signals with equity in the gate family (both versions reported)."""),
code("""analysis.equity_admission(store, model, lods, PLAN.ids, PLAN.alpha)"""),

md("""## 11. Per-dimension records (spec section 10)
One record per generator table and gating dimension: verdict, evidence strength, the limits of detection for
the flaw families that target the dimension, the table-specific privacy limit, and seed agreement."""),
code("""records = analysis.dimension_records(DATASET, gen, lods, table_lods, agreement)
store.write('records', 'dimension_records', {'records': records})
for r in records[:6]:
    print(r['text'], '\\n')"""),

md("""## 12. Figures
To be added once real results exist: dose-response curves (Figure 1), claimed vs observed detection
(Figure 2), seed stability (Figure 3). They will read only the saved result files."""),
]

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata = {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}}
nbf.write(nb, 'STB_v05_main_experiment.ipynb')
print('wrote STB_v05_main_experiment.ipynb with', len(cells), 'cells')
