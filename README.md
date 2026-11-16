# SynTrustBench v0.5: artifact

Code, preregistration, saved evidence and analysis scripts for the paper *SynTrustBench: Benchmarking Synthetic
Clinical Data with Calibrated False-Positive Rates and Certified Limits of Detection*.

Every number, table and the dose-response figure in the paper can be recomputed from the saved evidence in a few
minutes on a laptop, without patient-level data and without rerunning the experiment. The full experiment can
also be rerun from scratch.

## Quick check: recompute the paper's results (minutes)

```bash
./setup.sh                                                   # pinned Python 3.12 environment (10-20 min, once)
.venv/bin/python scripts/reproduce_paper_numbers.py         # writes results/tables/<dataset>/*.csv
.venv/bin/python scripts/make_dose_figure.py                # writes results/figures/fig_dose.pdf
```

The tables are already included in `results/tables/`, so they can be read without running anything.

## Where each result comes from

| Paper | File (in `results/tables/cdc/` and `results/tables/uci/`) |
|---|---|
| H3: honest false alarms (Section 5.1) | `h3_false_alarms.csv` (row `bases = all`, `dimension = gate`) |
| Table 3: limits of detection (Section 5.2) | `limits_of_detection.csv` (`matrix = honest`) |
| Validation on fresh bases (Section 5.3) | `validation.csv` (rows with `at_lod = True`) |
| Transfer to generator tables (Section 5.3) | `transfer_gap.csv`, `limits_of_detection.csv` (`matrix` = a generator) |
| Table-specific limits (Section 5.3) | `table_specific_limits.csv` |
| Table 4: head-to-head with SDMetrics, H1 (Section 5.4) | `h1_head_to_head.csv` (STB rows `(privacy only)` / `(fidelity only)`) |
| DCR comparison (Section 5.4) | `dcr_comparison.csv` |
| Table 5: generator verdicts, seed agreement (Section 5.5) | `generator_verdicts.csv`, `seed_agreement.csv` |
| H4: ranking stability (Section 5.5) | `h4_ranking_stability.csv` |
| H2 co-firing, shortcut check (Section 5.6) | `h2_cofiring.csv`, `a6_shortcut_agreement.csv` |
| Equity admission (Section 5.6) | `equity_admission.csv` |
| Figure 2 | `results/figures/fig_dose.pdf` |

Per-table verdict records (verdict, evidence strength, limits, seed agreement) are in the evidence at
`<dataset>/records/dimension_records.json`.

## Contents

| Path | What it is |
|---|---|
| `PREREGISTRATION_v0.5.md` | The preregistration, frozen 12 Nov 2026 before the first confirmatory run. SHA-256 `bb0a9605f75cb26b830fa93bff5822d03a06834f29ae30a5aeea29a294fa3140`. Its revision log records the 11 Nov revision of H1, H3 and H4. |
| `stb/` | SynTrustBench v0.5: null runs, calibration, signals, planted flaws, limits of detection, comparators, analysis. |
| `syntrustbench_campaign/`, `stb_cdc/` | Scoring library shared with the concurrent study, and the CDC adapter. |
| `vendor/frozen_v030/` | The frozen `syntrustbench` v0.3 package of the earlier release (commit `3dd0f7c`), installed with `--no-deps`. |
| `data/` | The two public datasets (CDC Diabetes Health Indicators; UCI Diabetes 130-US Hospitals), with SHA-256 checks. |
| `STB_v05_main_experiment.ipynb`, `STB_uci.ipynb` | The notebooks that produced the confirmatory run (CDC and UCI), with their outputs. |
| `results/evidence/stb-v0.5-full_json.tar.gz` | Saved evidence of the confirmatory run: raw signals and verdicts of every scored table (JSON only; no tables of patient rows). Absolute paths of the run machine are replaced by `<artifact-root>`; nothing else is changed. |
| `results/tables/`, `results/figures/` | Outputs of the two scripts above. |
| `setup.sh`, `run.sh`, `README_RUN.md` | Environment setup and the full rerun (see below). |

## Rerunning the full experiment

See `README_RUN.md`. In short: `./setup.sh`, then `./run.sh selftest`, `./run.sh repro` (must pass),
`./run.sh full cdc` and `./run.sh full uci`. The confirmatory run was scored on one Linux x86-64 server with
16 parallel jobs of 4 threads; on that machine each dataset took several days, dominated by the AIM fits. Results are
written to `runs/stb-v0.5-full/<dataset>/`, with a manifest that records code hashes, the plan, the platform,
package versions and the preregistration hash, and that refuses to mix platforms or code versions. Each table
draws from its own seeded random streams, so results do not depend on the number of workers.

## Deviations from the preregistration

1. **AIM on CDC** (15 Nov 2026): fits for seeds 17 and 59 were stopped after more than 60 hours; AIM is reported on
   seeds 29, 43 and 71. The surviving seeds are those whose fits finished, so they may not be representative.
   The CDC notebook's first cell records this and skips the two fits.
2. **AIM on UCI** (15 Nov 2026): because AIM fits on CDC took 11 to more than 60 hours, AIM was not started on
   UCI; UCI results cover Gaussian copula, CTGAN and TVAE. The UCI notebook's first cell records this.
3. For H4, Kendall's tau-b is computed for each seed pair over the generators both seeds contain
   (`stb/confidence.py: ranking_stability`).

The stopped and skipped AIM fits are being rerun with the preregistered settings.

## Licences

The vendored v0.3 package is MIT-licensed (`vendor/frozen_v030/LICENSE`). CDC Diabetes Health Indicators and UCI Diabetes 130-US Hospitals: CC BY 4.0, from the UCI Machine
Learning Repository.
