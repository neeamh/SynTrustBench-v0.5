# SynTrustBench v0.5: preregistration (FROZEN 12 Nov 2026, before the first confirmatory run)

Freeze before the first confirmatory run: record this file's SHA-256 in every run manifest.
Amendments are logged at the bottom with a date and a reason, and only before the data they govern are scored.

## 1. Question and dividing line
STB asks: **which whole benchmark gives verdicts you can trust, and what does each verdict rule out?**
The gate paper (concurrent, anonymous) asked which rule best combines a fixed set of signals. STB reuses
everything that measures the data and drops everything built to compare combining rules.

- **Reused, cited:** the data pipeline (earliest encounter, hashed person-disjoint 40/40/20 roles, an untouched
  held-out set), the signal implementations, null-run resampling and full-conformal maxT calibration, the v0.4.2
  flaw operators, and the infrastructure (ledger, manifests, hashes, one-writer rule, JUDGEMENT_CALLS, reproduce
  script).
- **Dropped:** the combining-rule comparison (arithmetic, geometric and harmonic means; matched-cost judges;
  per-base differences; the sign test), the exp(-z/t) score mapping and its temperature sensitivity, and every
  gate-paper figure, table and passage.
- **Code base:** `stb_code/syntrustbench_campaign`, copied from `runs/main_v5/notebook_library` (hashes in
  `SNAPSHOT_PROVENANCE.json`). The v0.4.2 folder is never written to.

## 2. Datasets
- **CDC Diabetes Health Indicators** (BRFSS 2015; UCI id 891, downloaded through UCI's link to the Kaggle release;
  SHA-256 19f367e3...d1d0). **Primary evidence.** 253,680 respondents, one row per respondent; about 14% of rows
  are exact duplicates of another respondent's answers and are kept.
- **UCI Diabetes 130-US Hospitals**, earliest encounter (71,518 patients). Replication. The gate paper's frozen
  splits are reused, and that reuse is disclosed.

## 3. Benchmarks compared
STB v0.5 and SDMetrics 0.33.0 (the Quality Report and the privacy metrics fixed in §7). Each competitor statistic
is calibrated on the same honest fit runs by the same full-conformal rule, dimension by dimension; a native
threshold is also recorded where the tool publishes one.
**SynthEval is not included.** It was not installed and its metric list could not be fixed before this
preregistration was frozen; adding it later would let its configuration be chosen after seeing data. Comparing
against one external benchmark is a stated limitation.

## 4. Verdicts and confidence (spec v1 + amendments v1.1/v1.2)
Five dimensions, three gating (fidelity, utility, privacy); equity and robustness are diagnostics.
Verdict states: Pass / Conditional / Fail / NotEvaluated.
- **Calibration is STB's own.** On each dataset the full-conformal maxT rule (cited from the gate paper, not
  re-derived) is refitted on STB's own n_fit = 100 null runs. The gate paper's frozen model is never used.
  At alpha = 0.05, r = floor(0.05 x 101) = 5.
- **Evidence strength (Fail, Conditional):** a per-dimension full-conformal p-value against the 100 fit runs,
  using the verdict's own pooled standardization. Resolution 1/101. Every Fail has e <= 5/101 (tested in
  `stb/calibrate.py`).
- **Limit of detection (Pass):** fixed-sequence one-sided Clopper-Pearson, pi = 0.95, gamma = 0.95, over a
  descending dose grid. A probit curve in log dose is descriptive only.
- **Table-specific privacy LoD:** for every generator table that passes privacy.
- **Seed agreement:** modal verdict and k/5; rankings use each dimension's pooled maximum z.
- **Random streams (v0.5):** each table gets one seed, seed_for("stb-v0.5", table_key), from which a
  SeedSequence spawns one independent stream per dimension. Calibration nulls, bases, planted tables and
  generator tables are all scored by this same evaluator (`stb/dimensions.py`).
- **Reproduction requirement, before any server run:** signals that use no randomness equal the v0.4.2 code
  path exactly; signals that use randomness agree within seed-to-seed spread (|z| <= 4); a dimension scored
  alone equals its value inside a full profile (`stb/repro_test.py`).
- **Target-only scoring and its check (amendment A6).** Planted tables re-score only the flaw's target dimension;
  the other gating dimensions are copied from the base's dose 0. A fixed random 5% of replicates per family
  (seeded by family name) are re-scored in full with the same planted tables and random streams, and agreement
  of the **target** verdict is reported (`analysis.a6_agreement`). If it is below 99% for a family, that family's
  LoD is recomputed from full evaluations before any claim is made. (Off-target verdicts are not compared: the
  shortcut copies them from the unflawed base by design.)
- **Environment.** Python 3.12 with the exact versions in `requirements-stb-v0.5.txt` (the library's
  `environment.PINS` plus threadpoolctl). Every manifest records all pinned versions, and a results folder refuses
  a second environment.
- **Raw signals are saved for every evaluated table.** Verdicts, evidence and LoDs are computed from saved
  evidence.
- **One platform.** Identical inputs give slightly different classifier-based signals on Apple ARM and on
  Linux x86 (e.g. two-sample AUC differs by ~0.0015 on a UCI null). All calibration runs, bases, planted
  tables and generator tables are therefore scored on the SJU server only; the platform is recorded in every
  manifest, and a run that mixes platforms is invalid.
- **Stored-value check (done 9 Nov, UCI null 51, read-only):** on Linux x86, STB v0.5 equals the gate
  paper's own v0.4.2 code path on every deterministic signal; both differ from the Mac-stored values by the
  same platform amounts; random signals sit within seed spread (|z| <= 1.5 with 30 reseeds).

## 5. Null runs (both datasets, same structure; 350 per dataset)
- Run ids are fixed now: **0-99 calibration (fit)**, **100-249 estimation bases**, **250-349 validation
  bases**. Every replicate uses a distinct base; validation uses fresh bases, not just fresh flaw seeds.
- **CDC:** all 350 are generated by STB from the frozen CDC splits.
- **UCI:** all 350 are generated by STB from the frozen UCI 40/40/20 splits (a copy inside STB's folder,
  hash-verified). No gate-paper run (main_v5, supplementary) is reused or read, so STB's UCI evidence shares
  no null runs with the gate paper.

## 6. Flaw families, doses, targets
| Family | Target | Dose grid |
|---|---|---|
| copy_random | privacy | 1, 3, 10, 30 rows; 0.1, 0.25, 0.5, 1, 2, 5, 10, 25 % |
| copy_outlier (top 5% of T by 10th-NN distance) | privacy | same |
| near_copy, **primary** (copy, then change exactly one field to another valid value: counts +/-1 within range, categories to another level) | privacy | same |
| noisy_copy, **secondary, run if budget allows** (spec v1 literal: copy, then add N(0, (0.05 IQR)^2) to every numeric field, unrounded; categories unchanged) | privacy | same |
| copy_test (outside flaw, Ganev: every released row is a real held-out record) | privacy | single dose; detection rate only. **Predicted: missed by every benchmark, STB included** (no excess matches against training; real data has perfect fidelity). Reported as a shared blind spot. |
| shuffle (one predictor; UCI number_inpatient, CDC BMI) | fidelity | 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0 |
| marginal_shift (every count column, d x max(IQR, 1)) | fidelity | 0.05, 0.1, 0.2, 0.3, 0.5, 1.0 IQR |
| constraint_violation (reported, no claim) | fidelity | 0.1, 0.5, 1, 2, 5, 10 % |
| label_flip | utility | 2, 5, 10, 20, 30, 50 % |
| outcome_decouple | utility | 5, 10, 20, 40, 70, 100 % |
| subgroup_degrade (UCI African-American; CDC age 80+) | equity (diagnostic) | 10, 25, 50, 75, 90, 100 % |
| copy_control (every released row copied from T) | privacy | single dose (100 %); detection rate only; must be about 100 % |

Replicates: R = 150 for privacy, 100 for fidelity and utility; V = 100 fresh validation replicates.
Matrices: honest (primary); gc and ctgan (privacy families only, built from fresh samples of the 5 fitted
models; a base is eligible only if it passes privacy at dose 0; at least 20 eligible bases).

## 7. Head-to-head statistics
- **Primary comparators (see the full table):** NewRowSynthesis for privacy; the Quality Report score for fidelity.
  NewRowSynthesis is computed by a vectorized implementation of SDMetrics 0.33.0's exact rule (SDMetrics' own
  row-by-row version took over 20 minutes per full-size CDC table); `stb/competitors.py` self-test requires the two
  to agree exactly on randomized tables, and the server run repeats that check before scoring.
- **Budget for the certified comparison:** NewRowSynthesis and the Quality Report are scored on copy_random,
  near_copy, shuffle and marginal_shift, on the first 100 estimation bases of each (100 >= 59 replicates, so a 95%
  LoD can be certified). **Every** benchmark's head-to-head LoD, STB's included, uses exactly those 100 bases.
- **DCR comparison (supporting figure, detection rates only, no LoD):** DCRBaselineProtection and
  DCROverfittingProtection are too slow for the full grid. They are scored at doses 0, 1%, 5% and 25% of copy_random
  and near_copy, on the first 30 estimation bases, calibrated on the 100 fit runs, and reported as detection rates
  with Wilson intervals. SDMetrics' own subsampling draws 2,000 rows of BOTH the released table and the training
  table, so a copied row is recognised only if its donor is in the training sample. To compare like with like,
  STB's privacy dimension is also scored at that budget (2,000 released rows against 2,000 training rows), calibrated
  on the same kind of samples of the fit runs. The figure shows DCR, STB at DCR's budget, and STB as shipped.
  DCR is also scored on every real generator table. **Prediction:** DCRBaselineProtection compares *median*
  distances, so it should stay blind until a large share of rows is copied; STB at the same budget should detect
  copying earlier.
- STB is reported as shipped (gate family) and as a single-dimension family at alpha = 0.05; the H1 LoD ratio uses
  the latter.

## 8. Equity admission (diagnostic to gating)
Rule, fixed now: equity is admitted to the gate on a dataset if and only if (a) it is evaluable on every one of
the 100 calibration runs and on every estimation base at dose 0, (b) an exact one-sided binomial test finds no
excess of honest non-acceptance over 5% at level 0.05 on the estimation bases, and (c) subgroup_degrade has an
official LoD within the tested range. The outcome is reported either way.
**Commitment:** if the rule admits equity on CDC, the CDC gate family is refitted with equity included, and every
CDC verdict, evidence strength and limit of detection is recomputed from the saved signals. Both the
three-dimension and the four-dimension CDC results are reported, the latter as primary. UCI is unaffected unless
the rule also admits equity there, in which case the same applies.

## 9. Hypotheses
- **H1:** at the matched false-positive rule of §3, STB's single-dimension privacy family (§7) has an official
  LoD **no higher** than NewRowSynthesis on copy_random and **strictly lower** on near_copy, on each dataset.
  LoDs are compared as grid doses, all estimated on the same 100 bases (§7); a family with no certified LoD counts
  as above the largest dose, and two uncertified LoDs tie. **Predictions, written before any scoring:**
  - copy_random: NewRowSynthesis counts exact matches, so it should roughly **tie** STB (a confirmed tie, not a loss);
  - near_copy: one changed field defeats exact matching, so STB's proximity signal should give the **lower** LoD;
  - CDC: about 9% of honest rows match a training row by chance; STB corrects for that baseline and the
    competitors do not, so their calibrated LoDs should be higher there;
  - fidelity (shuffle, marginal_shift): **no prediction of an STB advantage**; a tie with the Quality Report is
    plausible for simple marginal shifts.
- **H2 (descriptive):** for each single planted flaw, how often its own dimension fails and how often the other
  gating dimensions also fail, by dose, measured on the fully re-scored A6 replicates only (the target-only
  shortcut cannot see off-target effects). Co-firing is not counted as an error: a flipped label also changes
  the outcome's distribution, so fidelity may rightly fire. The claim is limited to sensitivity of the target
  dimension plus a co-firing profile. (The combined U/P/F flaws are not run in v0.5.)
- **H3 (design check):** on each dataset, the number of the 250 honest bases (ids 100-349) on which the gate
  returns Fail is tested against the nominal bound 5/101 with an exact one-sided binomial test at level 0.05; H3
  holds if neither dataset's test rejects. Each gating dimension's Fail rate is reported with an exact one-sided
  95% upper bound, without a separate test. NotEvaluated verdicts are counted and reported separately, not as
  Fails. If equity is admitted on a dataset (§8), H3 for the refitted gate uses only the 100 validation bases
  (ids 250-349), because the estimation bases were used to admit it.
- **H4 (risky):** for each benchmark and each dimension it scores, generators are ranked within each seed (STB: the
  dimension's pooled maximum z; SDMetrics: its calibrated statistic), and Kendall's tau-b is computed for each of
  the 10 pairs of seeds. A ranking is **unstable** if the mean tau-b is below 2/3 (with four generators: on average
  more than one of the six generator pairs reverses order between two seeds). H4 holds if every benchmark's ranking
  is unstable on every dimension it scores. Reported either way.

## 10. Generators
GC, CTGAN, TVAE, AIM (epsilon = 1); TabDDPM if the GPU environment works. 5 seeds each, both datasets,
default hyperparameters.

## Amendment log
(none yet; this file is still a draft)

Draft revisions before freezing (9 Nov, review): SynthEval removed (§3); combined U/P/F flaws removed and H2
restated on single flaws (§6, §9); copy_control added as a single-dose family; primary vs secondary comparators
and the matched-bases head-to-head budget (§7); predictions added to H1 and copy_test (§6, §9); the A6 check and
the environment lock (§4).

Draft revisions before freezing (11 Nov, before any confirmatory scoring): H1 restated as non-inferiority on
copy_random and superiority on near_copy, so that the hypothesis matches its own predictions, with the LoD
comparison rule made explicit; H3 given an exact binomial criterion, Fails and NotEvaluated counted separately,
and restricted to validation bases for a gate refitted after equity admission; H4 given a numeric threshold on
Kendall's tau-b. At the time of this revision only toy-size smoke runs on a Mac had been executed; they are not
evidence (§4) and no change above was made in response to their outputs.
