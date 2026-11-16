# SynTrustBench v0.5: run bundle

Everything needed to run the v0.5 experiment on another machine: the code (`stb/`, `syntrustbench_campaign/`,
`stb_cdc/`), the frozen data with its hashes (`data/`), the frozen `syntrustbench` v0.3 package
(`vendor/frozen_v030`, commit `3dd0f7c`), the pinned environment, the notebook and the preregistration.

Built on 12 Nov 2026 from `Synthetic Data/kaggle-upload/stb_code`. The code, data and other notebook cells are
identical to that folder; only the notebook's settings cell differs (it now reads the settings from environment
variables, so a run can start without editing anything). Results and logs are not included: they stay on the
machine that produced them.

## What the machine needs

- **Linux x86_64** for the confirmatory run (Ubuntu 20.04 or newer). macOS is fine for smoke runs.
  A Windows desktop works through WSL2 with Ubuntu: run everything inside the Ubuntu terminal.
- Internet access during setup (packages from PyPI and the PyTorch CPU index). No admin rights are needed:
  if Python 3.12 is missing, `setup.sh` installs it for your user with `uv`.
- Disk: about 3 GB for the environment, plus space for results.
- Cores: the full run uses (physical cores ÷ 4) jobs × 4 threads. The SJU server (64 cores) runs 16 jobs at once.

## Steps

```bash
# 0. Get the folder onto the machine (private git repo, scp or a USB drive), then:
cd SynTrustBench_run

# 1. One-time setup (10-20 minutes). Ends with a check of Python, pins, imports and data hashes.
./setup.sh

# 2. Checks required before the first confirmatory run on this machine (prereg §4)
./run.sh selftest          # calibration: every Fail has e <= 5/101
./run.sh repro             # reproduction test; must end with "REPRODUCTION TEST PASSED"

# 3. Smoke runs (toy sizes, minutes, no evidence). UCI has not been smoke-tested yet: do it before the full run.
./run.sh smoke uci
./run.sh smoke cdc

# 4. The confirmatory run, inside tmux so a dropped connection does not stop it
tmux new -s stb
./run.sh full cdc          # detach with Ctrl-b then d; reattach with: tmux attach -t stb
./run.sh full uci          # after CDC, or in a second tmux window if the machine has the cores

# Progress at any time (finished jobs per stage, failures)
./run.sh status
```

A run that stops (power, reboot, error) is resumed by running the same command again: every job writes its own
result file and finished jobs are skipped. The executed notebook and its log go to `logs/`, the results to
`runs/<plan>/<dataset>/`.

## Rules that keep the run valid

1. **Freeze the preregistration first.** `PREREGISTRATION_v0.5.md` is still marked DRAFT. Freeze it (change the
   header, record its SHA-256, timestamp it by email or commit) before step 4; every results manifest records
   its hash, and `scripts/verify_bundle.py` warns while it says DRAFT.
2. **One machine per results folder.** Classifier-based signals differ slightly between machines, so all of one
   dataset's confirmatory results must come from one machine; the manifest refuses to mix platforms or code
   versions. Do not start CDC on one computer and finish it on another.
3. **The preregistration names the SJU server.** Running the confirmatory run on any other machine (for example
   an office desktop) needs a dated revision of the preregistration before scoring, like the 11 Nov revision.
4. **Do not edit `stb/`, `syntrustbench_campaign/` or `stb_cdc/` between runs.** Their hashes go into the
   manifest; a change means a new results folder.

## Putting the folder in a private git repository

The largest file is 23 MB, so it fits in an ordinary GitHub repository. Keep the repository **private**: the
code names the lab, which matters for double-blind review.

```bash
cd SynTrustBench_run
git init && git add . && git commit -m "SynTrustBench v0.5 run bundle"
gh repo create SynTrustBench-run --private --source . --push     # or create it on github.com and push
# on the other machine:
git clone <repo url> && cd SynTrustBench-run && ./setup.sh
```

`.gitignore` keeps `.venv/`, `runs/` and `logs/` out of the repository.

## Files

| Path | What it is |
|---|---|
| `STB_v05_main_experiment.ipynb` | the experiment; `run.sh` executes it headless |
| `build_notebook.py` | source of the notebook (edit cells here, then `python build_notebook.py`) |
| `stb/` | STB v0.5 modules (planting, calibration, limits of detection, analysis) |
| `syntrustbench_campaign/` | signal implementations reused from the gate paper's library (snapshot, see `SNAPSHOT_PROVENANCE.json`) |
| `stb_cdc/` | CDC adapter, used by the reproduction test |
| `vendor/frozen_v030/` | frozen `syntrustbench` v0.3 package, installed with `--no-deps` |
| `data/` | frozen splits for CDC and UCI with `split_manifest.json` hashes, and the CDC mirror |
| `requirements-stb-v0.5.txt` | the pinned environment (prereg §4) |
| `scripts/verify_bundle.py` | checks Python, platform, pins, imports, data hashes and the preregistration |
| `scripts/progress.py` | what `./run.sh status` prints |
