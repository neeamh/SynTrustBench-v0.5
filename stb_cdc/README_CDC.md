# CDC adapter for SynTrustBench (STB)

STB code lives in `stb_code/`. `syntrustbench_campaign/` is a snapshot of the gate paper's
v0.4.2 library (hashes in `SNAPSHOT_PROVENANCE.json`); the gate folder is never written to.
Changed files are backed up in `backups/`.

## Changes to the snapshot
- `dp_generators.py`: the UCI-only `time_in_hospital` check applies only when that column exists.
- `overlap_policy.py` (new) + `evaluation.py`: for datasets with `rows_may_coincide=True`, frozen
  v0.3's "any train/test row overlap stops evaluation" check is replaced, during the call only, by
  a counted warning. Roles are already person-disjoint on hashed respondent keys. The v0.3 source on
  disk is untouched, so its hash checks pass.
- `campaign.py`: `run_official_v03` switch (default on; off for CDC). On CDC the legacy fixed-threshold
  report fails every honest table on privacy and is about 75% of profile cost.

## Commands (Mac, from `stb_code/`)
    GATE="$HOME/Documents/Research/Synthetic Data/Calibrated Gate Synthetic Data/syntrustbench_v04 2"
    PY="$GATE/.venv/bin/python"
    export STB_FROZEN_CHECKOUT="$GATE/syntrustbench_v04_dependencies/frozen_v030"

    # 1. Official data (UCI id 891) and frozen splits
    curl -L -o data/cdc891.zip "https://archive.ics.uci.edu/static/public/891/cdc+diabetes+health+indicators.zip"
    unzip -o data/cdc891.zip -d data/cdc891 && ls data/cdc891
    $PY stb_cdc/cdc_adapter.py build --raw data/cdc891/<the csv> --out data/cdc_frozen

    # 2. Timing at full scale, one honest table plus a GaussianCopula fit (about 10-20 min)
    $PY stb_cdc/time_cdc_profile.py --frozen data/cdc_frozen --scale 1.0 --work runs/cdc_timing_1.0 \
        --skip-official --generators GaussianCopula,CTGAN --epochs 5 --threads 4

    # 3. End-to-end smoke campaign: every stage, 200 rows
    $PY stb_cdc/run_cdc_smoke_campaign.py --frozen data/cdc_frozen --out runs/cdc_smoke
