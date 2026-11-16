#!/usr/bin/env bash
# Run inside tmux so a dropped connection does not kill the job:  tmux new -s stb   (detach: Ctrl-b d)
#
#   ./run.sh repro              reproduction test (prereg §4), required once per machine before a full run
#   ./run.sh selftest           calibration self-test (every Fail has e <= 5/101)
#   ./run.sh smoke cdc|uci      toy sizes, minutes, no evidence
#   ./run.sh full  cdc|uci      the preregistered run (Linux x86_64 only); re-running resumes, finished jobs are skipped
#   ./run.sh status             finished jobs per stage and failures, for every results folder
#
# Optional overrides: STB_WORKERS, STB_THREADS (default: physical cores / 4 workers x 4 threads).
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck disable=SC1091
source .venv/bin/activate
mkdir -p logs
STAMP="$(date +%Y%m%d-%H%M%S)"

case "${1:-}" in
  repro)
    python -m stb.repro_test --frozen data/cdc_frozen --scale 0.1 --work "runs/repro_${STAMP}" 2>&1 | tee "logs/repro_${STAMP}.log"
    exit "${PIPESTATUS[0]}" ;;
  selftest)
    python -m stb.calibrate 2>&1 | tee "logs/selftest_${STAMP}.log"
    exit "${PIPESTATUS[0]}" ;;
  status)
    python scripts/progress.py; exit 0 ;;
  smoke|full) ;;
  *) sed -n 2,10p "$0"; exit 2 ;;
esac

MODE="$1"; DATASET="${2:-}"
case "$DATASET" in cdc|uci) ;; *) echo "dataset must be cdc or uci"; exit 2 ;; esac
export STB_SMOKE=$([ "$MODE" = smoke ] && echo 1 || echo 0)
export STB_DATASET="$DATASET"
python scripts/verify_bundle.py --quiet

NAME="STB_${MODE}_${DATASET}_${STAMP}"
echo "Executing the notebook -> logs/${NAME}.ipynb   (live log: logs/${NAME}.log)"
jupyter nbconvert --to notebook --execute STB_v05_main_experiment.ipynb \
  --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 \
  --output-dir logs --output "${NAME}" 2>&1 | tee "logs/${NAME}.log"
