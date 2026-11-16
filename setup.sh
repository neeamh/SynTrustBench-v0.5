#!/usr/bin/env bash
# One-time setup on a fresh machine (Linux x86_64 for the confirmatory run; macOS works for smoke runs).
# Creates .venv with Python 3.12 and the exact pinned environment, installs the frozen syntrustbench v0.3
# package from vendor/, then checks everything with scripts/verify_bundle.py.
set -euo pipefail
cd "$(dirname "$0")"

PY="$(command -v python3.12 || true)"
if [ -z "$PY" ]; then
  echo "python3.12 not found; installing it with uv (user-level, no admin rights needed)"
  if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
  uv python install 3.12
  PY="$(uv python find 3.12)"
fi
echo "Using $PY ($("$PY" --version))"

"$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

# Scoring uses the CPU build of torch (prereg §4); on Linux take it from the CPU index to avoid a 2 GB CUDA wheel.
if [ "$(uname -s)" = "Linux" ]; then
  pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
fi
pip install -r requirements-stb-v0.5.txt
pip install --no-deps ./vendor/frozen_v030

python scripts/verify_bundle.py
echo
echo "Setup done. Next: ./run.sh repro   (required before the first confirmatory run, prereg §4)"
