"""End-to-end smoke campaign on CDC: every stage, tiny sizes. Not paper evidence.

Exercises cohort loading, null runs, calibration, all generators (incl. DP AIM),
every flaw family, the judges, analysis and report with the CDC schema, so
schema problems surface before the server runs.

Usage (from stb_code/, with the pinned venv):
    STB_FROZEN_CHECKOUT="<gate folder>/syntrustbench_v04 2/syntrustbench_v04_dependencies/frozen_v030" \
    python stb_cdc/run_cdc_smoke_campaign.py --frozen data/cdc_frozen --out runs/cdc_smoke
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdc_adapter import cdc_config  # noqa: E402
from syntrustbench_campaign.campaign import Campaign  # noqa: E402
from syntrustbench_campaign.config import STAGES  # noqa: E402
from syntrustbench_campaign.environment import Environment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--frozen', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--checkout', default=os.environ.get('STB_FROZEN_CHECKOUT'))
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--skip', default='', help='comma list of stages to skip (e.g. smoke)')
    args = parser.parse_args()
    if not args.checkout:
        raise SystemExit('Set --checkout or STB_FROZEN_CHECKOUT to the frozen v0.3 checkout (frozen_v030).')
    skip = set(filter(None, args.skip.split(',')))
    stages = tuple(s for s in STAGES if s not in skip)
    cfg = cdc_config(args.frozen, output_root=str(Path(args.out).resolve()), checkout_dir=args.checkout,
                     smoke=True, dry_run=False, workers=1, threads=args.threads, stages=stages)
    print('scientific fingerprint:', cfg.fingerprint[:16], '->', cfg.root)
    Environment(cfg).setup()
    campaign = Campaign(cfg)
    times = {}
    for stage in STAGES:
        start = time.perf_counter()
        result = campaign.run_stage(stage)
        times[stage] = round(time.perf_counter() - start, 1)
        print(f'{stage:>11}: {result.get("state")} ({result.get("jobs")} jobs, '
              f'{result.get("incomplete", 0)} incomplete) in {times[stage]} s', flush=True)
    audit = campaign.completion_audit()
    print(json.dumps({'stage_seconds': times, 'completed': audit['completed'], 'failed': audit['failed'],
                      'unavailable': audit['unavailable'], 'planned': audit['planned'],
                      'incomplete': audit['incomplete'][:20]}, indent=2))
    if audit['failed']:
        print('Failures: see', cfg.root / 'failures.jsonl')


if __name__ == '__main__':
    main()
