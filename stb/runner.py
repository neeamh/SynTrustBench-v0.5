"""Run many independent jobs in parallel, with pinned threads and resumable results.

A job is a picklable ``Job(stage, job_id, fn, kwargs)``. ``fn(**kwargs)`` returns a dict, which the worker
writes to the result store. Jobs whose result file exists are skipped. A failing job writes its traceback to
``<root>/failures/<job>.json`` and never stops the others.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
import multiprocessing
import os
import time
import traceback
from typing import Callable

from .store import ResultStore

THREAD_VARS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
               'NUMEXPR_NUM_THREADS', 'LOKY_MAX_CPU_COUNT')


@dataclass(frozen=True)
class Job:
    stage: str
    job_id: str
    fn: Callable
    kwargs: dict = field(default_factory=dict)


def _pin_threads(threads: int) -> None:
    for var in THREAD_VARS:
        os.environ[var] = str(threads)


def _execute(job: Job, store: ResultStore, threads: int) -> tuple[str, str, float]:
    _pin_threads(threads)
    from threadpoolctl import threadpool_limits
    start = time.perf_counter()
    try:
        with threadpool_limits(limits=threads):
            payload = job.fn(**job.kwargs)
        payload = {'job_id': job.job_id, 'seconds': round(time.perf_counter() - start, 2), **payload}
        store.write(job.stage, job.job_id, payload)
        return job.job_id, 'ok', payload['seconds']
    except Exception:  # noqa: BLE001 - recorded, never swallowed silently
        store.write('failures', f'{job.stage}/{job.job_id}', {'stage': job.stage, 'job_id': job.job_id,
                                                              'traceback': traceback.format_exc()})
        return job.job_id, 'failed', round(time.perf_counter() - start, 2)


def run_jobs(jobs: list[Job], store: ResultStore, *, workers: int = 1, threads: int = 4,
             progress_every: int = 10) -> dict:
    """Run every job not yet done. Returns counts; failures are listed under <root>/failures."""
    todo = [j for j in jobs if not store.done(j.stage, j.job_id)]
    summary = {'planned': len(jobs), 'skipped_done': len(jobs) - len(todo), 'ok': 0, 'failed': 0}
    if not todo:
        return summary
    start = time.perf_counter()
    if workers <= 1:
        results = (_execute(j, store, threads) for j in todo)
        iterator = results
    else:
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'))
        futures = [pool.submit(_execute, j, store, threads) for j in todo]
        iterator = (f.result() for f in as_completed(futures))
    for i, (job_id, state, seconds) in enumerate(iterator, 1):
        summary['ok' if state == 'ok' else 'failed'] += 1
        if state == 'failed':
            print(f'  FAILED {job_id} (see failures/)', flush=True)
        if i % progress_every == 0 or i == len(todo):
            elapsed = time.perf_counter() - start
            print(f'  {i}/{len(todo)} done, {elapsed / 60:.1f} min elapsed', flush=True)
    if workers > 1:
        pool.shutdown()
    return summary
