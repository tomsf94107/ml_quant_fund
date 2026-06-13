"""
analysis/explode_parallel.py - parallel drop-in for explode_panels().
Each base feature is independent (reads own panel + static bucket_map, writes
only {feat}__* keys) -> ProcessPoolExecutor across features. ~7h -> ~1h target.

Correctness contract: output dict must be IDENTICAL to serial explode_panels
(same keys, same values, same NaN positions). Verified by harness before prod use.
"""
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

# Worker imports the SAME operators the serial path uses - single source of truth.
from features.alpha_transformations import (
    signed_power, scale, cs_rank, cs_zscore, cs_demean,
    ts_mean, ts_std, ts_rank, ts_delta, ts_max, ts_min,
    ts_argmax, ts_decay_linear, group_neutralize,
)

_TS_OPS = {
    "ts_mean": ts_mean, "ts_std": ts_std, "ts_rank": ts_rank,
    "ts_delta": ts_delta, "ts_max": ts_max, "ts_min": ts_min,
    "ts_argmax": ts_argmax, "ts_decay_linear": ts_decay_linear,
}


def _explode_one(args):
    """Pure worker: one base feature -> its alpha sub-dict. Mirrors serial body EXACTLY."""
    feat_name, panel, bucket_map, ts_windows = args
    out = {}
    def _try(key, fn):
        try:
            out[key] = fn()
        except Exception as e:
            log.warning(f"  {key} failed: {e}")
    _try(f"{feat_name}__signed_power_p05", lambda: signed_power(panel, p=0.5))
    _try(f"{feat_name}__scale", lambda: scale(panel))
    _try(f"{feat_name}__cs_rank", lambda: cs_rank(panel))
    _try(f"{feat_name}__cs_zscore", lambda: cs_zscore(panel))
    _try(f"{feat_name}__cs_demean", lambda: cs_demean(panel))
    for op_name, op_fn in _TS_OPS.items():
        for w in ts_windows:
            _try(f"{feat_name}__{op_name}__w{w}", lambda op_fn=op_fn, w=w: op_fn(panel, window=w))
    if bucket_map is not None:
        _try(f"{feat_name}__group_neutralize", lambda: group_neutralize(panel, bucket_map))
    return out


def explode_panels_parallel(
    panels: dict,
    bucket_map: Optional[dict] = None,
    ts_windows: tuple = (5, 10, 20),
    max_workers: Optional[int] = None,
    verbose: bool = True,
) -> dict:
    if max_workers is None:
        max_workers = max(1, min(10, (os.cpu_count() or 4) - 4))  # leave 4 cores headroom
    tasks = [(fn, p, bucket_map, ts_windows) for fn, p in panels.items()]
    alphas = {}
    done = 0
    n = len(tasks)
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_explode_one, t): t[0] for t in tasks}
        for fut in as_completed(futs):
            sub = fut.result()
            alphas.update(sub)
            done += 1
            if verbose and done % 10 == 0:
                log.info(f"[{done}/{n}] features exploded ({max_workers} workers)")
    return alphas
