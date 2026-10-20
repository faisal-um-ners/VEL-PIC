"""Parametric sweeps (COMSOL-style): run one model for a list of parameter values,
in parallel worker processes, and tabulate time-averaged results."""
from __future__ import annotations

import copy
import csv
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ..expr import Evaluator
from ..model.model import complete


def sweep_values(model):
    sw = model["sweep"]
    ev = Evaluator(model.get("parameters", {}))
    vals = ev.value(sw["values"])
    return [float(v) for v in np.atleast_1d(np.asarray(vals, float))]


def _run_one(args):
    model, out_dir, threads = args
    from .simulation import Simulation
    model = copy.deepcopy(model)
    model["parallel"]["mode"] = "threads" if threads > 1 else "serial"
    model["parallel"]["threads"] = threads
    sim = Simulation(model, out_dir=out_dir, log=lambda m: None)
    sim.run()
    H, cols = sim.history_array()
    n = len(H)
    tail = H[int(0.7 * n):] if n > 3 else H
    return cols, tail.mean(axis=0).tolist(), H[-1].tolist()


def run_sweep(model, base_dir, log=print):
    m = complete(model)
    sw = m["sweep"]
    name = sw["parameter"].strip()
    if not name:
        raise ValueError("Sweep: no parameter name given")
    if name not in m.get("parameters", {}):
        raise ValueError(f"Sweep: '{name}' is not a global parameter")
    vals = sweep_values(m)
    workers = max(1, int(sw.get("workers", 2) or 1))
    ncpu = os.cpu_count() or 1
    threads = max(1, ncpu // workers)
    jobs = []
    root = os.path.join(base_dir, f"sweep_{name}")
    os.makedirs(root, exist_ok=True)
    for i, v in enumerate(vals):
        mm = copy.deepcopy(m)
        mm["parameters"][name] = repr(v)
        mm["sweep"]["enabled"] = False
        jobs.append((mm, os.path.join(root, f"run_{i:03d}"), threads))
    log(f"Sweep over {name} = {vals} with {workers} workers x {threads} threads")
    results = []
    if workers == 1:
        for j in jobs:
            results.append(_run_one(j))
            log(f"  finished {len(results)}/{len(jobs)}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for k, r in enumerate(ex.map(_run_one, jobs)):
                results.append(r)
                log(f"  finished {k + 1}/{len(jobs)}")
    cols = results[0][0]
    path = os.path.join(root, "sweep_summary.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([name] + [f"avg_{c}" for c in cols])
        for v, r in zip(vals, results):
            w.writerow([v] + r[1])
    log(f"Sweep summary written to {path}")
    return path
