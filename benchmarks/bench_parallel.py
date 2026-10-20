"""Parallel performance: numba threads (1..N) and MPI ranks on a 2D-ZY ES thermal plasma.

    python benchmarks/bench_parallel.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import RESULTS, new_model, plt, record, row  # noqa: E402
from picstudio.model.model import save  # noqa: E402


def model(ppc=64, steps=200, solver="electrostatic"):
    m = new_model("2D-ZY", solver)
    m["domain"].update(z_min="0", z_max="0.02", nz="128", y_min="0", y_max="0.02", ny="128")
    for s in ("z_min", "z_max", "y_min", "y_max"):
        m["boundaries"][s].update(field="periodic", particles="periodic")
    m["species"] = [{"name": "e", "preset": "electron"}]
    m["loaders"] = [dict(name="p", species="e", z0="0", z1="0.02", y0="0", y1="0.02", density="1e14",
                         ppc=str(ppc), temperature="5")]
    m["simulation"].update(n_steps=str(steps), t_end="1", dt="2e-11")
    m["diagnostics"].update(history_every="100", field_every="100000", particle_every="100000",
                            live_every="100000")
    return m


def timed_run(m, mode, threads, ranks=1):
    d = tempfile.mkdtemp()
    p = os.path.join(d, "m.pic.json")
    m = json.loads(json.dumps(m))
    m["parallel"].update(mode=mode, threads=threads)
    save(m, p)
    cmd = [sys.executable, "-m", "picstudio", "run", p, "--out", os.path.join(d, "o")]
    if ranks > 1:
        exe = shutil.which("mpiexec") or shutil.which("mpirun")
        if not exe:
            return None
        cmd = [exe, "-n", str(ranks)] + cmd + ["--mode", "mpi"]
    # warm-up (JIT cache) then timed run
    subprocess.run(cmd, capture_output=True, text=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    ms = None
    for line in r.stdout.splitlines():
        if "ms/step" in line:
            ms = float(line.split("(")[1].split(" ms/step")[0])
    shutil.rmtree(d, ignore_errors=True)
    return ms or wall * 1e3 / 200


def main():
    import numba
    ncores = numba.config.NUMBA_NUM_THREADS
    m = model()
    npart = 128 * 128 * 64
    res = {}
    threads = [1] + [t for t in (2, 4, 8, 16) if t <= ncores]
    for t in threads:
        res[f"threads={t}"] = timed_run(m, "serial" if t == 1 else "threads", t)
    if ncores >= 2:
        r = timed_run(m, "mpi", 1, ranks=2)
        if r:
            res["MPI 2 ranks x 1 thread"] = r
    base = res["threads=1"]
    rows = []
    for k, v in res.items():
        print(f"{k:28s} {v:8.2f} ms/step   speed-up {base / v:5.2f}   "
              f"{npart / (v * 1e-3) / 1e6:6.2f} M particle-steps/s")
    for k, v in res.items():
        if k != "threads=1":
            n = int(k.split("=")[1]) if k.startswith("threads") else 2
            rows.append(row(f"{k}: speed-up over 1 thread (ideal {n})", base / v, n, 0.6))
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.bar(list(res), [base / v for v in res.values()])
    ax.set_ylabel("speed-up")
    ax.set_title(f"2D ES, {npart / 1e6:.1f} M particles, {ncores} cores")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    record("20_parallel", f"Parallel speed-up (numba threads / MPI), {ncores}-core machine", rows, fig,
           notes=f"Single-thread throughput: {npart / (base * 1e-3) / 1e6:.2f} M particle-steps/s "
                 f"(push + deposit + scikit-fem Poisson on 128x128).")


if __name__ == "__main__":
    main()
