"""Command-line interface:  python -m picstudio <command> ...

Commands
  gui   [project.pic.json]          start the graphical user interface
  run   project.pic.json [options]  run a simulation (use mpiexec -n N for MPI)
  sweep project.pic.json            run the parametric sweep defined in the project
  check project.pic.json            validate a project file
  examples [dest]                   list / copy the example projects
  info                              versions, threads, MPI availability
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys


def _examples_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, "..", "examples"), os.path.join(here, "examples")):
        if os.path.isdir(cand):
            return os.path.abspath(cand)
    return None


def new_fill(m, name):
    for o in m["objects"]:
        if o.get("material") == "lossy" and name in (None, "", o.get("name")):
            return o.get("fill")
    return None


def main(argv=None):
    from . import APP_NAME
    p = argparse.ArgumentParser(prog="picstudio", description=f"{APP_NAME} - 1D/2D (ZY, RZ) particle-in-cell studio (package picstudio)")
    sub = p.add_subparsers(dest="cmd")
    g = sub.add_parser("gui")
    g.add_argument("project", nargs="?")
    r = sub.add_parser("run")
    r.add_argument("project")
    r.add_argument("--out", default=None, help="output folder (default from project)")
    r.add_argument("--threads", type=int, default=None)
    r.add_argument("--mode", choices=["serial", "threads", "mpi"], default=None)
    r.add_argument("--steps", type=int, default=None, help="override number of steps")
    s = sub.add_parser("sweep")
    s.add_argument("project")
    q = sub.add_parser("qcal", help="calibrate the loaded Q of a lossy object by cold-test ring-down")
    q.add_argument("project")
    q.add_argument("--object", default=None)
    q.add_argument("--measure-only", action="store_true")
    c = sub.add_parser("check")
    c.add_argument("project")
    e = sub.add_parser("examples")
    e.add_argument("dest", nargs="?")
    sub.add_parser("info")
    a = p.parse_args(argv)

    if a.cmd in (None, "gui"):
        from .gui.app import main as gui_main
        return gui_main(getattr(a, "project", None))

    from .model import model as M
    if a.cmd == "run":
        mdl = M.load(a.project)
        if a.threads is not None:
            mdl["parallel"]["threads"] = a.threads
        if a.mode is not None:
            mdl["parallel"]["mode"] = a.mode
        if a.steps is not None:
            mdl["simulation"]["n_steps"] = str(a.steps)
        errs = [m for lvl, m in M.validate(mdl) if lvl == "error"]
        if errs:
            print("Model errors:\n  " + "\n  ".join(errs))
            return 2
        from .core.simulation import Simulation
        sim = Simulation(mdl, out_dir=a.out)
        ok = sim.run()
        return 0 if ok else 1
    if a.cmd == "sweep":
        mdl = M.load(a.project)
        from .core.sweep import run_sweep
        run_sweep(mdl, os.path.dirname(os.path.abspath(a.project)))
        return 0
    if a.cmd == "qcal":
        mdl = M.load(a.project)
        from .core.qcal import calibrate_q, measure_q
        if a.measure_only:
            ev = M.evaluator(mdl)
            obj = [o for o in mdl["objects"] if o.get("material") == "lossy"][0]
            Q, f = measure_q(mdl, ev.num(obj["frequency"]))
            print(f"Measured loaded Q = {Q:.4g} at f = {f / 1e9:.5g} GHz")
            return 0
        new, hist = calibrate_q(mdl, a.object)
        M.save(new, a.project)
        fill, Qm, f = hist[-1]
        print(f"QCAL_RESULT fill={fill:.6g} Q={Qm:.6g} f={f:.6g}")
        print(f"Calibrated: fill = {new_fill(new, a.object)}; last measured Q = {Qm:.4g} at {f / 1e9:.5g} GHz. "
              f"Saved to {a.project}")
        return 0
    if a.cmd == "check":
        mdl = M.load(a.project)
        bad = False
        for lvl, msg in M.validate(mdl):
            print(f"[{lvl}] {msg}")
            bad |= lvl == "error"
        return 1 if bad else 0
    if a.cmd == "examples":
        d = _examples_dir()
        files = sorted(f for f in os.listdir(d) if f.endswith(".pic.json")) if d else []
        if a.dest:
            os.makedirs(a.dest, exist_ok=True)
            for f in files:
                shutil.copy(os.path.join(d, f), a.dest)
            print(f"Copied {len(files)} examples to {a.dest}")
        else:
            for f in files:
                print(f)
        return 0
    if a.cmd == "info":
        import numba
        import numpy
        import scipy
        import skfem
        from . import APP_NAME, __version__
        print(f"{APP_NAME} {__version__}")
        print(f"python {sys.version.split()[0]}, numpy {numpy.__version__}, scipy {scipy.__version__}, "
              f"scikit-fem {skfem.__version__}, numba {numba.__version__}")
        print(f"numba threads available: {numba.config.NUMBA_NUM_THREADS}")
        try:
            import mpi4py
            print(f"mpi4py {mpi4py.__version__} available")
        except ImportError:
            print("mpi4py not installed (MPI mode unavailable; threads still work)")
        try:
            import PySide6
            print(f"PySide6 {PySide6.__version__} available")
        except ImportError:
            print("PySide6 not installed (GUI unavailable)")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
