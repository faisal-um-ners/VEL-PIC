"""Shared helpers for the VEXPACE validation benchmarks."""
from __future__ import annotations

import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)
WORK = os.path.join(HERE, "work")

from picstudio import constants as C  # noqa: E402,F401
from picstudio.core.simulation import Simulation  # noqa: E402
from picstudio.model.model import new_model  # noqa: E402,F401


def run(model, name, steps=None):
    if steps is not None:
        model["simulation"]["n_steps"] = str(steps)
    d = model["diagnostics"]
    d.setdefault("field_every", "1000000")
    t0 = time.time()
    sim = Simulation(model, out_dir=os.path.join(WORK, name), log=lambda m: None)
    sim.run()
    sim.wall = time.time() - t0
    return sim


def quiet_diag(m, h=10, f=10 ** 7, p=10 ** 7):
    m["diagnostics"].update(history_every=str(h), field_every=str(f), particle_every=str(p),
                            live_every=str(10 ** 7))


def hist(sim, key):
    H, cols = sim.history_array()
    return H[:, cols.index("t")], H[:, cols.index(key)]


def tail_mean(sim, key, frac=0.4):
    t, v = hist(sim, key)
    return float(np.mean(v[int(len(v) * (1 - frac)):]))


def fit_frequency(t, s):
    """Dominant angular frequency of a sampled signal (FFT peak refined by least squares)."""
    from scipy.optimize import curve_fit
    s = np.asarray(s) - np.mean(s)
    n = len(s) * 16
    F = np.abs(np.fft.rfft(s * np.hanning(len(s)), n))
    f = np.fft.rfftfreq(n, t[1] - t[0])
    w0 = 2 * np.pi * f[np.argmax(F[1:]) + 1]

    def model(tt, A, w, ph, B):
        return A * np.cos(w * tt + ph) + B
    try:
        p, _ = curve_fit(model, t, s, p0=[np.std(s) * 1.4, w0, 0.0, 0.0], maxfev=20000)
        return abs(p[1])
    except Exception:  # noqa: BLE001
        return w0


def fit_growth(t, y, lo, hi):
    """Exponential growth rate of y between amplitudes lo..hi (fraction of max), via log-linear fit."""
    y = np.asarray(y)
    ymax = y.max()
    sel = (y > lo * ymax) & (y < hi * ymax) & (t < t[np.argmax(y)])
    if sel.sum() < 4:
        return float("nan"), sel
    p = np.polyfit(t[sel], np.log(y[sel]), 1)
    return p[0], sel


def record(name, title, rows, fig=None, notes=""):
    """rows: list of dicts with keys case, value, reference, unit, error (rel), passed."""
    data = {"name": name, "title": title, "rows": rows, "notes": notes,
            "passed": all(r.get("passed", True) for r in rows)}
    with open(os.path.join(RESULTS, f"{name}.json"), "w") as fh:
        json.dump(data, fh, indent=2, default=float)
    if fig is not None:
        fig.savefig(os.path.join(RESULTS, f"{name}.png"), dpi=130)
        plt.close(fig)
    print(f"== {title}")
    for r in rows:
        print(f"   {r['case']:<45s} sim = {r['value']:.5g}  ref = {r['reference']:.5g}  "
              f"err = {100 * r['error']:+.2f}%  {'PASS' if r['passed'] else 'FAIL'}")
    return data


def row(case, value, reference, tol, unit="", absolute=False):
    err = (value - reference) if absolute else (value / reference - 1 if reference != 0 else value)
    return {"case": case, "value": float(value), "reference": float(reference), "unit": unit,
            "error": float(err), "tol": tol, "passed": bool(abs(err) <= tol)}
