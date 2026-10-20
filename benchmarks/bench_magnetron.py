"""Vaned magnetrons (2D EM, r-theta cross-section in ZY): hot tests against published thresholds and the
A6 experiment.

  * A6 relativistic magnetron (Bekefi & Orzechowski 1976; Palevsky & Bekefi 1979) at the experimental
    operating point V = 360 kV, B = 0.8 T: must oscillate in the pi mode near the cold-test pi-mode
    frequency (2.31 GHz in this model; ~2.3 GHz measured).
  * Planar vane magnetron at V = 170 kV: inside the Hull / Buneman-Hartree window (B = 0.16 T) it must
    oscillate in the pi mode; above the Buneman-Hartree field (B = 0.19 T, V < V_BH) it must stay quiet.
    Relativistic planar Hull and BH conditions (Lau et al. 2010):
        B_H = (m c / e d) sqrt(gamma^2 - 1),
        e V_BH = e v_ph B d - m c^2 (1 - sqrt(1 - v_ph^2 / c^2)),  v_ph = w p / pi (pi mode).
  * A6 cylindrical thresholds (smooth-bore Hull: Lovelace & Ott 1974; cylindrical BH, pi mode n = 3).

    python benchmarks/bench_magnetron.py              # runs the three cases (~1-2 h on 2 cores)
    python benchmarks/bench_magnetron.py --from DIR   # analyse existing runs DIR/<case>
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import C, ROOT, WORK, plt, record, row  # noqa: E402
from picstudio.post.reader import Results  # noqa: E402

CASES = {
    "a6": dict(file="a6_magnetron_em.pic.json", params={"V0": "360e3", "B0": "0.8"}, t_end="40e-9",
               probe="probe_cavity2_Ey", port="P_port", f_pi=2.313e9, label="A6, 360 kV, 0.8 T",
               length=0.072),
    "planar": dict(file="planar_magnetron_em.pic.json", params={"V0": "170e3", "B0": "0.16"}, t_end="25e-9",
                   probe="probe_cav2_Ez", port="P_port", f_pi=5.85e9, label="planar, 170 kV, 0.16 T (in window)",
                   length=None),
    "planar_bh": dict(file="planar_magnetron_em.pic.json", params={"V0": "170e3", "B0": "0.19"}, t_end="25e-9",
                      probe="probe_cav2_Ez", port="P_port", f_pi=5.85e9,
                      label="planar, 170 kV, 0.19 T (above BH field)", length=None),
}


def gamma(V):
    return 1 + C.e * V / (C.me * C.c ** 2)


def planar_thresholds(V, d, f, p):
    BH = C.me * C.c / (C.e * d) * np.sqrt(gamma(V) ** 2 - 1)
    vph = 2 * np.pi * f * p / np.pi
    corr = C.me * C.c ** 2 / C.e * (1 - np.sqrt(1 - (vph / C.c) ** 2))
    B_BH = (V + corr) / (vph * d)              # magnetic field at which V = V_BH
    return BH, B_BH, vph


def a6_thresholds(V, rc, ra, f, n=3):
    de = (ra * ra - rc * rc) / (2 * ra)
    Bstar = C.me * C.c / (C.e * de) * np.sqrt(gamma(V) ** 2 - 1)
    w = 2 * np.pi * f
    corr = C.me * C.c ** 2 / C.e * (1 - np.sqrt(1 - (w * ra / (n * C.c)) ** 2))
    B_BH = (V + corr) / ((w / n) * (ra * ra - rc * rc) / 2)
    return Bstar, B_BH


def run_case(key):
    from picstudio.core.simulation import Simulation
    from picstudio.model.model import load
    c = CASES[key]
    m = load(os.path.join(ROOT, "examples", c["file"]))
    m["parameters"].update(c["params"])
    m["simulation"]["t_end"] = c["t_end"]
    m["diagnostics"].update(field_every="5000", particle_every="5000", live_every="100000000")
    out = os.path.join(WORK, "magnetron_" + key)
    Simulation(m, out_dir=out).run()
    return out


def analyse(key, out, axs):
    c = CASES[key]
    res = Results(out)
    H, cols = res.history()
    t = H[:, cols.index("t")]
    s = H[:, cols.index(c["probe"])]
    late = t > 0.6 * t[-1]
    ss = s[late] - s[late].mean()
    n = len(ss) * 8
    F = np.abs(np.fft.rfft(ss * np.hanning(len(ss)), n))
    f = np.fft.rfftfreq(n, np.median(np.diff(t[late])))
    band = (f > 0.5 * c["f_pi"]) & (f < 1.5 * c["f_pi"])          # fundamental near the pi mode
    fpk = f[band][np.argmax(F[band])]
    rms = float(np.sqrt(np.mean(ss ** 2)))
    P = H[:, cols.index(c["port"])]
    Pavg = float(P[late].mean())
    Ia = float(-H[late, cols.index("I_anode_block")].mean()) if "I_anode_block" in cols else float("nan")
    if axs is not None:
        axs[0].plot(t * 1e9, s / 1e6, lw=0.6)
        axs[0].set_xlabel("t (ns)")
        axs[0].set_ylabel("cavity E (MV/m)")
        axs[0].set_title(c["label"], fontsize=9)
        full = (f > 0.2 * c["f_pi"]) & (f < 3 * c["f_pi"])
        axs[1].semilogy(f[full] / 1e9, F[full] + 1e-30)
        axs[1].axvline(c["f_pi"] / 1e9, color="k", ls="--", lw=0.8, label="cold-test pi mode")
        axs[1].set_xlabel("f (GHz)")
        axs[1].set_title(f"spectrum: fundamental {fpk / 1e9:.3f} GHz", fontsize=9)
        axs[1].legend(fontsize=7)
        p = res.particles(-1)
        if p is not None and len(p.get("electrons__z", [])):
            axs[2].scatter(p["electrons__z"] * 100, p["electrons__y"] * 100, s=0.2, c="C3", rasterized=True)
            g = res.geometry_data()
            if g is not None:
                axs[2].contour(0.5 * (g["z"][1:] + g["z"][:-1]) * 100, 0.5 * (g["y"][1:] + g["y"][:-1]) * 100,
                               (g["cell_mat"] == 1).T.astype(float), levels=[0.5], colors="k", linewidths=0.6)
            axs[2].set_aspect("equal")
            axs[2].set_xlabel("z (cm)")
            axs[2].set_ylabel("y (cm)")
            axs[2].set_title("electrons (spokes)", fontsize=9)
    return dict(f=fpk, rms=rms, P=Pavg, Ia=Ia)


def main():
    src = None
    if "--from" in sys.argv:
        src = sys.argv[sys.argv.index("--from") + 1]
    keys = list(CASES)
    fig, axs = plt.subplots(len(keys), 3, figsize=(15, 4.2 * len(keys)))
    out = {}
    for k, key in enumerate(keys):
        d = os.path.join(src, key) if src else run_case(key)
        out[key] = analyse(key, d, axs[k])
    fig.tight_layout()
    a6, pl, pb = out["a6"], out["planar"], out["planar_bh"]
    Bs, Bbh6 = a6_thresholds(360e3, 1.58e-2, 2.11e-2, 2.313e9)
    BHp, Bbhp, vph = planar_thresholds(170e3, 1e-2, 5.85e9, 1e-2)
    rows = [
        row("A6 (360 kV, 0.8 T): fundamental / cold-test pi-mode frequency (oscillates in the pi mode)",
            a6["f"] / 2.313e9, 1.0, 0.10),
        row("A6: fundamental vs measured ~2.3 GHz (Palevsky & Bekefi 1979)", a6["f"] / 2.3e9, 1.0, 0.10),
        row("planar, inside Hull-BH window: fundamental / cold-test pi mode", pl["f"] / 5.85e9, 1.0, 0.10),
        {"case": "planar, B above the BH field: cavity field rms / in-window rms (quiet, < 0.1)",
         "value": pb["rms"] / max(pl["rms"], 1e-300), "reference": 0.0, "unit": "",
         "error": pb["rms"] / max(pl["rms"], 1e-300), "tol": 0.1, "passed": bool(pb["rms"] < 0.1 * pl["rms"])},
    ]
    L = CASES["a6"]["length"]
    notes = (f"A6 thresholds at 360 kV: smooth-bore Hull B* = {Bs:.3f} T (Lovelace-Ott), pi-mode BH field "
             f"{Bbh6:.3f} T; the run at 0.8 T sits in the window. 2D result per metre of axial length x "
             f"{L * 100:.1f} cm: anode current {a6['Ia'] * L / 1e3:.2f} kA, port power {a6['P'] * L / 1e6:.0f} MW, "
             f"electronic efficiency {a6['P'] / max(a6['Ia'] * 360e3, 1e-30) * 100:.1f} %. Experiment "
             f"(Bekefi & Orzechowski 1976): ~12 kA, ~1.7 GW at ~360 kV, ~0.8 T - the 2D model has no axial "
             f"end loss and its output (loss port, loaded Q = 250) is not the experimental extraction, so "
             f"current and power are indicative only; the frequency and mode are the benchmark. Planar "
             f"magnetron at 170 kV: Hull field B_H = {BHp:.3f} T, pi-mode BH field {Bbhp:.3f} T "
             f"(v_ph = {vph / C.c:.3f} c); B = 0.16 T is inside the window (oscillates, "
             f"f = {pl['f'] / 1e9:.3f} GHz), B = 0.19 T is above the BH field (V < V_BH: no oscillation, "
             f"field rms ratio {pb['rms'] / max(pl['rms'], 1e-300):.3f}).")
    record("21_magnetrons", "Vaned magnetrons (2D EM): A6 experiment and Hull / Buneman-Hartree window",
           rows, fig, notes=notes)


if __name__ == "__main__":
    main()
