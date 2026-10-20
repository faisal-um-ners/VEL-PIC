"""Limiting current of small emitters with low emission energy:
Lau, Chernin, Heri & Zhang, arXiv:2607.24590 (2026).

1. Steady confined-flow solver (picstudio.core.confined_flow, paper Eq. 7) against the paper's
   POISSON values (Table 1), the packing-density table and the circular patch.
2. PIC (optional, ``--pic``): the shipped examples yylau_2026_* run just below and just above the
   steady limit; below it all current must reach the anode, above it a virtual cathode must return
   part of the current to the cathode (the paper's XOOPIC test).

    python benchmarks/bench_lau2026.py          # steady solver (about 1 minute)
    python benchmarks/bench_lau2026.py --pic    # + PIC runs (about 30-60 minutes on 2 cores)
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
from common import ROOT, WORK, plt, record, row  # noqa: E402
from picstudio.core.confined_flow import ConfinedFlow, jaffe  # noqa: E402

# paper Table 1 (POISSON): (test, w/d, p/d, Delta, J_max/J_CL)
TABLE1 = [("E1", 0.1, 5.12, 1e-2, 5.191), ("E2", 0.05, 2.56, 1e-3, 4.829), ("E3", 0.01, 2.048, 1e-2, 32.863),
          ("E5", 0.01, 2.048, 1e-4, 9.226), ("E6", 0.01, 2.048, 1e-5, 7.824)]


def steady():
    rows = []
    res = []
    for name, w, p, D, ref in TABLE1:
        cf = ConfinedFlow("strip", w=w, p=p, delta=D)
        J = cf.jmax()
        res.append((name, w, D, J, ref, cf))
        rows.append(row(f"strip {name} (w/d={w:g}, p/d={p:g}, Delta={D:g}) vs paper POISSON", J, ref, 0.01))
    # packing density (tutorial Table, <J_max> = (w/p) J_max/(f_J J_CL)), w/d = 0.01, Delta = 0.01
    cf = ConfinedFlow("strip", w=0.01, p=0.5, delta=0.01)
    J = cf.jmax()
    ref = 0.446 * jaffe(0.01) * 0.5 / 0.01
    rows.append(row("strip w/d=0.01, p/d=0.5, Delta=0.01 (packing table <J>=0.446)", J, ref, 0.01))
    # circular patch, Delta = 1e-5: tutorial self-consistent value (finite-volume Newton, R/d = 0.1)
    cf = ConfinedFlow("disk", R=0.1, r_max=1.0, delta=1e-5)
    J = cf.jmax()
    rows.append(row("disk R/d=0.1, Delta=1e-5 vs tutorial (2.723)", J, 2.723, 0.03))
    first = 1.328 / np.sqrt(0.1) * (1 + 0.34 * 0.1 ** 1.5)
    rows.append(row("disk: self-consistent / first-order Eq.(B10) (tutorial 0.62)", J / first, 0.62, 0.03))
    # figure: centre-line potential of E1 and E3 at J_max
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for name, w, D, J, ref, cf in res:
        if name in ("E1", "E3"):
            z, ph = cf.centerline()
            ax[0].plot(z, ph / D, label=f"{name}: J = {J:.3f} J_CL")
    ax[0].set_xlim(0, 0.06)
    ax[0].set_ylim(-1.0, 3)
    ax[0].axhline(-1, color="k", lw=0.6, ls=":")
    ax[0].set_xlabel("z / d")
    ax[0].set_ylabel("phi / (Delta V)")
    ax[0].set_title("centre-line potential at J_max (minimum above -Delta)")
    ax[0].legend()
    names = [r[0] for r in res]
    ax[1].bar(np.arange(len(res)) - 0.2, [r[4] for r in res], 0.4, label="paper POISSON")
    ax[1].bar(np.arange(len(res)) + 0.2, [r[3] for r in res], 0.4, label="this code (FEM, Eq. 7)")
    ax[1].set_xticks(range(len(res)), names)
    ax[1].set_ylabel("J_max / J_CL")
    ax[1].legend()
    fig.tight_layout()
    record("24_lau2026_steady", "Lau et al. 2026: limiting current of small emitters (steady Eq. 7)", rows,
           fig=fig, notes="Strip array (planar) and circular patch (RZ), confined flow with injection energy "
                          "Delta = E_in/eV. Finite-element Picard iteration with warm starts and bisection "
                          "(threshold 1e-3 min(Delta, 1e-2)), tensor mesh refined across the emitter and "
                          "graded towards the cathode.")


def pic():
    import json
    from picstudio import load_model
    from picstudio.core.simulation import Simulation
    cases = [("yylau_2026_strip_caseA_zy.pic.json", 5.197, "strip w/d=0.1"),
             ("yylau_2026_disk_rz.pic.json", 5.872, "disk R/d=0.1")]
    rows = []
    out = {}
    for fn, Jref, label in cases:
        for fac in (0.96, 1.04):
            m = load_model(os.path.join(ROOT, "examples", fn))
            m["parameters"]["Jn"] = repr(fac * Jref)
            m["emitters"][0]["particles_per_cell"] = "1"
            s = Simulation(m, out_dir=os.path.join(WORK, f"lau_{fn[:-9]}_{fac}"), log=lambda x: None)
            s.run()
            H, c = s.history_array()
            k = H[:, 1] > 0.6 * H[-1, 1]
            em = H[k, c.index("Iemit_cathode")].mean()
            an = H[k, c.index("I_anode")].mean() / em
            rt = H[k, c.index("I_cathode")].mean() / em
            out[f"{label} {fac}"] = dict(anode=an, returned=rt)
            if fac < 1:
                rows.append(row(f"PIC {label}, J = 0.96 J_max: transmitted fraction", an, 1.0, 0.01))
            else:
                rows.append({"case": f"PIC {label}, J = 1.04 J_max: returned fraction (> 2 %)",
                             "value": float(rt), "reference": 0.0, "unit": "", "error": float(rt),
                             "tol": 0.02, "passed": bool(rt > 0.02)})
    record("25_lau2026_pic", "Lau et al. 2026: PIC onset of the virtual cathode at the steady J_max", rows,
           notes="Shipped examples yylau_2026_strip_caseA_zy and yylau_2026_disk_rz (z-only motion, refined "
                 "mesh), injected current 4 % below / above the steady limit of Eq. (7). " + json.dumps(out))


if __name__ == "__main__":
    steady()
    if "--pic" in sys.argv:
        pic()
