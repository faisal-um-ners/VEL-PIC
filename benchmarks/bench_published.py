"""Benchmarks against classical published results (planar and cylindrical).

32  Pierce diode instability (1D ES, planar): growth rate vs the exact root of the Pierce (1944)
    dispersion relation (stable for alpha < pi, purely growing for pi < alpha < 2 pi, oscillatory
    growth for 2 pi < alpha < ~2.9 pi; Godfrey 1987, Birdsall & Langdon).
34  Space-charge-limited emission with the EM solver (planar Child-Langmuir, coaxial Langmuir-Blodgett).
33  Cylindrical Hull cutoff of a smooth-bore (coaxial) magnetron diode (RZ, inner cathode, axial B):
    B* = (m c / e d_e) sqrt(gamma^2 - 1), d_e = (r_a^2 - r_c^2) / (2 r_a)  (Lovelace & Ott 1974).

    python benchmarks/bench_published.py [pierce] [coax_hull]
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
from common import C, hist, new_model, plt, quiet_diag, record, row, run, tail_mean  # noqa: E402


def _r(x):
    return repr(float(x))


# ----------------------------------------------------------------------------------------------
def pierce_det(w, alpha):
    """Determinant of the cold-beam Pierce-diode boundary-value problem (units v0 = L = 1, w_p = alpha):
    phi = a + b z + c+ exp(i(w + alpha) z) + c- exp(i(w - alpha) z) with phi(0) = phi(1) = 0 and no
    density / velocity perturbation of the injected beam at z = 0."""
    Dp, Dm = 1j * (w + alpha), 1j * (w - alpha)
    M = np.array([[1, 0, 1, 1],
                  [1, 1, np.exp(Dp), np.exp(Dm)],
                  [0, 0, Dp ** 2, Dm ** 2],
                  [0, 1 / w, -Dp / alpha, Dm / alpha]], complex)
    return np.linalg.det(M)


def pierce_root(alpha, guess):
    from scipy.optimize import fsolve

    def f(x):
        D = pierce_det(x[0] + 1j * x[1], alpha)
        return [D.real, D.imag]
    r = fsolve(f, guess, xtol=1e-14)
    return r[0] + 1j * r[1]


def _pierce_model(alpha, nz=200, amp=1e-3, tcross=12.0):
    v0, n0 = 1e6, 1e13
    wp = np.sqrt(n0 * C.e ** 2 / (C.eps0 * C.me))
    L = alpha * v0 / wp
    m = new_model("1D-Z", "electrostatic")
    m["domain"].update(z_min="0", z_max=_r(L), nz=str(nz))
    for s in ("z_min", "z_max"):
        m["boundaries"][s].update(field="conductor", voltage="0", particles="absorb")
    m["species"] = [{"name": "e", "preset": "electron"},
                    {"name": "ion", "preset": "custom", "charge": "1", "mass": "1e6*mp"}]   # immobile background
    m["loaders"] = [dict(name="e0", species="e", z0="0", z1=_r(L), y0="0", y1="1", density=_r(n0), ppc="64",
                         quiet_start=True, drift=[_r(v0), "0", "0"], perturbation="velocity",
                         pert_amplitude=_r(amp * v0), pert_mode="0.5"),
                    dict(name="i0", species="ion", z0="0", z1=_r(L), y0="0", y1="1", density=_r(n0), ppc="64",
                         quiet_start=True)]
    m["emitters"] = [{"name": "gun", "species": "e", "model": "beam", "z0": "0", "y0": "0", "z1": "0", "y1": "1",
                      "normal": "+z", "current_density": _r(C.e * n0 * v0), "energy": _r(0.5 * C.me * v0 ** 2 / C.e),
                      "temperature": "0", "particles_per_cell": "64"}]
    m["simulation"].update(dt=_r(0.25 * L / nz / v0), t_end=_r(tcross * L / v0), initial_field="poisson")
    quiet_diag(m, 4)
    return m, L, v0


def _growth(t, W, lo, hi):
    """Least-squares growth rate of the field amplitude sqrt(W) while W grows from lo to hi times its
    saturated value (linear phase)."""
    Ws = W.max()
    i0 = int(np.argmax(W > lo * Ws))
    i1 = int(np.argmax(W > hi * Ws))
    k = np.zeros(len(t), bool)
    k[i0:i1] = True
    p = np.polyfit(t[k], 0.5 * np.log(W[k]), 1)
    return p[0], k


def pierce():
    rows = []
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    cases = [(1.5, (0.0, 0.7)), (2.5, (1.4, 0.6))]
    notes = []
    for a, guess in cases:
        w = pierce_root(a * np.pi, list(guess))
        m, L, v0 = _pierce_model(a * np.pi)
        sim = run(m, f"pierce_{a}")
        t, W = hist(sim, "W_field")
        tn = t * v0 / L
        g, k = _growth(tn, W, 1e-4, 3e-2)
        rows.append(row(f"Pierce diode alpha = {a:g} pi: growth rate (v0/L) vs exact dispersion root", g, w.imag, 0.06))
        notes.append(f"alpha = {a:g} pi: exact w = {w.real:.4f} {w.imag:+.4f}i v0/L, PIC growth {g:.4f}")
        if abs(w.real) > 1e-6:
            # oscillatory growth: W ~ exp(2 gamma t) (1 + cos(2 w t)); 2w from the spectrum of the
            # detrended log W over the linear phase
            tt, lw = tn[k], np.log(W[k])
            r = lw - np.polyval(np.polyfit(tt, lw, 1), tt)
            dtn = np.median(np.diff(tt))
            F = np.abs(np.fft.rfft((r - r.mean()) * np.hanning(len(r)), 32 * len(r)))
            fr = np.fft.rfftfreq(32 * len(r), dtn)
            wr = np.pi * fr[np.argmax(F[1:]) + 1]
            rows.append(row(f"Pierce diode alpha = {a:g} pi: oscillation frequency (v0/L)", wr, abs(w.real), 0.03))
            notes[-1] += f", frequency {wr:.4f}"
        i = 0 if a < 2 else 1
        ax[i].semilogy(tn, W, lw=1, label="VEL-PIC field energy")
        t0 = tn[k][0]
        ax[i].semilogy(tn[k], W[k][0] * np.exp(2 * w.imag * (tn[k] - t0)), "k--", label=f"exact 2*gamma = {2 * w.imag:.3f} v0/L")
        ax[i].set_xlabel("t v0 / L")
        ax[i].set_ylabel("W_field (J/m^2)")
        ax[i].set_title(f"Pierce diode, alpha = {a:g} pi")
        ax[i].legend(fontsize=7)
    fig.tight_layout()
    return record("32_pierce_instability", "Pierce diode instability (1D ES, planar) vs the exact dispersion relation",
                  rows, fig, notes="Cold beam injected into a grounded gap with an immobile neutralising background, "
                                   "alpha = w_p L / v0. " + "; ".join(notes) + ". The determinant root also "
                                   "reproduces the classical stability map (no growing root for alpha < pi, purely "
                                   "growing for pi < alpha < 2 pi, stable again near alpha = 2.9 pi).")


# ----------------------------------------------------------------------------------------------
def coax_hull(ratios=(0.8, 0.9, 0.95, 1.05, 1.1, 1.2)):
    V, a, b = 10e3, 5e-3, 10e-3
    gam = 1 + C.e * V / (C.me * C.c ** 2)
    de = (b * b - a * a) / (2 * b)
    Bs = C.me * C.c / (C.e * de) * np.sqrt(gam ** 2 - 1)
    nr, nz = 64, 8
    Lz = (b - a) * nz / nr
    Ia = []
    I0 = None
    for r in (0.0,) + tuple(ratios):
        m = new_model("2D-RZ", "electrostatic")
        m["domain"].update(z_min="0", z_max=_r(Lz), nz=str(nz), y_min=_r(a), y_max=_r(b), ny=str(nr))
        for s in ("z_min", "z_max"):
            m["boundaries"][s].update(field="periodic", particles="periodic")
        m["boundaries"]["y_min"].update(field="conductor", voltage="0", particles="absorb")
        m["boundaries"]["y_max"].update(field="conductor", voltage=_r(V), particles="absorb")
        m["species"] = [{"name": "e", "preset": "electron"}]
        m["emitters"] = [{"name": "k", "species": "e", "model": "scl", "z0": "0", "y0": _r(a), "z1": _r(Lz),
                          "y1": _r(a), "normal": "+y", "particles_per_cell": "4"}]
        if r > 0:
            m["external_fields"] = [{"name": "B", "type": "uniform", "B": [_r(r * Bs), "0", "0"]}]
        m["simulation"].update(t_end="5e-9")
        quiet_diag(m, 20)
        sim = run(m, f"coax_hull_{r}")
        I = -tail_mean(sim, "I_y_max", 0.5)
        if r == 0:
            I0 = I
        else:
            Ia.append(I / I0)
    Ia = np.array(Ia)
    ratios = list(ratios)
    below = [x for x, rr in zip(Ia, ratios) if rr < 1]
    above = [x for x, rr in zip(Ia, ratios) if rr > 1]
    rows = [row("B = 0.95 B*: anode current / B = 0 current (conducting, > 0.3)", min(Ia[ratios.index(0.95)], 1.0),
                1.0, 0.7),
            row("B = 1.05 B*: anode current / B = 0 current (insulated, < 0.02)", Ia[ratios.index(1.05)], 0.0, 0.02,
                absolute=True)]
    # cutoff position by interpolation between the last conducting and first insulated point
    k = int(np.argmax(np.array(ratios) > 1))
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot(ratios, Ia, "o-", label="VEL-PIC (RZ, SCL emission)")
    ax.axvline(1.0, color="k", ls="--", label="Hull cutoff B* (Lovelace & Ott)")
    ax.set_xlabel("B / B*")
    ax.set_ylabel("anode current / current at B = 0")
    ax.legend(fontsize=8)
    ax.set_title(f"Coaxial smooth-bore magnetron: B* = {Bs * 1e3:.1f} mT")
    return record("33_coax_hull_cutoff", "Cylindrical Hull cutoff of a smooth-bore (coaxial) magnetron diode (RZ)",
                  rows, fig, notes=f"Inner cathode r_c = 5 mm, anode r_a = 10 mm, V = 10 kV, axial B, SCL emission. "
                                   f"Lovelace-Ott: d_e = (r_a^2 - r_c^2)/(2 r_a) = {de * 1e3:.2f} mm, B* = "
                                   f"{Bs * 1e3:.2f} mT (the planar formula with d = r_a - r_c would give "
                                   f"{C.me * C.c / (C.e * (b - a)) * np.sqrt(gam ** 2 - 1) * 1e3:.2f} mT). "
                                   f"Anode current / I(B=0) at B/B* = {ratios}: {np.round(Ia, 4).tolist()}.")


# ----------------------------------------------------------------------------------------------
def em_scl():
    """Space-charge-limited emission with the EM (Yee) solver: planar Child-Langmuir and the coaxial
    Langmuir-Blodgett law (electrodes held at their potentials by ideal voltage sources)."""
    rows = []
    notes = []
    V, d = 1e4, 1e-3
    gam = 1 + C.e * V / (C.me * C.c ** 2)
    JCL = 4 * C.eps0 / 9 * np.sqrt(2 * C.e / C.me) * V ** 1.5 / d ** 2
    # relativistic correction (Jory & Trivelpiece 1969) is -0.37 % at 10 kV; use the exact 1D value
    from scipy.integrate import quad
    def jt(V):
        g0 = 1 + C.e * V / (C.me * C.c ** 2)
        I = quad(lambda g: (g * g - 1) ** -0.25, 1, g0)[0]
        return C.eps0 * C.me * C.c ** 3 / (2 * C.e * d ** 2) * I ** 2
    Jref = jt(V)
    for nz in (100, 200):
        m = new_model("1D-Z", "electromagnetic")
        m["domain"].update(z_min="0", z_max=_r(d), nz=str(nz))
        m["boundaries"]["z_min"].update(field="conductor", voltage="0")
        m["boundaries"]["z_max"].update(field="conductor", voltage=_r(V))
        m["species"] = [{"name": "e", "preset": "electron"}]
        m["emitters"] = [{"name": "k", "species": "e", "model": "scl", "z0": "0", "particles_per_cell": "10"}]
        m["simulation"].update(t_end="0.4e-9")
        quiet_diag(m, 20)
        sim = run(m, f"em_scl_1d_{nz}")
        J = -tail_mean(sim, "I_z_max", 0.4)
        Jret = -tail_mean(sim, "I_z_min", 0.4)
        rows.append(row(f"EM 1D Child-Langmuir, {nz} cells: J_anode / J (relativistic CL)", J / Jref, 1.0,
                        0.035 if nz == 100 else 0.025))
        notes.append(f"1D {nz} cells: J/J_ref = {J / Jref:.4f}, returned current {Jret / J:.2e}")
    # coax, inner cathode a, anode b = 2a, periodic in z
    a, b = 1e-3, 2e-3
    beta2 = 0.2793           # Langmuir-Blodgett beta^2 for r_a / r_c = 2
    ILB = 8 * np.pi * C.eps0 / 9 * np.sqrt(2 * C.e / C.me) * V ** 1.5 / (b * beta2)     # A per metre
    nr, nzc = 100, 10
    Lz = (b - a) * nzc / nr
    m = new_model("2D-RZ", "electromagnetic")
    m["domain"].update(z_min="0", z_max=_r(Lz), nz=str(nzc), y_min=_r(a), y_max=_r(b), ny=str(nr))
    for s_ in ("z_min", "z_max"):
        m["boundaries"][s_].update(field="periodic", particles="periodic")
    m["boundaries"]["y_min"].update(field="conductor", voltage="0")
    m["boundaries"]["y_max"].update(field="conductor", voltage=_r(V))
    m["species"] = [{"name": "e", "preset": "electron"}]
    m["emitters"] = [{"name": "k", "species": "e", "model": "scl", "z0": "0", "y0": _r(a), "z1": _r(Lz), "y1": _r(a),
                      "normal": "+y", "particles_per_cell": "6"}]
    m["simulation"].update(t_end="0.4e-9")
    quiet_diag(m, 20)
    sim = run(m, "em_scl_coax")
    I = -tail_mean(sim, "I_y_max", 0.4) / Lz
    rows.append(row("EM coaxial diode (b/a = 2, 100 cells): I/L vs Langmuir-Blodgett", I / ILB, 1.0, 0.04))
    notes.append(f"coax: I/L = {I:.1f} A/m, Langmuir-Blodgett {ILB:.1f} A/m")
    return record("34_em_scl", "Space-charge-limited emission with the EM (Yee) solver: planar and cylindrical",
                  rows, notes="Electrodes held at their potentials (ideal voltage sources, fields/circuit.py); "
                              "Gauss-law emitter with the half-cell surface-charge correction. " + "; ".join(notes)
                              + ". The Gauss-law emitter converges at first order in the cell size near the "
                                "cathode, as in the ES solver.")


SUITE = {"pierce": pierce, "coax_hull": coax_hull, "em_scl": em_scl}

if __name__ == "__main__":
    for n in (sys.argv[1:] or list(SUITE)):
        SUITE[n]()
