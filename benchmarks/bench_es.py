"""Electrostatic benchmarks: Poisson solver, Child-Langmuir (1D/2D/RZ/coax), plasma waves,
two-stream and beam-plasma instabilities, emission laws, Hull cutoff."""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from common import C, new_model, plt, quiet_diag, record, row, run, tail_mean, fit_frequency, fit_growth, hist

JCL = lambda V, d: 4 * C.eps0 / 9 * np.sqrt(2 * C.e / C.me) * V ** 1.5 / d ** 2  # noqa: E731


# ----------------------------------------------------------------------------
def poisson_convergence():
    from picstudio.expr import Evaluator
    from picstudio.fields.poisson import PoissonFEM
    from picstudio.mesh.grid import Grid, Materials
    from picstudio.model.model import default_boundaries
    ev = Evaluator()
    ns = [16, 32, 64, 128]
    errs = {"1D-Z": [], "2D-ZY": [], "2D-RZ": [], "coax RZ": [], "dielectric 1D": []}
    for n in ns:
        g = Grid("1D-Z", 0, 1, n)
        P = PoissonFEM(g, Materials(g, [], default_boundaries("1D-Z"), ev))
        ex = np.sin(np.pi * g.z)
        phi = P.solve(C.eps0 * np.pi ** 2 * ex * g.node_volume, [0, 0])
        errs["1D-Z"].append(np.abs(phi - ex).max())
        g = Grid("2D-ZY", 0, 1, n, 0, 1, n)
        P = PoissonFEM(g, Materials(g, [], default_boundaries(), ev))
        Z, Y = g.nodes()
        ex = np.sin(np.pi * Z) * np.sin(np.pi * Y)
        phi = P.solve(C.eps0 * 2 * np.pi ** 2 * ex * g.node_volume, [0] * 4)
        errs["2D-ZY"].append(np.abs(phi - ex).max())
        g = Grid("2D-RZ", 0, 1, n, 0, 1, n)
        P = PoissonFEM(g, Materials(g, [], default_boundaries("2D-RZ"), ev))
        Z, R = g.nodes()
        ex = (1 - R ** 2) * np.sin(np.pi * Z)
        rho = C.eps0 * (4 + np.pi ** 2 * (1 - R ** 2)) * np.sin(np.pi * Z)
        phi = P.solve(rho * g.node_volume / (2 * np.pi), [0] * 3)
        errs["2D-RZ"].append(np.abs(phi - ex).max())
        a, b = 0.25, 1.0
        g = Grid("2D-RZ", 0, 1, n, a, b, n)
        bd = default_boundaries("2D-RZ")
        for s in ("z_min", "z_max"):
            bd[s]["field"] = "neumann"
        bd["y_min"].update(field="conductor", voltage="1")
        bd["y_max"].update(field="conductor", voltage="0")
        P = PoissonFEM(g, Materials(g, [], bd, ev))
        phi = P.solve(np.zeros(g.shape), [1, 0])
        errs["coax RZ"].append(np.abs(phi[n // 2] - np.log(b / g.y) / np.log(b / a)).max())
        # two-layer dielectric capacitor (eps 1 | 4), V=1 across [0,1], interface at 0.5
        g = Grid("1D-Z", 0, 1, n)
        bd = default_boundaries("1D-Z")
        bd["z_max"]["voltage"] = "1"
        mats = Materials(g, [{"name": "d", "shape": "rectangle", "z0": "0.5", "z1": "1", "y0": "-1", "y1": "2",
                              "material": "dielectric", "eps_r": "4"}], bd, ev)
        P = PoissonFEM(g, mats)
        phi = P.solve(np.zeros(g.shape), [0, 1])
        E1 = 1 / (0.5 + 0.5 / 4)
        exact = np.where(g.z <= 0.5, E1 * g.z, E1 * 0.5 + E1 / 4 * (g.z - 0.5))
        errs["dielectric 1D"].append(np.abs(phi - exact).max() + 1e-16)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    rows = []
    h = 1 / np.array(ns)
    for k, e in errs.items():
        ax.loglog(h, e, "o-", label=k)
        if k == "dielectric 1D":
            rows.append(row("dielectric interface (exact for P1): max error", max(e[-1], 1e-16), 0.0, 1e-10,
                            absolute=True))
            continue
        order = np.polyfit(np.log(h), np.log(e), 1)[0]
        rows.append(row(f"{k}: convergence order", order, 2.0, 0.1))
    ax.loglog(h, 0.5 * h ** 2, "k--", label="h^2")
    ax.set_xlabel("h")
    ax.set_ylabel("max error")
    ax.legend(fontsize=8)
    ax.set_title("scikit-fem Poisson solver convergence")
    return record("01_poisson_convergence", "Poisson (scikit-fem) convergence: manufactured & analytic solutions",
                  rows, fig)


# ----------------------------------------------------------------------------
def child_langmuir_1d():
    rows = []
    fig, ax = plt.subplots(figsize=(5.5, 4))
    d = 1e-3
    for V, rel in ((1e3, False), (1e5, True)):
        m = new_model("1D-Z", "electrostatic")
        m["domain"].update(z_min="0", z_max=str(d), nz="100")
        m["boundaries"]["z_min"].update(field="conductor", voltage="0")
        m["boundaries"]["z_max"].update(field="conductor", voltage=str(V))
        m["species"] = [{"name": "e", "preset": "electron"}]
        m["emitters"] = [{"name": "k", "species": "e", "model": "scl", "z0": "0", "particles_per_cell": "10"}]
        vfin = np.sqrt(2 * C.e * V / C.me) if not rel else C.c * np.sqrt(1 - 1 / (1 + V / 511e3) ** 2)
        m["simulation"].update(t_end=str(12 * d / vfin))
        quiet_diag(m, 20)
        sim = run(m, f"cl1d_{int(V)}")
        t, I = hist(sim, "I_z_max")
        J = -tail_mean(sim, "I_z_max", 0.4)
        if rel:
            # exact relativistic planar SCL (Jory & Trivelpiece 1969) by numerical quadrature
            ref = _relativistic_cl(V, d)
            lab = "relativistic CL (Jory-Trivelpiece)"
        else:
            ref = JCL(V, d)
            lab = "J_CL"
        ax.plot(t * 1e9, -I / ref, label=f"V = {V/1e3:g} kV")
        rows.append(row(f"V = {V / 1e3:g} kV, J_anode / {lab}", J / ref, 1.0, 0.03))
    ax.axhline(1, color="k", ls="--")
    ax.set_xlabel("t (ns)")
    ax.set_ylabel("J / J_theory")
    ax.set_ylim(0, 1.5)
    ax.legend()
    ax.set_title("1D Child-Langmuir: anode current density")
    return record("02_child_langmuir_1d", "Child-Langmuir law, planar 1D (non-relativistic & relativistic)", rows, fig)


def _relativistic_cl(V, d):
    """Exact relativistic SCL current density: d = int_0^phi dphi'/sqrt(2 J/eps0 * (sqrt(g^2-1))^{-1}... )."""
    mc2 = C.me * C.c ** 2 / C.e
    from scipy.integrate import quad

    def gap(J):
        k = 2 * J / (C.eps0 * C.c)

        def integrand(g):
            # E^2 = k * (sqrt(g^2-1)) * mc2 (from Poisson first integral), dphi = mc2 dg
            return mc2 / np.sqrt(k * mc2 * np.sqrt(g * g - 1))
        val, _ = quad(integrand, 1.0, 1 + V / mc2, limit=200)
        return val - d
    J0 = JCL(V, d)
    return brentq(gap, 0.05 * J0, 2 * J0)


# ----------------------------------------------------------------------------
def child_langmuir_2d(nz=48):
    d, V = 1e-3, 1e3
    rows = []
    res = {"strip": [], "disk": []}
    for geom, ratios in (("2D-ZY", (1.0, 2.0, 4.0)), ("2D-RZ", (1.0, 4.0))):
        for ratio in ratios:
            W = ratio * d
            m = new_model(geom, "electrostatic")
            ye = W / 2 if geom == "2D-ZY" else W
            Ly = ye + 3 * d
            ny = int(round(Ly / (d / nz)))
            m["domain"].update(z_min="0", z_max=str(d), nz=str(nz), y_min="0", y_max=str(Ly), ny=str(ny))
            m["boundaries"]["z_min"].update(field="conductor", voltage="0")
            m["boundaries"]["z_max"].update(field="conductor", voltage=str(V))
            m["boundaries"]["y_max"].update(field="neumann", particles="absorb")
            if geom == "2D-ZY":
                m["boundaries"]["y_min"].update(field="neumann", particles="reflect")
            # Luginsland et al. (1996) and Lau (2001): electrons confined to z by an infinite magnetic field
            m["species"] = [{"name": "e", "preset": "electron", "motion": "z_only"}]
            m["emitters"] = [{"name": "k", "species": "e", "model": "scl", "z0": "0", "y0": "0", "z1": "0",
                              "y1": str(ye), "particles_per_cell": "4"}]
            m["simulation"].update(t_end="1.5e-9")
            quiet_diag(m, 20)
            sim = run(m, f"cl2d_{geom}_{ratio}")
            A = ye if geom == "2D-ZY" else np.pi * W ** 2
            J = -tail_mean(sim, "I_z_max", 0.4) / A / JCL(V, d)
            if geom == "2D-ZY":
                th = 1 + 0.3145 / ratio - 0.0004 / ratio ** 2
                rows.append(row(f"strip W/d = {ratio:g}: J/J_CL vs Lau (2001)", J, th, 0.04))
                res["strip"].append((1 / ratio, J, th))
            else:
                th = 1 + 1 / (4 * ratio)
                rows.append(row(f"disk R/d = {ratio:g}: J/J_CL vs Lau 1+d/4R (approx.)", J, th, 0.10))
                res["disk"].append((1 / ratio, J, th))
    fig, ax = plt.subplots(figsize=(5.5, 4))
    x = np.linspace(0, 1.1, 50)
    ax.plot(x, 1 + 0.3145 * x - 0.0004 * x ** 2, "C0-", label="Lau strip")
    ax.plot(x, 1 + x / 4, "C1-", label="Lau disk 1+d/4R")
    for k, c in (("strip", "C0"), ("disk", "C1")):
        if res[k]:
            a = np.array(res[k])
            ax.plot(a[:, 0], a[:, 1], "o", color=c, label=f"VEL-PIC {k}")
    ax.set_xlabel("d/W  or  d/R")
    ax.set_ylabel("J / J_CL(1D)")
    ax.legend()
    ax.set_title("2D space-charge-limited emission")
    return record("03_child_langmuir_2d", "2D Child-Langmuir: finite strip (ZY) and disk (RZ) emitters", rows, fig,
                  notes="Electrons confined to z (motion = z_only, infinite axial B) as assumed by Luginsland "
                        "et al. (1996) and Lau (2001). Lau's disk formula is an approximation (valid for "
                        "d/R < 2 to a few %); the strip law fits PIC data closely.")


# ----------------------------------------------------------------------------
def langmuir_blodgett():
    """Radial SCL flow in a coaxial diode (RZ): compare with Langmuir-Blodgett."""
    V, a = 1000.0, 1e-3
    rows = []
    xs, ys, ref = [], [], []
    for b_over_a in (2.0, 4.0):
        b = b_over_a * a
        m = new_model("2D-RZ", "electrostatic")
        ny = int(round(48 * (b - 0) / (b - a)))
        m["domain"].update(z_min="0", z_max=str(0.25e-3), nz="6", y_min="0", y_max=str(b), ny=str(ny))
        for s in ("z_min", "z_max"):
            m["boundaries"][s].update(field="periodic", particles="periodic")
        m["boundaries"]["y_max"].update(field="conductor", voltage=str(V))
        m["objects"] = [{"name": "cathode", "shape": "rectangle", "z0": "-1", "z1": "1", "y0": "0",
                         "y1": str(a), "material": "conductor", "voltage": "0"}]
        m["species"] = [{"name": "e", "preset": "electron"}]
        m["emitters"] = [{"name": "k", "species": "e", "model": "scl", "z0": "0", "y0": str(a), "z1": "0.25e-3",
                          "y1": str(a), "normal": "+y", "particles_per_cell": "8"}]
        m["simulation"].update(t_end=str(1.5e-9 * b_over_a))
        quiet_diag(m, 20)
        sim = run(m, f"coax_{b_over_a}")
        IL = -tail_mean(sim, "I_y_max", 0.4) / 0.25e-3
        g = np.log(b / a)
        beta = g - 0.4 * g ** 2 + 0.091667 * g ** 3 - 0.014242 * g ** 4 + 0.0016793 * g ** 5
        th = 8 * np.pi * C.eps0 / 9 * np.sqrt(2 * C.e / C.me) * V ** 1.5 / (b * beta ** 2)
        rows.append(row(f"b/a = {b_over_a:g}: current per length (A/m)", IL, th, 0.05))
        xs.append(b_over_a)
        ys.append(IL)
        ref.append(th)
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.plot(xs, ref, "k-s", label="Langmuir-Blodgett")
    ax.plot(xs, ys, "ro", label="VEL-PIC (RZ)")
    ax.set_xlabel("b/a")
    ax.set_ylabel("I/L (A/m)")
    ax.legend()
    ax.set_title("Coaxial (radial) SCL diode")
    return record("04_langmuir_blodgett", "Radial SCL flow in a coaxial diode (RZ) vs Langmuir-Blodgett", rows, fig)


# ----------------------------------------------------------------------------
def plasma_oscillation():
    rows = []
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    n0 = 1e15
    wp = np.sqrt(n0 * C.e ** 2 / (C.eps0 * C.me))
    for geom, solver in (("1D-Z", "electrostatic"), ("2D-ZY", "electrostatic"), ("2D-RZ", "electrostatic"),
                         ("1D-Z", "electromagnetic"), ("2D-ZY", "electromagnetic")):
        m = new_model(geom, solver)
        # EM: choose the cell so that w_p*dt ~ 0.04 (dt is set by the Courant limit)
        L = 0.01 if solver == "electrostatic" else 64 * 0.06 * C.c / wp
        m["domain"].update(z_min="0", z_max=str(L), nz="64", y_min="0", y_max=str(L / 4), ny="8")
        for s in ("z_min", "z_max"):
            m["boundaries"][s].update(field="periodic", particles="periodic")
        if geom == "2D-ZY":
            for s in ("y_min", "y_max"):
                m["boundaries"][s].update(field="periodic", particles="periodic")
        if geom == "2D-RZ":
            m["boundaries"]["y_max"].update(field="neumann", particles="reflect")
        m["species"] = [{"name": "e", "preset": "electron"}]
        m["loaders"] = [dict(name="p", species="e", z0="0", z1=str(L), y0="0", y1=str(L / 4), density=str(n0),
                             ppc="64", quiet_start=True, perturbation="velocity",
                             pert_amplitude=str(2e4 if solver == "electrostatic" else 2e5),
                             pert_mode="1")]
        if solver == "electrostatic":
            m["simulation"].update(t_end=str(12 * 2 * np.pi / wp), dt=str(0.02 / wp))
        else:
            m["simulation"].update(t_end=str(12 * 2 * np.pi / wp))
        quiet_diag(m, 1 if solver == "electrostatic" else 4)
        sim = run(m, f"osc_{geom}_{solver}")
        t, W = hist(sim, "W_field")
        w = fit_frequency(t, W) / 2
        # finite-dz correction for CIC/centered differences is ~ (k dz)^2 small; cold plasma -> w = wp
        rows.append(row(f"{geom} {'ES' if solver == 'electrostatic' else 'EM'}: w / w_p", w / wp, 1.0, 0.01))
        ax.plot(t * wp / (2 * np.pi), W / W.max(), label=f"{geom} {'ES' if solver == 'electrostatic' else 'EM'}", lw=1)
    ax.set_xlabel("t w_p / 2 pi")
    ax.set_ylabel("W_field / max")
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("Cold plasma (Langmuir) oscillation")
    return record("05_plasma_oscillation", "Cold plasma oscillation frequency (ES & EM, 1D/ZY/RZ)", rows, fig)


# ----------------------------------------------------------------------------
def two_stream():
    n0, v0 = 1e14, 1e6
    wpb = np.sqrt(n0 / 2 * C.e ** 2 / (C.eps0 * C.me))
    k = np.sqrt(3) / 2 * wpb / v0
    L = 2 * np.pi / k
    m = new_model("1D-Z", "electrostatic")
    m["domain"].update(z_min="0", z_max=str(L), nz="64")
    for s in ("z_min", "z_max"):
        m["boundaries"][s].update(field="periodic", particles="periodic")
    m["species"] = [{"name": "r", "preset": "electron"}, {"name": "l", "preset": "electron"}]
    com = dict(z0="0", z1=str(L), y0="0", y1="1", density=str(n0 / 2), ppc="1000", quiet_start=True,
               perturbation="velocity", pert_mode="1")
    m["loaders"] = [dict(name="r", species="r", drift=[str(v0), "0", "0"], pert_amplitude=str(1e-5 * v0), **com),
                    dict(name="l", species="l", drift=[str(-v0), "0", "0"], pert_amplitude=str(-1e-5 * v0), **com)]
    m["simulation"].update(t_end=str(40 / wpb), dt=str(0.02 / wpb))
    quiet_diag(m, 2)
    sim = run(m, "two_stream")
    t, W = hist(sim, "W_field")
    amp = np.sqrt(W)
    gam, sel = fit_growth(t, amp, 1e-3, 0.2)
    # exact growth of the simulated mode from the cold dispersion relation
    x = k * v0 / wpb
    gth = wpb * np.sqrt(np.sqrt(4 * x ** 2 + 1) - x ** 2 - 1)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.semilogy(t * wpb, amp / amp.max(), label="VEL-PIC sqrt(W_field)")
    t0 = t[sel][len(t[sel]) // 2]
    ax.semilogy(t * wpb, amp[sel][len(t[sel]) // 2] / amp.max() * np.exp(gth * (t - t0)), "k--",
                label=f"theory gamma = {gth / wpb:.3f} w_pb")
    ax.set_ylim(1e-5, 2)
    ax.set_xlabel("t w_pb")
    ax.legend()
    ax.set_title("Two-stream instability")
    return record("06_two_stream", "Two-stream instability growth rate (1D ES)",
                  [row("growth rate gamma / w_pb", gam / wpb, gth / wpb, 0.05)], fig)


def beam_plasma():
    np0, alpha, vb = 1e14, 1e-3, 1e7
    wp = np.sqrt(np0 * C.e ** 2 / (C.eps0 * C.me))
    L = 2 * np.pi * vb / wp
    m = new_model("1D-Z", "electrostatic")
    m["domain"].update(z_min="0", z_max=str(L), nz="64")
    for s in ("z_min", "z_max"):
        m["boundaries"][s].update(field="periodic", particles="periodic")
    m["species"] = [{"name": "p", "preset": "electron"}, {"name": "b", "preset": "electron"}]
    m["loaders"] = [dict(name="p", species="p", z0="0", z1=str(L), y0="0", y1="1", density=str(np0), ppc="400",
                         quiet_start=True),
                    dict(name="b", species="b", z0="0", z1=str(L), y0="0", y1="1", density=str(alpha * np0),
                         ppc="400", quiet_start=True, drift=[str(vb), "0", "0"], perturbation="velocity",
                         pert_amplitude=str(1e-6 * vb), pert_mode="1")]
    m["simulation"].update(t_end=str(160 / wp), dt=str(0.05 / wp))
    quiet_diag(m, 4)
    sim = run(m, "beam_plasma")
    t, W = hist(sim, "W_field")
    amp = np.sqrt(W)
    gam, sel = fit_growth(t, amp, 3e-3, 0.15)
    # exact root of the cold dispersion relation 1 = wp^2/w^2 + a wp^2/(w - k vb)^2 at k = wp/vb
    from numpy.polynomial import polynomial as Pn
    kv = wp / vb * vb
    # (w^2)(w-kv)^2 - wp^2 (w-kv)^2 - a wp^2 w^2 = 0
    p1 = Pn.polymul([0, 0, 1], Pn.polypow([-kv, 1], 2))
    p2 = Pn.polymul([wp ** 2], Pn.polypow([-kv, 1], 2))
    p3 = [0, 0, alpha * wp ** 2]
    roots = Pn.polyroots(Pn.polysub(Pn.polysub(p1, p2), p3))
    gth = max(r.imag for r in roots)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.semilogy(t * wp, amp / amp.max(), label="VEL-PIC")
    t0 = t[sel][len(t[sel]) // 2]
    ax.semilogy(t * wp, amp[sel][len(t[sel]) // 2] / amp.max() * np.exp(gth * (t - t0)), "k--",
                label=f"theory {gth / wp:.4f} w_p")
    ax.set_ylim(1e-4, 2)
    ax.set_xlabel("t w_p")
    ax.legend()
    ax.set_title("Weak beam-plasma instability (n_b/n_p = 1e-3)")
    return record("07_beam_plasma", "Beam-plasma instability growth rate (1D ES)",
                  [row("growth rate gamma / w_p", gam / wp, gth / wp, 0.08)], fig,
                  notes=f"weak-beam asymptote sqrt(3)/2^(4/3) alpha^(1/3) = {np.sqrt(3) / 2 ** (4 / 3) * alpha ** (1 / 3):.4f}")


# ----------------------------------------------------------------------------
def emission_laws():
    rows = []
    d = 1e-3
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    # thermionic: W = 4.5 eV, 2500 K
    Vs = [30, 100, 300, 1000, 3000]
    Js = []
    from picstudio.sources.emitters import richardson_J, fowler_nordheim_J
    for V in Vs:
        m = new_model("1D-Z", "electrostatic")
        m["domain"].update(z_min="0", z_max=str(d), nz="200")
        m["boundaries"]["z_min"].update(field="conductor", voltage="0")
        m["boundaries"]["z_max"].update(field="conductor", voltage=str(V))
        m["species"] = [{"name": "e", "preset": "electron"}]
        m["emitters"] = [{"name": "k", "species": "e", "model": "thermionic", "z0": "0", "work_function": "4.5",
                          "cathode_temperature": "2500", "particles_per_cell": "40"}]
        v = np.sqrt(2 * C.e * V / C.me)
        m["simulation"].update(t_end=str(max(15 * d / v, 2e-9)))
        quiet_diag(m, 20)
        sim = run(m, f"thermionic_{V}")
        Js.append(-tail_mean(sim, "I_z_max", 0.4))
    Js = np.array(Js)
    Vs = np.array(Vs, float)
    Jrd = richardson_J(2500, 4.5, Vs / d)
    Jcl = JCL(Vs, d)
    ax = axs[0]
    VV = np.logspace(1, 3.7, 100)
    ax.loglog(VV, JCL(VV, d), "k--", label="Child-Langmuir")
    ax.loglog(VV, richardson_J(2500, 4.5, VV / d), "b:", label="Richardson-Schottky (E=V/d)")
    ax.loglog(Vs, Js, "ro", label="VEL-PIC")
    ax.set_xlabel("V (V)")
    ax.set_ylabel("J (A/m^2)")
    ax.legend(fontsize=8)
    ax.set_title("Thermionic diode, W=4.5 eV, T=2500 K")
    rows.append(row("thermionic, V=3 kV (T-limited): J / J_RDS", Js[-1] / Jrd[-1], 1.0, 0.05))
    # Langmuir's thermal correction of the SCL law: J = J_CL(V - V_m) (1 + 2.66/sqrt(eta)),
    # eta = e(V - V_m)/kT, potential minimum V_m = -(kT/e) ln(J_s/J)  (x_m neglected)
    kT = C.kB * 2500 / C.e
    Jl = Jcl[0]
    for _ in range(20):
        Vm = -kT * np.log(richardson_J(2500, 4.5, 0.0) / Jl)
        eta = (Vs[0] - Vm) / kT
        Jl = JCL(Vs[0] - Vm, d) * (1 + 2.66 / np.sqrt(eta))
    rows.append(row("thermionic, V=30 V (SCL): J / Langmuir thermal-SCL estimate", Js[0] / Jl, 1.0, 0.08))
    # Fowler-Nordheim, low current regime: J must follow the FN law at F = V/d
    dfn = 1e-6
    for V in (3000.0, 4000.0):
        m = new_model("1D-Z", "electrostatic")
        m["domain"].update(z_min="0", z_max=str(dfn), nz="100")
        m["boundaries"]["z_min"].update(field="conductor", voltage="0")
        m["boundaries"]["z_max"].update(field="conductor", voltage=str(V))
        m["species"] = [{"name": "e", "preset": "electron"}]
        m["emitters"] = [{"name": "k", "species": "e", "model": "fowler_nordheim", "z0": "0",
                          "work_function": "4.5", "particles_per_cell": "20"}]
        v = np.sqrt(2 * C.e * V / C.me)
        m["simulation"].update(t_end=str(8 * dfn / v))
        quiet_diag(m, 10)
        sim = run(m, f"fn_{V}")
        J = -tail_mean(sim, "I_z_max", 0.4)
        Jfn = fowler_nordheim_J(V / dfn, 4.5)
        rows.append(row(f"Fowler-Nordheim F = {V / dfn:.1e} V/m: J / J_FN", J / Jfn, 1.0, 0.05))
        axs[1].plot(V / dfn, J, "ro")
    F = np.linspace(2.5e9, 6e9, 50)
    axs[1].semilogy(F, fowler_nordheim_J(F, 4.5), "k-", label="Murphy-Good FN")
    axs[1].set_xlabel("F (V/m)")
    axs[1].set_ylabel("J (A/m^2)")
    axs[1].legend()
    axs[1].set_title("Field emission (space charge negligible)")
    return record("08_emission_laws", "Thermionic and Fowler-Nordheim emission with self-consistent space charge",
                  rows, fig)


# ----------------------------------------------------------------------------
def hull_cutoff():
    V, d = 10e3, 5e-3
    gam = 1 + C.e * V / (C.me * C.c ** 2)
    BH = C.me * C.c / (C.e * d) * np.sqrt(gam ** 2 - 1)
    ratios = [0.6, 0.85, 0.95, 1.05, 1.2]
    Ia = []
    for r in ratios:
        m = new_model("2D-ZY", "electrostatic")
        m["domain"].update(z_min="0", z_max=str(d), nz="64", y_min="0", y_max="1e-3", ny="12")
        m["boundaries"]["z_min"].update(field="conductor", voltage="0")
        m["boundaries"]["z_max"].update(field="conductor", voltage=str(V))
        for s in ("y_min", "y_max"):
            m["boundaries"][s].update(field="periodic", particles="periodic")
        m["species"] = [{"name": "e", "preset": "electron"}]
        m["emitters"] = [{"name": "k", "species": "e", "model": "scl", "z0": "0", "y0": "0", "z1": "0",
                          "y1": "1e-3", "particles_per_cell": "4"}]
        m["external_fields"] = [{"name": "B", "type": "uniform", "B": ["0", "0", str(r * BH)]}]
        m["simulation"].update(t_end="5e-9")
        quiet_diag(m, 20)
        sim = run(m, f"hull_{r}")
        Ia.append(-tail_mean(sim, "I_z_max", 0.5) / 1e-3)
    Ia = np.array(Ia) / JCL(V, d)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot(ratios, Ia, "o-", label="VEL-PIC")
    ax.axvline(1.0, color="k", ls="--", label="Hull cutoff B_H")
    ax.set_xlabel("B / B_H")
    ax.set_ylabel("J_anode / J_CL")
    ax.legend()
    ax.set_title(f"Crossed-field (planar magnetron) diode, B_H = {BH * 1e3:.1f} mT")
    below = Ia[ratios.index(0.85)]
    above = Ia[ratios.index(1.05)]
    rows = [row("J_anode/J_CL at B = 0.85 B_H (must conduct, >0.3)", min(below, 1.0), 1.0, 0.7),
            row("J_anode/J_CL at B = 1.05 B_H (cut off, <0.02)", above, 0.0, 0.02, absolute=True)]
    return record("09_hull_cutoff", "Hull cutoff in a planar crossed-field diode (ZY, magnetized)", rows, fig)
