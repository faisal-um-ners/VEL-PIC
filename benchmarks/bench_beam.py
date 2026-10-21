"""Beam-dynamics / space-charge benchmarks (RZ): drift-tube expansion and solenoid (Brillouin) transport,
compared with the paraxial envelope equation."""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from common import C, new_model, plt, quiet_diag, record, row, run
from picstudio.fields.external import loop_field

I0A = 4 * np.pi * C.eps0 * C.me * C.c ** 3 / C.e     # Alfven-type current 17.045 kA


def _edge_radius(sim, zbins):
    sp = sim.species[0]
    z, r, w = sp.z, sp.y, sp.w
    idx = np.digitize(z, zbins) - 1
    out = np.full(len(zbins) - 1, np.nan)
    for b in range(len(zbins) - 1):
        s = idx == b
        if s.sum() > 20:
            out[b] = np.sqrt(2 * np.average(r[s] ** 2, weights=w[s]))
    return out


def _beam_model(solenoid):
    Eb, I, a, Rt, Lz = 10e3, 0.05, 2e-3, 10e-3, 0.2
    m = new_model("2D-RZ", "electrostatic")
    m["domain"].update(z_min="0", z_max=str(Lz), nz="400", y_min="0", y_max=str(Rt), ny="40")
    for s in ("z_min", "z_max", "y_max"):
        m["boundaries"][s].update(field="conductor", voltage="0")
    m["species"] = [{"name": "beam", "preset": "electron"}]
    m["emitters"] = [{"name": "gun", "species": "beam", "model": "beam", "z0": "0", "y0": "0", "z1": "0",
                      "y1": str(a), "current_density": str(I / (np.pi * a ** 2)), "energy": str(Eb),
                      "particles_per_cell": "6"}]
    gam = 1 + Eb * C.e / (C.me * C.c ** 2)
    beta = np.sqrt(1 - 1 / gam ** 2)
    BB = 0.0
    if solenoid:
        nb = I / (C.e * beta * C.c * np.pi * a ** 2)
        wp = np.sqrt(nb * C.e ** 2 / (C.eps0 * C.me * gam))
        BB = np.sqrt(2 * gam) * C.me * wp / C.e
        m["external_fields"] = [{"name": "sol", "type": "solenoid", "zc": "0.02", "z_end": "0.5",
                                 "radius": "0.03", "ampere_turns": str(BB * 0.48 / C.mu0)}]
    m["simulation"].update(t_end="10e-9")
    quiet_diag(m, 50)
    return m, dict(I=I, a=a, gam=gam, beta=beta, Lz=Lz, BB=BB)


def _envelope(p, Bz_of_z):
    K = 2 * p["I"] / (I0A * (p["beta"] * p["gam"]) ** 3)
    kfac = C.e / (2 * p["gam"] * C.me * p["beta"] * C.c)

    def rhs(z, y):
        a, ap = y
        kL = kfac * Bz_of_z(z)
        return [ap, K / a - kL ** 2 * a]
    sol = solve_ivp(rhs, (0, p["Lz"]), [p["a"], 0.0], max_step=1e-4, dense_output=True)
    return sol, K


def drift_and_solenoid():
    rows = []
    fig, ax = plt.subplots(figsize=(6.5, 4))
    zb = np.linspace(0, 0.2, 41)
    zc = 0.5 * (zb[1:] + zb[:-1])
    for sol_on in (False, True):
        m, p = _beam_model(sol_on)
        sim = run(m, f"beam_{'sol' if sol_on else 'drift'}")
        a_sim = _edge_radius(sim, zb)
        if sol_on:
            zs = np.linspace(0.02, 0.5, max(10, int(0.48 / (0.05 * 0.03)) + 1))
            Ieach = p["BB"] * 0.48 / C.mu0 / len(zs)

            def Bz(z):
                return sum(loop_field(z, 0.0, zc_, 0.03, Ieach)[0] for zc_ in zs)
        else:
            def Bz(z):
                return 0.0
        env, K = _envelope(p, Bz)
        a_th = env.sol(zc)[0]
        ok = np.isfinite(a_sim) & (zc > 0.01) & (zc < 0.19)
        err = np.max(np.abs(a_sim[ok] / a_th[ok] - 1))
        lab = "solenoid (Brillouin)" if sol_on else "drift tube"
        rows.append(row(f"{lab}: max |a_PIC/a_envelope - 1| over z", err, 0.0, 0.08 if sol_on else 0.03, absolute=True))
        ax.plot(zc * 100, a_sim * 1e3, "o", ms=3, label=f"VEXPACE, {lab}")
        zz = np.linspace(0, 0.2, 300)
        ax.plot(zz * 100, env.sol(zz)[0] * 1e3, "-", label=f"envelope eq., {lab}")
    ax.set_xlabel("z (cm)")
    ax.set_ylabel("beam edge radius sqrt(2<r^2>) (mm)")
    ax.legend(fontsize=8)
    ax.set_title(f"10 keV, 50 mA beam (perveance K = {K:.2e})")
    return record("15_beam_envelope", "Space-charge beam transport (RZ): drift expansion & Brillouin solenoid",
                  rows, fig, notes="The solenoid case uses the paraxial (linear-field, constant beta_z) envelope model; the residual difference comes from the fringe-field non-linearity and rotation-induced axial slowing included in PIC. Envelope equation a'' = K/a - (eB/2 gamma m beta c)^2 a, K = 2I/(I_A beta^3 gamma^3)")


def relativistic_selffield():
    """500 keV, 200 A beam in RZ: EM (self-consistent B) vs ES with beam self-B vs envelope (1/gamma^2)."""
    Eb, I, a, Rt, Lz = 500e3, 200.0, 5e-3, 20e-3, 0.2
    gam = 1 + Eb * C.e / (C.me * C.c ** 2)
    beta = np.sqrt(1 - 1 / gam ** 2)
    zb = np.linspace(0, Lz, 21)
    zc = 0.5 * (zb[1:] + zb[:-1])
    out = {}
    for label, solver, selfB in (("ES, no self-B", "electrostatic", False),
                                 ("ES + beam self-B", "electrostatic", True),
                                 ("EM (Yee)", "electromagnetic", False)):
        m = new_model("2D-RZ", solver)
        m["domain"].update(z_min="0", z_max=str(Lz), nz="400", y_min="0", y_max=str(Rt), ny="80")
        for s in ("z_min", "z_max", "y_max"):
            m["boundaries"][s].update(field="conductor", voltage="0")
        m["species"] = [{"name": "beam", "preset": "electron", "beam_self_B": selfB}]
        m["emitters"] = [{"name": "gun", "species": "beam", "model": "beam", "z0": "0", "y0": "0", "z1": "0",
                          "y1": str(a), "energy": str(Eb), "particles_per_cell": "4",
                          # smooth 1 ns sin^2 current rise (an abrupt 200 A step rings the drift tube's TM modes)
                          "current_density": f"{I / (np.pi * a ** 2)!r}*sin(0.5*pi*ramp(t, 1e-9))**2"}]
        m["simulation"].update(t_end="3.5e-9")
        if solver == "electromagnetic":
            m["simulation"].update(cfl="0.9", initial_field="none")
        quiet_diag(m, 100)
        sim = run(m, "selfB_" + solver + str(selfB))
        out[label] = _edge_radius(sim, zb)
    p = dict(I=I, a=a, gam=gam, beta=beta, Lz=Lz)
    env, K = _envelope(p, lambda z: 0.0)
    a_th = env.sol(zc)[0]
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, v in out.items():
        ax.plot(zc * 100, v * 1e3, "o-", ms=3, label=k)
    zz = np.linspace(0, Lz, 200)
    ax.plot(zz * 100, env.sol(zz)[0] * 1e3, "k--", label="envelope, K ~ 1/(beta gamma)^3")
    ax.set_xlabel("z (cm)")
    ax.set_ylabel("edge radius (mm)")
    ax.legend(fontsize=8)
    ax.set_title("500 keV, 200 A beam: magnetic self-field (1/gamma^2) check")
    ok = (zc > 0.01) & (zc < 0.15)
    rows = [row("EM: max |a/a_envelope - 1| (z < 15 cm)", np.max(np.abs(out["EM (Yee)"][ok] / a_th[ok] - 1)), 0, 0.10,
                absolute=True),
            row("ES+self-B: max |a/a_envelope - 1| (z < 15 cm)",
                np.max(np.abs(out["ES + beam self-B"][ok] / a_th[ok] - 1)), 0, 0.05, absolute=True)]
    return record("22_relativistic_selffield", "Relativistic beam self-fields: EM vs ES+self-B vs envelope (RZ)",
                  rows, fig, notes="Without the magnetic self-field (ES, no self-B) the beam blows up 1/(1-beta^2) "
                                   "= 3.9x faster; EM reproduces the 1/gamma^2 space-charge reduction. The beam current rises "
                                   "smoothly (sin^2, 1 ns) so that the TM modes of the drift tube are not rung by a "
                                   "step turn-on; edge radius measured at t = 3.5 ns.")
