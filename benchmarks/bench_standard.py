"""Standard, widely used PIC verification tests (Birdsall & Langdon; Vay 2008; WarpX test suite).

  26  particle pushers vs WarpX 26.09 (Boris, Vay, Higuera-Cary): identical trajectories, plus the
      textbook pusher properties (Boris gyro-phase error, E x B drift, relativistic E x B drift)
  27  Landau damping (k lambda_D = 0.5) and Bohm-Gross / thermal Langmuir dispersion
      (k lambda_D = 0.3) against the exact roots of the Vlasov dispersion relation
  28  electromagnetic wave in a plasma: w^2 = w_p^2 + c^2 k^2
  29  energy conservation of a thermal plasma (EM, energy-conserving vs momentum-conserving gather)

    python benchmarks/bench_standard.py [pushers landau emwave energy]
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
from common import C, fit_frequency, hist, new_model, plt, quiet_diag, record, row, run  # noqa: E402
from picstudio.particles import kernels as K  # noqa: E402


def _r(x):
    return repr(float(x))

DATA = os.path.join(HERE, "warpx_compare", "data", "pushers")


# ----------------------------------------------------------------------------
def _warpx_cases():
    v = 19.974984355438178 / np.sqrt(1 + 19.974984355438178 ** 2) * C.c
    return {
        "gyro": dict(u=(0.0, 0.0, 0.01), E=(0, 0, 0), B=(0.0, 0.1, 0.0), dt=5e-12, n=400),
        "exb": dict(u=(0.0, 0.0, 0.0), E=(1e5, 0, 0), B=(0.0, 0.1, 0.0), dt=5e-12, n=400),
        "relexb": dict(u=(0.0, 0.0, 19.974984355438178), E=(v, 0, 0), B=(0.0, 1.0, 0.0), dt=2e-12, n=400),
    }


def _ours(cs, pusher):
    """Single electron in uniform fields with VEXPACE's kernels, WarpX ordering (PushP(-dt/2) at t=0).
    Our 2D (z, y, x) corresponds to WarpX XZ (z, x, -y) - both right-handed."""
    z, y = np.zeros(1), np.zeros(1)
    ux, uy, uz = np.array(cs["u"]) * C.c
    u0, u1, u2 = np.array([uz]), np.array([ux]), np.array([-uy])
    Ex, Ey, Ez = cs["E"]
    Bx, By, Bz = cs["B"]
    P = np.array([[Ez], [Ex], [-Ey], [Bz], [Bx], [-By]], float)
    qm, dt = -C.e / C.me, cs["dt"]
    pn = {"boris": 0, "vay": 1, "higuera": 2}[pusher]
    K.push_fields_2d(z, y, u0, u1, u2, 1, P, qm, -0.5 * dt, -1.0, False, K.INV_C2, pn, False)
    tr = [(0.0, 0.0)]
    for _ in range(cs["n"]):
        K.push_fields_2d(z, y, u0, u1, u2, 1, P, qm, dt, -1.0, False, K.INV_C2, pn, True)
        tr.append((y[0], z[0]))
    return np.array(tr)


def pushers():
    rows = []
    cases = _warpx_cases()
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for k, (name, cs) in enumerate(cases.items()):
        for pusher in ("boris", "vay", "higuera"):
            f = os.path.join(DATA, f"{name}_{pusher}.txt")
            if not os.path.exists(f):
                continue
            W = np.loadtxt(f)
            wx = W[:, [2, 6]]
            tr = _ours(cs, pusher)
            scale = np.abs(wx).max()
            err = np.abs(tr - wx).max() / scale
            # the WarpX reduced diagnostic is written with 9 significant digits
            rows.append(row(f"{name}, {pusher}: max |VEXPACE - WarpX| / path size", err, 0.0, 1e-7, absolute=True))
            ax[k].plot(wx[:, 1] * 1e3, wx[:, 0] * 1e3, "-", lw=2.5, alpha=0.35, label=f"WarpX {pusher}")
            ax[k].plot(tr[::8, 1] * 1e3, tr[::8, 0] * 1e3, ".", ms=3, label=f"VEXPACE {pusher}")
        ax[k].set_title(name)
        ax[k].set_xlabel("z (mm)")
        ax[k].set_ylabel("x (mm)")
    ax[0].legend(fontsize=6)
    # textbook properties -------------------------------------------------------
    cs = cases["gyro"]
    tr = _ours(cs, "boris")
    wc = C.e * 0.1 / C.me / np.sqrt(1 + 0.01 ** 2)
    phase = np.unwrap(np.arctan2(tr[1:, 0] - tr[1:, 0].mean(), tr[1:, 1] - tr[1:, 1].mean()))
    w_num = abs(np.polyfit(np.arange(len(phase)) * cs["dt"], phase, 1)[0])
    w_boris = 2 / cs["dt"] * np.arctan(wc * cs["dt"] / 2)
    rows.append(row("Boris gyro-frequency vs (2/dt) atan(w_c dt/2)", w_num, w_boris, 2e-4))
    cs = cases["exb"]
    tr = _ours(cs, "boris")
    T = 2 * np.pi * C.me / (C.e * 0.1)
    nper = int(round(T / cs["dt"]))
    nfull = (len(tr) - 1) // nper * nper
    vd = (tr[nfull, 1] - tr[0, 1]) / (nfull * cs["dt"])
    rows.append(row("E x B drift velocity vs E/B", vd, 1e5 / 0.1, 0.01))
    cs = cases["relexb"]
    for pusher in ("vay", "higuera"):
        tr = _ours(cs, pusher)
        dev = np.abs(tr[:, 0]).max() / np.abs(tr[:, 1]).max()
        rows.append(row(f"relativistic E x B (gamma=20), {pusher}: transverse deviation / path", dev, 0.0,
                        1e-9, absolute=True))
    tr = _ours(cs, "boris")
    dev_b = np.abs(tr[:, 0]).max() / np.abs(tr[:, 1]).max()
    fig.tight_layout()
    return record("26_pushers_warpx", "Particle pushers: identical to WarpX (Boris, Vay, Higuera-Cary)", rows, fig,
                  notes="Single electron in uniform fields, WarpX 26.09 2D run (algo.particle_pusher, "
                        "algo.maxwell_solver = none, ParticleExtrema reduced diagnostic every step) against "
                        "VEXPACE's push kernels with the same initial half-step push-back. The WarpX inputs and "
                        f"outputs are stored in benchmarks/warpx_compare/data/pushers. In the relativistic E x B "
                        f"case the Boris pusher drifts sideways by {dev_b:.2e} of the path (its known error, "
                        "identical in WarpX), whereas Vay and Higuera-Cary keep the particle on a straight line.")


# ----------------------------------------------------------------------------
def vlasov_root(kld, guess):
    """Complex root w/w_p of 1 + (1 + zeta Z(zeta)) / (k lambda_D)^2 = 0, zeta = w/(sqrt(2) k v_th)."""
    from scipy.optimize import fsolve
    from scipy.special import wofz

    def Z(zz):
        return 1j * np.sqrt(np.pi) * wofz(zz)

    def f(x):
        w = x[0] + 1j * x[1]
        zeta = w / (np.sqrt(2) * kld)
        D = 1 + (1 + zeta * Z(zeta)) / kld ** 2
        return [D.real, D.imag]
    r = fsolve(f, guess, xtol=1e-13)
    return r[0], r[1]


def _thermal_es(kld, ppc, tmax, name, amp=0.01):
    n0, Te = 1e15, 1.0
    wp = np.sqrt(n0 * C.e ** 2 / (C.eps0 * C.me))
    ld = np.sqrt(C.eps0 * Te / (n0 * C.e))
    k = kld / ld
    L = 2 * np.pi / k
    m = new_model("1D-Z", "electrostatic")
    m["domain"].update(z_min="0", z_max=_r(L), nz="64")
    for s in ("z_min", "z_max"):
        m["boundaries"][s].update(field="periodic", particles="periodic")
    m["species"] = [{"name": "e", "preset": "electron"}]
    m["loaders"] = [dict(name="p", species="e", z0="0", z1=_r(L), y0="0", y1="1", density=_r(n0), ppc=str(ppc),
                         quiet_start=True, temperature=_r(Te), perturbation="density", pert_amplitude=_r(amp),
                         pert_mode="1")]
    m["simulation"].update(t_end=_r(tmax / wp), dt=_r(0.05 / wp))
    quiet_diag(m, 1)
    sim = run(m, name)
    t, W = hist(sim, "W_field")
    return t * wp, W


def _peaks(t, W):
    i = np.where((W[1:-1] > W[:-2]) & (W[1:-1] >= W[2:]))[0] + 1
    return t[i], W[i]


def landau():
    rows = []
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    # Landau damping, k lambda_D = 0.5
    wr, wi = vlasov_root(0.5, [1.4, -0.15])
    t, W = _thermal_es(0.5, 4000, 22.0, "landau_05", amp=0.05)
    tp, Wp = _peaks(t, W)
    sel = (tp > 1.0) & (tp < 14.5)
    slope = np.polyfit(tp[sel], np.log(Wp[sel]), 1)[0]
    gam = slope / 2
    w_meas = np.pi / np.mean(np.diff(tp[sel]))
    rows.append(row("Landau damping k lambda_D = 0.5: damping rate gamma / w_p", gam, wi, 0.05))
    rows.append(row("Landau damping k lambda_D = 0.5: frequency w / w_p", w_meas, wr, 0.02))
    ax[0].semilogy(t, W / W[0], lw=0.8, label="VEXPACE")
    ax[0].semilogy(t, np.exp(2 * wi * t), "k--", lw=1, label=f"exact: gamma = {wi:.4f} w_p")
    ax[0].plot(tp[sel], Wp[sel] / W[0], "o", ms=3)
    ax[0].set_xlabel("t w_p")
    ax[0].set_ylabel("field energy / initial")
    ax[0].set_title("Landau damping, k lambda_D = 0.5")
    ax[0].legend(fontsize=7)
    # Bohm-Gross / thermal Langmuir wave, k lambda_D = 0.3
    wr3, wi3 = vlasov_root(0.3, [1.15, -0.01])
    t, W = _thermal_es(0.3, 8000, 30.0, "bohm_gross_03")
    w3 = fit_frequency(t, W) / 2
    rows.append(row("thermal Langmuir wave k lambda_D = 0.3: w / w_p (exact Vlasov root)", w3, wr3, 0.01))
    ax[1].plot(t, W / W.max(), lw=0.8)
    ax[1].set_xlabel("t w_p")
    ax[1].set_ylabel("field energy / max")
    ax[1].set_title(f"k lambda_D = 0.3: w = {w3:.4f} w_p (exact {wr3:.4f})")
    fig.tight_layout()
    return record("27_landau_bohm_gross", "Landau damping and thermal Langmuir dispersion (1D ES)", rows, fig,
                  notes=f"Maxwellian electrons on a fixed neutralising background, 5% density perturbation (k lambda_D = 0.5) / 1% (k lambda_D = 0.3), quiet start in position and velocity, "
                        f"64 cells per wavelength, References are the exact complex "
                        f"roots of the Vlasov-Poisson dispersion relation 1 + (1 + zeta Z(zeta))/(k lambda_D)^2 "
                        f"= 0: k lambda_D = 0.5 -> w = {wr:.4f} {wi:+.4f}i w_p; k lambda_D = 0.3 -> "
                        f"w = {wr3:.4f} {wi3:+.4f}i w_p (the fluid Bohm-Gross estimate sqrt(1 + 3 k^2 lambda_D^2) = "
                        f"{np.sqrt(1.27):.4f} is 3 % low at this k lambda_D; VEXPACE gives {w3:.4f}).")


# ----------------------------------------------------------------------------
def emwave():
    """Transverse EM wave in a cold plasma: w^2 = w_p^2 + c^2 k^2 (1D EM, periodic)."""
    from picstudio.core.simulation import Simulation
    rows = []
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    n0 = 1e18
    wp = np.sqrt(n0 * C.e ** 2 / (C.eps0 * C.me))
    for ck_wp in (1.0, 2.0):
        k = ck_wp * wp / C.c
        L = 2 * np.pi / k
        m = new_model("1D-Z", "electromagnetic")
        m["domain"].update(z_min="0", z_max=_r(L), nz="128")
        for s in ("z_min", "z_max"):
            m["boundaries"][s].update(field="periodic", particles="periodic")
        m["species"] = [{"name": "e", "preset": "electron"}]
        m["loaders"] = [dict(name="p", species="e", z0="0", z1=_r(L), y0="0", y1="1", density=_r(n0),
                             ppc="32", quiet_start=True)]
        m["simulation"].update(t_end=_r(40 * 2 * np.pi / wp), initial_field="none")
        quiet_diag(m, 2)
        m["probes"] = [dict(name=q, kind="point", quantity=q, z=_r(L / 4), y="0") for q in ("Ex", "Ey")]
        sim = Simulation(m, out_dir=os.path.join(HERE, "work", f"emwave_{ck_wp}"), log=lambda x: None)
        sp = sim.species[0]
        sp.u1[:] = 1e4 * np.sin(k * sp.z)          # transverse velocity perturbation
        sim.run()
        t, W = hist(sim, "W_field")
        H, cols = sim.history_array()
        E = max((H[:, cols.index(f"probe_{q}")] for q in ("Ex", "Ey")), key=np.std)
        w = fit_frequency(t, E)        # E has no static part (B and the current do: magnetostatic mode)
        # Yee + leapfrog discrete dispersion (for reference): small at 128 cells per wavelength
        th = np.sqrt(1 + ck_wp ** 2)
        rows.append(row(f"EM wave in plasma, ck = {ck_wp:g} w_p: w / sqrt(w_p^2 + c^2 k^2)", w / wp / th, 1.0, 0.01))
        ax.plot(t * wp, W / W.max(), lw=0.8, label=f"ck = {ck_wp:g} w_p")
    ax.set_xlabel("t w_p")
    ax.set_ylabel("field energy / max")
    ax.set_xlim(0, 10)
    ax.legend(fontsize=7)
    return record("28_em_wave_in_plasma", "Electromagnetic wave dispersion in a cold plasma (1D EM)", rows, fig,
                  notes="Transverse velocity perturbation u_y = A sin(kz) in a periodic cold plasma; the "
                        "transverse E at z = L/4 oscillates at w, compared with w^2 = w_p^2 + c^2 k^2.")


# ----------------------------------------------------------------------------
def energy():
    """Total-energy conservation of a warm plasma, EM 1D, dz = lambda_D (WarpX test
    'energy_conserving_thermal_plasma' idea): energy-conserving vs momentum-conserving gathering."""
    rows = []
    res = {}
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    n0, Te = 1e18, 100.0
    wp = np.sqrt(n0 * C.e ** 2 / (C.eps0 * C.me))
    ld = np.sqrt(C.eps0 * Te / (n0 * C.e))
    for fg, jf in (("energy-conserving", 0), ("momentum-conserving", 0), ("energy-conserving", 1)):
        L = 64 * ld
        m = new_model("1D-Z", "electromagnetic")
        m["domain"].update(z_min="0", z_max=_r(L), nz="64")
        for s in ("z_min", "z_max"):
            m["boundaries"][s].update(field="periodic", particles="periodic")
        m["species"] = [{"name": "e", "preset": "electron"}]
        m["loaders"] = [dict(name="p", species="e", z0="0", z1=_r(L), y0="0", y1="1", density=_r(n0),
                             ppc="64", quiet_start=True, temperature=_r(Te))]
        m["simulation"].update(t_end=_r(300 / wp), initial_field="none", field_gathering=fg,
                               current_filter=str(jf))
        quiet_diag(m, 10)
        key = fg + (" + J filter (WarpX defaults)" if jf else "")
        sim = run(m, f"energy_{fg}_{jf}")
        H, cols = sim.history_array()
        t = H[:, 1] * wp
        Et = H[:, cols.index("KE_e")] + H[:, cols.index("W_field")]
        dE = (Et[-1] - Et[0]) / Et[0]
        res[key] = dE
        ax.plot(t, Et / Et[0] - 1, label=f"{key}: {100 * dE:+.3f} %")
    rows.append(row("energy-conserving gather: total energy change after 300/w_p", res["energy-conserving"], 0.0,
                    0.02, absolute=True))
    rows.append(row("energy-conserving + 1-pass J filter (WarpX defaults): total energy change", 
                    res["energy-conserving + J filter (WarpX defaults)"], 0.0, 0.02, absolute=True))
    ax.set_xlabel("t w_p")
    ax.set_ylabel("relative change of total energy")
    ax.legend(fontsize=7)
    return record("29_energy_conservation", "Energy conservation of a thermal plasma (1D EM, dz = lambda_D)", rows,
                  fig, notes=f"64 cells of one Debye length, 64 particles per cell, T = 100 eV. Total energy change: "
                             f"energy-conserving {100 * res['energy-conserving']:+.2f} %, momentum-conserving "
                             f"{100 * res['momentum-conserving']:+.2f} % (grid heating), energy-conserving with WarpX's 1-pass "
                             f"bilinear J filter {100 * res['energy-conserving + J filter (WarpX defaults)']:+.2f} %.")


SUITE = {"pushers": pushers, "landau": landau, "emwave": emwave, "energy": energy}

if __name__ == "__main__":
    for n in (sys.argv[1:] or list(SUITE)):
        SUITE[n]()
