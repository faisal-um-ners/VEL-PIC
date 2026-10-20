"""Electromagnetic benchmarks: cavity modes, absorbing boundaries, Gauss-law conservation,
laser propagation in plasma, plasma wakefield."""
from __future__ import annotations

import numpy as np
from scipy.special import j0, j1, jn_zeros

from common import C, Simulation, WORK, fit_frequency, hist, new_model, plt, quiet_diag, record, row, run
import os


def cavities():
    rows = []
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    # RZ pillbox TM010 and TM011
    R, L = 0.05, 0.04
    for mode in ("TM010", "TM011"):
        m = new_model("2D-RZ", "electromagnetic")
        m["domain"].update(z_min="0", z_max=str(L), nz="32", y_min="0", y_max=str(R), ny="40")
        for s in ("z_min", "z_max", "y_max"):
            m["boundaries"][s].update(field="conductor")
        m["simulation"].update(t_end="30e-9", cfl="0.95")
        quiet_diag(m, 1)
        m["probes"] = [{"name": "p", "quantity": "Ez", "z": str(L / 3), "y": str(0.2 * R)}]
        sim = Simulation(m, out_dir=os.path.join(WORK, "cav"), log=lambda x: None)
        x01 = jn_zeros(0, 1)[0]
        zh = 0.5 * (sim.grid.z[1:] + sim.grid.z[:-1])
        if mode == "TM010":
            sim.yee.E0[:] = j0(x01 * sim.grid.y / R)[None, :]
            fth = x01 * C.c / (2 * np.pi * R)
        else:
            sim.yee.E0[:] = np.cos(np.pi * zh / L)[:, None] * j0(x01 * sim.grid.y / R)[None, :]
            fth = C.c / (2 * np.pi) * np.sqrt((x01 / R) ** 2 + (np.pi / L) ** 2)
        sim.run()
        t, s = hist(sim, "probe_p")
        f = fit_frequency(t, s) / (2 * np.pi)
        rows.append(row(f"RZ pillbox {mode} frequency (GHz)", f / 1e9, fth / 1e9, 0.005))
        axs[0].plot(t * 1e9, s / np.abs(s).max(), lw=0.8, label=mode)
    axs[0].set_xlabel("t (ns)")
    axs[0].set_title("RZ pillbox cavity probe E_z")
    axs[0].legend()
    # ZY rectangular cavity: TM11 (out-of-plane Ex) and TE21 (in-plane, B_x)
    Lz, Ly = 0.06, 0.04
    for mode in ("TMx 1,1", "TEx 2,1"):
        m = new_model("2D-ZY", "electromagnetic")
        m["domain"].update(z_min="0", z_max=str(Lz), nz="48", y_min="0", y_max=str(Ly), ny="32")
        for s in ("z_min", "z_max", "y_min", "y_max"):
            m["boundaries"][s].update(field="conductor")
        m["simulation"].update(t_end="20e-9", cfl="0.95")
        quiet_diag(m, 1)
        qty = "Ex" if mode.startswith("TM") else "Bx"
        m["probes"] = [{"name": "p", "quantity": qty, "z": str(0.3 * Lz), "y": str(0.3 * Ly)}]
        sim = Simulation(m, out_dir=os.path.join(WORK, "cav"), log=lambda x: None)
        g = sim.grid
        if mode.startswith("TM"):
            sim.yee.E2[:] = np.sin(np.pi * g.z / Lz)[:, None] * np.sin(np.pi * g.y / Ly)[None, :]
            fth = C.c / 2 * np.hypot(1 / Lz, 1 / Ly)
        else:
            zh = 0.5 * (g.z[1:] + g.z[:-1])
            yh = 0.5 * (g.y[1:] + g.y[:-1])
            sim.yee.B2[:] = np.cos(2 * np.pi * zh / Lz)[:, None] * np.cos(np.pi * yh / Ly)[None, :]
            fth = C.c / 2 * np.hypot(2 / Lz, 1 / Ly)
        sim.run()
        t, s = hist(sim, "probe_p")
        f = fit_frequency(t, s) / (2 * np.pi)
        rows.append(row(f"ZY rectangular cavity {mode} frequency (GHz)", f / 1e9, fth / 1e9, 0.005))
        axs[1].plot(t * 1e9, s / np.abs(s).max(), lw=0.8, label=mode)
    axs[1].set_xlabel("t (ns)")
    axs[1].set_title("ZY rectangular cavity probes")
    axs[1].legend()
    return record("10_em_cavities", "EM cavity eigenfrequencies (Yee FDTD, RZ and ZY)", rows, fig)


def absorber():
    """Reflection from the graded absorbing layer (1D normal incidence, 2D Gaussian beam)."""
    rows = []
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    lam = 1e-6
    for geom in ("1D-Z", "2D-ZY"):
        m = new_model(geom, "electromagnetic")
        m["domain"].update(z_min="0", z_max=str(30 * lam), nz=str(600), y_min=str(-10 * lam), y_max=str(10 * lam),
                           ny="200")
        for s in ("z_min", "z_max", "y_min", "y_max"):
            m["boundaries"][s].update(field="absorbing", absorber_cells=20)
        m["sources"] = [{"name": "L", "type": "laser", "z_position": str(2 * lam), "wavelength": str(lam),
                         "a0": "0.01", "duration": "8e-15", "t_peak": "16e-15", "polarization": "x",
                         "waist": "0" if geom == "1D-Z" else str(3 * lam), "focus_z": str(15 * lam)}]
        m["simulation"].update(t_end="200e-15", cfl="0.95")
        quiet_diag(m, 2)
        sim = run(m, f"absorber_{geom}")
        t, W = hist(sim, "W_field")
        Wmax = W.max()
        Wend = W[-1]
        rows.append(row(f"{geom}: residual energy after absorption / peak", Wend / Wmax, 0.0, 1e-3, absolute=True))
        ax.semilogy(t * 1e15, W / Wmax + 1e-12, label=geom)
    ax.set_xlabel("t (fs)")
    ax.set_ylabel("field energy / max")
    ax.legend()
    ax.set_title("Laser pulse leaving through absorbing boundaries")
    return record("11_absorbing_boundary", "Graded absorbing boundary performance", rows, fig)


def gauss_law():
    rows = []
    for geom in ("1D-Z", "2D-ZY", "2D-RZ"):
        m = new_model(geom, "electromagnetic")
        m["domain"].update(z_min="0", z_max="0.01", nz="32", y_min="0", y_max="0.01", ny="32")
        for s in ("z_min", "z_max"):
            m["boundaries"][s].update(field="periodic")
        if geom != "1D-Z":
            m["boundaries"]["y_max"].update(field="conductor")
        if geom == "2D-ZY":
            m["boundaries"]["y_min"].update(field="conductor")
        m["species"] = [{"name": "e", "preset": "electron"}, {"name": "i", "preset": "proton"}]
        m["loaders"] = [{"name": "pe", "species": "e", "z0": "0", "z1": "0.006", "y0": "0.002", "y1": "0.006",
                         "density": "1e15", "ppc": "8", "temperature": "1000", "drift": ["3e7", "1e7", "2e7"]},
                        {"name": "pi", "species": "i", "z0": "0.003", "z1": "0.01", "y0": "0", "y1": "0.008",
                         "density": "1e15", "ppc": "4", "temperature": "10000"}]
        m["simulation"].update(n_steps="100", t_end="1")
        quiet_diag(m, 1000)
        sim = Simulation(m, out_dir=os.path.join(WORK, "gauss"), log=lambda x: None)
        r0, dv = _gauss_residual(sim)
        for _ in range(100):
            sim.step()
        r1, dv = _gauss_residual(sim)
        rows.append(row(f"{geom}: change of (rho/eps0 - div E) / max|div E| over 100 steps",
                        np.abs(r1 - r0).max() / np.abs(dv).max(), 0.0, 1e-10, absolute=True))
    return record("12_gauss_law", "Charge conservation of the EM current deposition (Esirkepov)", rows)


def _gauss_residual(s):
    g = s.grid
    y = s.yee
    Q, _ = s._deposit_charge()
    rho = Q / g.node_volume
    d = np.zeros(g.shape)
    if g.dim == 1:
        d[1:-1] = (y.E0[1:] - y.E0[:-1]) / g.dz
        r = (rho / C.eps0 - d)[1:-1]
        return r, d
    E0, E1 = y.E0, y.E1
    if g.rz:
        rh = y.rh
        d[1:-1, 1:-1] = ((E0[1:, 1:-1] - E0[:-1, 1:-1]) * g.wr[None, 1:-1] + 2 * np.pi * (
            E1[1:-1, 1:] * rh[None, 1:] - E1[1:-1, :-1] * rh[None, :-1]) * g.dz) / (g.dz * g.wr[None, 1:-1])
        d[1:-1, 0] = ((E0[1:, 0] - E0[:-1, 0]) * g.wr[0] + 2 * np.pi * E1[1:-1, 0] * rh[0] * g.dz) / (g.dz * g.wr[0])
        return (rho / C.eps0 - d)[1:-1, :-1], d
    d[1:-1, 1:-1] = (E0[1:, 1:-1] - E0[:-1, 1:-1]) / g.dz + (E1[1:-1, 1:] - E1[1:-1, :-1]) / g.dy
    return (rho / C.eps0 - d)[1:-1, 1:-1], d


def laser_plasma():
    """Group velocity in an underdense slab and reflection from an overdense slab (1D EM)."""
    rows = []
    lam = 0.8e-6
    w = 2 * np.pi * C.c / lam
    nc = C.eps0 * C.me * w ** 2 / C.e ** 2
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for nfrac in (0.5, 4.0):
        m = new_model("1D-Z", "electromagnetic")
        Lx = 80 * lam
        m["domain"].update(z_min="0", z_max=str(Lx), nz=str(3200))
        for s in ("z_min", "z_max"):
            m["boundaries"][s].update(field="absorbing", absorber_cells=40)
        m["species"] = [{"name": "e", "preset": "electron"}, {"name": "i", "preset": "proton"}]
        m["loaders"] = [dict(name="slab", species="e", z0=str(20 * lam), z1=str(70 * lam), y0="0", y1="1",
                             density=str(nfrac * nc), ppc="32", quiet_start=True),
                        dict(name="ions", species="i", z0=str(20 * lam), z1=str(70 * lam), y0="0", y1="1",
                             density=str(nfrac * nc), ppc="32", quiet_start=True)]
        m["sources"] = [{"name": "L", "type": "laser", "z_position": str(2 * lam), "wavelength": str(lam),
                         "a0": "0.01", "duration": "12e-15", "t_peak": "25e-15", "polarization": "x"}]
        z1, z2 = 30 * lam, 60 * lam
        m["probes"] = [{"name": "a", "quantity": "Ex", "z": str(z1), "y": "0"},
                       {"name": "b", "quantity": "Ex", "z": str(z2), "y": "0"}]
        m["power_lines"] = [{"name": "S", "orientation": "z", "position": str(10 * lam), "start": "0", "end": "1"}]
        m["simulation"].update(t_end=str(450e-15 if nfrac < 1 else 250e-15), cfl="0.95")
        quiet_diag(m, 1)
        sim = run(m, f"laser_{nfrac}")
        if nfrac < 1:
            t, a = hist(sim, "probe_a")
            _, b = hist(sim, "probe_b")
            from scipy.signal import hilbert
            ea, eb = np.abs(hilbert(a)), np.abs(hilbert(b))
            ta, tb = _peak_time(t, ea), _peak_time(t, eb)
            vg = (z2 - z1) / (tb - ta)
            vth = C.c * np.sqrt(1 - nfrac)
            rows.append(row("underdense n = 0.5 n_c: group velocity / c", vg / C.c, vth / C.c, 0.02))
            axs[0].plot(t * 1e15, ea, label="z = 30 lambda")
            axs[0].plot(t * 1e15, eb, label="z = 60 lambda")
            axs[0].set_title(f"n = 0.5 n_c: v_g = {vg / C.c:.3f} c (theory {vth / C.c:.3f} c)")
            axs[0].set_xlabel("t (fs)")
            axs[0].legend()
        else:
            t, S = hist(sim, "P_S")
            dt = t[1] - t[0]
            Ein = S[S > 0].sum() * dt
            Eout = -S[S < 0].sum() * dt
            rows.append(row("overdense n = 4 n_c: reflected / incident energy", Eout / Ein, 1.0, 0.03))
            axs[1].plot(t * 1e15, S, lw=0.6)
            axs[1].set_title(f"n = 4 n_c: R = {Eout / Ein:.3f}")
            axs[1].set_xlabel("t (fs)")
            axs[1].set_ylabel("Poynting flux at z = 10 lambda (W/m^2)")
    return record("13_laser_plasma", "Laser propagation in plasma: dispersion and critical-density reflection",
                  rows, fig)


def _peak_time(t, e):
    i = int(np.argmax(e))
    i = min(max(i, 1), len(e) - 2)
    y0, y1, y2 = e[i - 1], e[i], e[i + 1]
    den = y0 - 2 * y1 + y2
    off = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    return t[i] + off * (t[1] - t[0])


def wakefield():
    np0 = 1e22
    wp = np.sqrt(np0 * C.e ** 2 / (C.eps0 * C.me))
    kp = wp / C.c
    lp = 2 * np.pi / kp
    alpha = 0.05
    Lb = lp / 2
    m = new_model("1D-Z", "electromagnetic")
    Lx = 6 * lp
    m["domain"].update(z_min="0", z_max=str(Lx), nz=str(1800))
    for s in ("z_min", "z_max"):
        m["boundaries"][s].update(field="absorbing", absorber_cells=40)
    m["species"] = [{"name": "p", "preset": "electron"}, {"name": "b", "preset": "electron"}]
    ub = C.c * np.sqrt((1 + 1e9 * C.e / (C.me * C.c ** 2)) ** 2 - 1)
    m["loaders"] = [dict(name="p", species="p", z0="0", z1=str(Lx), y0="0", y1="1", density=str(np0), ppc="32",
                         quiet_start=True),
                    dict(name="b", species="b", z0=str(0.5 * lp), z1=str(0.5 * lp + Lb), y0="0", y1="1",
                         density=str(alpha * np0), ppc="64", quiet_start=True, drift=[str(ub), "0", "0"])]
    m["simulation"].update(t_end=str(4.0 * lp / C.c), cfl="0.98", smoothing=1, initial_field="none")
    m["diagnostics"].update(history_every="100", field_every="100000", particle_every="1000000",
                            live_every="1000000")
    sim = run(m, "wakefield")
    Ez = sim.F[0]
    z = sim.grid.z
    zb = 0.5 * lp + C.c * sim.t
    sel = (z < zb - Lb - 0.25 * lp) & (z > zb - Lb - 2.5 * lp)
    Emax = np.abs(Ez[sel]).max()
    Eth = C.e * alpha * np0 / (C.eps0 * kp) * 2 * np.sin(kp * Lb / 2)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot((z - zb) / lp, Ez / Eth, lw=1, label="VEL-PIC E_z")
    xi = np.linspace(-3, 0, 400) * lp
    # linear theory behind a flat-top bunch: Ez = -Eth*cos(kp*(xi + Lb/2)) (amplitude Eth)
    ax.axhline(1, color="k", ls="--", lw=0.8)
    ax.axhline(-1, color="k", ls="--", lw=0.8, label="linear theory amplitude")
    ax.set_xlim(-3, 0.5)
    ax.set_xlabel("(z - z_head) / lambda_p")
    ax.set_ylabel("E_z / E_lin")
    ax.legend(fontsize=8)
    ax.set_title("Beam-driven plasma wake (n_b = 0.05 n_p, k_p L = pi)")
    return record("14_plasma_wakefield", "Relativistic beam-driven plasma wakefield amplitude (1D EM)",
                  [row("wake amplitude / linear theory", Emax / Eth, 1.0, 0.06)], fig,
                  notes="The ~5 % excess over linear theory is a weakly non-linear effect (n_b/n_p = 0.05); WarpX gives the same amplitude (benchmark 19).")
