"""Code-to-code comparison with WarpX: smooth-bore magnetron diodes (crossed-field gaps).

    E  planar crossed-field gap (2D: WarpX XZ, VEXPACE ZY), B below and above the Hull cutoff
    F  cylindrical (coaxial) crossed-field gap (RZ), inner cathode, axial B, below and above the
       cylindrical Hull cutoff of Lovelace & Ott

Electrons are injected from the cathode at a fixed current density (temperature-limited, below the
space-charge limit) with a small normal energy; both codes use the electrostatic solver, linear
shape, Boris push, energy-conserving gather and the same time step.  Compared: the potential profile
across the gap and the number of electrons in the gap (which doubles when the flow is magnetically
insulated, since the electrons return to the cathode).

    WARPX_BIN=~/ext/warpx/build/bin python benchmarks/warpx_compare/compare_crossed_field.py
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from common import C, hist, new_model, plt, quiet_diag, record, row, run  # noqa: E402
from compare_warpx import WDIR, last_plotfile  # noqa: E402
from compare_warpx import run_warpx as _run_warpx  # noqa: E402


def run_warpx(dim, name, text):
    """Run WarpX, or reuse an existing run with identical inputs (REUSE_WARPX=1)."""
    d = os.path.join(WDIR, name)
    if os.environ.get("REUSE_WARPX") and os.path.exists(os.path.join(d, "inputs")) \
            and open(os.path.join(d, "inputs")).read() == text and glob.glob(os.path.join(d, "diags", "diag1*")):
        return d
    return _run_warpx(dim, name, text)


def _r(x):
    return repr(float(x))


def _gamma(V):
    return 1 + C.e * V / (C.me * C.c ** 2)


def hull_planar(V, d):
    return C.me * C.c / (C.e * d) * np.sqrt(_gamma(V) ** 2 - 1)


def hull_coax(V, a, b):
    """Lovelace & Ott (1974): inner cathode a, outer anode b, effective gap (b^2 - a^2)/(2b)."""
    return C.me * C.c / (C.e * (b * b - a * a) / (2 * b)) * np.sqrt(_gamma(V) ** 2 - 1)


def _warpx_count(d):
    fn = os.path.join(d, "diags", "reducedfiles", "PN.txt")
    A = np.loadtxt(fn)
    return A[:, 1], A[:, -1]          # time, total weight of electrons


# ---------------------------------------------------------------------------------------------
def planar(ratios=(0.6, 1.3)):
    V, d = 10e3, 5e-3
    nz, nx = 64, 8
    Lx = d * nx / nz
    BH = hull_planar(V, d)
    JCL = 4 * C.eps0 / 9 * np.sqrt(2 * C.e / C.me) * V ** 1.5 / d ** 2
    J = 0.25 * JCL
    E0 = 2.0                                          # injection energy (eV)
    u0 = np.sqrt(_gamma(E0) ** 2 - 1)
    vmax = C.c * np.sqrt(1 - 1 / _gamma(V) ** 2)
    dt = 0.2 * (d / nz) / vmax
    nstep = int(2.2e-9 / dt)
    out = {}
    for r in ratios:
        B = r * BH
        wx = f"""
max_step = {nstep}
warpx.const_dt = {dt:.17g}
warpx.do_electrostatic = labframe
warpx.use_filter = 0
warpx.self_fields_required_precision = 1e-7
warpx.self_fields_absolute_tolerance = 1e-3
algo.particle_shape = 1
algo.particle_pusher = boris
amr.n_cell = {nx} {nz}
amr.max_grid_size = 64
amr.blocking_factor = 8
amr.max_level = 0
geometry.dims = 2
geometry.prob_lo = 0.0 0.0
geometry.prob_hi = {Lx:.17g} {d:.17g}
boundary.field_lo = periodic pec
boundary.field_hi = periodic pec
boundary.potential_lo_z = 0.0
boundary.potential_hi_z = {V:.17g}
boundary.particle_lo = periodic absorbing
boundary.particle_hi = periodic absorbing
particles.species_names = electrons
particles.B_ext_particle_init_style = constant
particles.B_external_particle = 0.0 {B:.17g} 0.0
electrons.charge = -q_e
electrons.mass = m_e
electrons.injection_style = NFluxPerCell
electrons.num_particles_per_cell = 4
electrons.surface_flux_pos = 1e-12
electrons.flux_normal_axis = z
electrons.flux_direction = 1
electrons.flux_profile = constant
electrons.flux = {J / C.e:.17g}
electrons.momentum_distribution_type = constant
electrons.uz = {u0:.17g}
diagnostics.diags_names = diag1
diag1.intervals = {nstep}
diag1.diag_type = Full
diag1.fields_to_plot = phi rho
warpx.reduced_diags_names = PN
PN.type = ParticleNumber
PN.intervals = 20
"""
        dd = run_warpx("2d", f"xfield_{r}", wx)
        f, (xc, zc), _ = last_plotfile(dd)
        phiw = f["phi"]
        phiw = phiw.mean(axis=0) if phiw.shape[0] == len(xc) else phiw.mean(axis=1)
        tw, Nw = _warpx_count(dd)
        # VEXPACE (ZY; out-of-plane component x = -y of WarpX XZ)
        m = new_model("2D-ZY", "electrostatic")
        m["domain"].update(z_min="0", z_max=_r(d), nz=str(nz), y_min="0", y_max=_r(Lx), ny=str(nx))
        m["boundaries"]["z_min"].update(field="conductor", voltage="0")
        m["boundaries"]["z_max"].update(field="conductor", voltage=_r(V))
        for s in ("y_min", "y_max"):
            m["boundaries"][s].update(field="periodic", particles="periodic")
        m["species"] = [{"name": "e", "preset": "electron"}]
        m["emitters"] = [{"name": "k", "species": "e", "model": "beam", "z0": "0", "y0": "0", "z1": "0",
                          "y1": _r(Lx), "current_density": _r(J), "energy": _r(E0), "particles_per_cell": "4"}]
        m["external_fields"] = [{"name": "B", "type": "uniform", "B": ["0", "0", _r(-B)]}]
        m["simulation"].update(n_steps=str(nstep), t_end="1", dt=_r(dt), field_gathering="energy-conserving",
                               initial_field="poisson")
        quiet_diag(m, 20)
        sim = run(m, f"xfield_{r}")
        zp = sim.grid.z
        phip = sim.phi.mean(axis=1)
        tp, Np = hist(sim, "N_e")
        sp = sim.species[0]
        Np = Np * (float(sp._w[:sp.n].mean()) if sp.n else 0.0)   # macro-particle count -> physical (equal weights)
        out[r] = dict(zc=zc, phiw=phiw, zp=zp, phip=phip, tw=tw, Nw=Nw, tp=tp, Np=Np, B=B)
    return out, dict(V=V, d=d, BH=BH, J=J, JCL=JCL)


def coax(ratios=(0.6, 1.3)):
    V, a, b = 10e3, 5e-3, 10e-3
    nr, nz = 64, 8
    Lz = (b - a) * nz / nr
    Bs = hull_coax(V, a, b)
    # Langmuir-Blodgett scale for the injected current: use a fraction of the 1D CL density at the cathode
    JCL = 4 * C.eps0 / 9 * np.sqrt(2 * C.e / C.me) * V ** 1.5 / (b - a) ** 2
    J = 0.25 * JCL
    E0 = 2.0
    u0 = np.sqrt(_gamma(E0) ** 2 - 1)
    vmax = C.c * np.sqrt(1 - 1 / _gamma(V) ** 2)
    dr = (b - a) / nr
    dt = 0.2 * dr / vmax
    nstep = int(2.2e-9 / dt)
    out = {}
    for r in ratios:
        B = r * Bs
        wx = f"""
max_step = {nstep}
warpx.const_dt = {dt:.17g}
warpx.do_electrostatic = labframe
warpx.use_filter = 0
warpx.self_fields_required_precision = 1e-7
warpx.self_fields_absolute_tolerance = 1e-3
warpx.n_rz_azimuthal_modes = 1
algo.particle_shape = 1
algo.particle_pusher = boris
amr.n_cell = {2 * nr} {nz}
amr.max_grid_size = 128
amr.blocking_factor = 8
amr.max_level = 0
geometry.dims = RZ
geometry.prob_lo = 0.0 0.0
geometry.prob_hi = {b:.17g} {Lz:.17g}
boundary.field_lo = none periodic
boundary.field_hi = pec periodic
boundary.potential_hi_x = 0.0
boundary.particle_lo = none periodic
boundary.particle_hi = absorbing periodic
# inner cathode r < a as an embedded boundary at -V, anode at 0 (WarpX's RZ Poisson solver needs
# r_min = 0, and with an embedded boundary the potential is applied on the EB)
warpx.eb_implicit_function = "-(x**2-{a:.17g}**2)"
warpx.eb_potential(x,y,z,t) = "{-V:.17g}"
particles.species_names = electrons
particles.B_ext_particle_init_style = constant
particles.B_external_particle = 0.0 0.0 {B:.17g}
electrons.charge = -q_e
electrons.mass = m_e
electrons.injection_style = NFluxPerCell
electrons.num_particles_per_cell = 4
electrons.surface_flux_pos = {a * (1 + 1e-9):.17g}
electrons.flux_normal_axis = r
electrons.flux_direction = 1
electrons.flux_profile = constant
electrons.flux = {J / C.e:.17g}
electrons.momentum_distribution_type = constant
electrons.ux = {u0:.17g}
diagnostics.diags_names = diag1
diag1.intervals = {nstep}
diag1.diag_type = Full
diag1.fields_to_plot = phi rho
warpx.reduced_diags_names = PN
PN.type = ParticleNumber
PN.intervals = 20
"""
        dd = run_warpx("rz", f"xcoax_{r}", wx)
        f, (rc, zc), _ = last_plotfile(dd)
        phiw = f["phi"]
        phiw = phiw.mean(axis=1) if phiw.shape[0] == len(rc) else phiw.mean(axis=0)
        tw, Nw = _warpx_count(dd)
        m = new_model("2D-RZ", "electrostatic")
        m["domain"].update(z_min="0", z_max=_r(Lz), nz=str(nz), y_min=_r(a), y_max=_r(b), ny=str(nr))
        for s in ("z_min", "z_max"):
            m["boundaries"][s].update(field="periodic", particles="periodic")
        m["boundaries"]["y_min"].update(field="conductor", voltage=_r(-V), particles="absorb")
        m["boundaries"]["y_max"].update(field="conductor", voltage="0", particles="absorb")
        m["species"] = [{"name": "e", "preset": "electron"}]
        m["emitters"] = [{"name": "k", "species": "e", "model": "beam", "z0": "0", "y0": _r(a), "z1": _r(Lz),
                          "y1": _r(a), "normal": "+y", "current_density": _r(J), "energy": _r(E0),
                          "particles_per_cell": "4"}]
        m["external_fields"] = [{"name": "B", "type": "uniform", "B": [_r(B), "0", "0"]}]
        m["simulation"].update(n_steps=str(nstep), t_end="1", dt=_r(dt), field_gathering="energy-conserving",
                               initial_field="poisson")
        quiet_diag(m, 20)
        sim = run(m, f"xcoax_{r}")
        rp = sim.grid.y
        phip = sim.phi.mean(axis=0)
        tp, Np = hist(sim, "N_e")
        sp = sim.species[0]
        Np = Np * (float(sp._w[:sp.n].mean()) if sp.n else 0.0)   # macro-particle count -> physical
        out[r] = dict(zc=rc, phiw=phiw, zp=rp, phip=phip, tw=tw, Nw=Nw, tp=tp, Np=Np, B=B)
    return out, dict(V=V, a=a, b=b, Bs=Bs, J=J)


def _report(name, title, res, prm, xlabel, Bname, notes):
    rows = []
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    V = prm["V"]
    for k, (r, o) in enumerate(sorted(res.items())):
        inside = (o["zc"] >= o["zp"].min()) & (o["zc"] <= o["zp"].max())
        phip_c = np.interp(o["zc"][inside], o["zp"], o["phip"])
        dphi = np.abs(phip_c - o["phiw"][inside]).max() / V
        late_w = o["tw"] > 0.45 * o["tw"][-1]
        late_p = o["tp"] > 0.45 * o["tp"][-1]
        Nw, Np = o["Nw"][late_w].mean(), o["Np"][late_p].mean()
        state = "insulated" if r > 1 else "conducting"
        rows.append(row(f"B = {r:g} {Bname} ({state}): max |phi_VEXPACE - phi_WarpX| / V", dphi, 0.0, 0.01,
                        absolute=True))
        rows.append(row(f"B = {r:g} {Bname} ({state}): electrons in the gap, VEXPACE / WarpX", Np / Nw, 1.0, 0.03))
        c = f"C{k}"
        ax[0].plot(o["zc"] * 1e3, o["phiw"] / V, "-", color=c, lw=2, alpha=0.5, label=f"WarpX, B = {r:g} {Bname}")
        ax[0].plot(o["zp"] * 1e3, o["phip"] / V, ":", color=c, lw=2, label=f"VEXPACE, B = {r:g} {Bname}")
        ax[1].plot(o["tw"] * 1e9, o["Nw"], "-", color=c, lw=2, alpha=0.5, label=f"WarpX, B = {r:g} {Bname}")
        ax[1].plot(o["tp"] * 1e9, o["Np"], ":", color=c, lw=2, label=f"VEXPACE, B = {r:g} {Bname}")
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel("phi / V")
    ax[0].legend(fontsize=7)
    ax[1].set_xlabel("t (ns)")
    ax[1].set_ylabel("electrons in the gap (per unit length)")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    return record(name, title, rows, fig, notes=notes)


def main(which=("planar", "coax")):
    if "planar" in which:
        res, prm = planar()
        _report("30_warpx_planar_crossed_field",
                "WarpX comparison E: planar smooth-bore magnetron gap (2D), below / above Hull cutoff",
                res, prm, "z (mm)", "B_H",
                f"V = 10 kV, d = 5 mm, Hull field B_H = {prm['BH'] * 1e3:.2f} mT (relativistic planar). Electrons "
                f"injected at J = 0.25 J_CL with 2 eV normal energy, periodic 8-cell strip along the cathode. "
                f"WarpX 26.09 2D (XZ, ES labframe, use_filter = 0) vs VEXPACE 2D-ZY, same dt, shape, pusher and "
                f"energy-conserving gather. Above B_H all electrons return to the cathode (twice the charge in "
                f"the gap, the potential depressed near the cathode). Compared after the start-up, t = 1-2.2 ns, while the "
                f"insulated hub is still laminar (later it develops the 2D diocotron/turbulent hub in both codes).")
    if "coax" in which:
        res, prm = coax()
        _report("31_warpx_coax_crossed_field",
                "WarpX comparison F: cylindrical smooth-bore magnetron gap (RZ), below / above Hull cutoff",
                res, prm, "r (mm)", "B*",
                f"Coaxial gap, inner cathode a = 5 mm, anode b = 10 mm, V = 10 kV, axial B. Hull field "
                f"B* = (m c / e d_e) sqrt(gamma^2 - 1), d_e = (b^2 - a^2)/(2b) (Lovelace & Ott 1974) = "
                f"{prm['Bs'] * 1e3:.2f} mT. Electrons injected radially at J = 0.25 J_CL(b - a), 2 eV. WarpX 26.09 RZ "
                f"(r_min = a, ES labframe) vs VEXPACE 2D-RZ (annular domain).")


if __name__ == "__main__":
    main(sys.argv[1:] or ("planar", "coax"))
