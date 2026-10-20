"""Code-to-code comparison with WarpX (https://github.com/BLAST-WarpX/warpx).

Set WARPX_BIN to the folder containing the WarpX executables (warpx.1d.*, warpx.rz.*), e.g.
    WARPX_BIN=~/src/warpx/build/bin python benchmarks/warpx_compare/compare_warpx.py
The same physical problems are run with VEL-PIC and WarpX:
    A  1D ES two-stream instability            (field-energy history)
    B  1D ES Pierce diode at J_CL               (potential profile)
    C  RZ ES space-charge beam expansion         (beam edge radius vs z)
    D  1D EM beam-driven plasma wakefield        (E_z line-out)
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from amrex_plotfile import read_plotfile  # noqa: E402
from common import C, WORK, hist, new_model, plt, quiet_diag, record, row, run  # noqa: E402

BIN = os.environ.get("WARPX_BIN", os.path.expanduser("~/ext/warpx/build/bin"))
WDIR = os.path.join(WORK, "warpx")


def exe(dim):
    c = sorted(glob.glob(os.path.join(BIN, f"warpx.{dim}.*")))
    c = [x for x in c if os.access(x, os.X_OK) and not x.endswith(".py")]
    return c[0] if c else None


def run_warpx(dim, name, text):
    d = os.path.join(WDIR, name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    with open(os.path.join(d, "inputs"), "w") as fh:
        fh.write(text)
    env = dict(os.environ, OMP_NUM_THREADS=str(os.cpu_count() or 1))
    r = subprocess.run([exe(dim), "inputs"], cwd=d, capture_output=True, text=True, env=env)
    open(os.path.join(d, "log.txt"), "w").write(r.stdout + r.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"WarpX {name} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return d


def last_plotfile(d, prefix="diags/diag1"):
    pf = sorted(glob.glob(os.path.join(d, prefix + "*")))
    pf = [p for p in pf if os.path.isdir(p)]
    return read_plotfile(pf[-1])


# ---------------------------------------------------------------------------------------------
def two_stream():
    n0, v0 = 1e14, 1e6
    wpb = np.sqrt(n0 / 2 * C.e ** 2 / (C.eps0 * C.me))
    k = np.sqrt(3) / 2 * wpb / v0
    L = 2 * np.pi / k
    dt = 0.02 / wpb
    nstep = int(round(40 / wpb / dt))
    wx = f"""
max_step = {nstep}
warpx.const_dt = {dt:.17g}
amr.n_cell = 64
amr.max_grid_size = 64
amr.blocking_factor = 8
amr.max_level = 0
geometry.dims = 1
geometry.prob_lo = 0.
geometry.prob_hi = {L:.17g}
boundary.field_lo = periodic
boundary.field_hi = periodic
boundary.particle_lo = periodic
boundary.particle_hi = periodic
warpx.do_electrostatic = labframe
algo.particle_shape = 1
particles.species_names = b1 b2 ions
my_constants.v0 = {v0:.17g}
my_constants.kk = {k:.17g}
b1.charge = -q_e
b1.mass = m_e
b1.injection_style = NUniformPerCell
b1.num_particles_per_cell_each_dim = 1000
b1.profile = constant
b1.density = {n0 / 2:.17g}
b1.momentum_distribution_type = parse_momentum_function
b1.momentum_function_ux(x,y,z) = "0"
b1.momentum_function_uy(x,y,z) = "0"
b1.momentum_function_uz(x,y,z) = "(v0 + 1e-5*v0*sin(kk*z))/clight"
b2.charge = -q_e
b2.mass = m_e
b2.injection_style = NUniformPerCell
b2.num_particles_per_cell_each_dim = 1000
b2.profile = constant
b2.density = {n0 / 2:.17g}
b2.momentum_distribution_type = parse_momentum_function
b2.momentum_function_ux(x,y,z) = "0"
b2.momentum_function_uy(x,y,z) = "0"
b2.momentum_function_uz(x,y,z) = "(-v0 - 1e-5*v0*sin(kk*z))/clight"
ions.charge = q_e
ions.mass = m_p
ions.injection_style = NUniformPerCell
ions.num_particles_per_cell_each_dim = 1
ions.profile = constant
ions.density = {n0:.17g}
ions.momentum_distribution_type = at_rest
ions.do_not_push = 1
warpx.reduced_diags_names = FE
FE.type = FieldEnergy
FE.intervals = 2
diagnostics.diags_names = diag1
diag1.intervals = 100000
diag1.diag_type = Full
diag1.fields_to_plot = Ez
"""
    d = run_warpx("1d", "two_stream", wx)
    fe = np.loadtxt(os.path.join(d, "diags", "reducedfiles", "FE.txt"))
    tw, Ww = fe[:, 1], fe[:, 2]
    # VEL-PIC
    m = new_model("1D-Z", "electrostatic")
    m["domain"].update(z_min="0", z_max=str(float(L)), nz="64")
    for s in ("z_min", "z_max"):
        m["boundaries"][s].update(field="periodic", particles="periodic")
    m["species"] = [{"name": "r", "preset": "electron"}, {"name": "l", "preset": "electron"}]
    com = dict(z0="0", z1=str(float(L)), y0="0", y1="1", density=str(float(n0 / 2)), ppc="1000", quiet_start=True,
               perturbation="velocity", pert_mode="1")
    m["loaders"] = [dict(name="r", species="r", drift=[str(float(v0)), "0", "0"], pert_amplitude=str(float(1e-5 * v0)), **com),
                    dict(name="l", species="l", drift=[str(float(-v0)), "0", "0"], pert_amplitude=str(float(-1e-5 * v0)), **com)]
    m["simulation"].update(n_steps=str(nstep), t_end="1", dt=str(float(dt)))
    quiet_diag(m, 2)
    sim = run(m, "wx_two_stream")
    tp, Wp = hist(sim, "W_field")
    # growth rates (sqrt W) in the linear phase
    from common import fit_growth
    gp, _ = fit_growth(tp, np.sqrt(Wp), 1e-3, 0.2)
    gw, _ = fit_growth(tw, np.sqrt(Ww), 1e-3, 0.2)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.semilogy(tw * wpb, Ww, "k-", lw=2, label="WarpX")
    ax.semilogy(tp * wpb, Wp, "r--", lw=1.5, label="VEL-PIC")
    ax.set_xlabel("t w_pb")
    ax.set_ylabel("field energy (J/m^2)")
    ax.legend()
    ax.set_title("Two-stream: WarpX vs VEL-PIC")
    sat_p, sat_w = Wp.max(), Ww.max()
    return record("16_warpx_two_stream", "WarpX comparison A: 1D ES two-stream instability",
                  [row("growth rate gamma/w_pb  (WarpX reference)", gp / wpb, gw / wpb, 0.03),
                   row("saturated field energy (WarpX reference)", sat_p, sat_w, 0.10)], fig)


def pierce_diode():
    V, d = 1000.0, 1e-3
    JCL = 4 * C.eps0 / 9 * np.sqrt(2 * C.e / C.me) * V ** 1.5 / d ** 2
    nz = 128
    vf = np.sqrt(2 * C.e * V / C.me)
    dt = 0.4 * (d / nz) / vf
    nstep = 3000
    wx = f"""
max_step = {nstep}
warpx.const_dt = {dt:.17g}
warpx.do_electrostatic = labframe
algo.particle_shape = 1
amr.n_cell = {nz}
amr.max_level = 0
geometry.dims = 1
geometry.prob_lo = 0.0
geometry.prob_hi = {d:.17g}
boundary.field_lo = pec
boundary.field_hi = pec
boundary.potential_lo_z = 0.0
boundary.potential_hi_z = {V:.17g}
boundary.particle_lo = absorbing
boundary.particle_hi = absorbing
particles.species_names = electrons
electrons.charge = -q_e
electrons.mass = m_e
electrons.do_continuous_injection = 1
electrons.injection_style = NFluxPerCell
electrons.num_particles_per_cell = 15
electrons.surface_flux_pos = 0
electrons.flux_normal_axis = z
electrons.flux_direction = 1
electrons.flux_profile = constant
electrons.flux = {JCL / C.e:.17g}
electrons.momentum_distribution_type = constant
diagnostics.diags_names = diag1
diag1.intervals = {nstep}
diag1.diag_type = Full
diag1.fields_to_plot = Ez rho phi
"""
    dd = run_warpx("1d", "pierce", wx)
    f, (zc,), _ = last_plotfile(dd)
    phiw = f["phi"]
    # VEL-PIC: beam emitter with zero energy at J_CL
    m = new_model("1D-Z", "electrostatic")
    m["domain"].update(z_min="0", z_max=str(float(d)), nz=str(nz))
    m["boundaries"]["z_max"]["voltage"] = str(float(V))
    m["species"] = [{"name": "e", "preset": "electron"}]
    m["emitters"] = [{"name": "k", "species": "e", "model": "beam", "z0": "0", "current_density": str(float(JCL)),
                      "energy": "0", "particles_per_cell": "15"}]
    m["simulation"].update(n_steps=str(nstep), t_end="1", dt=str(float(dt)))
    quiet_diag(m, 50)
    sim = run(m, "wx_pierce")
    zp, phip = sim.grid.z, sim.phi
    th = V * (zp / d) ** (4 / 3)
    phip_c = np.interp(zc, zp, phip)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot(zp * 1e3, th, "k-", lw=2, label="theory V (z/d)^{4/3}")
    ax.plot(zc * 1e3, phiw, "b--", label="WarpX")
    ax.plot(zp * 1e3, phip, "r:", lw=2, label="VEL-PIC")
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("phi (V)")
    ax.legend()
    ax.set_title("Pierce diode at J_CL: potential")
    return record("17_warpx_pierce", "WarpX comparison B: 1D ES Pierce diode injected at J_CL",
                  [row("max |phi_PIC - phi_WarpX| / V", np.abs(phip_c - phiw).max() / V, 0.0, 0.02, absolute=True),
                   row("max |phi_PIC - phi_theory| / V", np.abs(phip - th).max() / V, 0.0, 0.02, absolute=True)],
                  fig)


def rz_beam():
    Eb, I, a, Rt, Lz = 10e3, 0.05, 2e-3, 10e-3, 0.2
    gam = 1 + Eb * C.e / (C.me * C.c ** 2)
    ub = np.sqrt(gam ** 2 - 1)
    dt = 2e-12
    nstep = 5000
    wx = f"""
max_step = {nstep}
warpx.const_dt = {dt:.17g}
warpx.do_electrostatic = labframe
warpx.n_rz_azimuthal_modes = 1
algo.particle_shape = 1
amr.n_cell = 40 400
amr.max_grid_size = 400
amr.blocking_factor = 8
amr.max_level = 0
geometry.dims = RZ
geometry.prob_lo = 0.0 0.0
geometry.prob_hi = {Rt:.17g} {Lz:.17g}
boundary.field_lo = none pec
boundary.field_hi = pec pec
boundary.potential_lo_z = 0.0
boundary.potential_hi_z = 0.0
boundary.potential_hi_x = 0.0
boundary.particle_lo = none absorbing
boundary.particle_hi = absorbing absorbing
particles.species_names = beam
beam.charge = -q_e
beam.mass = m_e
beam.injection_style = NFluxPerCell
beam.num_particles_per_cell = 6
beam.surface_flux_pos = 1e-9
beam.flux_normal_axis = z
beam.flux_direction = 1
beam.xmin = 0.0
beam.xmax = {a:.17g}
beam.flux_profile = constant
beam.flux = {I / (C.e * np.pi * a ** 2):.17g}
beam.momentum_distribution_type = constant
beam.uz = {ub:.17g}
diagnostics.diags_names = diag1
diag1.intervals = {nstep}
diag1.diag_type = Full
diag1.fields_to_plot = rho
"""
    dd = run_warpx("rz", "rz_beam", wx)
    f, (rc, zc), _ = last_plotfile(dd)
    rho = f["rho"] if "rho" in f else f[[k for k in f if k.startswith("rho")][0]]
    if rho.ndim == 2 and rho.shape[0] != len(rc):
        rho = rho.T
    num = (rho * rc[:, None] ** 3).sum(axis=0)
    den = (rho * rc[:, None]).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        aw = np.sqrt(2 * num / den)
    # VEL-PIC
    import bench_beam
    m, p = bench_beam._beam_model(False)
    m["simulation"].update(n_steps=str(nstep), t_end="1", dt=str(float(dt)))
    m["emitters"][0]["particles_per_cell"] = "6"
    sim = run(m, "wx_rz_beam")
    zb = np.linspace(0, Lz, 41)
    zcc = 0.5 * (zb[1:] + zb[:-1])
    ap = bench_beam._edge_radius(sim, zb)
    env, K = bench_beam._envelope(p, lambda z: 0.0)
    awi = np.interp(zcc, zc, aw)
    ok = np.isfinite(ap) & np.isfinite(awi) & (zcc > 0.01) & (zcc < 0.19)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(zc * 100, aw * 1e3, "b-", label="WarpX (from rho)")
    ax.plot(zcc * 100, ap * 1e3, "ro", ms=3, label="VEL-PIC")
    zz = np.linspace(0, Lz, 200)
    ax.plot(zz * 100, env.sol(zz)[0] * 1e3, "k--", label="envelope equation")
    ax.set_xlabel("z (cm)")
    ax.set_ylabel("edge radius (mm)")
    ax.legend()
    ax.set_title("RZ beam expansion: WarpX vs VEL-PIC")
    return record("18_warpx_rz_beam", "WarpX comparison C: RZ ES space-charge beam expansion",
                  [row("max |a_PIC/a_WarpX - 1| over z", np.abs(ap[ok] / awi[ok] - 1).max(), 0.0, 0.05,
                       absolute=True)], fig)


def wakefield():
    np0, alpha = 1e22, 0.05
    wp = np.sqrt(np0 * C.e ** 2 / (C.eps0 * C.me))
    kp = wp / C.c
    lp = 2 * np.pi / kp
    Lb, Lx, nz = lp / 2, 6 * lp, 1800
    ub = np.sqrt((1 + 1e9 * C.e / (C.me * C.c ** 2)) ** 2 - 1)
    dz = Lx / nz
    dt = 0.98 * dz / C.c
    nstep = int(round(4.0 * lp / C.c / dt))
    wx = f"""
max_step = {nstep}
warpx.const_dt = {dt:.17g}
amr.n_cell = {nz}
amr.max_grid_size = {nz}
amr.blocking_factor = 8
amr.max_level = 0
geometry.dims = 1
geometry.prob_lo = 0.
geometry.prob_hi = {Lx:.17g}
boundary.field_lo = absorbing_silver_mueller
boundary.field_hi = absorbing_silver_mueller
boundary.particle_lo = absorbing
boundary.particle_hi = absorbing
algo.particle_shape = 1
algo.current_deposition = esirkepov
warpx.use_filter = 1
particles.species_names = plasma bunch
plasma.charge = -q_e
plasma.mass = m_e
plasma.injection_style = NUniformPerCell
plasma.num_particles_per_cell_each_dim = 32
plasma.profile = constant
plasma.density = {np0:.17g}
plasma.momentum_distribution_type = at_rest
bunch.charge = -q_e
bunch.mass = m_e
bunch.injection_style = NUniformPerCell
bunch.num_particles_per_cell_each_dim = 64
bunch.profile = constant
bunch.density = {alpha * np0:.17g}
bunch.zmin = {0.5 * lp:.17g}
bunch.zmax = {0.5 * lp + Lb:.17g}
bunch.momentum_distribution_type = constant
bunch.uz = {ub:.17g}
diagnostics.diags_names = diag1
diag1.intervals = {nstep}
diag1.diag_type = Full
diag1.fields_to_plot = Ez
"""
    dd = run_warpx("1d", "wakefield", wx)
    f, (zc,), tw = last_plotfile(dd)
    Ezw = f["Ez"]
    m = new_model("1D-Z", "electromagnetic")
    m["domain"].update(z_min="0", z_max=str(float(Lx)), nz=str(nz))
    for s in ("z_min", "z_max"):
        m["boundaries"][s].update(field="absorbing", absorber_cells=40)
    m["species"] = [{"name": "p", "preset": "electron"}, {"name": "b", "preset": "electron"}]
    m["loaders"] = [dict(name="p", species="p", z0="0", z1=str(float(Lx)), y0="0", y1="1", density=str(float(np0)), ppc="32",
                         quiet_start=True),
                    dict(name="b", species="b", z0=str(float(0.5 * lp)), z1=str(float(0.5 * lp + Lb)), y0="0", y1="1",
                         density=str(float(alpha * np0)), ppc="64", quiet_start=True, drift=[str(float(ub * C.c)), "0", "0"])]
    m["simulation"].update(n_steps=str(nstep), t_end="1", dt=str(float(dt)), current_filter=1, initial_field="none")  # = WarpX use_filter + EC gather
    quiet_diag(m, 100)
    sim = run(m, "wx_wake")
    zp, Ezp = sim.grid.z, sim.F[0]
    Eth = C.e * alpha * np0 / (C.eps0 * kp) * 2 * np.sin(kp * Lb / 2)
    zhead = 0.5 * lp + Lb + C.c * sim.t
    sel = (zc > zhead - 3 * lp) & (zc < zhead - Lb - 0.2 * lp)
    Ep = np.interp(zc, zp, Ezp)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot((zc - zhead) / lp, Ezw / Eth, "k-", lw=2, label="WarpX")
    ax.plot((zp - zhead) / lp, Ezp / Eth, "r--", label="VEL-PIC")
    ax.set_xlim(-3.5, 0.5)
    ax.set_xlabel("(z - z_head)/lambda_p")
    ax.set_ylabel("E_z / E_linear")
    ax.legend()
    ax.set_title("Beam-driven wake: WarpX vs VEL-PIC")
    diff = np.abs(Ep[sel] - Ezw[sel]).max() / Eth
    return record("19_warpx_wakefield", "WarpX comparison D: 1D EM beam-driven plasma wakefield",
                  [row("max |Ez_PIC - Ez_WarpX| / E_lin behind bunch", diff, 0.0, 0.08, absolute=True),
                   row("wake amplitude PIC / WarpX", np.abs(Ep[sel]).max(), np.abs(Ezw[sel]).max(), 0.05)], fig)


if __name__ == "__main__":
    if exe("1d") is None:
        print(f"WarpX executables not found in {BIN}; set WARPX_BIN")
        sys.exit(1)
    todo = sys.argv[1:] or ["two_stream", "pierce_diode", "rz_beam", "wakefield"]
    for n in todo:
        try:
            globals()[n]()
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
