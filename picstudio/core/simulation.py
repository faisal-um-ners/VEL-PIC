"""The PIC simulation engine: builds every component from a model and runs the loop.

Electrostatic cycle (per step)
    gather E(x^n) + applied fields -> Boris push -> particle boundaries ->
    emission -> charge deposit -> FEM Poisson solve (scikit-fem) -> E^{n+1}

Electromagnetic cycle (per step)
    gather E^n, B^n -> Boris push -> boundaries (clip) -> emission ->
    Esirkepov current deposit -> antennas -> B^{n+1/2} -> E^{n+1} -> B^{n+1}
"""
from __future__ import annotations

import math
import os
import time

import numpy as np

from .. import constants as C
from ..diagnostics.output import Output
from ..fields.external import ExternalFields
from ..fields.fdtd import Yee1D, Yee2D, binomial_filter
from ..fields.poisson import PoissonFEM, gradient_field
from ..mesh.grid import CONDUCTOR, DIELECTRIC, Grid, Materials, grid_from_model
from ..model.model import complete, evaluator
from ..particles import kernels as K
from ..particles.species import Species
from ..sources.emitters import Emitter
from ..sources.em_sources import WaveSource
from ..sources.loaders import load_particles
from .parallel import configure_threads, get_comm

PBC = {"absorb": 0, "reflect": 1, "periodic": 2}
COMP_NAMES = {"1D-Z": ("z", "y", "x"), "2D-ZY": ("z", "y", "x"), "2D-RZ": ("z", "r", "theta")}
GUARD = 2


def decimate_minmax(H, nbins=2000):
    """Reduce a (n, ncol) time series to about 2*nbins rows while keeping, for every column, the
    minimum and maximum inside each bin (so oscillation envelopes and spikes stay visible)."""
    n = len(H)
    if n <= 2 * nbins or H.ndim != 2:
        return H
    edges = np.linspace(0, n, nbins + 1).astype(int)
    lo = np.minimum.reduceat(H, edges[:-1], axis=0)
    hi = np.maximum.reduceat(H, edges[:-1], axis=0)
    first, last = H[edges[:-1]], H[edges[1:] - 1]
    out = np.empty((2 * nbins, H.shape[1]))
    out[0::2], out[1::2] = lo, hi
    out[0::2, :2], out[1::2, :2] = first[:, :2], last[:, :2]      # step, t: bin start / end
    return out


class Simulation:
    def __init__(self, model, out_dir=None, comm=None, log=None, write_output=True,
                 base_dir=None):
        self.model = m = complete(model)
        self.ev = ev = evaluator(m)
        self.log_fn = log
        self.comm = comm if comm is not None else get_comm(m["parallel"]["mode"])
        self.rank, self.size = self.comm.rank, self.comm.size
        self.nchunk = configure_threads(m["parallel"]["mode"], int(m["parallel"].get("threads", 0) or 0),
                                        self.comm)
        sim, dom = m["simulation"], m["domain"]
        self.geometry = sim["geometry"]
        self.em = sim["solver"] == "electromagnetic"
        self.rel = bool(sim["relativistic"])
        self.smooth = int(sim.get("smoothing", 0) or 0)
        self.j_filter = int(sim.get("current_filter", 0) or 0)
        g = self.grid = grid_from_model(m, ev, axis_correction=bool(sim.get("axis_correction", True)))
        if self.em and not g.uniform:
            raise ValueError("Mesh refinement is available for the electrostatic solver only "
                             "(the Yee FDTD solver needs a uniform grid). Disable the refinement regions.")
        bnd = m["boundaries"]
        if g.rz and g.on_axis:
            bnd["y_min"]["field"] = "axis"
        self.bnd = bnd
        self.mats = Materials(g, m["objects"], bnd, ev)
        self.pz = bnd["z_min"]["field"] == "periodic"
        self.py = g.dim == 2 and bnd["y_min"]["field"] == "periodic"
        self.hand = 1.0 if g.rz else -1.0
        self.comp = COMP_NAMES[self.geometry]
        # particle boundary codes
        pb = []
        for side in ("z_min", "z_max", "y_min", "y_max"):
            b = bnd[side]
            if b["field"] == "periodic":
                pb.append(2)
            elif b["field"] == "axis":
                pb.append(3)
            else:
                pb.append(PBC.get(b["particles"], 0))
        self.pbc = np.array(pb, dtype=np.int64)
        self.species = [Species(s, ev, i) for i, s in enumerate(m["species"])]
        self.sp_by_name = {s.name: s for s in self.species}
        self.vfuncs = [ev.func_t(cnd["voltage"]) for cnd in self.mats.conductors]
        self.v_td = any(ev.is_time_dependent(cnd["voltage"]) for cnd in self.mats.conductors)
        self.log(g.info())
        t0 = time.time()
        self.local_es = None
        if hasattr(g, "local_mesh"):
            from ..fields.local_es import LocalElectrostatics
            self.local_es = LocalElectrostatics(g, self.mats, m, ev)
            self.poisson = self.local_es
            self.log(f"Fixed local mesh: {len(g.local_mesh.points):,} nodes, "
                     f"{len(g.local_mesh.triangles):,} P1 triangles")
        else:
            self.poisson = PoissonFEM(g, self.mats, self.pz, self.py)
        if self.local_es is not None:
            self.log(f"Local P1 Poisson operator assembled and factorized "
                     f"({len(g.local_mesh.points):,} nodes, {len(self.mats.conductors)} conductors) "
                     f"in {time.time() - t0:.2f} s")
        else:
            self.log(f"scikit-fem Poisson operator assembled and factorized ({g.nnodes} nodes, "
                     f"{len(self.mats.conductors)} conductors) in {time.time() - t0:.2f} s")
        self.ext = ExternalFields(m["external_fields"], g, ev)
        self.has_obj = bool((self.mats.cell_obj >= 0).any())
        self.obj_names = [o.get("name", f"obj{i}") for i, o in enumerate(m["objects"])]
        # objects that collect particles (vacuum / lossy regions never do -> no always-zero columns)
        self.obj_cur = [i for i, o in enumerate(m["objects"])
                        if o.get("material", "conductor") in ("conductor", "dielectric", "absorber")]
        self.obj_mat = [o.get("material", "conductor") for o in m["objects"]]
        self.obj_collect = [bool(o.get("collect_charge", True)) for o in m["objects"]]
        self.rng = np.random.default_rng(int(sim.get("seed", 12345)) + 7919 * self.rank)
        # time step -------------------------------------------------------------
        self.t_end = ev.num(sim["t_end"])
        self.dt = self._choose_dt()
        ns = sim.get("n_steps", "")
        self.n_steps = int(ev.num(ns)) if str(ns).strip() not in ("", "None") else \
            int(math.ceil(self.t_end / self.dt - 1e-9))
        self.t = 0.0
        self.step_count = 0
        # particles -----------------------------------------------------------------
        for ld in m["loaders"]:
            sp = self.sp_by_name[ld["species"]]
            if self.local_es is not None:
                from ..fields.local_es import load_local_particles
                n = load_local_particles(ld, sp, self.local_es, ev, self.rng, self.rank, self.size)
            else:
                n = load_particles(ld, sp, g, self.mats, ev, self.rng, self.rank, self.size)
            self.log(f"Loaded {self.comm.allsum(n)} macro-particles of '{sp.name}' ({ld['name']})")
        if self.local_es is not None:
            from ..fields.local_es import LocalEmitter
            self.emitters = [LocalEmitter(self.local_es, e, self.sp_by_name[e["species"]],
                              g, self.mats, ev, sim["solver"], objects=m["objects"])
                             for e in m["emitters"]]
        else:
            self.emitters = [Emitter(e, self.sp_by_name[e["species"]], g, self.mats, ev, sim["solver"],
                                     objects=m["objects"]) for e in m["emitters"]]
        for e in self.emitters:
            self.log(f"Emitter '{e.name}' ({e.model}): {e.npanel} panels"
                     f"{', reaction surface field' if e.react_idx and not self.em else ''}")
            if e.model == "scl" and not self.em and not e.react_idx:
                self.log(f"  WARNING: emitter '{e.name}' is not aligned with conductor nodes; the Gauss-law "
                         f"emission falls back to a finite-difference surface field (less stable). Use an "
                         f"axis-aligned segment on a conductor or 'surface = object'.")
        # fields --------------------------------------------------------------------
        self.F = np.zeros((6,) + g.shape)
        self.F_gather = np.zeros_like(self.F)      # fields gathered at nodes (energy-conserving mode)
        self.pusher = {"boris": 0, "vay": 1, "higuera": 2}.get(str(sim.get("particle_pusher", "boris")), 0)
        fg = str(sim.get("field_gathering", "auto"))
        self.gather_ec = (fg == "energy-conserving") or (fg == "auto" and self.em)
        if self.local_es is not None and fg == "auto":
            self.gather_ec = True
            self.log("Local P1 mesh: auto gathers E directly from the triangle potential gradient.")
        if self.gather_ec and self.em and self.smooth:
            self.gather_ec = False
            self.log("  note: smoothing of the gathered fields requires momentum-conserving gathering; "
                     "using it for this run")
        self.ic2 = K.INV_C2 if self.rel else 0.0
        self.phi = np.zeros(g.shape)
        self.rho = np.zeros(g.shape)
        self.static_rhs = np.zeros(g.shape)       # charge stuck on dielectrics (ES)
        self.Bself = None
        self.vol_density = g.node_volume.copy()
        if self.pz:
            self.vol_density[0] *= 2
            self.vol_density[-1] *= 2
        if self.py:
            self.vol_density[:, 0] *= 2
            self.vol_density[:, -1] *= 2
        if self.em:
            self.yee = Yee1D(g, self.mats, bnd) if g.dim == 1 else Yee2D(g, self.mats, bnd)
            self.axis_factor = 3.0 if (g.rz and g.axis_correction) else 4.0
            self.sources = [WaveSource(s, g, self.yee, ev) for s in m["sources"]]
            self.static_E = None
            self._j_dirty = True
            self._deposited = False
            self._smooth_mask = (self.mats.node_cond < 0).astype(float)
            self._per_axes = tuple(a for a, p in ((0, self.pz), (1, self.py)) if p)
            self.circuit = None
            if str(sim.get("em_conductors", "held")).startswith("held") and self.mats.conductors:
                from ..fields.circuit import ConductorCircuit
                self.circuit = ConductorCircuit(self.yee, g, self.mats, self.pz, self.py)
                if not self.circuit.active:
                    self.circuit = None
        else:
            self.circuit = None
        # diagnostics config -----------------------------------------------------------
        d = m["diagnostics"]
        self.h_every = max(1, int(ev.num(d["history_every"])))
        self.f_every = max(1, int(ev.num(d["field_every"])))
        self.p_every = max(1, int(ev.num(d["particle_every"])))
        self.live_every = max(1, int(ev.num(d["live_every"])))
        self.max_dump = int(ev.num(d["max_dump_particles"]))
        try:
            self.max_particles = float(ev.num(sim.get("max_particles", "2e7") or 0))
        except Exception:  # noqa: BLE001
            self.max_particles = 0.0
        self.probes = [(p["name"], p["quantity"], ev.num(p["z"]), ev.num(p["y"])) for p in m["probes"]]
        # line probes: (name, qty, along, z0, y0, z1, y1, record_min, record_max)
        self.line_probes = []
        for p in m.get("line_probes", []):
            along = p.get("along", "z")
            a, b = ev.num(p.get("start", "0")), ev.num(p.get("end", "0"))
            if along == "y" and g.dim == 2:
                zc = ev.num(p.get("z", "0"))
                z0, y0, z1, y1 = zc, a, zc, b
            else:
                along = "z"
                yc = ev.num(p.get("y", "0")) if g.dim == 2 else 0.0
                z0, y0, z1, y1 = a, yc, b, yc
            ext = p.get("extremum", "min")
            self.line_probes.append((p["name"], p["quantity"], along, z0, y0, z1, y1,
                                     ext in ("min", "min and max"), ext in ("max", "min and max")))
        self.plines = m["power_lines"]
        if out_dir is None:
            base = base_dir or (os.path.dirname(m["_path"]) if m.get("_path") else os.getcwd())
            out_dir = os.path.join(base, d["output_dir"] or "output")
        self.out_dir = out_dir
        self.out = Output(out_dir, enabled=write_output and self.rank == 0)
        self.coll_side = np.zeros(5)
        self.coll_obj = np.zeros(max(1, len(m["objects"])))
        self.emit_acc = np.zeros(max(1, len(self.emitters)))
        self.hist_t0 = 0.0
        self.history_rows = []
        self.timers = {"push": 0.0, "deposit": 0.0, "field": 0.0, "emit": 0.0, "diag": 0.0}
        self._initial_fields()

    # ==========================================================================
    def log(self, msg):
        if self.rank != 0:
            return
        if self.log_fn:
            self.log_fn(msg)
        else:
            print(msg, flush=True)

    def _kern(self, kernel, n):
        if self.nchunk == 1 or n < K.SERIAL_THRESHOLD:
            return kernel.serial, 1
        return kernel, self.nchunk

    def voltages(self, t):
        return [f(t) for f in self.vfuncs]

    def _choose_dt(self):
        m, ev, g = self.model, self.ev, self.grid
        sim = m["simulation"]
        cfl = ev.num(sim.get("cfl", 0.95))
        if str(sim.get("dt", "auto")).strip().lower() not in ("auto", ""):
            return ev.num(sim["dt"])
        h = g.hmin
        if self.em:
            if g.dim == 1:
                return cfl * g.dz / C.c
            if g.rz:
                # cylindrical Yee (m = 0) stability limit, as WarpX CylindricalYeeAlgorithm::ComputeMaxDt
                # (R. Lehe: the on-axis update tightens the radial term by the factor 1.2105)
                return cfl / (C.c * math.sqrt(1.2105 / g.dy ** 2 + 1 / g.dz ** 2))
            return cfl / (C.c * math.sqrt(1 / g.dz ** 2 + 1 / g.dy ** 2))
        vmax = 0.0
        ts = np.linspace(0, self.t_end, 7)
        Vmax = max([0.0] + [abs(f(t)) for f in self.vfuncs for t in ts])
        for sp in self.species:
            gam = 1 + abs(sp.q) * Vmax / (sp.m * C.c ** 2)
            vmax = max(vmax, C.c * math.sqrt(1 - 1 / gam ** 2))
        wp2 = 0.0
        for ld in m["loaders"]:
            sp = self.sp_by_name.get(ld["species"])
            if sp is None:
                continue
            zz = np.linspace(ev.num(ld["z0"]), ev.num(ld["z1"]), 11)
            yy = np.linspace(ev.num(ld["y0"]), ev.num(ld["y1"]), 11) if g.dim == 2 else np.zeros(1)
            Z, Y = np.meshgrid(zz, yy)
            n = np.max(np.abs(np.broadcast_to(ev.value(ld["density"], z=Z, y=Y, r=Y, t=0.0), Z.shape)))
            wp2 += n * sp.q ** 2 / (C.eps0 * sp.m)
            T = ev.num(ld.get("temperature", 0))
            dr = np.abs([ev.num(v) for v in ld.get("drift", [0, 0, 0])]).max()
            vmax = max(vmax, dr + 4 * math.sqrt(max(T, 0) * C.e / sp.m))
        for em in m["emitters"]:
            sp = self.sp_by_name.get(em["species"])
            if sp is None:
                continue
            E = ev.num(em.get("energy", 0)) + 4 * ev.num(em.get("temperature", 0))
            gam = 1 + E * C.e / (sp.m * C.c ** 2)
            vmax = max(vmax, C.c * math.sqrt(1 - 1 / gam ** 2))
        cands = []
        if vmax > 0:
            cands.append(0.5 * cfl * h / vmax)
        if wp2 > 0:
            cands.append(0.2 / math.sqrt(wp2))
        Bm = self.ext.max_B()
        gyro = [sp for sp in self.species if not (sp.z_only or sp.y_only)]   # infinite B: no gyration
        if Bm > 0 and gyro:
            wc = max(abs(sp.q) * Bm / sp.m for sp in gyro)
            cands.append(0.25 / wc)                 # Boris push resolves the cyclotron motion (w_c dt <= 0.25)
        if not cands:
            return self.t_end / 1000
        return min(cands)

    # ==========================================================================
    def _deposit_charge(self):
        g = self.grid
        Q = np.zeros(g.shape)
        per = {}
        for sp in self.species:
            if sp.n == 0:
                q = np.zeros(g.shape)
            elif not g.uniform:            # refined mesh: bilinear weights in logical coordinates
                zl, yl = self._logical(sp._z[:sp.n], sp._y[:sp.n])
                if g.dim == 1:
                    kd, nc = self._kern(K.deposit_1d, sp.n)
                    q = kd(zl, sp._w, sp.n, sp.q, 0.0, 1.0, g.nz, nc)
                else:
                    kd, nc = self._kern(K.deposit_2d, sp.n)
                    q = kd(zl, yl, sp._w, sp.n, sp.q, 0.0, 0.0, 1.0, 1.0, g.nz, g.ny, nc)
            elif g.dim == 1:
                kd, nc = self._kern(K.deposit_1d, sp.n)
                q = kd(sp._z, sp._w, sp.n, sp.q, g.z_min, g.dz, g.nz, nc)
            else:
                kd, nc = self._kern(K.deposit_2d, sp.n)
                q = kd(sp._z, sp._y, sp._w, sp.n, sp.q, g.z_min, g.y_min, g.dz, g.dy,
                       g.nz, g.ny, nc)
            per[sp.name] = q
            Q += q
        if self.size > 1:
            Q = self.comm.allreduce(Q)
        return Q, per

    def _logical(self, z, y=None):
        g = self.grid
        zl = np.ascontiguousarray(g.lz(z))
        yl = np.ascontiguousarray(g.ly(y)) if (g.dim == 2 and y is not None) else zl
        return zl, yl

    def density(self, Q):
        """Charge density from raw node charges (periodic duplicates folded)."""
        Qf = self._fold_periodic(Q.copy())
        return Qf / self.vol_density

    def _fold_periodic(self, Q):
        if self.pz:
            if Q.ndim == 1:
                Q[0] += Q[-1]
                Q[-1] = Q[0]
            else:
                Q[0, :] += Q[-1, :]
                Q[-1, :] = Q[0, :]
        if self.py:
            Q[:, 0] += Q[:, -1]
            Q[:, -1] = Q[:, 0]
        return Q

    def _es_solve(self, t):
        if self.local_es is not None:
            self.local_es.update(self, t)
            return
        g = self.grid
        Q, per = self._deposit_charge()
        Q = Q + self.static_rhs
        if self.smooth:
            Q = binomial_filter(Q, self.smooth)
        self.rho = self.density(Q)
        rhs = Q / (2 * np.pi) if g.rz else Q
        self.rhs_fem = rhs
        self.phi = self.poisson.solve(rhs, self.voltages(t))
        E = gradient_field(self.phi, g, self.pz, self.py)
        self.F[:] = 0.0
        for c_, Ec in enumerate(E):
            self.F[c_] = Ec
        # relativistic beam self magnetic field
        for sp in self.species:
            if not sp.beam_self_B or g.dim == 1:
                continue
            qs = per[sp.name]
            if self.size > 1:
                qs = self.comm.allreduce(qs)
            ws = self.comm.allsum(float(sp.w.sum())) if sp.n or self.size > 1 else 0.0
            if ws == 0:
                continue
            gam = sp.gamma(self.rel)
            beta = self.comm.allsum(float((sp.w * sp.u0 / gam).sum())) / ws / C.c
            rhs_s = qs / (2 * np.pi) if g.rz else qs
            phis = self.poisson.solve(rhs_s, None)
            Es = gradient_field(phis, g, self.pz, self.py)
            if g.rz:
                self.F[5] += beta * Es[1] / C.c        # B_theta = beta E_r / c
            else:
                self.F[5] += -beta * Es[1] / C.c       # B_x = -beta E_y / c
        ext = self.ext.nodal(t)
        if ext is not None:
            self.F += ext
        if self.gather_ec:
            self.F_gather[:] = self.F
            for c_, Ec in enumerate(E):
                self.F_gather[c_] -= Ec

    def _em_fields(self, t):
        g = self.grid
        self.yee.nodal(self.F)
        if self.smooth:
            # Smoothing is applied to the fields seen by the particles (not to J), which keeps
            # the current deposition exactly charge conserving.  Conductor nodes are excluded.
            m = self._smooth_mask
            den = binomial_filter(m, self.smooth, periodic=self._per_axes)
            den = np.where(den > 0, den, 1.0)
            for c_ in range(6):
                self.F[c_] = binomial_filter(self.F[c_] * m, self.smooth, periodic=self._per_axes) / den * m \
                    + self.F[c_] * (1 - m)
        Fy = self.F[:2].copy() if self.emitters else None
        F_yee = self.F.copy() if self.gather_ec else None
        if self.vfuncs:
            V = self.voltages(t)
            if any(v != 0 for v in V):
                if self.static_E is None or self.v_td:
                    phiL = self.poisson.laplace_field(V)
                    self.phi = phiL
                    self.static_E = gradient_field(phiL, g, self.pz, self.py)
                for c_, Ec in enumerate(self.static_E):
                    self.F[c_] += Ec
        ext = self.ext.nodal(t)
        if ext is not None:
            self.F += ext
        self.F_static = (self.F[:2] - Fy) if Fy is not None else None
        if F_yee is not None:
            np.subtract(self.F, F_yee, out=self.F_gather)    # static electrode + applied fields only

    def _initial_fields(self):
        g = self.grid
        if not self.em:
            self._es_solve(0.0)
        else:
            Q, _ = self._deposit_charge()
            if np.abs(Q).max() > 0 and self.model["simulation"].get("initial_field", "poisson") == "poisson":
                if self.circuit is not None and self.circuit.active:
                    phi0 = self.circuit.grounded_potential(Q)     # exact discrete Gauss law on the Yee grid
                else:
                    rhs = Q / (2 * np.pi) if g.rz else Q
                    phi0 = self.poisson.solve(rhs, None)
                self.yee.set_from_potential(phi0)
            self.rho = self.density(Q)
            self._em_fields(0.0)
        # leapfrog: push the momenta back by half a step with the full Lorentz force
        # (as WarpX does at initialisation, PushP(-dt/2))
        for sp in self.species:
            if sp.n == 0:
                continue
            P = self._gather(sp)
            z, y, u0, u1, u2, w = sp.arrays()
            if g.dim == 1:
                K.push_fields_1d(z, u0, u1, u2, sp.n, P, sp.qm, -0.5 * self.dt, self.hand, self.ic2,
                                 self.pusher, False)
            else:
                K.push_fields_2d(z, y, u0, u1, u2, sp.n, P, sp.qm, -0.5 * self.dt, self.hand, g.rz,
                                 self.ic2, self.pusher, False)
            if sp.z_only:
                u1[:] = 0.0
                u2[:] = 0.0
            elif sp.y_only:
                u0[:] = 0.0
                u2[:] = 0.0

    def _gather(self, sp):
        """Fields at the particles (6, n): nodal part + (energy-conserving) self field."""
        g = self.grid
        n = sp.n
        if self.local_es is not None:
            return self.local_es.gather(self, sp)
        zl = np.ascontiguousarray(g.lz(sp._z[:n]))
        yl = np.ascontiguousarray(g.ly(sp._y[:n])) if g.dim == 2 else zl
        P = np.empty((6, n))
        Fg = self.F_gather if self.gather_ec else self.F
        if g.dim == 1:
            K.gather_nodal_1d(Fg, zl, n, g.nz, P)
        else:
            K.gather_nodal_2d(Fg, zl, yl, n, g.nz, g.ny, P)
        if self.gather_ec:
            if not self.em:
                if g.dim == 1:
                    K.gather_ec_es_1d(self.phi, zl, g.hz, n, g.nz, P)
                else:
                    K.gather_ec_es_2d(self.phi, zl, yl, g.hz, g.hy, n, g.nz, g.ny, P)
            else:
                y_ = self.yee
                if g.dim == 1:
                    K.gather_ec_yee_1d(y_.E0, y_.E1, y_.E2, y_.B1, y_.B2, zl, n, g.nz, P)
                else:
                    K.gather_ec_yee_2d(y_.E0, y_.E1, y_.E2, y_.B0, y_.B1, y_.B2, zl, yl, n, g.nz, g.ny, P)
        return P

    def _interp(self, A, z, y):
        if self.local_es is not None:
            return self.local_es.interp(self, A, z, y)
        g = self.grid
        if not g.uniform:
            return g.interp(A, z, y)
        if g.dim == 1:
            return K.interp_1d(A, np.ascontiguousarray(z), g.z_min, g.dz, g.nz)
        return K.interp_2d(A, np.ascontiguousarray(z), np.ascontiguousarray(y), g.z_min, g.y_min,
                           g.dz, g.dy, g.nz, g.ny)

    # ==========================================================================
    def _push(self, sp):
        g = self.grid
        z, y, u0, u1, u2, w = sp.arrays()
        if sp.z_only:                 # infinite axial B: no transverse motion
            u1[:] = 0.0
            u2[:] = 0.0
            y_old = y.copy() if g.dim == 2 else None
        elif sp.y_only:               # infinite B along y: motion along y only
            u0[:] = 0.0
            u2[:] = 0.0
            z_old = z.copy()
        P = self._gather(sp)
        if g.dim == 1:
            K.push_fields_1d(z, u0, u1, u2, sp.n, P, sp.qm, self.dt, self.hand, self.ic2, self.pusher, True)
        else:
            K.push_fields_2d(z, y, u0, u1, u2, sp.n, P, sp.qm, self.dt, self.hand, g.rz, self.ic2,
                             self.pusher, True)
        if sp.z_only:
            u1[:] = 0.0
            u2[:] = 0.0
            if y_old is not None:
                y[:] = y_old
        elif sp.y_only:
            u0[:] = 0.0
            u2[:] = 0.0
            z[:] = z_old

    def _boundaries(self, sp, wrap=True):
        g = self.grid
        z, y, u0, u1, u2, w = sp.arrays()
        flag = np.zeros(sp.n, dtype=np.int32)
        if self.local_es is not None:
            # Reuse the existing physical wall/periodic rules. Material hits
            # are looked up on the actual triangles, not the display grid.
            K.boundary_2d(z, y, u0, u1, sp.n, g.z_min, g.z_max, g.y_min, g.y_max,
                          self.pbc, wrap, self.mats.cell_obj, g.dz, g.dy, g.nz, g.ny, False, flag)
            return self.local_es.boundary_objects(z[:sp.n], y[:sp.n], flag)
        if not g.uniform:            # refined mesh: apply the walls in logical coordinates
            n = sp.n
            zl, yl = self._logical(z[:n], y[:n])
            zl, yl = zl.copy(), yl.copy()
            if g.dim == 1:
                K.boundary_1d(zl, u0, n, 0.0, float(g.nz), self.pbc[0], self.pbc[1], wrap, flag)
                if self.has_obj:
                    ic = np.floor(zl).astype(np.int64)
                    ok = (flag == 0) & (ic >= 0) & (ic < g.nz)
                    obj = np.full(n, -1)
                    obj[ok] = self.mats.cell_obj[ic[ok]]
                    hit = obj >= 0
                    flag[hit] = 10 + obj[hit]
            else:
                K.boundary_2d(zl, yl, u0, u1, n, 0.0, float(g.nz), 0.0, float(g.ny), self.pbc, wrap,
                              self.mats.cell_obj, 1.0, 1.0, g.nz, g.ny, self.has_obj, flag)
                y[:n] = g.py(yl)
            z[:n] = g.pz(zl)
            return flag
        if g.dim == 1:
            K.boundary_1d(z, u0, sp.n, g.z_min, g.z_max, self.pbc[0], self.pbc[1], wrap, flag)
            if self.has_obj:
                ic = ((sp.z - g.z_min) / g.dz).astype(np.int64)
                ok = (flag == 0) & (ic >= 0) & (ic < g.nz)
                obj = np.full(sp.n, -1)
                obj[ok] = self.mats.cell_obj[ic[ok]]
                hit = obj >= 0
                flag[hit] = 10 + obj[hit]
        else:
            K.boundary_2d(z, y, u0, u1, sp.n, g.z_min, g.z_max, g.y_min, g.y_max, self.pbc, wrap,
                          self.mats.cell_obj, g.dz, g.dy, g.nz, g.ny, self.has_obj, flag)
        return flag

    def _account(self, sp, flag):
        lost = flag != 0
        if not lost.any():
            return
        f = flag[lost]
        qw = sp.q * sp.w[lost]
        side = f < 10
        if side.any():
            self.coll_side += np.bincount(f[side], weights=qw[side], minlength=5)[:5]
        obj = ~side
        if obj.any():
            k = f[obj] - 10
            self.coll_obj += np.bincount(k, weights=qw[obj], minlength=len(self.coll_obj))[:len(self.coll_obj)]
            if not self.em:
                # charge stays on dielectric surfaces
                for oi in np.unique(k):
                    if self.obj_mat[oi] == "dielectric" and self.obj_collect[oi]:
                        sel = np.nonzero(lost)[0][obj][k == oi]
                        self._stick_charge(sp, sel)

    def _stick_charge(self, sp, idx):
        if self.local_es is not None:
            q = self.local_es.mesh.deposit(sp.z[idx], sp.y[idx], sp.q*sp.w[idx])
            if self.size > 1:
                q = self.comm.allreduce(q)
            self.local_es.stuck += q
            return
        g = self.grid
        z = np.ascontiguousarray(sp.z[idx])
        w = np.ascontiguousarray(sp.w[idx])
        if g.dim == 1:
            if not g.uniform:
                q = K.deposit_1d.serial(np.ascontiguousarray(g.lz(z)), w, len(z), sp.q, 0.0, 1.0, g.nz, 1)
            else:
                q = K.deposit_1d.serial(z, w, len(z), sp.q, g.z_min, g.dz, g.nz, 1)
        else:
            y = np.ascontiguousarray(sp.y[idx])
            if not g.uniform:
                q = K.deposit_2d.serial(np.ascontiguousarray(g.lz(z)), np.ascontiguousarray(g.ly(y)), w,
                                        len(z), sp.q, 0.0, 0.0, 1.0, 1.0, g.nz, g.ny, 1)
            else:
                q = K.deposit_2d.serial(z, y, w, len(z), sp.q, g.z_min, g.y_min, g.dz, g.dy, g.nz, g.ny, 1)
        if self.size > 1:
            q = self.comm.allreduce(q)
        self.static_rhs += q

    def _emit(self, t_emit):
        new = []
        efield = None
        reaction = None
        yee = static = qn = None
        if self.em:
            efield = self.F[:2] if self.grid.dim == 2 else self.F[:1]
            yee = self.yee
            static = self.F_static
            if any(em.model == "scl" for em in self.emitters):
                # free charge on the nodes (folded across periodic seams) for the surface field
                qn, _ = self._deposit_charge()
                qn = qn.copy()
                if self.pz:
                    qn[0] += qn[-1]
                    qn[-1] = qn[0]
                if self.py and self.grid.dim == 2:
                    qn[:, 0] += qn[:, -1]
                    qn[:, -1] = qn[:, 0]
        elif self.emitters and self.local_es is None:
            reaction = self.poisson.reaction(self.phi, self.rhs_fem)
        for k, em in enumerate(self.emitters):
            out = em.emit(t_emit, self.dt, self.rng, self.step_count, self.rank, self.size,
                          phi=None if self.em else self.phi, efield=efield, reaction=reaction,
                          yee=yee, static_E=static, qnodes=qn)
            if out is None:
                continue
            self.emit_acc[k] += em.sp.q * out["w"].sum()
            new.append((em.sp, out))
        return new

    # ==========================================================================
    def step(self):
        g = self.grid
        dt = self.dt
        t = self.t
        tm = time.perf_counter()
        if not self.em:
            for sp in self.species:
                if sp.n:
                    self._push(sp)
            self.timers["push"] += time.perf_counter() - tm
            for sp in self.species:
                if sp.n:
                    flag = self._boundaries(sp, wrap=True)
                    self._account(sp, flag)
                    sp.keep(flag == 0)
            te = time.perf_counter()
            for sp, out in self._emit(t):
                sp.add(out["z"], out["y"], out["u0"], out["u1"], out["u2"], out["w"])
            self.timers["emit"] += time.perf_counter() - te
            tf = time.perf_counter()
            self._es_solve(t + dt)
            self.timers["field"] += time.perf_counter() - tf
        else:
            self._em_step(t, dt)
        self.t = t + dt
        self.step_count += 1

    def _em_step(self, t, dt):
        g = self.grid
        yee = self.yee
        tm = time.perf_counter()
        old = {}
        for sp in self.species:
            if sp.n:
                old[sp.name] = (sp.z.copy(), sp.y.copy())
                self._push(sp)
        self.timers["push"] += time.perf_counter() - tm
        td = time.perf_counter()
        if g.dim == 1:
            NZ = g.nz + 1 + 2 * GUARD
            Fz = np.zeros(NZ)
            Q1 = np.zeros(NZ)
            Q2 = np.zeros(NZ)
        else:
            NZ, NY = g.nz + 1 + 2 * GUARD, g.ny + 1 + 2 * GUARD
            Fz = np.zeros((NZ, NY))
            Fy = np.zeros((NZ, NY))
            Q2 = np.zeros((NZ, NY))
        flags = {}
        self._deposited = False

        def dep(zo, yo, zn, yn, u0, u1, u2, w, n, q):
            if n == 0:
                return
            self._deposited = True
            if g.dim == 1:
                ke, nc = self._kern(K.esirkepov_1d, n)
                a, b, c_ = ke(zo, zn, u0, u1, u2, w, n, q, g.z_min, g.dz, NZ, GUARD, nc, self.rel)
                Fz[:] += a
                Q1[:] += b
                Q2[:] += c_
            else:
                ke, nc = self._kern(K.esirkepov_2d, n)
                a, b, c_ = ke(zo, yo, zn, yn, u0, u1, u2, w, n, q, g.z_min, g.y_min, g.dz,
                              g.dy, NZ, NY, GUARD, nc, self.rel)
                Fz[:] += a
                Fy[:] += b
                Q2[:] += c_

        for sp in self.species:
            if not sp.n:
                continue
            zo, yo = old[sp.name]
            flag = self._boundaries(sp, wrap=False)
            flags[sp.name] = flag
            z, y, u0, u1, u2, w = sp.arrays()
            if (flag != 0).any() and ((flag >= 1) & (flag <= 4)).any():
                if g.dim == 1:
                    K.clip_to_box_1d(zo, z, flag, sp.n, g.z_min, g.z_max)
                else:
                    K.clip_to_box_2d(zo, yo, z, y, flag, sp.n, g.z_min, g.z_max, g.y_min, g.y_max)
            dep(zo, yo, z, y, u0, u1, u2, w, sp.n, sp.q)
        self.timers["deposit"] += time.perf_counter() - td
        # bookkeeping: remove absorbed, wrap periodic
        for sp in self.species:
            if sp.name in flags:
                flag = flags[sp.name]
                self._account(sp, flag)
                sp.keep(flag == 0)
                if self.pz:
                    self._kern(K.wrap_periodic, sp.n)[0](sp._z, sp.n, g.z_min, g.z_max)
                if self.py:
                    self._kern(K.wrap_periodic, sp.n)[0](sp._y, sp.n, g.y_min, g.y_max)
        # emission (deposit their current from the birth point)
        te = time.perf_counter()
        for sp, out in self._emit(t):
            n = len(out["w"])
            yb = out["y_birth"] if g.dim == 2 else out["z_birth"] * 0
            yy = out["y"] if g.dim == 2 else out["z"] * 0
            dep(out["z_birth"], yb, out["z"], yy, out["u0"], out["u1"], out["u2"], out["w"], n, sp.q)
            sp.add(out["z"], out["y"], out["u0"], out["u1"], out["u2"], out["w"])
        self.timers["emit"] += time.perf_counter() - te
        if self.size > 1:
            Fz = self.comm.allreduce(Fz)
            Q2 = self.comm.allreduce(Q2)
            if g.dim == 1:
                Q1 = self.comm.allreduce(Q1)
            else:
                Fy = self.comm.allreduce(Fy)
        tf = time.perf_counter()
        if self._j_dirty or self._deposited:
            yee.zero_J()
            self._j_dirty = False
        if self._deposited:
            if g.dim == 1:
                yee.set_J(Fz, Q1, Q2, GUARD, dt)
            else:
                yee.set_J(Fz, Fy, Q2, GUARD, dt)
            self._j_dirty = True
        if self.sources:
            self._j_dirty = True
        for s in self.sources:
            s.apply(yee, t + 0.5 * dt)
        if self.j_filter and self._j_dirty:
            from ..fields.fdtd import filter_current
            filter_current(yee, self.j_filter, (g.nz,) if g.dim == 1 else (g.nz, g.ny),
                           (self.pbc[0] == 2,) if g.dim == 1 else (self.pbc[0] == 2, self.pbc[2] == 2))
        yee.half_B(dt)
        if g.dim == 1:
            yee.full_E(dt)
        else:
            yee.full_E(dt, self.axis_factor)
        yee.half_B(dt)
        if self.circuit is not None:
            self.circuit.apply()          # ideal voltage sources: conductors held at their potentials
        self._em_fields(t + dt)
        self.timers["field"] += time.perf_counter() - tf

    # ==========================================================================
    # diagnostics
    # ==========================================================================
    def history_columns(self):
        cols = ["step", "t"]
        for sp in self.species:
            cols += [f"N_{sp.name}", f"KE_{sp.name}"]
        cols += ["W_field"]
        cols += [f"I_{(self.bnd[s].get('name') or '').strip() or s}".replace(" ", "_")
                 for s in self._current_sides()]
        cols += [f"I_{self.obj_names[i]}".replace(" ", "_") for i in self.obj_cur]
        cols += [f"Iemit_{e.name}".replace(" ", "_") for e in self.emitters]
        cols += [f"V_{c_['name']}".replace(" ", "_") for c_ in self.mats.conductors]
        cols += [f"probe_{p[0]}" for p in self.probes]
        for p in self.line_probes:
            cols += self._line_cols(p)
        cols += [f"P_{pl['name']}" for pl in self.plines]
        return cols

    def history_units(self):
        """SI unit of every history column (per unit depth in ZY, per unit area in 1D)."""
        g = self.grid
        per = "" if g.rz else ("/m" if g.dim == 2 else "/m²")
        qu = {"E": "V/m", "B": "T", "phi": "V", "rho": "C/m³"}
        lpos, lval = set(), {}
        for lp in self.line_probes:
            cs = self._line_cols(lp)
            lval[cs[0]] = lp[1]                       # line average
            for vk, pk in zip(cs[1::2], cs[2::2]):
                lval[vk] = lp[1]
                lpos.add(pk)
        units = []
        for c_ in self.history_columns():
            if c_ == "t":
                u = "s"
            elif c_.startswith("KE_") or c_ == "W_field":
                u = "J" + per
            elif c_.startswith(("I_", "Iemit_")):
                u = "A" + per
            elif c_.startswith("V_"):
                u = "V"
            elif c_.startswith("P_"):
                u = "W" + per
            elif c_ in lpos:
                u = "m"
            elif c_.startswith("probe_"):
                q = [p[1] for p in self.probes if f"probe_{p[0]}" == c_] + [qv for k, qv in lval.items() if k == c_]
                q = q[0] if q else ""
                u = qu.get(q, qu.get(q[:1], ""))
            else:
                u = ""
            units.append(u)
        return units

    def _current_sides(self):
        """Domain sides that can collect particles (periodic sides and the RZ axis are skipped)."""
        sides = ("z_min", "z_max") if self.grid.dim == 1 else ("z_min", "z_max", "y_min", "y_max")
        return [sd for k, sd in enumerate(sides) if self.pbc[k] not in (2, 3)]

    def field_energy(self):
        if self.em:
            return self.yee.energy()
        return self.poisson.energy(self.phi)

    @staticmethod
    def _line_cols(lp):
        name, along, cols = lp[0], lp[2], [f"probe_{lp[0]}_avg"]
        if lp[7]:
            cols += [f"probe_{name}_min", f"probe_{name}_{along}min"]
        if lp[8]:
            cols += [f"probe_{name}_max", f"probe_{name}_{along}max"]
        return cols

    def _line_profile(self, qty, z0, y0, z1, y1):
        """Samples of a field along the segment (z0,y0)-(z1,y1): (s in [0,1], z, y, values)."""
        g = self.grid
        names = {"Ez": 0, "Ey": 1, "Ex": 2, "Bz": 3, "By": 4, "Bx": 5}
        A = self.F[names[qty]] if qty in names else (self.phi if qty == "phi" else self.rho)
        L = math.hypot(z1 - z0, (y1 - y0) if g.dim == 2 else 0.0)
        ncell = max(L / g.hmin, 1.0)
        M = int(min(max(4 * ncell, 64), 20000))
        s_ = np.linspace(0.0, 1.0, M + 1)
        zs = z0 + s_ * (z1 - z0)
        ys = y0 + s_ * (y1 - y0) if g.dim == 2 else np.zeros_like(zs)
        v = np.asarray(self._interp(A, zs, ys) if g.dim == 2 else self._interp(A, zs, None), float)
        # samples inside a conductor (PEC: no field there) are excluded -> NaN
        if self.local_es is not None:
            ids, _ = self.local_es.mesh.locate(zs, ys)
            v = np.where(self.local_es.P.solid[ids], np.nan, v)
            return s_, zs, ys, v
        if not hasattr(self, "_interior_f"):
            from ..mesh.grid import conductor_interior_nodes
            self._interior_f = conductor_interior_nodes(self.mats.cell_mat).astype(float)
        if self._interior_f.any():
            sv = np.asarray(self._interp(self._interior_f, zs, ys) if g.dim == 2
                            else self._interp(self._interior_f, zs, None), float)
            v = np.where(sv > 0.999, np.nan, v)
        return s_, zs, ys, v

    @staticmethod
    def _extremum(s_, v, sgn):
        """(value, s) of the extremum, refined by a parabola through the three samples around it."""
        k = int(np.argmax(sgn * v))
        M = len(v) - 1
        if 0 < k < M:
            f0, f1, f2 = v[k - 1], v[k], v[k + 1]
            den = f0 - 2 * f1 + f2
            if np.isfinite(den) and den != 0:
                d = 0.5 * (f0 - f2) / den
                if abs(d) <= 1:
                    return f1 - 0.25 * (f0 - f2) * d, s_[k] + d * (s_[1] - s_[0])
        return v[k], s_[k]

    def _line_probe(self, lp):
        """History values of a line probe: [min, position of min][, max, position of max]."""
        _, qty, along, z0, y0, z1, y1, rmin, rmax = lp
        s_, zs, ys, v = self._line_profile(qty, z0, y0, z1, y1)
        ok = np.isfinite(v)                       # NaN = inside a conductor
        trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        if not ok.any():
            nan = float("nan")
            return [nan] + [nan, nan] * (int(bool(rmin)) + int(bool(rmax)))
        den = trap(ok.astype(float), s_)
        avg = trap(np.where(ok, v, 0.0), s_) / den if den > 0 else float(v[ok].mean())
        out = [float(avg)]                                                # line average (vacuum part)
        for on, sgn in ((rmin, -1.0), (rmax, 1.0)):
            if on:
                val, sk = self._extremum(s_, np.where(ok, v, -sgn * np.inf), sgn)
                pos = (z0 + sk * (z1 - z0)) if along == "z" else (y0 + sk * (y1 - y0))
                out += [float(val), float(pos)]
        return out

    def _probe(self, qty, z, y):
        names = {"Ez": 0, "Ey": 1, "Ex": 2, "Bz": 3, "By": 4, "Bx": 5}
        if qty in names:
            A = self.F[names[qty]]
        elif qty == "phi":
            A = self.phi
        else:
            A = self.rho
        return float(self._interp(A, np.array([z]), np.array([y]))[0])

    def power_through(self, pl):
        """Poynting power through an axis-aligned line (W, per unit depth in ZY, per m^2 in 1D)."""
        g = self.grid
        F = self.F
        if self.em:
            # radiated power: self-consistent (Yee) fields only, applied static fields excluded
            F = np.zeros_like(self.F)
            self.yee.nodal(F)
        ev = self.ev
        pos = ev.num(pl["position"])
        if g.dim == 1:
            i = int(round((pos - g.z_min) / g.dz))
            return float((F[2][i] * F[4][i] - F[1][i] * F[5][i]) / C.mu0)
        a, b = sorted((ev.num(pl["start"]), ev.num(pl["end"])))
        if pl["orientation"] == "z":
            i = min(max(int(round((pos - g.z_min) / g.dz)), 0), g.nz)
            ys = g.y
            sel = (ys >= a - 1e-12) & (ys <= b + 1e-12)
            if g.rz:
                Sz = F[1][i] * F[5][i] - F[2][i] * F[4][i]
                wgt = 2 * np.pi * ys
            else:
                Sz = F[2][i] * F[4][i] - F[1][i] * F[5][i]
                wgt = np.ones_like(ys)
            return float(np.trapezoid((Sz * wgt)[sel], ys[sel]) / C.mu0) if sel.sum() > 1 else 0.0
        j = min(max(int(round((pos - g.y_min) / g.dy)), 0), g.ny)
        zs = g.z
        sel = (zs >= a - 1e-12) & (zs <= b + 1e-12)
        if g.rz:
            Sy = F[2][:, j] * F[3][:, j] - F[0][:, j] * F[5][:, j]
            wgt = 2 * np.pi * g.y[j]
        else:
            Sy = F[0][:, j] * F[5][:, j] - F[2][:, j] * F[3][:, j]
            wgt = 1.0
        return float(np.trapezoid(Sy[sel] * wgt, zs[sel]) / C.mu0) if sel.sum() > 1 else 0.0

    def history_row(self):
        row = [self.step_count, self.t]
        for sp in self.species:
            n = self.comm.allsum(sp.n)
            ke = self._kern(K.kinetic_energy, sp.n)[0](sp._u0, sp._u1, sp._u2, sp._w, sp.n, sp.m, self.rel) if sp.n else 0.0
            ke = self.comm.allsum(ke)
            if self.grid.rz:
                pass  # weights already include 2*pi*r (real particles)
            row += [n, ke]
        row.append(self.field_energy())
        dtint = max(self.t - self.hist_t0, 1e-300)
        cs = self.comm.allreduce(self.coll_side.copy()) if self.size > 1 else self.coll_side
        co = self.comm.allreduce(self.coll_obj.copy()) if self.size > 1 else self.coll_obj
        ea = self.comm.allreduce(self.emit_acc.copy()) if self.size > 1 else self.emit_acc
        allsides = ("z_min", "z_max", "y_min", "y_max")
        row += [cs[1 + allsides.index(sd)] / dtint for sd in self._current_sides()]
        row += [co[i] / dtint for i in self.obj_cur]
        row += list(ea[:len(self.emitters)] / dtint)
        row += self.voltages(self.t)
        row += [self._probe(q, z, y) for (_, q, z, y) in self.probes]
        for lp in self.line_probes:
            row += self._line_probe(lp)
        row += [self.power_through(pl) for pl in self.plines]
        self.coll_side[:] = 0
        self.coll_obj[:] = 0
        self.emit_acc[:] = 0
        self.hist_t0 = self.t
        return row

    def field_snapshot(self):
        g = self.grid
        d = {"t": self.t, "step": self.step_count, "z": g.z, "y": g.y, "phi": self.phi,
             "rho": self.rho, "geometry": self.geometry,
             "cell_mat": self.mats.cell_mat.astype(np.int8)}     # lets the plots hide conductor interiors
        if not self.em and self.poisson is not None:
            # space-charge potential = total minus the applied (Laplace) part of the electrodes
            try:
                d["phi_sc"] = self.phi - self.poisson.laplace_field(self.voltages(self.t))
            except Exception:  # noqa: BLE001
                pass
        for c_, nm in enumerate(("E", "E", "E", "B", "B", "B")):
            d[f"{nm}{self.comp[c_ % 3]}"] = self.F[c_]
        if self.local_es is not None:
            d.update(self.local_es.snapshot(self))
        elif self.em and len(self.species):
            Q, per = self._deposit_charge()
            d["rho"] = self.density(Q)
            for k, v in per.items():
                d[f"rho_{k}"] = self.density(self.comm.allreduce(v) if self.size > 1 else v)
        elif len(self.species) > 1:
            _, per = self._deposit_charge()
            for k, v in per.items():
                d[f"rho_{k}"] = self.density(self.comm.allreduce(v) if self.size > 1 else v)
        return d

    def particle_snapshot(self, nmax):
        d = {"t": self.t, "step": self.step_count}
        for sp in self.species:
            s = sp.snapshot(nmax, self.rng)
            for k, v in s.items():
                d[f"{sp.name}__{k}"] = v
        return d

    def geometry_data(self):
        g = self.grid
        data = {"z": g.z, "y": g.y, "node_cond": self.mats.node_cond, "cell_mat": self.mats.cell_mat,
                "cell_obj": self.mats.cell_obj, "eps_cell": self.mats.eps_cell, "geometry": self.geometry}
        if self.local_es is not None:
            data.update(local_points=g.local_mesh.points, local_triangles=g.local_mesh.triangles,
                        local_cell_mat=self.local_es.P.cell_mat,
                        local_node_cond=self.local_es.P.node_cond, mesh_type="local P1 triangles (fixed)")
        return data

    # ==========================================================================
    def run(self, callback=None, should_stop=None):
        out = self.out
        cols = self.history_columns()
        out.start_history(cols)
        out.write_json("run_info.json", {
            "name": self.model.get("name"), "geometry": self.geometry,
            "solver": "electromagnetic" if self.em else "electrostatic",
            "dt": self.dt, "n_steps": self.n_steps, "t_end": self.n_steps * self.dt,
            "grid": {"nz": self.grid.nz, "ny": self.grid.ny, "dz": self.grid.dz, "dy": self.grid.dy,
                     **({"mesh_type": "local P1 triangles (fixed)",
                         "local_nodes": len(self.local_es.mesh.points),
                         "local_triangles": len(self.local_es.mesh.triangles)}
                        if self.local_es is not None else {})},
            "species": [{"name": s.name, "q": s.q, "m": s.m} for s in self.species],
            "conductors": [c_["name"] for c_ in self.mats.conductors],
            "objects": self.obj_names, "components": self.comp,
            "mpi_ranks": self.size, "threads": self.nchunk,
        })
        if out.enabled:
            from ..model.model import save
            save(self.model, os.path.join(self.out_dir, "model.pic.json"))
        out.save_npz("", "geometry.npz", self.geometry_data())
        self.log(f"dt = {self.dt:.4g} s, steps = {self.n_steps}, MPI ranks = {self.size}, "
                 f"threads = {self.nchunk}")
        self._diag(force=True)
        tstart = time.time()
        last_print = 0.0
        stopped = False
        ctrl_every = max(1, min(self.live_every, 20))
        while self.step_count < self.n_steps:
            self.step()
            self._diag()
            if self.step_count % ctrl_every == 0:
                # pause / resume: the GUI creates / removes the file PAUSE in the output folder
                paused = bool(self.comm.bcast(out.pause_requested()))
                if paused:
                    if self.rank == 0:
                        print(f"PAUSED {self.step_count} {self.n_steps} {self.t:.6e}", flush=True)
                    self._write_live()
                    tp = time.time()
                    while paused:
                        time.sleep(0.2)
                        paused = bool(self.comm.bcast(out.pause_requested() and not out.stop_requested()))
                    tstart += time.time() - tp           # paused time is not run time
                    if self.rank == 0:
                        print(f"RESUMED {self.step_count} {self.n_steps} {self.t:.6e}", flush=True)
                if bool(self.comm.bcast(out.stop_requested())):
                    self.log("Stop requested - finishing")
                    stopped = True
                    break
                ntot = sum(sp.n for sp in self.species)
                ntot = self.comm.allreduce(ntot) if self.size > 1 else ntot
                if self.max_particles > 0 and ntot > self.max_particles:
                    self.log(f"  WARNING: {ntot} macro-particles exceed the particle limit "
                             f"({self.max_particles:.3g}, Study / Solver) - run stopped. Raise the limit, "
                             f"lower the emitted current or the particles per cell.")
                    stopped = True
                    break
            if self.step_count % self.live_every == 0 or self.step_count == self.n_steps:
                stop = out.stop_requested() or (should_stop is not None and should_stop())
                stop = bool(self.comm.bcast(stop))
                if callback is not None:
                    callback(self)
                self._write_live()          # collective (charge deposit); rank 0 writes
                el = time.time() - tstart
                if self.rank == 0 and (el - last_print > 0.5 or self.step_count == self.n_steps):
                    last_print = el
                    npart = sum(sp.n for sp in self.species) * self.size
                    print(f"PROGRESS {self.step_count} {self.n_steps} {self.t:.6e} {npart} {el:.2f}",
                          flush=True)
                if stop:
                    stopped = True
                    self.log("Stop requested - finishing")
                    break
        self._diag(force=True)
        out.close()
        el = time.time() - tstart
        self.log(f"Finished {self.step_count} steps in {el:.2f} s "
                 f"({1e3 * el / max(self.step_count, 1):.2f} ms/step). Timers: " +
                 ", ".join(f"{k}={v:.2f}s" for k, v in self.timers.items()))
        return not stopped

    def _diag(self, force=False):
        t0 = time.perf_counter()
        s = self.step_count
        if force or s % self.h_every == 0:
            row = self.history_row()
            self.history_rows.append(row)
            self.out.history(row)
            if len(self.history_rows) == 60 or (force and not getattr(self, "_refl_checked", False)):
                self._check_reflection()
        if force or s % self.f_every == 0:
            snap = self.field_snapshot()
            self.out.save_npz("fields", f"field_{s:07d}.npz", snap)
        if force or s % self.p_every == 0:
            self.out.save_npz("particles", f"part_{s:07d}.npz", self.particle_snapshot(self.max_dump))
        self.timers["diag"] += time.perf_counter() - t0

    def _check_reflection(self):
        """One-time warning when (almost) all emitted current returns to the emitting surface:
        the injected current exceeds the space-charge limit and a virtual cathode has formed."""
        if len(self.history_rows) < 15:
            return
        self._refl_checked = True
        try:
            cols = self.history_columns()
            H = np.array(self.history_rows[len(self.history_rows) // 3:])
            emit = sum(H[:, i].mean() for i, c_ in enumerate(cols) if c_.startswith("Iemit_"))
            if emit == 0:
                return
            coll = {c_: H[:, i].mean() for i, c_ in enumerate(cols) if c_.startswith("I_")}
            back = max(coll.items(), key=lambda kv: abs(kv[1]), default=(None, 0.0))
            n_alive = sum(self.comm.allsum(sp.n) for sp in self.species)
            if back[0] and abs(back[1]) > 0.95 * abs(emit) and n_alive < 5000:
                self.log(f"  WARNING: {abs(back[1] / emit) * 100:.0f} % of the emitted current returns to "
                         f"'{back[0][2:]}' - the injected current is above the space-charge limit and a "
                         "virtual cathode sits at the emitter. Lower the beam current density (about "
                         "J_CL = 2.33e-6 V^1.5/d^2 A/m^2 for a planar gap), raise its energy, or use the "
                         "'scl' emission model.")
        except Exception:  # noqa: BLE001
            pass

    def _write_live(self):
        if not self.out.enabled and self.size == 1:
            return
        d = self.field_snapshot()
        if self.rank != 0 or not self.out.enabled:
            return
        d.update(self.particle_snapshot(min(4000, self.max_dump)))
        d.update(self.line_snapshot())
        H = np.array(self.history_rows, dtype=float) if self.history_rows else np.zeros((0, 0))
        d["history"] = decimate_minmax(H, 2000)          # whole run, envelope-preserving
        d["history_tail"] = H[-4096:]                     # last raw samples (exact spectrum)
        d["history_columns"] = np.array(self.history_columns())
        d["history_units"] = np.array(self.history_units())
        d["progress"] = np.array([self.step_count, self.n_steps])
        d["t_end"] = self.n_steps * self.dt
        self.out.save_npz("", "live.npz", d, optional=True)

    def line_snapshot(self):
        """Profiles of the line probes for the live monitor: line_<name> (values), line_<name>_s
        (coordinate along the line) and line_probes (name|quantity|along|z0|y0|z1|y1|minmax)."""
        d, meta = {}, []
        for lp in self.line_probes:
            name, qty, along, z0, y0, z1, y1 = lp[:7]
            try:
                _, zs, ys, v = self._line_profile(qty, z0, y0, z1, y1)
            except Exception:  # noqa: BLE001
                continue
            d[f"line_{name}"] = v
            d[f"line_{name}_s"] = zs if along == "z" else ys
            ext = ("min" if lp[7] else "") + ("max" if lp[8] else "")
            meta.append(f"{name}|{qty}|{along}|{z0}|{y0}|{z1}|{y1}|{ext}")
        if meta:
            d["line_probes"] = np.array(meta)
        return d

    def history_array(self):
        return np.array(self.history_rows), self.history_columns()
