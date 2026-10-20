"""Surface emission models.

beam            prescribed J(t) [A/m^2], energy and temperature
scl             space-charge-limited emission by Gauss's law: each step the charge
                eps0*E_n*A needed to cancel the surface normal field is emitted
                (converges to the Child-Langmuir law)
fowler_nordheim Murphy-Good FN law with Forbes' approximation for v(f), t(f),
                field enhancement beta, local self-consistent surface field
thermionic      Richardson-Dushman with optional Schottky lowering,
                half-Maxwellian (flux-weighted) emission velocities
photo           prescribed J(t) with uniform excess-energy distribution and
                cosine (Lambertian) angular distribution

Emitting surfaces are either a straight segment or the whole (staircased)
surface of a conductor object.  The surface is split into panels (one per
cell face).  In the electrostatic solver the surface field is obtained from
the finite-element 'reaction' (induced surface charge = K phi - b on the
Dirichlet nodes), which is exact in the discrete sense and includes the
charge of freshly emitted particles, making Gauss-law emission stable.

Emitted macro-particles get a random emission time inside the step and are
advanced analytically (constant acceleration).
"""
from __future__ import annotations

import numpy as np

from .. import constants as C
from ..mesh.grid import CONDUCTOR, VACUUM


def fowler_nordheim_J(F, phi_w, beta=1.0):
    """Murphy-Good FN current density [A/m^2]; F = |E| [V/m], phi_w [eV]."""
    F = np.asarray(beta * np.abs(F), float)
    out = np.zeros_like(F)
    ok = F > 1e3
    if not ok.any():
        return out
    Fo = F[ok]
    f = np.clip(C.SCHOTTKY_C * Fo / phi_w ** 2, 1e-12, 1.0)
    v = 1 - f + f / 6 * np.log(f)
    t = 1 + f / 9 - f / 18 * np.log(f)
    out[ok] = C.FN_A * Fo ** 2 / (phi_w * t ** 2) * np.exp(-v * C.FN_B * phi_w ** 1.5 / Fo)
    return out


def richardson_J(T, W, E=0.0, A=C.A_RICHARDSON, schottky=True):
    """Richardson-Dushman current density [A/m^2] with Schottky lowering."""
    E = np.abs(np.asarray(E, float))
    dW = np.sqrt(C.SCHOTTKY_C * E) if schottky else 0.0
    kT = C.kB * T / C.e
    return A * T ** 2 * np.exp(-(W - dW) / kT)


class Emitter:
    def __init__(self, spec, species, grid, mats, ev, solver="electrostatic", objects=None):
        self.spec = spec
        self.name = spec["name"]
        self.sp = species
        self.g = g = grid
        self.mats = mats
        self.ev = ev
        self.model = spec["model"]
        self.solver = solver
        self.t_start = ev.num(spec.get("t_start", 0))
        self.t_stop = ev.num(spec.get("t_stop", 1e9))
        self.npc = max(1, int(round(ev.num(spec.get("particles_per_cell", 4)))))
        self.Jfunc = ev.func_t(spec.get("current_density", "0"))
        self.energy = ev.num(spec.get("energy", 0))
        self.T_eV = ev.num(spec.get("temperature", 0))
        self.W = ev.num(spec.get("work_function", 4.5))
        self.beta = ev.num(spec.get("beta", 1))
        self.Tc = ev.num(spec.get("cathode_temperature", 2000))
        self.AR = ev.num(spec.get("richardson", C.A_RICHARDSON))
        self.schottky = bool(spec.get("schottky", True))
        self.relax = ev.num(spec.get("relax", 1.0))
        self.emitted_charge = 0.0
        self.last_current = 0.0
        self.hmin = g.hmin
        self.eps = 1e-6 * self.hmin
        self.react_idx = None
        surface = spec.get("surface", "segment")
        if surface == "object" and g.dim == 2:
            self._panels_object(spec.get("object", ""), objects or [])
        else:
            self._panels_segment(spec)
        self._setup_yee_index()
        prof = spec.get("profile", "1")
        c = self.centers
        self.profile = np.asarray(ev.value(prof, z=c[:, 0], y=c[:, 1], r=c[:, 1], t=0.0),
                                  float) * np.ones(len(c))
        self.npanel = len(c)

    # ------------------------------------------------------------------
    # panel construction
    # ------------------------------------------------------------------
    def _panels_segment(self, spec):
        g, ev = self.g, self.ev
        normal = spec.get("normal", "auto")
        if g.dim == 1:
            z0 = ev.num(spec["z0"])
            c = np.array([[z0, 0.0]])
            if normal == "auto":
                n = np.array([1.0, 0.0]) if z0 - g.z_min <= g.z_max - z0 else np.array([-1.0, 0.0])
                n = self._orient(n, c[0], g.hz[g.cell_iz(z0)])
            else:
                n = np.array([1.0 if normal == "+z" else -1.0, 0.0])
            self.centers = c
            self.start = c.copy()
            self.tvec = np.array([[0.0, 0.0]])
            self.nvec = n[None, :]
            self.ds = np.array([0.0])
            self.area = np.array([1.0])
            self.h = np.array([0.5 * g.hz[g.cell_iz(z0 + 1e-9 * g.hmin * n[0])]])
            i = int(np.clip(g.node_iz(z0), 0, g.nz))
            if abs(g.z[i] - z0) < 1e-6 * g.hmin and self.mats.node_cond[i] >= 0:
                self.react_idx = "1d"
                self.react_i = i
            return
        p0 = np.array([ev.num(spec["z0"]), ev.num(spec["y0"])])
        p1 = np.array([ev.num(spec["z1"]), ev.num(spec["y1"])])
        L = np.hypot(*(p1 - p0))
        if L <= 0:
            raise ValueError(f"Emitter '{self.name}': zero-length segment")
        d = (p1 - p0) / L
        if abs(d[1]) < 1e-12 or abs(d[0]) < 1e-12:
            # axis-aligned: one panel per cell face (the mesh may be refined / non-uniform)
            along = 0 if abs(d[1]) < 1e-12 else 1
            nodes = g.z if along == 0 else g.y
            a, b = sorted((p0[along], p1[along]))
            inner = nodes[(nodes > a + 1e-9 * g.hmin) & (nodes < b - 1e-9 * g.hmin)]
            sk = np.unique(np.r_[a, inner, b]) - a
            if d[along] < 0:
                sk = (b - a) - sk[::-1]
            ends_s = sk
        else:
            npan_ = max(1, int(np.ceil(L / self.hmin - 1e-9)))
            ends_s = np.linspace(0.0, L, npan_ + 1)
        dsv = np.diff(ends_s)
        npan = len(dsv)
        start = p0[None, :] + ends_s[:-1, None] * d[None, :]
        centers = start + 0.5 * dsv[:, None] * d[None, :]
        if normal == "auto":
            n = np.array([-d[1], d[0]])
            n = self._orient(n, centers[npan // 2], self.hmin)
        else:
            n = np.array({"+z": (1, 0), "-z": (-1, 0), "+y": (0, 1), "-y": (0, -1)}[normal], float)
        self.centers, self.start = centers, start
        self.tvec = np.repeat(d[None, :], npan, axis=0)
        self.nvec = np.repeat(n[None, :], npan, axis=0)
        self.ds = dsv
        self.area = 2 * np.pi * np.abs(centers[:, 1]) * dsv if g.rz else dsv.copy()
        self.h = self._front_half_cell(centers, np.repeat(n[None, :], npan, axis=0))
        # reaction-based surface field (axis-aligned segment on conductor nodes)
        if not (abs(d[0]) < 1e-12 or abs(d[1]) < 1e-12):
            return
        ends = p0[None, :] + ends_s[:, None] * d[None, :]
        fi = g.lz(ends[:, 0])
        fj = g.ly(ends[:, 1])
        ii, jj = np.rint(fi).astype(int), np.rint(fj).astype(int)
        if (np.abs(fi - ii).max() > 1e-6 or np.abs(fj - jj).max() > 1e-6 or ii.min() < 0
                or jj.min() < 0 or ii.max() > g.nz or jj.max() > g.ny):
            return
        if not (self.mats.node_cond[ii, jj] >= 0).all():
            return
        if abs(d[0]) < 1e-12:
            area = g.wr[jj].copy()
        else:
            area = g.wz[ii] * (2 * np.pi * g.y[jj] if g.rz else 1.0)
        self.react_idx = "2d"
        self.ra, self.rb = (ii[:-1], jj[:-1]), (ii[1:], jj[1:])
        self.node_ij = (ii, jj)
        self.node_area = np.maximum(area, 1e-300)
        self.pan_nodes = (np.arange(npan), np.arange(1, npan + 1))

    def _panels_object(self, objname, objects):
        """Panels on every cell face between a solid cell of the object and vacuum."""
        g, mats = self.g, self.mats
        names = [o.get("name", "") for o in objects]
        if objname not in names:
            raise ValueError(f"Emitter '{self.name}': unknown object '{objname}'")
        k = names.index(objname)
        solid = (mats.cell_obj == k) & (mats.cell_mat != VACUUM)
        vac = mats.cell_mat == VACUUM
        P = []
        nz, ny = g.nz, g.ny
        # faces normal to z: between cells (i-1, j) and (i, j) at z = z_i, y in [y_j, y_j+1]
        a, b = solid[:-1, :], vac[1:, :]
        for i, j in zip(*np.nonzero(a & b)):          # solid left, vacuum right -> +z
            P.append((i + 1, j, i + 1, j + 1, 1.0, 0.0))
        a, b = vac[:-1, :], solid[1:, :]
        for i, j in zip(*np.nonzero(a & b)):          # vacuum left, solid right -> -z
            P.append((i + 1, j, i + 1, j + 1, -1.0, 0.0))
        a, b = solid[:, :-1], vac[:, 1:]
        for i, j in zip(*np.nonzero(a & b)):          # solid below -> +y
            P.append((i, j + 1, i + 1, j + 1, 0.0, 1.0))
        a, b = vac[:, :-1], solid[:, 1:]
        for i, j in zip(*np.nonzero(a & b)):          # solid above -> -y
            P.append((i, j + 1, i + 1, j + 1, 0.0, -1.0))
        if not P:
            raise ValueError(f"Emitter '{self.name}': object '{objname}' has no surface facing vacuum")
        P = np.array(P)
        ia, ja, ib, jb = (P[:, q].astype(int) for q in range(4))
        n = P[:, 4:6]
        A = np.c_[g.z[ia], g.y[ja]]
        B = np.c_[g.z[ib], g.y[jb]]
        ds = np.hypot(*(B - A).T)
        self.start = A
        self.centers = 0.5 * (A + B)
        self.tvec = (B - A) / ds[:, None]
        self.nvec = n
        self.ds = ds
        self.area = 2 * np.pi * np.abs(self.centers[:, 1]) * ds if g.rz else ds.copy()
        self.h = self._front_half_cell(self.centers, n)
        # node areas for the reaction field: half of each adjacent face area
        if (mats.node_cond[ia, ja] >= 0).all() and (mats.node_cond[ib, jb] >= 0).all():
            flat_a = ia * (ny + 1) + ja
            flat_b = ib * (ny + 1) + jb
            nodes, inv = np.unique(np.r_[flat_a, flat_b], return_inverse=True)
            area = np.bincount(inv, weights=np.r_[0.5 * self.area, 0.5 * self.area])
            self.react_idx = "2d"
            self.node_ij = (nodes // (ny + 1), nodes % (ny + 1))
            self.node_area = np.maximum(area, 1e-300)
            npan = len(ds)
            self.pan_nodes = (inv[:npan], inv[npan:])

    def _setup_yee_index(self):
        """Indices of the staggered Yee E component sitting half a cell in front of each panel
        (EM solver): this is the discrete surface field, so Gauss-law emission sees the full
        effect of the charge it emitted on the previous step (no lag, no overshoot)."""
        g = self.g
        self.yee_idx = None
        n = self.nvec
        A = self.start
        B = self.start + self.ds[:, None] * self.tvec
        if g.dim == 1:
            i = int(g.node_iz(A[0, 0]))
            if abs(g.z[min(max(i, 0), g.nz)] - A[0, 0]) > 1e-6 * g.hmin:
                return
            ie = i if n[0, 0] > 0 else i - 1
            if 0 <= ie < g.nz:
                self.yee_idx = ("1d", np.array([ie]), np.sign(n[:, 0]))
                self.yee_nodes = (np.array([i]), np.array([1.0]))      # surface node, control area
            return
        fz_a, fy_a = g.lz(A[:, 0]), g.ly(A[:, 1])
        fz_b, fy_b = g.lz(B[:, 0]), g.ly(B[:, 1])
        ia, ja, ib, jb = (np.rint(x).astype(int) for x in (fz_a, fy_a, fz_b, fy_b))
        if max(np.abs(fz_a - ia).max(), np.abs(fy_a - ja).max(), np.abs(fz_b - ib).max(),
               np.abs(fy_b - jb).max()) > 1e-6:
            return
        alongz = np.abs(n[:, 0]) > 0.5
        comp = np.where(alongz, 0, 1)
        i1 = np.where(alongz, np.where(n[:, 0] > 0, ia, ia - 1), ia)
        j1 = np.where(alongz, ja, np.where(n[:, 1] > 0, ja, ja - 1))
        i2 = np.where(alongz, i1, ib)
        j2 = np.where(alongz, jb, j1)
        ok0 = alongz & (i1 >= 0) & (i1 < g.nz) & (j1 >= 0) & (j2 <= g.ny)
        ok1 = ~alongz & (j1 >= 0) & (j1 < g.ny) & (i1 >= 0) & (i2 <= g.nz)
        if not (ok0 | ok1).all():
            return
        sign = np.where(alongz, np.sign(n[:, 0]), np.sign(n[:, 1]))
        self.yee_idx = ("2d", comp, i1, j1, i2, j2, sign)
        # surface nodes behind each of the two edges of a panel and their control-face areas
        # (the same areas the Yee solver uses for J: Gauss law  eps0 * E * A = node charge)
        na_i = np.where(alongz, np.where(n[:, 0] > 0, i1, i1 + 1), i1)
        na_j = np.where(alongz, j1, np.where(n[:, 1] > 0, j1, j1 + 1))
        nb_i = np.where(alongz, np.where(n[:, 0] > 0, i2, i2 + 1), i2)
        nb_j = np.where(alongz, j2, np.where(n[:, 1] > 0, j2, j2 + 1))
        self.yee_nodes2 = (na_i, na_j, nb_i, nb_j, alongz)

    def yee_surface_field(self, yee, static_E=None, qnodes=None):
        """Normal field at the emitting surface from the Yee grid.  The staggered E half a cell in
        front of the surface is corrected by the free charge sitting on the surface nodes (Gauss law
        over the half cell), so the result is the field AT the surface - the EM counterpart of the
        finite-element reaction used by the electrostatic solver.  Without this correction freshly
        emitted charge near the cathode is invisible to the surface field and the emitter overshoots."""
        g = self.g
        fac = (2 * np.pi) if g.rz else 1.0
        if self.yee_idx[0] == "1d":
            _, ie, sign = self.yee_idx
            En = yee.E0[ie] * sign
            if qnodes is not None:
                En = En - qnodes[self.yee_nodes[0]] / (C.eps0 * fac * self.yee_nodes[1])
        else:
            _, comp, i1, j1, i2, j2, sign = self.yee_idx
            Ea = np.empty(len(comp))
            Eb = np.empty(len(comp))
            m0 = comp == 0
            Ea[m0] = yee.E0[i1[m0], j1[m0]] * sign[m0]
            Eb[m0] = yee.E0[i2[m0], j2[m0]] * sign[m0]
            m1 = ~m0
            Ea[m1] = yee.E1[i1[m1], j1[m1]] * sign[m1]
            Eb[m1] = yee.E1[i2[m1], j2[m1]] * sign[m1]
            if qnodes is not None:
                na_i, na_j, nb_i, nb_j, alongz = self.yee_nodes2
                Az = np.asarray(yee.Az)
                Aa = np.where(alongz, Az[np.clip(na_j, 0, len(Az) - 1)],
                              yee.Ay[np.clip(i1, 0, yee.Ay.shape[0] - 1), np.clip(j1, 0, yee.Ay.shape[1] - 1)])
                Ab = np.where(alongz, Az[np.clip(nb_j, 0, len(Az) - 1)],
                              yee.Ay[np.clip(i2, 0, yee.Ay.shape[0] - 1), np.clip(j2, 0, yee.Ay.shape[1] - 1)])
                Ea = Ea - qnodes[na_i, na_j] / (C.eps0 * fac * Aa)
                Eb = Eb - qnodes[nb_i, nb_j] / (C.eps0 * fac * Ab)
            En = 0.5 * (Ea + Eb)
        if static_E is not None:
            from ..particles.kernels import interp_1d, interp_2d
            c = self.centers + self.h[:, None] * self.nvec
            if g.dim == 1:
                En = En + g.interp(static_E[0], c[:, 0]) * self.nvec[:, 0]
            else:
                Ez = g.interp(static_E[0], c[:, 0], c[:, 1])
                Ey = g.interp(static_E[1], c[:, 0], c[:, 1])
                En = En + Ez * self.nvec[:, 0] + Ey * self.nvec[:, 1]
        return En

    def _vacuum(self, pt):
        g = self.g
        z, y = pt
        if z < g.z_min or z > g.z_max:
            return False
        i = int(g.cell_iz(z))
        if g.dim == 1:
            return not self.mats.solid_cells[i]
        if y < g.y_min or y > g.y_max:
            return False
        j = int(g.cell_iy(y))
        return not self.mats.solid_cells[i, j]

    def _front_half_cell(self, centers, n):
        """Half the size of the cell in front of each panel (along its normal)."""
        g = self.g
        h = np.full(len(centers), 0.5 * self.hmin)
        eps = 1e-6 * g.hmin
        az = np.abs(n[:, 0]) > 0.5
        ay = np.abs(n[:, 1]) > 0.5
        if az.any():
            h[az] = 0.5 * g.hz[g.cell_iz(centers[az, 0] + eps * n[az, 0])]
        if ay.any() and g.dim == 2:
            h[ay] = 0.5 * g.hy[g.cell_iy(centers[ay, 1] + eps * n[ay, 1])]
        return h

    def _orient(self, n, c, h):
        if self._vacuum(c + 0.5 * h * n):
            return n
        if self._vacuum(c - 0.5 * h * n):
            return -n
        return n

    # ------------------------------------------------------------------
    def normal_field(self, phi=None, efield=None, reaction=None, yee=None, static_E=None, qnodes=None):
        """E.n at the panels (V/m)."""
        g = self.g
        from ..particles.kernels import interp_1d, interp_2d
        c = self.centers
        n = self.nvec
        if yee is not None and self.yee_idx is not None:
            return self.yee_surface_field(yee, static_E, qnodes)
        if reaction is not None and self.react_idx is not None:
            if self.react_idx == "1d":
                return np.array([reaction[self.react_i] / C.eps0])
            ii, jj = self.node_ij
            R = reaction[ii, jj] * (2 * np.pi if g.rz else 1.0)
            En_node = R / (C.eps0 * self.node_area)
            pa, pb = self.pan_nodes
            return 0.5 * (En_node[pa] + En_node[pb])
        if phi is not None:
            pa = c + self.eps * n
            pb = c + self.h[:, None] * n
            if g.dim == 1:
                fa = g.interp(phi, pa[:, 0])
                fb = g.interp(phi, pb[:, 0])
            else:
                fa = g.interp(phi, pa[:, 0], pa[:, 1])
                fb = g.interp(phi, pb[:, 0], pb[:, 1])
            return -(fb - fa) / (self.h - self.eps)
        pb = c + 2 * self.h[:, None] * n
        if g.dim == 1:
            return g.interp(efield[0], pb[:, 0]) * n[:, 0]
        Ez = g.interp(efield[0], pb[:, 0], pb[:, 1])
        Ey = g.interp(efield[1], pb[:, 0], pb[:, 1])
        return Ez * n[:, 0] + Ey * n[:, 1]

    def emit(self, t, dt, rng, step=0, rank=0, size=1, phi=None, efield=None, reaction=None, yee=None,
             static_E=None, qnodes=None):
        """Create particles emitted during the step (t, t+dt].  Returns dict or None."""
        if t < self.t_start or t > self.t_stop:
            self.last_current = 0.0
            return None
        sp = self.sp
        q, m = sp.q, sp.m
        En = self.normal_field(phi, efield, reaction, yee, static_E, qnodes)
        force_n = q * En                      # > 0: accelerated away from the surface
        accel = np.where(force_n > 0, np.abs(q / m * En), 0.0)
        A = self.area * self.profile
        model = self.model
        if model in ("beam", "photo"):
            Q = self.Jfunc(t) * A * dt
        elif model == "scl":
            Q = np.where(force_n > 0, self.relax * C.eps0 * np.abs(En) * A, 0.0)
        elif model == "fowler_nordheim":
            J = np.where(force_n > 0, fowler_nordheim_J(En, self.W, self.beta), 0.0)
            Q = J * A * dt
        elif model == "thermionic":
            Eacc = np.where(force_n > 0, En, 0.0)
            Q = richardson_J(self.Tc, self.W, Eacc, self.AR, self.schottky) * A * dt
        else:
            raise ValueError(model)
        Q = np.abs(np.asarray(Q, float)) * np.ones(self.npanel)
        self.last_current = Q.sum() / dt
        if size > 1:
            Q = np.where((np.arange(self.npanel) + step) % size == rank, Q, 0.0)
        idx = np.nonzero(Q > 0)[0]
        if len(idx) == 0:
            return None
        npc = self.npc
        pan = np.repeat(idx, npc)
        N = len(pan)
        w = Q[pan] / (abs(q) * npc)
        g = self.g
        s = rng.random(N)
        base = self.start[pan] + (s * self.ds[pan])[:, None] * self.tvec[pan]
        if g.rz:
            rc = np.abs(self.centers[pan, 1])
            with np.errstate(divide="ignore", invalid="ignore"):
                w = w * np.where(rc > 0, np.abs(base[:, 1]) / rc, 1.0)
        un = np.zeros(N)
        ut = np.zeros(N)
        uo = np.zeros(N)
        if model == "beam":
            gam = 1.0 + self.energy * C.e / (m * C.c ** 2)
            un[:] = C.c * np.sqrt(gam ** 2 - 1.0)
            if self.T_eV > 0:
                vt = np.sqrt(self.T_eV * C.e / m)
                un += vt * rng.standard_normal(N)
                ut += vt * rng.standard_normal(N)
                uo += vt * rng.standard_normal(N)
        elif model == "thermionic" or (model == "scl" and self.T_eV > 0):
            kT = C.kB * self.Tc if model == "thermionic" else self.T_eV * C.e
            vt = np.sqrt(kT / m)
            un = vt * np.sqrt(-2.0 * np.log(1.0 - rng.random(N)))
            ut = vt * rng.standard_normal(N)
            uo = vt * rng.standard_normal(N)
        elif model == "photo" and self.energy > 0:
            E = self.energy * rng.random(N)
            v = np.sqrt(2 * E * C.e / m)
            ct = np.sqrt(rng.random(N))
            st = np.sqrt(1 - ct ** 2)
            ph = 2 * np.pi * rng.random(N)
            un, ut, uo = v * ct, v * st * np.cos(ph), v * st * np.sin(ph)
        if g.dim == 1:
            ut[:] = 0.0
        tau = rng.random(N) * dt
        a = accel[pan]
        nv = self.nvec[pan]
        tv = self.tvec[pan]
        disp_n = np.maximum(un * tau + 0.5 * a * tau ** 2, 0.0) + self.eps
        birth = base + self.eps * nv
        pos = base + disp_n[:, None] * nv + (ut * tau)[:, None] * tv
        u_half = un + a * (tau - 0.5 * dt)
        u0 = u_half * nv[:, 0] + ut * tv[:, 0]
        u1 = u_half * nv[:, 1] + ut * tv[:, 1]
        if g.rz:
            pos[:, 1] = np.abs(pos[:, 1])
            birth[:, 1] = np.abs(birth[:, 1])
        self.emitted_charge += q * w.sum()
        return dict(z=pos[:, 0], y=pos[:, 1], u0=u0, u1=u1, u2=uo, w=w,
                    z_birth=birth[:, 0], y_birth=birth[:, 1])

    def outline(self):
        """Panel segments for drawing: array (P, 2, 2)."""
        a = self.start
        b = self.start + self.ds[:, None] * self.tvec
        return np.stack([a, b], axis=1)
