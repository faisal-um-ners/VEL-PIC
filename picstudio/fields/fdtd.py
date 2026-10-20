"""Yee finite-difference time-domain Maxwell solvers (1D-Z, 2D-ZY, 2D-RZ m=0).

Staggering (i: z index, j: y/r index), component order (z, y|r, x|theta):

    E0 (Ez)  (i+1/2, j)        B0 (Bz)  (i, j+1/2)
    E1 (Ey)  (i, j+1/2)        B1 (By)  (i+1/2, j)
    E2 (Ex)  (i, j)            B2 (Bx)  (i+1/2, j+1/2)

In 1D only the z-staggering applies (E1, E2 at nodes; E0, B1, B2 at half nodes).

Boundaries: conductor (PEC), neumann (PMC / symmetry plane), periodic,
absorbing (graded matched-loss layer backed by PEC, like MAGIC 'FREESPACE'),
axis (RZ).  Embedded conductors are staircased (tangential E = 0 on edges
whose both end nodes are inside a conductor).  Dielectrics enter through a
per-component relative permittivity.

The current comes from the charge-conserving Esirkepov deposit, mapped
onto Yee edges with areas consistent with the finite-volume divergence
(including the r-weighted areas and the on-axis volume in RZ), so Gauss's
law is preserved to round-off.
"""
from __future__ import annotations

import numpy as np

from ..constants import c, eps0, mu0
from . import fdtd_kernels as FK

C2 = c * c


def _bc_code(b):
    return b["field"]


def _ramp_profile(n_layer, n_total, side_lo, side_hi, half):
    """Normalized depth (0..1) into absorbing layers for node (half=False, length
    N+1) or half-node (half=True, length N) positions of an axis with N cells."""
    N = n_total if half else n_total - 1
    x = np.arange(n_total) + (0.5 if half else 0.0)
    d = np.zeros(n_total)
    if n_layer <= 0:
        return d
    if side_lo:
        d = np.maximum(d, (n_layer - x) / n_layer)
    if side_hi:
        d = np.maximum(d, (x - (N - n_layer)) / n_layer)
    return np.clip(d, 0.0, 1.0)


class _Absorber:
    """Graded matched loss: dE/dt = ... - s E,  dB/dt = ... - s B,  s = smax*d^3."""

    @staticmethod
    def smax(ncell, h, R=1e-6, m=3):
        L = ncell * h
        return -(m + 1) * c * np.log(R) / (2 * L) if L > 0 else 0.0

    @staticmethod
    def coeffs(s, dt):
        a = (1 - 0.5 * s * dt) / (1 + 0.5 * s * dt)
        b = dt / (1 + 0.5 * s * dt)
        return a, b


def _pad(A, axis, lo, hi):
    """Pad one ghost layer on each side of ``axis`` according to BC modes
    'periodic' (wrap), 'pmc' (odd mirror) or other (zero)."""
    A = np.moveaxis(A, axis, 0)
    out = np.zeros((A.shape[0] + 2,) + A.shape[1:])
    out[1:-1] = A
    if lo == "periodic":
        out[0] = A[-1]
    elif lo == "pmc":
        out[0] = -A[0]
    if hi == "periodic":
        out[-1] = A[0]
    elif hi == "pmc":
        out[-1] = -A[-1]
    return np.moveaxis(out, 0, axis)


def _mode(field):
    return {"periodic": "periodic", "neumann": "pmc"}.get(field, "zero")


# ============================================================================
class Yee1D:
    def __init__(self, grid, mats, boundaries):
        g = self.g = grid
        nz = g.nz
        self.E0 = np.zeros(nz)
        self.E1 = np.zeros(nz + 1)
        self.E2 = np.zeros(nz + 1)
        self.B1 = np.zeros(nz)
        self.B2 = np.zeros(nz)
        self.J0 = np.zeros(nz)
        self.J1 = np.zeros(nz + 1)
        self.J2 = np.zeros(nz + 1)
        bl, bh = boundaries["z_min"]["field"], boundaries["z_max"]["field"]
        self.periodic = bl == "periodic"
        self.mz = (_mode(bl), _mode(bh))
        pec = mats.node_cond >= 0
        if bl == "absorbing":
            pec[0] = True
        if bh == "absorbing":
            pec[-1] = True
        self.m_node = (~pec).astype(float)
        self.m_half = (~(pec[:-1] & pec[1:])).astype(float)
        epsc = mats.eps_cell
        self.er_node = np.ones(nz + 1)
        self.er_node[1:-1] = 0.5 * (epsc[1:] + epsc[:-1])
        self.er_node[0], self.er_node[-1] = epsc[0], epsc[-1]
        self.er_half = epsc.copy()
        nl = [boundaries[s].get("absorber_cells", 16) if boundaries[s]["field"] == "absorbing" else 0
              for s in ("z_min", "z_max")]
        nlay = max(nl)
        smax = _Absorber.smax(nlay, g.dz)
        self.s_node = smax * _ramp_profile(nlay, nz + 1, nl[0] > 0, nl[1] > 0, False) ** 3
        self.s_half = smax * _ramp_profile(nlay, nz, nl[0] > 0, nl[1] > 0, True) ** 3
        self.sE_half = self.s_half.copy()
        self.sE_node = self.s_node.copy()
        sig = getattr(mats, "sigma_cell", None)
        if sig is not None and sig.any():
            rc = sig / (eps0 * epsc)
            self.sE_half = self.sE_half + rc
            rp = np.pad(rc, 1, mode="constant")
            self.sE_node = self.sE_node + 0.5 * (rp[:-1] + rp[1:])
        self._dt = None

    def _coef(self, dt):
        if self._dt == dt:
            return
        self._dt = dt
        self.aEn, self.bEn = _Absorber.coeffs(self.sE_node, dt)
        self.aEh, self.bEh = _Absorber.coeffs(self.sE_half, dt)
        self.aBh, self.bBh = _Absorber.coeffs(self.s_half, 0.5 * dt)

    def zero_J(self):
        self.J0[:] = 0
        self.J1[:] = 0
        self.J2[:] = 0

    def set_J(self, Fz, Q1, Q2, G, dt):
        g = self.g
        nz = g.nz
        if self.periodic:
            Fz = _fold_face(Fz, G, nz)
            Q1 = _fold_node(Q1, G, nz)
            Q2 = _fold_node(Q2, G, nz)
        else:
            Fz, Q1, Q2 = Fz[G:G + nz], Q1[G:G + nz + 1], Q2[G:G + nz + 1]
        wz = g.wz.copy()
        if self.periodic:
            wz[0] = wz[-1] = g.dz
        self.J0 += Fz / dt
        self.J1 += Q1 / wz
        self.J2 += Q2 / wz

    def half_B(self, dt):
        self._coef(dt)
        dz = self.g.dz
        self.B2 = self.aBh * self.B2 + self.bBh * ((self.E1[1:] - self.E1[:-1]) / dz)
        self.B1 = self.aBh * self.B1 + self.bBh * (-(self.E2[1:] - self.E2[:-1]) / dz)

    def full_E(self, dt):
        self._coef(dt)
        dz = self.g.dz
        B2p = _pad(self.B2, 0, *self.mz)
        B1p = _pad(self.B1, 0, *self.mz)
        r1 = C2 / self.er_node * (B2p[1:] - B2p[:-1]) / dz - self.J1 / (eps0 * self.er_node)
        r2 = -C2 / self.er_node * (B1p[1:] - B1p[:-1]) / dz - self.J2 / (eps0 * self.er_node)
        self.E1 = (self.aEn * self.E1 + self.bEn * r1) * self.m_node
        self.E2 = (self.aEn * self.E2 + self.bEn * r2) * self.m_node
        self.E0 = (self.aEh * self.E0 - self.bEh * self.J0 / (eps0 * self.er_half)) * self.m_half
        if self.periodic:
            self.E1[-1] = self.E1[0]
            self.E2[-1] = self.E2[0]

    def nodal(self, out):
        out[0] = _avg_half_to_node(self.E0, 0, self.periodic)
        out[1] = self.E1
        out[2] = self.E2
        out[3] = 0.0
        out[4] = _avg_half_to_node(self.B1, 0, self.periodic)
        out[5] = _avg_half_to_node(self.B2, 0, self.periodic)

    def set_from_potential(self, phi):
        self.E0 = -(phi[1:] - phi[:-1]) / self.g.dz * self.m_half

    def energy(self):
        dz = self.g.dz
        We = 0.5 * eps0 * ((self.E0 ** 2).sum() * dz + ((self.E1 ** 2 + self.E2 ** 2) * self.g.wz).sum())
        Wb = 0.5 / mu0 * ((self.B1 ** 2 + self.B2 ** 2).sum() * dz)
        return We + Wb

    def source_plane(self, z):
        i = int(round((z - self.g.z_min) / self.g.dz))
        return min(max(i, 1), self.g.nz - 1)


# ============================================================================
class Yee2D:
    def __init__(self, grid, mats, boundaries):
        g = self.g = grid
        nz, ny = g.nz, g.ny
        self.rz = g.rz
        self.E0 = np.zeros((nz, ny + 1))
        self.E1 = np.zeros((nz + 1, ny))
        self.E2 = np.zeros((nz + 1, ny + 1))
        self.B0 = np.zeros((nz + 1, ny))
        self.B1 = np.zeros((nz, ny + 1))
        self.B2 = np.zeros((nz, ny))
        self.J0 = np.zeros_like(self.E0)
        self.J1 = np.zeros_like(self.E1)
        self.J2 = np.zeros_like(self.E2)
        bf = {s: boundaries[s]["field"] for s in ("z_min", "z_max", "y_min", "y_max")}
        if g.rz and g.on_axis:
            bf["y_min"] = "axis"
        self.bf = bf
        self.pz = bf["z_min"] == "periodic"
        self.py = bf["y_min"] == "periodic"
        self.mz = (_mode(bf["z_min"]), _mode(bf["z_max"]))
        self.my = (_mode(bf["y_min"]) if bf["y_min"] != "axis" else "zero", _mode(bf["y_max"]))
        # --- PEC masks -------------------------------------------------------
        pec = mats.node_cond >= 0
        if bf["z_min"] == "absorbing":
            pec[0, :] = True
        if bf["z_max"] == "absorbing":
            pec[-1, :] = True
        if bf["y_min"] == "absorbing":
            pec[:, 0] = True
        if bf["y_max"] == "absorbing":
            pec[:, -1] = True
        self.pec_node = pec
        self.m0 = (~(pec[:-1, :] & pec[1:, :])).astype(float)
        self.m1 = (~(pec[:, :-1] & pec[:, 1:])).astype(float)
        self.m2 = (~pec).astype(float)
        if g.rz and g.on_axis:
            self.m2[:, 0] = 0.0          # E_theta = 0 on axis
        # --- dielectric -------------------------------------------------------
        ec = mats.eps_cell
        ecp = np.pad(ec, 1, mode="edge")
        self.er0 = 0.5 * (ecp[1:-1, :-1] + ecp[1:-1, 1:])           # (nz, ny+1)
        self.er1 = 0.5 * (ecp[:-1, 1:-1] + ecp[1:, 1:-1])           # (nz+1, ny)
        self.er2 = 0.25 * (ecp[:-1, :-1] + ecp[1:, :-1] + ecp[:-1, 1:] + ecp[1:, 1:])
        # --- metric (RZ) ------------------------------------------------------
        r = g.y
        rh = g.y_min + (np.arange(ny) + 0.5) * g.dy
        self.r, self.rh = r, rh
        wz = g.wz.copy()
        wr = g.wr.copy()
        if self.pz:                      # periodic duplicates carry full cells after folding
            wz[0] = wz[-1] = g.dz
        if self.py:
            wr[0] = wr[-1] = g.dy
        self.vol = np.outer(wz, wr)
        if g.rz:
            # area of z-faces at node rows (2*pi*r*dr incl. axis/outer corrections), /2pi
            self.Az = wr / (2 * np.pi)                   # (ny+1)
            self.Ay = np.outer(wz, rh)                   # per (2 pi): r_{j+1/2} * wz
        else:
            self.Az = wr
            self.Ay = np.outer(wz, np.ones(ny))
        # --- absorbing layers ---------------------------------------------------
        nl = {s: (int(boundaries[s].get("absorber_cells", 16)) if bf[s] == "absorbing" else 0)
              for s in bf}
        lz = max(nl["z_min"], nl["z_max"])
        ly = max(nl["y_min"], nl["y_max"])
        sz_max = _Absorber.smax(lz, g.dz)
        sy_max = _Absorber.smax(ly, g.dy)
        szn = sz_max * _ramp_profile(lz, nz + 1, nl["z_min"] > 0, nl["z_max"] > 0, False) ** 3
        szh = sz_max * _ramp_profile(lz, nz, nl["z_min"] > 0, nl["z_max"] > 0, True) ** 3
        syn = sy_max * _ramp_profile(ly, ny + 1, nl["y_min"] > 0, nl["y_max"] > 0, False) ** 3
        syh = sy_max * _ramp_profile(ly, ny, nl["y_min"] > 0, nl["y_max"] > 0, True) ** 3
        self.sE0 = szh[:, None] + syn[None, :]
        self.sE1 = szn[:, None] + syh[None, :]
        self.sE2 = szn[:, None] + syn[None, :]
        self.sB0 = szn[:, None] + syh[None, :]
        self.sB1 = szh[:, None] + syn[None, :]
        self.sB2 = szh[:, None] + syh[None, :]
        # --- conductive (lossy) loads: dE/dt = ... - sigma/(eps0 eps_r) E ---------
        sig = getattr(mats, "sigma_cell", None)
        if sig is not None and sig.any():
            rc = sig / (eps0 * ec)
            rp = np.pad(rc, 1, mode="constant")
            self.sE0 = self.sE0 + 0.5 * (rp[1:-1, :-1] + rp[1:-1, 1:])
            self.sE1 = self.sE1 + 0.5 * (rp[:-1, 1:-1] + rp[1:, 1:-1])
            self.sE2 = self.sE2 + 0.25 * (rp[:-1, :-1] + rp[1:, :-1] + rp[:-1, 1:] + rp[1:, 1:])
        self.has_abs = (lz + ly) > 0
        self._dt = None

    def _coef(self, dt):
        if self._dt == dt:
            return
        self._dt = dt
        A = _Absorber.coeffs
        self.cE = [tuple(np.ascontiguousarray(x) for x in A(s, dt)) for s in (self.sE0, self.sE1, self.sE2)]
        self.cB = [tuple(np.ascontiguousarray(x) for x in A(s, 0.5 * dt))
                   for s in (self.sB0, self.sB1, self.sB2)]

    def zero_J(self):
        self.J0[:] = 0
        self.J1[:] = 0
        self.J2[:] = 0

    # ------------------------------------------------------------------
    def set_J(self, Fz, Fy, Q2, G, dt):
        g = self.g
        nz, ny = g.nz, g.ny
        # z-direction folding / slicing
        if self.pz:
            Fz = _fold_face(Fz, G, nz, axis=0)
            Fy = _fold_node(Fy, G, nz, axis=0)
            Q2 = _fold_node(Q2, G, nz, axis=0)
        else:
            Fz, Fy, Q2 = Fz[G:G + nz], Fy[G:G + nz + 1], Q2[G:G + nz + 1]
        if self.py:
            Fz = _fold_node(Fz, G, ny, axis=1)
            Fy = _fold_face(Fy, G, ny, axis=1)
            Q2 = _fold_node(Q2, G, ny, axis=1)
        else:
            Fz, Fy, Q2 = Fz[:, G:G + ny + 1], Fy[:, G:G + ny], Q2[:, G:G + ny + 1]
        if g.rz:
            self.J0 += Fz / (dt * 2 * np.pi * self.Az[None, :] * 1.0)
            self.J1 += Fy / (dt * 2 * np.pi * self.Ay)
            self.J2 += Q2 / self.vol
        else:
            self.J0 += Fz / (dt * self.Az[None, :])
            self.J1 += Fy / (dt * self.Ay)
            self.J2 += Q2 / self.vol

    # ------------------------------------------------------------------
    def half_B(self, dt):
        self._coef(dt)
        g = self.g
        (a0, b0), (a1, b1), (a2, b2) = self.cB
        FK.b_update(self.E0, self.E1, self.E2, self.B0, self.B1, self.B2, a0, b0, a1, b1, a2, b2,
                    g.dz, g.dy, self.rz, self.r, self.rh, bool(g.on_axis))

    def full_E(self, dt, axis_factor=3.0):
        self._coef(dt)
        g = self.g
        (a0, b0), (a1, b1), (a2, b2) = self.cE
        code = {"zero": 0, "periodic": 1, "pmc": 2}
        FK.e_update(self.E0, self.E1, self.E2, self.B0, self.B1, self.B2, self.J0, self.J1, self.J2,
                    a0, b0, a1, b1, a2, b2, self.er0, self.er1, self.er2, self.m0, self.m1, self.m2,
                    g.dz, g.dy, self.rz, self.r, self.rh, bool(g.on_axis), float(axis_factor),
                    code[self.mz[0]], code[self.mz[1]], code[self.my[0]], code[self.my[1]],
                    C2, 1.0 / eps0)
        if self.pz:
            self.E1[-1, :] = self.E1[0, :]
            self.E2[-1, :] = self.E2[0, :]
        if self.py:
            self.E0[:, -1] = self.E0[:, 0]
            self.E2[:, -1] = self.E2[:, 0]

    # ------------------------------------------------------------------
    def nodal(self, out):
        FK.nodal(self.E0, self.E1, self.E2, self.B0, self.B1, self.B2, out, self.pz, self.py,
                 bool(self.rz and self.g.on_axis))

    def set_from_potential(self, phi):
        g = self.g
        self.E0 = -(phi[1:, :] - phi[:-1, :]) / g.dz * self.m0
        self.E1 = -(phi[:, 1:] - phi[:, :-1]) / g.dy * self.m1

    def energy(self):
        F = np.zeros((6,) + self.g.shape)
        self.nodal(F)
        V = self.g.node_volume
        return float((0.5 * eps0 * (F[0] ** 2 + F[1] ** 2 + F[2] ** 2) * V).sum()
                     + (0.5 / mu0 * (F[3] ** 2 + F[4] ** 2 + F[5] ** 2) * V).sum())

    def source_plane(self, z):
        i = int(round((z - self.g.z_min) / self.g.dz))
        return min(max(i, 1), self.g.nz - 1)


# ============================================================================
def _avg_half_to_node(A, axis, periodic):
    A = np.moveaxis(A, axis, 0)
    n = A.shape[0]
    out = np.empty((n + 1,) + A.shape[1:])
    out[1:-1] = 0.5 * (A[1:] + A[:-1])
    if periodic:
        out[0] = out[-1] = 0.5 * (A[0] + A[-1])
    else:
        out[0] = A[0]
        out[-1] = A[-1]
    return np.moveaxis(out, 0, axis)


def _fold_node(A, G, n, axis=0):
    """Periodic fold of a node-centred array with G guard nodes (physical nodes 0..n)."""
    A = np.moveaxis(A, axis, 0)
    phys = A[G:G + n + 1].copy()
    for k in range(1, G + 1):            # guard below: physical node -k == n-k
        phys[n - k] += A[G - k]
    for k in range(1, A.shape[0] - (G + n + 1) + 1):   # guard above: node n+k == k
        phys[k] += A[G + n + k]
    phys[0] += phys[n]
    phys[n] = phys[0]
    return np.moveaxis(phys, 0, axis)


def _fold_face(A, G, n, axis=0):
    """Periodic fold of a face-centred array (faces k+1/2, physical k = 0..n-1)."""
    A = np.moveaxis(A, axis, 0)
    phys = A[G:G + n].copy()
    for k in range(1, G + 1):
        phys[n - k] += A[G - k]
    for k in range(0, A.shape[0] - (G + n)):
        phys[k] += A[G + n + k]
    return np.moveaxis(phys, 0, axis)


def binomial_filter(A, passes=1, axes=None, periodic=()):
    """1-2-1 smoothing along the given axes (interior points; edges kept)."""
    if passes <= 0:
        return A
    axes = range(A.ndim) if axes is None else axes
    B = A.copy()
    for _ in range(passes):
        for ax in axes:
            Bm = np.moveaxis(B, ax, 0)
            if Bm.shape[0] < 3:
                continue
            C_ = Bm.copy()
            if ax in periodic:
                Bm[:] = 0.25 * (np.roll(C_, 1, 0) + 2 * C_ + np.roll(C_, -1, 0))
            else:
                Bm[1:-1] = 0.25 * (C_[:-2] + 2 * C_[1:-1] + C_[2:])
    return B


def filter_current(yee, passes, ncell, periodic):
    """WarpX-style bilinear (1-2-1) filter of the deposited current density, ``passes`` times along
    every axis (WarpX ``warpx.use_filter`` / ``filter_npass_each_dir``; applied before the field
    update).  ``ncell`` = cells per axis, ``periodic`` = periodic flag per axis.  On a periodic axis the
    filter wraps around (a node-centred array carries the duplicated end node); elsewhere the edge
    values are kept."""
    if passes <= 0:
        return
    for name in ("J0", "J1", "J2"):
        A = getattr(yee, name)
        for _ in range(passes):
            for ax in range(A.ndim):
                Am = np.moveaxis(A, ax, 0)
                n = Am.shape[0]
                if n < 3:
                    continue
                if periodic[ax]:
                    if n == ncell[ax] + 1:          # node-centred: drop the duplicated end node
                        Cc = Am[:-1].copy()
                        Am[:-1] = 0.25 * (np.roll(Cc, 1, 0) + 2 * Cc + np.roll(Cc, -1, 0))
                        Am[-1] = Am[0]
                    else:
                        Cc = Am.copy()
                        Am[:] = 0.25 * (np.roll(Cc, 1, 0) + 2 * Cc + np.roll(Cc, -1, 0))
                else:
                    Cc = Am.copy()
                    Am[1:-1] = 0.25 * (Cc[:-2] + 2 * Cc[1:-1] + Cc[2:])
