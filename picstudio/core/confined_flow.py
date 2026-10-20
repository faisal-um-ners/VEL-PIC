"""Steady space-charge-limited confined flow from a small emitter (reference solver).

Model of Lau, Chernin, Heri & Zhang, "Solution to an unsolved problem in diodes: limiting current for
small emission area and low emission energy", arXiv:2607.24590 (2026), their Eq. (7):

    lap(phi) = (4/9) (J/J_CL) (phi + Delta)^(-1/2)   inside the beam column, 0 elsewhere,

in units of the gap d and voltage V (phi = 0 at the cathode z = 0, phi = 1 at the anode z = 1).
Electrons move along z only (infinite axial magnetic field), all with the injection energy
E_in = Delta * e V, from

  * a strip of width w in a periodic array of period p (planar, 'strip'): x in [0, p/2] with symmetry
    (dphi/dx = 0) at x = 0 and x = p/2;
  * a disk of radius R ('disk', axisymmetric r-z), outer surface r = r_max with dphi/dr = 0.

The limiting current J_max is the fold of the branch of steady states.  As in the paper (POISSON) it is
found by Picard iteration with warm starts plus bisection on J; the convergence threshold is scaled with
Delta, following the remark that near the cathode phi ~ Delta.  Finite elements (scikit-fem, Q1) on a
tensor mesh: uniform across the emitter, geometrically stretched outside, graded towards the cathode.
"""
from __future__ import annotations

import math

import numpy as np


def jaffe(delta):
    """Jaffe (1944) factor: 1D limiting current with injection energy, J_max = f_J J_CL."""
    return ((1 + delta) ** 0.5 + delta ** 0.5) ** 3


def j_cl(V, d, q=1.602176634e-19, m=9.1093837015e-31):
    eps0 = 8.8541878128e-12
    return 4 * eps0 / 9 * math.sqrt(2 * q / m) * V ** 1.5 / d ** 2


def _geom(n, first, total):
    """n cell widths growing geometrically from `first`, summing to `total`."""
    if n * first >= total:
        return np.full(n, total / n)
    lo, hi = 1.0 + 1e-12, 2.0
    while first * (hi ** n - 1) / (hi - 1) < total:
        hi *= 2
    for _ in range(200):
        q = 0.5 * (lo + hi)
        if first * (q ** n - 1) / (q - 1) > total:
            hi = q
        else:
            lo = q
    return first * q ** np.arange(n) * total / (first * (q ** n - 1) / (q - 1))


class ConfinedFlow:
    def __init__(self, geometry="strip", w=0.1, p=2.0, R=0.1, r_max=1.0, delta=0.01, ncol=8, nout=48,
                 nz=260, dz0=None):
        from skfem import Basis, BilinearForm, ElementQuad1, MeshQuad
        from skfem.helpers import dot, grad
        self.geometry, self.delta = geometry, float(delta)
        self.rz = geometry == "disk"
        a = R if self.rz else 0.5 * w            # emitter half width / radius
        outer = r_max if self.rz else 0.5 * p
        self.a = a
        x = np.concatenate([[0.0], np.cumsum(np.full(ncol, a / ncol))])
        if outer > a * (1 + 1e-9):
            x = np.concatenate([x, a + np.cumsum(_geom(nout, a / ncol, outer - a))])
        dz0 = dz0 or 0.05 * max(self.delta, 1e-7)
        z = np.concatenate([[0.0], np.cumsum(_geom(nz, dz0, 1.0))])
        z[-1] = 1.0
        self.x, self.z = x, z
        mesh = MeshQuad.init_tensor(x, z)
        self.basis = Basis(mesh, ElementQuad1(), intorder=4)
        rz = self.rz

        @BilinearForm
        def lap(u, v, w_):
            return dot(grad(u), grad(v)) * (w_.x[0] if rz else 1.0)

        K = lap.assemble(self.basis).tocsr()
        X, Z = mesh.p
        self.D = np.where((Z < 1e-14) | (Z > 1 - 1e-14))[0]
        self.I = np.setdiff1d(np.arange(mesh.p.shape[1]), self.D)
        self.phiD = np.where(Z[self.D] > 0.5, 1.0, 0.0)
        from scipy.sparse.linalg import splu
        self.lu = splu(K[self.I][:, self.I].tocsc())
        self.KID = K[self.I][:, self.D]
        self.vac = Z.copy()                       # vacuum solution phi = z
        self.X, self.Z = X, Z
        # quadrature data for the source term
        qx = self.basis.mapping.F(self.basis.quadrature[0])       # (2, nelem, nqp)
        self.inside = qx[0] < a * (1 - 1e-12)
        self.qr = qx[0] if rz else np.ones_like(qx[0])

    # ------------------------------------------------------------------
    def _rhs(self, phi, Jn):
        from skfem import LinearForm
        pq = np.asarray(self.basis.interpolate(phi))
        arg = pq + self.delta
        if np.any(arg[self.inside] <= 0):
            return None
        S = np.where(self.inside, 4.0 / 9.0 * Jn / np.sqrt(np.maximum(arg, 1e-300)), 0.0) * self.qr

        @LinearForm
        def src(v, w_):
            return w_["S"] * v

        return src.assemble(self.basis, S=S)

    def step(self, phi, Jn):
        b = self._rhs(phi, Jn)
        if b is None:
            return None
        new = np.empty_like(phi)
        new[self.D] = self.phiD
        new[self.I] = self.lu.solve(-b[self.I] - self.KID @ self.phiD)
        return new

    def solve(self, Jn, phi0=None, tol=None, max_iter=3000):
        """Picard iteration; returns (phi, converged)."""
        tol = tol or 1e-3 * min(self.delta, 1e-2)
        phi = self.vac.copy() if phi0 is None else phi0.copy()
        for _ in range(max_iter):
            new = self.step(phi, Jn)
            if new is None:
                return phi, False
            res = np.abs(new - phi).max()
            phi = new
            if res < tol:
                return phi, True
        return phi, False

    def jmax(self, guess=None, nbisect=12, log=None):
        """Limiting current J_max/J_CL (fold of the steady branch) by bisection with warm starts."""
        lo, philo = 0.0, self.vac.copy()
        hi = guess or self.estimate()
        # bracket: grow hi until it fails
        while True:
            phi, ok = self.solve(hi, philo)
            if not ok:
                break
            lo, philo = hi, phi
            hi *= 1.5
        for _ in range(nbisect):
            mid = 0.5 * (lo + hi)
            phi, ok = self.solve(mid, philo)
            if log:
                log(f"    J/J_CL = {mid:.5g}: {'steady' if ok else 'no steady state'}")
            if ok:
                lo, philo = mid, phi
            else:
                hi = mid
        self.phi = philo
        return 0.5 * (lo + hi)

    def estimate(self):
        """Starting guess from the paper's scaling laws (Eqs. 3-5), reduced to be below the fold."""
        if self.rz:
            return 0.5 * 1.328 / math.sqrt(self.a)
        w = 2 * self.a
        if w > 2 * math.pi * self.delta:
            return 0.5 * 9 / 8 / math.sqrt(w)
        return 0.5 * 3.302 * self.delta ** 0.5274 / w

    def centerline(self, phi=None):
        phi = self.phi if phi is None else phi
        sel = np.abs(self.X) < 1e-14
        o = np.argsort(self.Z[sel])
        return self.Z[sel][o], phi[sel][o]
