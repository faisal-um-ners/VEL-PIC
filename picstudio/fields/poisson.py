"""Finite-element Poisson solver built on scikit-fem.

Solves   -div( eps0*eps_r grad(phi) ) = rho   on the structured mesh with
bilinear (Q1) elements in 2D (Cartesian ZY or axisymmetric RZ with the r
weight in the weak form) and linear (P1) elements in 1D.

Weak form (RZ, the 2*pi cancels on both sides):
    int eps grad(phi).grad(v) r dr dz  =  int rho v r dr dz

The right-hand side is assembled directly from particles: for a macro
particle of charge q located at x_p,  b_i += q N_i(x_p)  (divided by 2*pi in
RZ).  This is exactly the bilinear particle weighting, so the deposit is
consistent with the finite-element discretisation (and handles the axis
without special cases).

Conductors are Dirichlet nodes.  For time-dependent voltages the solution is
built by superposition   phi = phi_rho + sum_k V_k(t) phi_k   where phi_k are
pre-computed Laplace solutions, so only one sparse LU factorisation is done
for the whole run.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from skfem import (Basis, BilinearForm, ElementLineP1, ElementQuad1, MeshLine,
                   MeshQuad, asm)
from skfem.helpers import dot, grad

from ..constants import eps0


class PoissonFEM:
    def __init__(self, grid, mats, periodic_z=False, periodic_y=False):
        self.grid = g = grid
        self.mats = mats
        self.periodic_z = periodic_z
        self.periodic_y = periodic_y and g.dim == 2
        N = g.nnodes

        # ---------------- mesh & basis --------------------------------------
        if g.dim == 1:
            mesh = MeshLine(g.z)
            elem = ElementLineP1()
            intorder = 2
        else:
            mesh = MeshQuad.init_tensor(g.z, g.y)
            elem = ElementQuad1()
            intorder = 3 if g.rz else 2
        basis = Basis(mesh, elem, intorder=intorder)
        self.mesh, self.basis = mesh, basis

        # permittivity per element (elements map 1:1 onto grid cells)
        cen = mesh.p[:, mesh.t].mean(axis=1)
        ic = g.cell_iz(cen[0])
        if g.dim == 1:
            eps_el = mats.eps_cell[ic]
        else:
            jc = g.cell_iy(cen[1])
            eps_el = mats.eps_cell[ic, jc]
        nqp = basis.X.shape[-1]
        eps_q = np.repeat(eps_el[:, None], nqp, axis=1)
        rz = g.rz

        @BilinearForm
        def laplace(u, v, w):
            wgt = w.x[1] if rz else 1.0
            return eps0 * w["epsr"] * dot(grad(u), grad(v)) * wgt

        K = asm(laplace, basis, epsr=eps_q).tocsr()

        # ---------------- reorder DOFs to grid (iz, iy) ordering ---------------
        if g.dim == 1:
            gidx = g.node_iz(mesh.p[0])
        else:
            iz = g.node_iz(mesh.p[0])
            iy = g.node_iy(mesh.p[1])
            gidx = iz * (g.ny + 1) + iy
        P = sp.csr_matrix((np.ones(N), (gidx, np.arange(N))), shape=(N, N))
        self.K = (P @ K @ P.T).tocsr()        # grid ordering

        # ---------------- periodic reduction ------------------------------------
        rep = np.arange(N).reshape(g.shape)
        if g.dim == 1:
            if periodic_z:
                rep[-1] = rep[0]
        else:
            if periodic_z:
                rep[-1, :] = rep[0, :]
            if self.periodic_y:
                rep[:, -1] = rep[:, 0]
            if periodic_z and self.periodic_y:
                rep[-1, -1] = rep[0, 0]
        rep = rep.ravel()
        uniq, red = np.unique(rep, return_inverse=True)
        self.R = sp.csr_matrix((np.ones(N), (np.arange(N), red)), shape=(N, len(uniq)))
        self.RT = self.R.T.tocsr()
        Kr = (self.RT @ self.K @ self.R).tocsr()
        Nr = len(uniq)

        cond_full = mats.node_cond.ravel()
        cond_red = -np.ones(Nr, dtype=int)
        for n in range(N):
            if cond_full[n] >= 0:
                cond_red[red[n]] = cond_full[n]
        self.cond_red = cond_red
        dmask = cond_red >= 0
        self.singular = not dmask.any()
        if self.singular:
            dmask = dmask.copy()
            dmask[0] = True          # pin the potential; neutralize the RHS
        self.dmask = dmask
        self.free = np.where(~dmask)[0]
        self.dir = np.where(dmask)[0]
        Kff = Kr[self.free][:, self.free].tocsc()
        self.Kfd = Kr[self.free][:, self.dir].tocsr()
        self.lu = spla.splu(Kff)
        # lumped volumes for neutralisation (FEM units: divide RZ by 2 pi)
        vol = g.node_volume.ravel() / (2 * np.pi if g.rz else 1.0)
        self.vol_red = self.R.T @ vol

        # ---------------- Laplace solutions for each conductor ------------------
        self.laplace = []
        for k in range(len(mats.conductors)):
            phiD = (cond_red[self.dir] == k).astype(float)
            if not phiD.any():
                self.laplace.append(np.zeros(N))
                continue
            phiF = self.lu.solve(-self.Kfd @ phiD)
            red_sol = np.zeros(Nr)
            red_sol[self.free] = phiF
            red_sol[self.dir] = phiD
            self.laplace.append(self.R @ red_sol)

    # ----------------------------------------------------------------------
    def solve(self, rhs_nodes, voltages=None):
        """rhs_nodes: node charges in FEM units (grid shape).  Returns phi (grid shape)."""
        g = self.grid
        b = self.RT @ np.asarray(rhs_nodes, float).ravel()
        if self.singular:
            b = b - self.vol_red * (b.sum() / self.vol_red.sum())
        sol = np.zeros(b.shape[0])
        sol[self.free] = self.lu.solve(b[self.free])
        if self.singular:
            sol -= (sol * self.vol_red).sum() / self.vol_red.sum()
        phi = self.R @ sol
        if voltages is not None:
            for k, V in enumerate(voltages):
                if V != 0.0:
                    phi += V * self.laplace[k]
        return phi.reshape(g.shape)

    def reaction(self, phi, rhs_nodes):
        """Nodal residual K phi - b: induced charge on Dirichlet (conductor) nodes,
        in FEM units (multiply by 2*pi in RZ for Coulombs)."""
        r = self.K @ phi.ravel() - np.asarray(rhs_nodes, float).ravel()
        return r.reshape(self.grid.shape)

    def laplace_field(self, voltages):
        g = self.grid
        phi = np.zeros(g.nnodes)
        for k, V in enumerate(voltages):
            if V != 0.0:
                phi += V * self.laplace[k]
        return phi.reshape(g.shape)

    def energy(self, phi):
        """Electrostatic energy 1/2 phi^T K phi (per unit area in 1D, per unit depth in ZY)."""
        p = phi.ravel()
        W = 0.5 * p @ (self.K @ p)
        return W * (2 * np.pi if self.grid.rz else 1.0)


def _pdiff(phi, x, axis):
    """Central difference -dphi/dx on a periodic axis (nodes 0 and N are the same point)."""
    ph = np.moveaxis(phi, axis, 0)
    h = np.diff(x)
    E = np.empty_like(ph)
    E[1:-1] = -(ph[2:] - ph[:-2]) / (h[1:] + h[:-1]).reshape((-1,) + (1,) * (ph.ndim - 1))
    E[0] = -(ph[1] - ph[-2]) / (h[0] + h[-1])
    E[-1] = E[0]
    return np.moveaxis(E, 0, axis)


def gradient_field(phi, grid, periodic_z=False, periodic_y=False):
    """Nodal E = -grad(phi) with 2nd-order differences (also on refined, non-uniform grids)."""
    g = grid
    zc = g.dz if g.uniform_z else g.z
    if g.dim == 1:
        Ez = _pdiff(phi, g.z, 0) if periodic_z else -np.gradient(phi, zc, edge_order=2)
        return [Ez]
    yc = g.dy if g.uniform_y else g.y
    Ez = _pdiff(phi, g.z, 0) if periodic_z else -np.gradient(phi, zc, axis=0, edge_order=2)
    Ey = _pdiff(phi, g.y, 1) if periodic_y else -np.gradient(phi, yc, axis=1, edge_order=2)
    if g.rz and g.on_axis:
        Ey[:, 0] = 0.0
    return [Ez, Ey]
