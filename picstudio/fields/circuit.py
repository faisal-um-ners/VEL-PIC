"""Electrode potentials in electromagnetic runs: ideal voltage sources on every conductor.

In a Yee FDTD solver a PEC conductor is *isolated*: its total charge only changes when particles
are absorbed on it or emitted from it, so as current flows from cathode to anode the gap capacitor
discharges and the electrode voltages drift.  A real diode or magnetron is driven by a power supply
that keeps every electrode at its potential.  VEL-PIC models that supply as an ideal (zero-impedance)
voltage source on each conductor:

  * the applied electrode voltages are carried by the electrostatic Laplace solution, superposed on
    the fields seen by the particles (as before);
  * the Yee fields must therefore keep every conductor at zero potential.  After each field update
    the floating potentials p_k of the conductors implied by the Yee field are computed and the
    corresponding electrostatic field  sum_k p_k (-grad phi_k)  is removed from E.

Everything uses the discrete operators of the Yee grid itself, so the correction is exact and
local charge conservation (the discrete Gauss law at the vacuum nodes) is not touched:

    M = (flux of eps_r E through the node control volume) o (E = -grad phi on the Yee edges)
    phi_k  : discrete harmonic function, phi_k = 1 on conductor k, 0 on the others, M phi_k = 0 at
             the free nodes (one sparse solve per conductor, done once)
    q_n    = eps0 * flux(E_yee)_n           charge implied by the Yee field at every node
    dQ_k   = sum_n phi_k(n) q_n             conductor charge in excess of the grounded state
                                            (Green's reciprocity)
    C_kj   = eps0 phi_k^T M phi_j           capacitance matrix
    p      = C^-1 dQ,     E_yee -= sum_k p_k (-grad phi_k)

The correction field is curl-free and divergence-free at every vacuum node, so B and the discrete
Gauss law are unchanged.  The same operator gives a Yee-consistent electrostatic initial field.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..constants import eps0


class ConductorCircuit:
    def __init__(self, yee, grid, mats, periodic_z=False, periodic_y=False):
        g = self.g = grid
        self.dim = g.dim
        self.yee = yee
        if g.dim == 1:
            nz = g.nz
            N = nz + 1
            m = yee.m_half
            er = yee.er_half
            i = np.arange(nz)
            # G0 phi = +(phi[i+1] - phi[i]) / dz * m[i]  (gradient; E = -G0 phi)
            G0 = sp.csr_matrix((np.r_[-m, m] / g.dz, (np.r_[i, i], np.r_[i, i + 1])), shape=(nz, N))
            # flux at node: +E0[i] at node i, -E0[i] at node i+1 (unit area)
            F0 = sp.csr_matrix((np.r_[er, -er], (np.r_[i, i + 1], np.r_[i, i])), shape=(N, nz))
            self.G = [(-G0).tocsr()]                   # E = G phi = -grad phi
            self.F = [F0]
            rep = np.arange(N)
            if periodic_z:
                rep[-1] = 0
            self.shapes = [(nz,)]
        else:
            nz, ny = g.nz, g.ny
            N = (nz + 1) * (ny + 1)
            idx = np.arange(N).reshape(nz + 1, ny + 1)
            wz, wr = g.wz, g.wr                      # control-volume face lengths (half at the ends)
            if g.rz:
                rh = g.y_min + (np.arange(ny) + 0.5) * g.dy
                Az = wr / (2 * np.pi)                # per 2 pi (as in the Yee solver)
                Ay = np.outer(wz, rh)
            else:
                Az = wr
                Ay = np.outer(wz, np.ones(ny))
            # z edges E0 (nz, ny+1)
            a = idx[:-1, :].ravel()
            b = idx[1:, :].ravel()
            m0 = yee.m0.ravel()
            e = np.arange(nz * (ny + 1))
            G0 = sp.csr_matrix((np.r_[m0, -m0] / g.dz, (np.r_[e, e], np.r_[b, a])), shape=(nz * (ny + 1), N))
            w0 = (yee.er0 * Az[None, :]).ravel()
            F0 = sp.csr_matrix((np.r_[w0, -w0], (np.r_[a, b], np.r_[e, e])), shape=(N, nz * (ny + 1)))
            # y edges E1 (nz+1, ny)
            a = idx[:, :-1].ravel()
            b = idx[:, 1:].ravel()
            m1 = yee.m1.ravel()
            e = np.arange((nz + 1) * ny)
            G1 = sp.csr_matrix((np.r_[m1, -m1] / g.dy, (np.r_[e, e], np.r_[b, a])), shape=((nz + 1) * ny, N))
            w1 = (yee.er1 * Ay).ravel()
            F1 = sp.csr_matrix((np.r_[w1, -w1], (np.r_[a, b], np.r_[e, e])), shape=(N, (nz + 1) * ny))
            # E = G phi with G = -grad (rows: edge = -(phi_b - phi_a)/h): built above as (m/h)(phi_b - phi_a) -> negate
            self.G = [(-G0).tocsr(), (-G1).tocsr()]
            self.F = [F0, F1]
            rep = idx.copy()
            if periodic_z:
                rep[-1, :] = rep[0, :]
            if periodic_y:
                rep[:, -1] = rep[:, 0]
            if periodic_z and periodic_y:
                rep[-1, -1] = rep[0, 0]
            rep = rep.ravel()
            self.shapes = [(nz, ny + 1), (nz + 1, ny)]
        uniq, red = np.unique(rep, return_inverse=True)
        Nr = len(uniq)
        self.R = sp.csr_matrix((np.ones(N), (np.arange(N), red)), shape=(N, Nr))
        M = sum(F @ G for F, G in zip(self.F, self.G))          # flux of eps_r E for E = G phi
        self.M = M.tocsr()
        Mr = (self.R.T @ self.M @ self.R).tocsr()
        # conductors that touch each other and have the same voltage are one electrode (e.g. the walls
        # of a cavity, or an anode block touching the outer boundary): group them
        nc = np.asarray(mats.node_cond)
        K0 = len(mats.conductors)
        parent = list(range(K0))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        volt = [str(c.get("voltage", "0")).replace(" ", "") for c in mats.conductors]
        pairs = []
        if nc.ndim == 1:
            pairs.append((nc[:-1], nc[1:]))
        else:
            pairs += [(nc[:-1, :], nc[1:, :]), (nc[:, :-1], nc[:, 1:]), (nc[:-1, :-1], nc[1:, 1:]),
                      (nc[1:, :-1], nc[:-1, 1:])]
        for A_, B_ in pairs:
            msk = (A_ >= 0) & (B_ >= 0) & (A_ != B_)
            for a_, b_ in set(zip(A_[msk].tolist(), B_[msk].tolist())):
                if volt[a_] == volt[b_]:
                    ra, rb = find(a_), find(b_)
                    if ra != rb:
                        parent[rb] = ra
        roots = sorted({find(k) for k in range(K0)})
        self.group_of = np.array([roots.index(find(k)) for k in range(K0)], int)
        cond_full = np.where(nc.ravel() >= 0, self.group_of[np.maximum(nc.ravel(), 0)], -1)
        cond = -np.ones(Nr, int)
        for n in np.where(cond_full >= 0)[0]:
            cond[red[n]] = cond_full[n]
        self.nk = len(roots)
        self.active = self.nk > 0 and np.any(cond >= 0)
        self.free = np.where(cond < 0)[0]
        self.dir = np.where(cond >= 0)[0]
        self.cond = cond
        if not self.active:
            return
        Mff = Mr[self.free][:, self.free].tocsc()
        self.Mfd = Mr[self.free][:, self.dir].tocsr()
        self.lu = spla.splu(Mff)
        # harmonic functions and their (Yee-edge) fields
        self.phi_red = np.zeros((self.nk, Nr))
        for k in range(self.nk):
            on = (cond[self.dir] == k).astype(float)
            if not on.any():
                continue
            self.phi_red[k, self.dir] = on
            self.phi_red[k, self.free] = self.lu.solve(-(self.Mfd @ on))
        self.phi_full = np.array([self.R @ p for p in self.phi_red])
        self.Ek = [[(G @ self.phi_full[k]).reshape(s) for G, s in zip(self.G, self.shapes)]
                   for k in range(self.nk)]
        Mphi = np.array([self.R.T @ (self.M @ self.phi_full[k]) for k in range(self.nk)])   # reduced nodes
        C = eps0 * (self.phi_red @ Mphi.T)
        self.C = 0.5 * (C + C.T)
        has = np.array([np.any(cond == k) for k in range(self.nk)])
        self.Cinv = np.zeros_like(self.C)
        if has.any():
            sub = np.ix_(has, has)
            self.Cinv[sub] = np.linalg.pinv(self.C[sub])
        self.last_p = np.zeros(self.nk)

    # ------------------------------------------------------------------
    def node_charge(self):
        """eps0 * flux(E_yee) at the reduced nodes: the charge implied by the Yee field."""
        y = self.yee
        E = [y.E0] if self.dim == 1 else [y.E0, y.E1]
        q = sum(F @ e.ravel() for F, e in zip(self.F, E))
        return eps0 * (self.R.T @ q)

    def floating_potentials(self):
        dQ = self.phi_red @ self.node_charge()
        return self.Cinv @ dQ

    def apply(self):
        """Remove the floating-conductor field from the Yee E (ideal voltage sources)."""
        if not self.active:
            return None
        p = self.floating_potentials()
        y = self.yee
        for k in range(self.nk):
            if p[k] == 0.0:
                continue
            y.E0 -= p[k] * self.Ek[k][0]
            if self.dim == 2:
                y.E1 -= p[k] * self.Ek[k][1]
        self.last_p = p
        return p

    def grounded_potential(self, Q_nodes):
        """Yee-consistent electrostatic potential of the node charges Q (Coulombs; per unit depth in
        ZY, per unit area in 1D, total ring charge in RZ) with every conductor grounded."""
        g = self.g
        q = np.asarray(Q_nodes, float).ravel() / (2 * np.pi if g.rz else 1.0)
        qr = self.R.T @ q
        phi = np.zeros(self.R.shape[1])
        if self.active:
            phi[self.free] = self.lu.solve(qr[self.free] / eps0)
        else:
            # no conductor: periodic / open system, pin the mean (neutralised)
            Mr = (self.R.T @ self.M @ self.R).tocsc()
            vol = np.ones_like(qr)
            qn = qr - vol * qr.sum() / vol.sum()
            A = Mr + sp.csr_matrix((np.ones(1), ([0], [0])), shape=Mr.shape)
            phi = spla.spsolve(A.tocsc(), qn / eps0)
        return (self.R @ phi).reshape(g.shape)
