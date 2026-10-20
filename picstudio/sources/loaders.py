"""Initial particle loading (plasmas, beams) with optional quiet start."""
from __future__ import annotations

import numpy as np

from .. import constants as C


def load_particles(spec, species, grid, mats, ev, rng, rank=0, size=1):
    g = grid
    z0 = max(ev.num(spec["z0"]), g.z_min)
    z1 = min(ev.num(spec["z1"]), g.z_max)
    if z1 <= z0:
        return 0
    ppc = max(1, int(round(ev.num(spec["ppc"]))))
    quiet = bool(spec.get("quiet_start", False))
    # positions are generated in logical (cell-index) coordinates, where every cell has unit size,
    # and mapped to physical coordinates: this is uniform within each cell also on refined meshes
    i0 = max(int(np.floor(g.lz(z0) + 1e-9)), 0)
    i1 = min(int(np.ceil(g.lz(z1) - 1e-9)), g.nz)
    ncz = i1 - i0
    if g.dim == 1:
        if quiet:
            frac = (np.arange(ppc) + 0.5) / ppc
            Lz = (np.arange(i0, i1)[:, None] + frac[None, :]).ravel()
        else:
            Lz = i0 + rng.random(ncz * ppc) * ncz
        ic = np.clip(np.floor(Lz).astype(int), 0, g.nz - 1)
        Z = g.pz(Lz)
        Y = np.zeros_like(Z)
        keep = (Z >= z0) & (Z < z1)
        dV = g.hz[ic] / ppc
        slot = np.arange(len(Z))
    else:
        y0 = max(ev.num(spec["y0"]), g.y_min)
        y1 = min(ev.num(spec["y1"]), g.y_max)
        if y1 <= y0:
            return 0
        j0 = max(int(np.floor(g.ly(y0) + 1e-9)), 0)
        j1 = min(int(np.ceil(g.ly(y1) - 1e-9)), g.ny)
        ncy = j1 - j0
        if quiet:
            nq = max(1, int(round(np.sqrt(ppc))))
            ppc = nq * nq
            f = (np.arange(nq) + 0.5) / nq
            fz, fy = np.meshgrid(f, f, indexing="ij")
            ci, cj = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing="ij")
            Lz = (ci.ravel()[:, None] + fz.ravel()[None, :]).ravel()
            Ly = (cj.ravel()[:, None] + fy.ravel()[None, :]).ravel()
        else:
            N = ncz * ncy * ppc
            Lz = i0 + rng.random(N) * ncz
            Ly = j0 + rng.random(N) * ncy
        ic = np.clip(np.floor(Lz).astype(int), 0, g.nz - 1)
        jc = np.clip(np.floor(Ly).astype(int), 0, g.ny - 1)
        Z, Y = g.pz(Lz), g.py(Ly)
        keep = (Z >= z0) & (Z < z1) & (Y >= y0) & (Y < y1)
        slot = np.arange(len(Z))
        area = g.hz[ic] * g.hy[jc] / ppc
        dV = area * (2 * np.pi * Y if g.rz else np.ones_like(Y))
    Z, Y, dV, slot = Z[keep], Y[keep], dV[keep], slot[keep]
    # remove particles inside solid cells
    if mats is not None and mats.solid_cells.any():
        ic = g.cell_iz(Z)
        if g.dim == 1:
            ok = ~mats.solid_cells[ic]
        else:
            ok = ~mats.solid_cells[ic, g.cell_iy(Y)]
        Z, Y, dV, slot = Z[ok], Y[ok], dV[ok], slot[ok]
    dens = ev.value(spec["density"], z=Z, y=Y, r=Y, t=0.0)
    dens = np.broadcast_to(np.asarray(dens, float), Z.shape)
    W = dens * dV
    ok = W > 0
    Z, Y, W, slot = Z[ok], Y[ok], W[ok], slot[ok]
    n = len(Z)
    T = ev.num(spec.get("temperature", 0))
    drift = [ev.num(v) for v in spec.get("drift", [0, 0, 0])]
    vth = np.sqrt(max(T, 0.0) * C.e / species.m)
    U = np.empty((3, n))
    quiet_v = quiet and vth > 0 and ppc >= 16
    if quiet_v:
        # quiet start in velocity (Birdsall & Langdon, Sec. 16-6): in every cell the ppc particles carry
        # the ppc equal-probability quantiles of the Maxwellian, independently shuffled per component
        from scipy.special import erfinv
        quant = np.sqrt(2.0) * erfinv(2.0 * (np.arange(ppc) + 0.5) / ppc - 1.0)
        blk, slot = slot // ppc, slot % ppc
        blk = np.unique(blk, return_inverse=True)[1] if n else blk
        nblk = int(blk.max()) + 1 if n else 0
    for c in range(3):
        if quiet_v:
            perm = np.argsort(rng.random((nblk, ppc)), axis=1)
            U[c] = drift[c] + vth * quant[perm[blk, slot]]
        else:
            U[c] = drift[c] + (vth * rng.standard_normal(n) if vth > 0 else 0.0)
    pert = spec.get("perturbation", "none")
    if pert != "none":
        A = ev.num(spec.get("pert_amplitude", 0))
        mode = ev.num(spec.get("pert_mode", 1))
        k = 2 * np.pi * mode / (z1 - z0)
        if pert == "density":
            W = W * (1.0 + A * np.cos(k * (Z - z0)))
        else:
            U[0] += A * np.sin(k * (Z - z0))
    if size > 1:
        sel = slice(rank, None, size)
        Z, Y, U, W = Z[sel], Y[sel], U[:, sel], W[sel]
    species.add(Z, Y, U[0], U[1], U[2], W)
    return len(Z)
