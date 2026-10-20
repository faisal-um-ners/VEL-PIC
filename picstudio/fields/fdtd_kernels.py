"""Numba kernels for the 2D Yee updates (Cartesian ZY and cylindrical RZ).

Ghost modes for boundary rows: 0 = zero (PEC/absorbing), 1 = periodic, 2 = PMC (odd mirror).
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(inline="always")
def _g_lo(mode, first, last):
    if mode == 1:
        return last
    if mode == 2:
        return -first
    return 0.0


@njit(inline="always")
def _g_hi(mode, first, last):
    if mode == 1:
        return first
    if mode == 2:
        return -last
    return 0.0


@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def b_update(E0, E1, E2, B0, B1, B2, a0, b0, a1, b1, a2, b2, dz, dy, rz, r, rh, on_axis):
    nz = B2.shape[0]
    ny = B2.shape[1]
    for i_ in prange(nz + 1):
        i = np.int64(i_)
        for j in range(ny + 1):
            if i < nz and j < ny:
                d0E1 = (E1[i + 1, j] - E1[i, j]) / dz
                d1E0 = (E0[i, j + 1] - E0[i, j]) / dy
                if rz:
                    B2[i, j] = a2[i, j] * B2[i, j] + b2[i, j] * (d1E0 - d0E1)
                else:
                    B2[i, j] = a2[i, j] * B2[i, j] + b2[i, j] * (d0E1 - d1E0)
            if i < nz:
                d0E2 = (E2[i + 1, j] - E2[i, j]) / dz
                if rz:
                    if on_axis and j == 0:
                        B1[i, j] = 0.0
                    else:
                        B1[i, j] = a1[i, j] * B1[i, j] + b1[i, j] * d0E2
                else:
                    B1[i, j] = a1[i, j] * B1[i, j] - b1[i, j] * d0E2
            if j < ny:
                if rz:
                    c0 = -(r[j + 1] * E2[i, j + 1] - r[j] * E2[i, j]) / (rh[j] * dy)
                else:
                    c0 = (E2[i, j + 1] - E2[i, j]) / dy
                B0[i, j] = a0[i, j] * B0[i, j] + b0[i, j] * c0


@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def e_update(E0, E1, E2, B0, B1, B2, J0, J1, J2, a0, b0, a1, b1, a2, b2, er0, er1, er2,
             m0, m1, m2, dz, dy, rz, r, rh, on_axis, axis_factor, zlo, zhi, ylo, yhi, c2, inv_eps0):
    nz = B2.shape[0]
    ny = B2.shape[1]
    for i_ in prange(nz + 1):
        i = np.int64(i_)
        for j in range(ny + 1):
            # ---------------- E0 at (i+1/2, j) ----------------
            if i < nz:
                if j < ny:
                    Bu = B2[i, j]
                else:
                    Bu = _g_hi(yhi, B2[i, 0], B2[i, ny - 1])
                if j > 0:
                    Bd = B2[i, j - 1]
                else:
                    Bd = _g_lo(ylo, B2[i, 0], B2[i, ny - 1])
                if rz:
                    if on_axis and j == 0:
                        curl = axis_factor * B2[i, 0] / dy
                    else:
                        ru = rh[j] if j < ny else rh[ny - 1] + dy
                        rd = rh[j - 1] if j > 0 else rh[0] - dy
                        curl = (ru * Bu - rd * Bd) / (r[j] * dy)
                else:
                    curl = -(Bu - Bd) / dy
                E0[i, j] = m0[i, j] * (a0[i, j] * E0[i, j]
                                       + b0[i, j] * (c2 * curl - J0[i, j] * inv_eps0) / er0[i, j])
            # ---------------- E1 at (i, j+1/2) ----------------
            if j < ny:
                if i < nz:
                    Bf = B2[i, j]
                else:
                    Bf = _g_hi(zhi, B2[0, j], B2[nz - 1, j])
                if i > 0:
                    Bb = B2[i - 1, j]
                else:
                    Bb = _g_lo(zlo, B2[0, j], B2[nz - 1, j])
                curl = (Bf - Bb) / dz
                if rz:
                    curl = -curl
                E1[i, j] = m1[i, j] * (a1[i, j] * E1[i, j]
                                       + b1[i, j] * (c2 * curl - J1[i, j] * inv_eps0) / er1[i, j])
            # ---------------- E2 at (i, j) ----------------
            if j < ny:
                B0u = B0[i, j]
            else:
                B0u = _g_hi(yhi, B0[i, 0], B0[i, ny - 1])
            if j > 0:
                B0d = B0[i, j - 1]
            else:
                B0d = _g_lo(ylo, B0[i, 0], B0[i, ny - 1])
            if i < nz:
                B1f = B1[i, j]
            else:
                B1f = _g_hi(zhi, B1[0, j], B1[nz - 1, j])
            if i > 0:
                B1b = B1[i - 1, j]
            else:
                B1b = _g_lo(zlo, B1[0, j], B1[nz - 1, j])
            if rz:
                curl = (B1f - B1b) / dz - (B0u - B0d) / dy
            else:
                curl = (B0u - B0d) / dy - (B1f - B1b) / dz
            E2[i, j] = m2[i, j] * (a2[i, j] * E2[i, j]
                                   + b2[i, j] * (c2 * curl - J2[i, j] * inv_eps0) / er2[i, j])


@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def nodal(E0, E1, E2, B0, B1, B2, out, pz, py, rz_axis):
    nz = B2.shape[0]
    ny = B2.shape[1]
    for i_ in prange(nz + 1):
        i = np.int64(i_)
        if i == 0:
            im = nz - 1 if pz else 0
            ip = 0
        elif i == nz:
            im = nz - 1
            ip = 0 if pz else nz - 1
        else:
            im = i - 1
            ip = i
        for j in range(ny + 1):
            if j == 0:
                jm = ny - 1 if py else 0
                jp = 0
            elif j == ny:
                jm = ny - 1
                jp = 0 if py else ny - 1
            else:
                jm = j - 1
                jp = j
            out[0, i, j] = 0.5 * (E0[im, j] + E0[ip, j])
            out[1, i, j] = 0.5 * (E1[i, jm] + E1[i, jp])
            out[2, i, j] = E2[i, j]
            out[3, i, j] = 0.5 * (B0[i, jm] + B0[i, jp])
            out[4, i, j] = 0.5 * (B1[im, j] + B1[ip, j])
            out[5, i, j] = 0.25 * (B2[im, jm] + B2[ip, jm] + B2[im, jp] + B2[ip, jp])
            if rz_axis and j == 0:
                out[1, i, j] = 0.0
                out[4, i, j] = 0.0
                out[5, i, j] = 0.0
