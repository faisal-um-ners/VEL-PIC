"""Numba-compiled particle kernels (gather/push, deposit, boundaries).

Component ordering of vectors (velocity u = gamma*v, E, B):
    1D-Z  : (z, y, x)       left-handed   -> hand = -1  (v x B gets a minus sign)
    2D-ZY : (z, y, x)       left-handed   -> hand = -1
    2D-RZ : (z, r, theta)   right-handed  -> hand = +1

All deposit kernels are thread-parallel over particle chunks with private
grid buffers (no race conditions), reduced at the end.
"""
from __future__ import annotations

import math

import numpy as np
import numba
from numba import get_num_threads, njit, prange

C_LIGHT = 299792458.0
INV_C2 = 1.0 / C_LIGHT ** 2
SERIAL_THRESHOLD = 20000      # below this many particles the serial kernels are used

_SERIAL = {}


def _dual(func):
    """Compile ``func`` twice: thread-parallel (returned) and serial (registered).
    The serial copy gets its own name so the two variants never share an on-disk cache entry."""
    import types
    fser = types.FunctionType(func.__code__, func.__globals__, func.__name__ + "_serial",
                              func.__defaults__, func.__closure__)
    fser.__qualname__ = func.__qualname__ + "_serial"
    par = njit(parallel=True, fastmath=True, cache=True, nogil=True)(func)
    ser = njit(parallel=False, fastmath=True, cache=True, nogil=True)(fser)
    _SERIAL[func.__name__] = ser
    par.serial = ser
    return par


class _Pair:
    """A parallel kernel with a ``.serial`` twin (same call signature)."""

    def __init__(self, par, ser):
        self.par = par
        self.serial = ser

    def __call__(self, *a):
        return self.par(*a)


def pick(kernel, n):
    """Return the serial variant of a dual kernel for small particle numbers."""
    return kernel.serial if n < SERIAL_THRESHOLD else kernel


# ============================================================================
#  Boris pusher
# ============================================================================
@njit(inline="always", fastmath=True)
def _boris(u0, u1, u2, E0, E1, E2, B0, B1, B2, h, hand, rel):
    um0 = u0 + h * E0
    um1 = u1 + h * E1
    um2 = u2 + h * E2
    if rel:
        g = math.sqrt(1.0 + (um0 * um0 + um1 * um1 + um2 * um2) * INV_C2)
    else:
        g = 1.0
    f = h * hand / g
    t0 = f * B0
    t1 = f * B1
    t2 = f * B2
    tt = t0 * t0 + t1 * t1 + t2 * t2
    up0 = um0 + (um1 * t2 - um2 * t1)
    up1 = um1 + (um2 * t0 - um0 * t2)
    up2 = um2 + (um0 * t1 - um1 * t0)
    s = 2.0 / (1.0 + tt)
    un0 = um0 + s * (up1 * t2 - up2 * t1) + h * E0
    un1 = um1 + s * (up2 * t0 - up0 * t2) + h * E1
    un2 = um2 + s * (up0 * t1 - up1 * t0) + h * E2
    return un0, un1, un2


@njit(fastmath=True, cache=True, nogil=True)
def _push_1d_body(p0, p1, z, u0, u1, u2, F, zmin, dz, nz, qm, dt, hand, rel):
    """F: (6, nz+1) nodal fields (E0,E1,E2,B0,B1,B2)."""
    h = 0.5 * qm * dt
    for p in range(p0, p1):
        fz = (z[p] - zmin) / dz
        i = int(fz)
        if i < 0:
            i = 0
        elif i > nz - 1:
            i = nz - 1
        a = fz - i
        b = 1.0 - a
        E0 = b * F[0, i] + a * F[0, i + 1]
        E1 = b * F[1, i] + a * F[1, i + 1]
        E2 = b * F[2, i] + a * F[2, i + 1]
        B0 = b * F[3, i] + a * F[3, i + 1]
        B1 = b * F[4, i] + a * F[4, i + 1]
        B2 = b * F[5, i] + a * F[5, i + 1]
        un0, un1, un2 = _boris(u0[p], u1[p], u2[p], E0, E1, E2, B0, B1, B2, h, hand, rel)
        u0[p] = un0
        u1[p] = un1
        u2[p] = un2
        if rel:
            g = math.sqrt(1.0 + (un0 * un0 + un1 * un1 + un2 * un2) * INV_C2)
        else:
            g = 1.0
        z[p] += un0 / g * dt

@njit(parallel=True, fastmath=True, cache=False, nogil=True)   # (parfor calling a jitted body cannot be cached)
def _push_1d_par(z, u0, u1, u2, n, F, zmin, dz, nz, qm, dt, hand, rel):
    nt = get_num_threads()
    ch = (n + nt - 1) // nt
    for t in prange(nt):
        _push_1d_body(t * ch, min(n, (t + 1) * ch), z, u0, u1, u2, F, zmin, dz, nz, qm, dt, hand, rel)


@njit(fastmath=True, cache=True, nogil=True)
def _push_1d_ser(z, u0, u1, u2, n, F, zmin, dz, nz, qm, dt, hand, rel):
    _push_1d_body(0, n, z, u0, u1, u2, F, zmin, dz, nz, qm, dt, hand, rel)


push_1d = _Pair(_push_1d_par, _push_1d_ser)


@njit(fastmath=True, cache=True, nogil=True)
def _push_2d_body(p0, p1, z, y, u0, u1, u2, F, zmin, ymin, dz, dy, nz, ny, qm, dt, hand, rz, rel):
    """F: (6, nz+1, ny+1) nodal fields.  rz=True uses the Boris rotation scheme."""
    h = 0.5 * qm * dt
    for p in range(p0, p1):
        fz = (z[p] - zmin) / dz
        i = int(fz)
        if i < 0:
            i = 0
        elif i > nz - 1:
            i = nz - 1
        a = fz - i
        fy = (y[p] - ymin) / dy
        j = int(fy)
        if j < 0:
            j = 0
        elif j > ny - 1:
            j = ny - 1
        b = fy - j
        w00 = (1.0 - a) * (1.0 - b)
        w10 = a * (1.0 - b)
        w01 = (1.0 - a) * b
        w11 = a * b
        E0 = w00 * F[0, i, j] + w10 * F[0, i + 1, j] + w01 * F[0, i, j + 1] + w11 * F[0, i + 1, j + 1]
        E1 = w00 * F[1, i, j] + w10 * F[1, i + 1, j] + w01 * F[1, i, j + 1] + w11 * F[1, i + 1, j + 1]
        E2 = w00 * F[2, i, j] + w10 * F[2, i + 1, j] + w01 * F[2, i, j + 1] + w11 * F[2, i + 1, j + 1]
        B0 = w00 * F[3, i, j] + w10 * F[3, i + 1, j] + w01 * F[3, i, j + 1] + w11 * F[3, i + 1, j + 1]
        B1 = w00 * F[4, i, j] + w10 * F[4, i + 1, j] + w01 * F[4, i, j + 1] + w11 * F[4, i + 1, j + 1]
        B2 = w00 * F[5, i, j] + w10 * F[5, i + 1, j] + w01 * F[5, i, j + 1] + w11 * F[5, i + 1, j + 1]
        un0, un1, un2 = _boris(u0[p], u1[p], u2[p], E0, E1, E2, B0, B1, B2, h, hand, rel)
        if rel:
            g = math.sqrt(1.0 + (un0 * un0 + un1 * un1 + un2 * un2) * INV_C2)
        else:
            g = 1.0
        z[p] += un0 / g * dt
        if rz:
            xx = y[p] + un1 / g * dt
            yy = un2 / g * dt
            rn = math.sqrt(xx * xx + yy * yy)
            if rn > 0.0:
                cs = xx / rn
                sn = yy / rn
                ur = cs * un1 + sn * un2
                ut = -sn * un1 + cs * un2
                un1 = ur
                un2 = ut
            y[p] = rn
        else:
            y[p] += un1 / g * dt
        u0[p] = un0
        u1[p] = un1
        u2[p] = un2

@njit(parallel=True, fastmath=True, cache=False, nogil=True)   # (parfor calling a jitted body cannot be cached)
def _push_2d_par(z, y, u0, u1, u2, n, F, zmin, ymin, dz, dy, nz, ny, qm, dt, hand, rz, rel):
    nt = get_num_threads()
    ch = (n + nt - 1) // nt
    for t in prange(nt):
        _push_2d_body(t * ch, min(n, (t + 1) * ch), z, y, u0, u1, u2, F, zmin, ymin, dz, dy, nz, ny,
                      qm, dt, hand, rz, rel)


@njit(fastmath=True, cache=True, nogil=True)
def _push_2d_ser(z, y, u0, u1, u2, n, F, zmin, ymin, dz, dy, nz, ny, qm, dt, hand, rz, rel):
    _push_2d_body(0, n, z, y, u0, u1, u2, F, zmin, ymin, dz, dy, nz, ny, qm, dt, hand, rz, rel)


push_2d = _Pair(_push_2d_par, _push_2d_ser)


# ----------------------------------------------------------------------------
# Refined (rectilinear, non-uniform) grids: the bilinear weights come from the logical coordinates
# zl, yl (fractional node index), the particle moves in physical coordinates.
@njit(fastmath=True, cache=True, nogil=True)
def _push_1d_body_l(p0, p1, z, zl, u0, u1, u2, F, nz, qm, dt, hand, rel):
    h = 0.5 * qm * dt
    for p in range(p0, p1):
        fz = zl[p]
        i = int(math.floor(fz))
        if i < 0:
            i = 0
        elif i > nz - 1:
            i = nz - 1
        a = fz - i
        b = 1.0 - a
        E0 = b * F[0, i] + a * F[0, i + 1]
        E1 = b * F[1, i] + a * F[1, i + 1]
        E2 = b * F[2, i] + a * F[2, i + 1]
        B0 = b * F[3, i] + a * F[3, i + 1]
        B1 = b * F[4, i] + a * F[4, i + 1]
        B2 = b * F[5, i] + a * F[5, i + 1]
        un0, un1, un2 = _boris(u0[p], u1[p], u2[p], E0, E1, E2, B0, B1, B2, h, hand, rel)
        u0[p] = un0
        u1[p] = un1
        u2[p] = un2
        if rel:
            g = math.sqrt(1.0 + (un0 * un0 + un1 * un1 + un2 * un2) * INV_C2)
        else:
            g = 1.0
        z[p] += un0 / g * dt


@njit(parallel=True, fastmath=True, cache=False, nogil=True)
def _push_1d_l_par(z, zl, u0, u1, u2, n, F, nz, qm, dt, hand, rel):
    nt = get_num_threads()
    ch = (n + nt - 1) // nt
    for t in prange(nt):
        _push_1d_body_l(t * ch, min(n, (t + 1) * ch), z, zl, u0, u1, u2, F, nz, qm, dt, hand, rel)


@njit(fastmath=True, cache=True, nogil=True)
def _push_1d_l_ser(z, zl, u0, u1, u2, n, F, nz, qm, dt, hand, rel):
    _push_1d_body_l(0, n, z, zl, u0, u1, u2, F, nz, qm, dt, hand, rel)


push_1d_l = _Pair(_push_1d_l_par, _push_1d_l_ser)


@njit(fastmath=True, cache=True, nogil=True)
def _push_2d_body_l(p0, p1, z, y, zl, yl, u0, u1, u2, F, nz, ny, qm, dt, hand, rz, rel):
    h = 0.5 * qm * dt
    for p in range(p0, p1):
        fz = zl[p]
        i = int(math.floor(fz))
        if i < 0:
            i = 0
        elif i > nz - 1:
            i = nz - 1
        a = fz - i
        fy = yl[p]
        j = int(math.floor(fy))
        if j < 0:
            j = 0
        elif j > ny - 1:
            j = ny - 1
        b = fy - j
        w00 = (1.0 - a) * (1.0 - b)
        w10 = a * (1.0 - b)
        w01 = (1.0 - a) * b
        w11 = a * b
        E0 = w00 * F[0, i, j] + w10 * F[0, i + 1, j] + w01 * F[0, i, j + 1] + w11 * F[0, i + 1, j + 1]
        E1 = w00 * F[1, i, j] + w10 * F[1, i + 1, j] + w01 * F[1, i, j + 1] + w11 * F[1, i + 1, j + 1]
        E2 = w00 * F[2, i, j] + w10 * F[2, i + 1, j] + w01 * F[2, i, j + 1] + w11 * F[2, i + 1, j + 1]
        B0 = w00 * F[3, i, j] + w10 * F[3, i + 1, j] + w01 * F[3, i, j + 1] + w11 * F[3, i + 1, j + 1]
        B1 = w00 * F[4, i, j] + w10 * F[4, i + 1, j] + w01 * F[4, i, j + 1] + w11 * F[4, i + 1, j + 1]
        B2 = w00 * F[5, i, j] + w10 * F[5, i + 1, j] + w01 * F[5, i, j + 1] + w11 * F[5, i + 1, j + 1]
        un0, un1, un2 = _boris(u0[p], u1[p], u2[p], E0, E1, E2, B0, B1, B2, h, hand, rel)
        if rel:
            g = math.sqrt(1.0 + (un0 * un0 + un1 * un1 + un2 * un2) * INV_C2)
        else:
            g = 1.0
        z[p] += un0 / g * dt
        if rz:
            xx = y[p] + un1 / g * dt
            yy = un2 / g * dt
            rn = math.sqrt(xx * xx + yy * yy)
            if rn > 0.0:
                cs = xx / rn
                sn = yy / rn
                ur = cs * un1 + sn * un2
                ut = -sn * un1 + cs * un2
                un1 = ur
                un2 = ut
            y[p] = rn
        else:
            y[p] += un1 / g * dt
        u0[p] = un0
        u1[p] = un1
        u2[p] = un2


@njit(parallel=True, fastmath=True, cache=False, nogil=True)
def _push_2d_l_par(z, y, zl, yl, u0, u1, u2, n, F, nz, ny, qm, dt, hand, rz, rel):
    nt = get_num_threads()
    ch = (n + nt - 1) // nt
    for t in prange(nt):
        _push_2d_body_l(t * ch, min(n, (t + 1) * ch), z, y, zl, yl, u0, u1, u2, F, nz, ny, qm, dt, hand,
                        rz, rel)


@njit(fastmath=True, cache=True, nogil=True)
def _push_2d_l_ser(z, y, zl, yl, u0, u1, u2, n, F, nz, ny, qm, dt, hand, rz, rel):
    _push_2d_body_l(0, n, z, y, zl, yl, u0, u1, u2, F, nz, ny, qm, dt, hand, rz, rel)


push_2d_l = _Pair(_push_2d_l_par, _push_2d_l_ser)


# ============================================================================
#  Field interpolation (diagnostics, emission)
# ============================================================================
@njit(cache=True, nogil=True)
def interp_2d(A, zq, yq, zmin, ymin, dz, dy, nz, ny):
    out = np.empty(zq.shape[0])
    for p in range(zq.shape[0]):
        fz = (zq[p] - zmin) / dz
        i = int(math.floor(fz))
        i = min(max(i, 0), nz - 1)
        a = min(max(fz - i, 0.0), 1.0)
        fy = (yq[p] - ymin) / dy
        j = int(math.floor(fy))
        j = min(max(j, 0), ny - 1)
        b = min(max(fy - j, 0.0), 1.0)
        out[p] = ((1 - a) * (1 - b) * A[i, j] + a * (1 - b) * A[i + 1, j]
                  + (1 - a) * b * A[i, j + 1] + a * b * A[i + 1, j + 1])
    return out


@njit(cache=True, nogil=True)
def interp_1d(A, zq, zmin, dz, nz):
    out = np.empty(zq.shape[0])
    for p in range(zq.shape[0]):
        fz = (zq[p] - zmin) / dz
        i = int(math.floor(fz))
        i = min(max(i, 0), nz - 1)
        a = min(max(fz - i, 0.0), 1.0)
        out[p] = (1 - a) * A[i] + a * A[i + 1]
    return out


# ============================================================================
#  Charge deposition  (node charges  sum q w N_i)
# ============================================================================
@_dual
def deposit_1d(z, w, n, q, zmin, dz, nz, nchunk):
    buf = np.zeros((nchunk, nz + 1))
    chunk = (n + nchunk - 1) // nchunk
    for t in prange(nchunk):
        s = t * chunk
        e = min(n, s + chunk)
        for p in range(s, e):
            fz = (z[p] - zmin) / dz
            i = int(fz)
            if i < 0:
                i = 0
            elif i > nz - 1:
                i = nz - 1
            a = fz - i
            qw = q * w[p]
            buf[t, i] += qw * (1.0 - a)
            buf[t, i + 1] += qw * a
    out = np.zeros(nz + 1)
    for t in range(nchunk):
        out += buf[t]
    return out


@_dual
def deposit_2d(z, y, w, n, q, zmin, ymin, dz, dy, nz, ny, nchunk):
    buf = np.zeros((nchunk, nz + 1, ny + 1))
    chunk = (n + nchunk - 1) // nchunk
    for t in prange(nchunk):
        s = t * chunk
        e = min(n, s + chunk)
        for p in range(s, e):
            fz = (z[p] - zmin) / dz
            i = int(fz)
            if i < 0:
                i = 0
            elif i > nz - 1:
                i = nz - 1
            a = fz - i
            fy = (y[p] - ymin) / dy
            j = int(fy)
            if j < 0:
                j = 0
            elif j > ny - 1:
                j = ny - 1
            b = fy - j
            qw = q * w[p]
            buf[t, i, j] += qw * (1.0 - a) * (1.0 - b)
            buf[t, i + 1, j] += qw * a * (1.0 - b)
            buf[t, i, j + 1] += qw * (1.0 - a) * b
            buf[t, i + 1, j + 1] += qw * a * b
    out = np.zeros((nz + 1, ny + 1))
    for t in range(nchunk):
        out += buf[t]
    return out


@_dual
def deposit_2d_weighted(z, y, w, val, n, q, zmin, ymin, dz, dy, nz, ny, nchunk):
    """Deposit q*w*val (e.g. current densities for the ES beam self-field)."""
    buf = np.zeros((nchunk, nz + 1, ny + 1))
    chunk = (n + nchunk - 1) // nchunk
    for t in prange(nchunk):
        s = t * chunk
        e = min(n, s + chunk)
        for p in range(s, e):
            fz = (z[p] - zmin) / dz
            i = min(max(int(fz), 0), nz - 1)
            a = fz - i
            fy = (y[p] - ymin) / dy
            j = min(max(int(fy), 0), ny - 1)
            b = fy - j
            qw = q * w[p] * val[p]
            buf[t, i, j] += qw * (1.0 - a) * (1.0 - b)
            buf[t, i + 1, j] += qw * a * (1.0 - b)
            buf[t, i, j + 1] += qw * (1.0 - a) * b
            buf[t, i + 1, j + 1] += qw * a * b
    out = np.zeros((nz + 1, ny + 1))
    for t in range(nchunk):
        out += buf[t]
    return out


# ============================================================================
#  Charge-conserving current deposition (Esirkepov 2001) in index space
# ============================================================================
@_dual
def esirkepov_1d(zo, zn, u0, u1, u2, w, n, q, zmin, dz, NZ, G, nchunk, rel):
    """Returns (Fz, Q1, Q2):  Fz[k] = charge crossing face k+1/2 during the step,
    Q1/Q2 = sum q w v_{1,2} * (S0+S1)/2 at nodes.  Arrays include G guard nodes."""
    Fz = np.zeros((nchunk, NZ))
    Q1 = np.zeros((nchunk, NZ))
    Q2 = np.zeros((nchunk, NZ))
    chunk = (n + nchunk - 1) // nchunk
    S0 = np.zeros((nchunk, 4))
    S1 = np.zeros((nchunk, 4))
    for t in prange(nchunk):
        s = t * chunk
        e = min(n, s + chunk)
        for p in range(s, e):
            xo = (zo[p] - zmin) / dz + G
            xn = (zn[p] - zmin) / dz + G
            i0 = int(math.floor(xo))
            for k in range(4):
                node = i0 - 1 + k
                S0[t, k] = max(0.0, 1.0 - abs(xo - node))
                S1[t, k] = max(0.0, 1.0 - abs(xn - node))
            qw = q * w[p]
            if rel:
                g = math.sqrt(1.0 + (u0[p] * u0[p] + u1[p] * u1[p] + u2[p] * u2[p]) * INV_C2)
            else:
                g = 1.0
            acc = 0.0
            for k in range(4):
                node = i0 - 1 + k
                if node < 0 or node >= NZ:
                    continue
                acc -= qw * (S1[t, k] - S0[t, k])
                Fz[t, node] += acc
                sh = 0.5 * (S0[t, k] + S1[t, k])
                Q1[t, node] += qw * u1[p] / g * sh
                Q2[t, node] += qw * u2[p] / g * sh
    oF = np.zeros(NZ)
    o1 = np.zeros(NZ)
    o2 = np.zeros(NZ)
    for t in range(nchunk):
        oF += Fz[t]
        o1 += Q1[t]
        o2 += Q2[t]
    return oF, o1, o2


@_dual
def esirkepov_2d(zo, yo, zn, yn, u0, u1, u2, w, n, q, zmin, ymin, dz, dy, NZ, NY, G, nchunk, rel):
    """Charge fluxes through z-faces (Fz[i,j]: face i+1/2 at node row j), y-faces
    (Fy[i,j]: face j+1/2) and out-of-plane moment Q2 = sum q w v2 W at nodes.
    Requires |displacement| < 1 cell per step.  Arrays include G guard nodes."""
    Fz = np.zeros((nchunk, NZ, NY))
    Fy = np.zeros((nchunk, NZ, NY))
    Q2 = np.zeros((nchunk, NZ, NY))
    S0x = np.zeros((nchunk, 4))
    S1x = np.zeros((nchunk, 4))
    S0y = np.zeros((nchunk, 4))
    S1y = np.zeros((nchunk, 4))
    chunk = (n + nchunk - 1) // nchunk
    for t in prange(nchunk):
        s = t * chunk
        e = min(n, s + chunk)
        for p in range(s, e):
            xo = (zo[p] - zmin) / dz + G
            xn = (zn[p] - zmin) / dz + G
            yo_ = (yo[p] - ymin) / dy + G
            yn_ = (yn[p] - ymin) / dy + G
            i0 = int(math.floor(xo))
            j0 = int(math.floor(yo_))
            for k in range(4):
                nd = i0 - 1 + k
                S0x[t, k] = max(0.0, 1.0 - abs(xo - nd))
                S1x[t, k] = max(0.0, 1.0 - abs(xn - nd))
                nd = j0 - 1 + k
                S0y[t, k] = max(0.0, 1.0 - abs(yo_ - nd))
                S1y[t, k] = max(0.0, 1.0 - abs(yn_ - nd))
            qw = q * w[p]
            if rel:
                g = math.sqrt(1.0 + (u0[p] * u0[p] + u1[p] * u1[p] + u2[p] * u2[p]) * INV_C2)
            else:
                g = 1.0
            v2 = u2[p] / g
            # z-fluxes
            for l in range(4):
                jj = j0 - 1 + l
                if jj < 0 or jj >= NY:
                    continue
                dSy = S1y[t, l] - S0y[t, l]
                acc = 0.0
                for k in range(4):
                    ii = i0 - 1 + k
                    dSx = S1x[t, k] - S0x[t, k]
                    acc -= qw * dSx * (S0y[t, l] + 0.5 * dSy)
                    if ii >= 0 and ii < NZ:
                        Fz[t, ii, jj] += acc
            # y-fluxes
            for k in range(4):
                ii = i0 - 1 + k
                if ii < 0 or ii >= NZ:
                    continue
                dSx = S1x[t, k] - S0x[t, k]
                acc = 0.0
                for l in range(4):
                    jj = j0 - 1 + l
                    dSy = S1y[t, l] - S0y[t, l]
                    acc -= qw * dSy * (S0x[t, k] + 0.5 * dSx)
                    if jj >= 0 and jj < NY:
                        Fy[t, ii, jj] += acc
            # out-of-plane
            for k in range(4):
                ii = i0 - 1 + k
                if ii < 0 or ii >= NZ:
                    continue
                dSx = S1x[t, k] - S0x[t, k]
                for l in range(4):
                    jj = j0 - 1 + l
                    if jj < 0 or jj >= NY:
                        continue
                    dSy = S1y[t, l] - S0y[t, l]
                    W = (S0x[t, k] * S0y[t, l] + 0.5 * dSx * S0y[t, l]
                         + 0.5 * S0x[t, k] * dSy + dSx * dSy / 3.0)
                    Q2[t, ii, jj] += qw * v2 * W
    oz = np.zeros((NZ, NY))
    oy = np.zeros((NZ, NY))
    o2 = np.zeros((NZ, NY))
    for t in range(nchunk):
        oz += Fz[t]
        oy += Fy[t]
        o2 += Q2[t]
    return oz, oy, o2


# ============================================================================
#  Particle boundary conditions
# ============================================================================
@njit(cache=True, nogil=True)
def boundary_1d(z, u0, n, zmin, zmax, bc_lo, bc_hi, wrap, flag):
    """bc: 0 absorb, 1 reflect, 2 periodic.  flag: 0 keep, 1 z_min, 2 z_max."""
    L = zmax - zmin
    for p in range(n):
        f = 0
        zz = z[p]
        if zz < zmin:
            if bc_lo == 2:
                if wrap:
                    z[p] = zz + L * math.ceil((zmin - zz) / L)
            elif bc_lo == 1:
                z[p] = 2 * zmin - zz
                u0[p] = -u0[p]
            else:
                f = 1
        elif zz >= zmax:
            if bc_hi == 2:
                if wrap:
                    z[p] = zz - L * (1 + math.floor((zz - zmax) / L))
            elif bc_hi == 1:
                z[p] = 2 * zmax - zz
                u0[p] = -u0[p]
                if z[p] >= zmax:
                    z[p] = zmax - 1e-12 * L
            else:
                f = 2
        flag[p] = f


@njit(cache=True, nogil=True)
def boundary_2d(z, y, u0, u1, n, zmin, zmax, ymin, ymax, bc, wrap, cell_obj, dz, dy, nz, ny,
                has_obj, flag):
    """bc[4] for (z_min, z_max, y_min, y_max): 0 absorb, 1 reflect, 2 periodic, 3 axis.
    flag: 0 keep, 1..4 boundary side, 10+k absorbed by object k."""
    Lz = zmax - zmin
    Ly = ymax - ymin
    for p in range(n):
        f = 0
        zz = z[p]
        if zz < zmin:
            if bc[0] == 2:
                if wrap:
                    z[p] = zz + Lz * math.ceil((zmin - zz) / Lz)
            elif bc[0] == 1:
                z[p] = 2 * zmin - zz
                u0[p] = -u0[p]
            else:
                f = 1
        elif zz >= zmax:
            if bc[1] == 2:
                if wrap:
                    z[p] = zz - Lz * (1 + math.floor((zz - zmax) / Lz))
            elif bc[1] == 1:
                z[p] = 2 * zmax - zz
                u0[p] = -u0[p]
                if z[p] >= zmax:
                    z[p] = zmax - 1e-12 * Lz
            else:
                f = 2
        if f == 0:
            yy = y[p]
            if yy < ymin:
                if bc[2] == 2:
                    if wrap:
                        y[p] = yy + Ly * math.ceil((ymin - yy) / Ly)
                elif bc[2] == 1 or bc[2] == 3:
                    y[p] = 2 * ymin - yy
                    u1[p] = -u1[p]
                else:
                    f = 3
            elif yy >= ymax:
                if bc[3] == 2:
                    if wrap:
                        y[p] = yy - Ly * (1 + math.floor((yy - ymax) / Ly))
                elif bc[3] == 1:
                    y[p] = 2 * ymax - yy
                    u1[p] = -u1[p]
                    if y[p] >= ymax:
                        y[p] = ymax - 1e-12 * Ly
                else:
                    f = 4
        if f == 0 and has_obj:
            i = int((z[p] - zmin) / dz)
            j = int((y[p] - ymin) / dy)
            if i >= 0 and i < nz and j >= 0 and j < ny:
                k = cell_obj[i, j]
                if k >= 0:
                    f = 10 + k
        flag[p] = f


@njit(cache=True, nogil=True)
def clip_to_box_2d(zo, yo, zn, yn, flag, n, zmin, zmax, ymin, ymax):
    """For particles leaving through a boundary, move the end point onto the boundary
    (for charge-conserving current deposition before removal)."""
    for p in range(n):
        f = flag[p]
        if f < 1 or f > 4:
            continue
        dzp = zn[p] - zo[p]
        dyp = yn[p] - yo[p]
        s = 1.0
        if zn[p] < zmin and dzp != 0.0:
            s = min(s, (zmin - zo[p]) / dzp)
        if zn[p] > zmax and dzp != 0.0:
            s = min(s, (zmax - zo[p]) / dzp)
        if yn[p] < ymin and dyp != 0.0:
            s = min(s, (ymin - yo[p]) / dyp)
        if yn[p] > ymax and dyp != 0.0:
            s = min(s, (ymax - yo[p]) / dyp)
        s = max(0.0, min(1.0, s))
        zn[p] = min(max(zo[p] + s * dzp, zmin), zmax)
        yn[p] = min(max(yo[p] + s * dyp, ymin), ymax)


@njit(cache=True, nogil=True)
def clip_to_box_1d(zo, zn, flag, n, zmin, zmax):
    for p in range(n):
        if flag[p] == 1:
            zn[p] = zmin
        elif flag[p] == 2:
            zn[p] = zmax


@_dual
def wrap_periodic(x, n, xmin, xmax):
    L = xmax - xmin
    for p in prange(n):
        v = x[p]
        if v < xmin:
            x[p] = v + L * math.ceil((xmin - v) / L)
            if x[p] >= xmax:
                x[p] = xmin
        elif v >= xmax:
            x[p] = v - L * (1 + math.floor((v - xmax) / L))
            if x[p] < xmin:
                x[p] = xmin


@_dual
def kinetic_energy(u0, u1, u2, w, n, m, rel):
    tot = 0.0
    for p in prange(n):
        uu = u0[p] * u0[p] + u1[p] * u1[p] + u2[p] * u2[p]
        if rel:
            g = math.sqrt(1.0 + uu * INV_C2)
            tot += w[p] * m * uu / (g + 1.0)
        else:
            tot += w[p] * 0.5 * m * uu
    return tot


# ============================================================================
#  WarpX-consistent gather + push (separate steps, as in WarpX PushPX)
#
#  Field gathering:
#    * momentum-conserving: bilinear (order-1) interpolation of node-centred fields;
#    * energy-conserving (WarpX default, `algo.field_gathering = energy-conserving` with
#      `interpolation.galerkin_scheme = 1`): every Yee component is gathered from its own staggered
#      location, with the shape order lowered by one in the staggered direction(s)
#      (WarpX Source/Particles/Gather/FieldGather.H, doGatherShapeN<1, 1>).  For the electrostatic
#      solver this is exactly E = -grad of the bilinear interpolant of phi at the particle.
#  Particle push (WarpX Source/Particles/Pusher): Boris (UpdateMomentumBoris), Vay
#  (UpdateMomentumVay) and Higuera-Cary (UpdateMomentumHigueraCary), followed by UpdatePosition.
#  WarpX is Copyright (c) 2018 The Regents of the University of California (LBNL) and LLNS
#  (BSD-3-Clause-LBNL; full notice in THIRD_PARTY_NOTICES.md); the formulas below follow its
#  implementation.
#  All positions here are logical coordinates (fractional node index, uniform spacing 1).
# ============================================================================
@njit(inline="always")
def _node1(x, nmax):
    """Order-1 shape on nodes: (index, weight of index+1)."""
    i = int(math.floor(x))
    if i < 0:
        i = 0
    elif i > nmax - 1:
        i = nmax - 1
    return i, x - i


@njit(inline="always")
def _cell0(x, ncell):
    """Order-0 (Galerkin-reduced) shape on cell centres: index of the cell containing x."""
    i = int(math.floor(x))
    if i < 0:
        i = 0
    elif i > ncell - 1:
        i = ncell - 1
    return i


@njit(fastmath=True, cache=True, nogil=True, parallel=True)
def gather_nodal_1d(F, zl, n, nz, P):
    for p in prange(n):
        i, a = _node1(zl[p], nz)
        b = 1.0 - a
        for c in range(6):
            P[c, p] = b * F[c, i] + a * F[c, i + 1]


@njit(fastmath=True, cache=True, nogil=True, parallel=True)
def gather_nodal_2d(F, zl, yl, n, nz, ny, P):
    for p in prange(n):
        i, a = _node1(zl[p], nz)
        j, b = _node1(yl[p], ny)
        w00 = (1.0 - a) * (1.0 - b)
        w10 = a * (1.0 - b)
        w01 = (1.0 - a) * b
        w11 = a * b
        for c in range(6):
            P[c, p] = (w00 * F[c, i, j] + w10 * F[c, i + 1, j] + w01 * F[c, i, j + 1]
                       + w11 * F[c, i + 1, j + 1])


@njit(fastmath=True, cache=True, nogil=True, parallel=True)
def gather_ec_es_1d(phi, zl, hz, n, nz, P):
    """Energy-conserving ES gather: Ez = -dphi/dz of the linear interpolant (adds to P[0])."""
    for p in prange(n):
        i = _cell0(zl[p], nz)
        P[0, p] += -(phi[i + 1] - phi[i]) / hz[i]


@njit(fastmath=True, cache=True, nogil=True, parallel=True)
def gather_ec_es_2d(phi, zl, yl, hz, hy, n, nz, ny, P):
    """Energy-conserving ES gather: E = -grad of the bilinear interpolant of phi (adds to P[0:2])."""
    for p in prange(n):
        ic = _cell0(zl[p], nz)
        jc = _cell0(yl[p], ny)
        i, a = _node1(zl[p], nz)
        j, b = _node1(yl[p], ny)
        # Ez: staggered (cell) in z -> order 0; node in y -> order 1
        dz0 = (phi[ic + 1, j] - phi[ic, j]) / hz[ic]
        dz1 = (phi[ic + 1, j + 1] - phi[ic, j + 1]) / hz[ic]
        P[0, p] += -((1.0 - b) * dz0 + b * dz1)
        # Ey: node in z -> order 1; staggered in y -> order 0
        dy0 = (phi[i, jc + 1] - phi[i, jc]) / hy[jc]
        dy1 = (phi[i + 1, jc + 1] - phi[i + 1, jc]) / hy[jc]
        P[1, p] += -((1.0 - a) * dy0 + a * dy1)


@njit(fastmath=True, cache=True, nogil=True, parallel=True)
def gather_ec_yee_1d(E0, E1, E2, B1, B2, zl, n, nz, P):
    """Energy-conserving gather from the 1D Yee grid (adds to P).  E0, B1, B2 at half nodes
    (order 0), E1, E2 at nodes (order 1)."""
    for p in prange(n):
        ic = _cell0(zl[p], nz)
        i, a = _node1(zl[p], nz)
        P[0, p] += E0[ic]
        P[1, p] += (1.0 - a) * E1[i] + a * E1[i + 1]
        P[2, p] += (1.0 - a) * E2[i] + a * E2[i + 1]
        P[4, p] += B1[ic]
        P[5, p] += B2[ic]


@njit(fastmath=True, cache=True, nogil=True, parallel=True)
def gather_ec_yee_2d(E0, E1, E2, B0, B1, B2, zl, yl, n, nz, ny, P):
    """Energy-conserving (Galerkin) gather from the 2D Yee grid (adds to P).
    E0 (i+1/2, j), E1 (i, j+1/2), E2 (i, j), B0 (i, j+1/2), B1 (i+1/2, j), B2 (i+1/2, j+1/2)."""
    for p in prange(n):
        ic = _cell0(zl[p], nz)
        jc = _cell0(yl[p], ny)
        i, a = _node1(zl[p], nz)
        j, b = _node1(yl[p], ny)
        P[0, p] += (1.0 - b) * E0[ic, j] + b * E0[ic, j + 1]
        P[1, p] += (1.0 - a) * E1[i, jc] + a * E1[i + 1, jc]
        P[2, p] += ((1.0 - a) * (1.0 - b) * E2[i, j] + a * (1.0 - b) * E2[i + 1, j]
                    + (1.0 - a) * b * E2[i, j + 1] + a * b * E2[i + 1, j + 1])
        P[3, p] += (1.0 - a) * B0[i, jc] + a * B0[i + 1, jc]
        P[4, p] += (1.0 - b) * B1[ic, j] + b * B1[ic, j + 1]
        P[5, p] += B2[ic, jc]


@njit(inline="always", fastmath=True)
def _vay(ux, uy, uz, Ex, Ey, Ez, Bx, By, Bz, qmdt, ic2):
    econst = qmdt
    bconst = 0.5 * qmdt
    inv_gamma = 1.0 / math.sqrt(1.0 + (ux * ux + uy * uy + uz * uz) * ic2)
    taux = bconst * Bx
    tauy = bconst * By
    tauz = bconst * Bz
    tausq = taux * taux + tauy * tauy + tauz * tauz
    uxpr = ux + econst * Ex + (uy * tauz - uz * tauy) * inv_gamma
    uypr = uy + econst * Ey + (uz * taux - ux * tauz) * inv_gamma
    uzpr = uz + econst * Ez + (ux * tauy - uy * taux) * inv_gamma
    gprsq = 1.0 + (uxpr * uxpr + uypr * uypr + uzpr * uzpr) * ic2
    ust = (uxpr * taux + uypr * tauy + uzpr * tauz) * math.sqrt(ic2)
    sigma = gprsq - tausq
    gisq = 2.0 / (sigma + math.sqrt(sigma * sigma + 4.0 * (tausq + ust * ust)))
    bg = bconst * math.sqrt(gisq)
    tx = bg * Bx
    ty = bg * By
    tz = bg * Bz
    s = 1.0 / (1.0 + tausq * gisq)
    tu = tx * uxpr + ty * uypr + tz * uzpr
    return (s * (uxpr + tx * tu + uypr * tz - uzpr * ty),
            s * (uypr + ty * tu + uzpr * tx - uxpr * tz),
            s * (uzpr + tz * tu + uxpr * ty - uypr * tx))


@njit(inline="always", fastmath=True)
def _higuera(ux, uy, uz, Ex, Ey, Ez, Bx, By, Bz, qmdt, ic2):
    qmt = 0.5 * qmdt
    umx = ux + qmt * Ex
    umy = uy + qmt * Ey
    umz = uz + qmt * Ez
    gamma = 1.0 + (umx * umx + umy * umy + umz * umz) * ic2
    betax = qmt * Bx
    betay = qmt * By
    betaz = qmt * Bz
    betam = betax * betax + betay * betay + betaz * betaz
    sigma = gamma - betam
    ust = (umx * betax + umy * betay + umz * betaz) * math.sqrt(ic2)
    gamma = 1.0 / math.sqrt(0.5 * (sigma + math.sqrt(sigma * sigma + 4.0 * (betam + ust * ust))))
    tx = gamma * betax
    ty = gamma * betay
    tz = gamma * betaz
    s = 1.0 / (1.0 + (tx * tx + ty * ty + tz * tz))
    umt = umx * tx + umy * ty + umz * tz
    upx = s * (umx + umt * tx + umy * tz - umz * ty)
    upy = s * (umy + umt * ty + umz * tx - umx * tz)
    upz = s * (umz + umt * tz + umx * ty - umy * tx)
    return (upx + qmt * Ex + upy * tz - upz * ty,
            upy + qmt * Ey + upz * tx - upx * tz,
            upz + qmt * Ez + upx * ty - upy * tx)


@njit(inline="always", fastmath=True)
def _momentum(u0, u1, u2, E0, E1, E2, B0, B1, B2, qm, dt, hand, ic2, pusher):
    """One momentum update with WarpX's pushers; hand = +-1 orientation of the (z, y, x) basis."""
    if pusher == 1:
        return _vay(u0, u1, u2, E0, E1, E2, hand * B0, hand * B1, hand * B2, qm * dt, ic2)
    if pusher == 2:
        return _higuera(u0, u1, u2, E0, E1, E2, hand * B0, hand * B1, hand * B2, qm * dt, ic2)
    # Boris (UpdateMomentumBoris, full push)
    econst = 0.5 * qm * dt
    u0 += econst * E0
    u1 += econst * E1
    u2 += econst * E2
    inv_gamma = 1.0 / math.sqrt(1.0 + (u0 * u0 + u1 * u1 + u2 * u2) * ic2)
    tx = econst * inv_gamma * hand * B0
    ty = econst * inv_gamma * hand * B1
    tz = econst * inv_gamma * hand * B2
    tsqi = 2.0 / (1.0 + tx * tx + ty * ty + tz * tz)
    sx = tx * tsqi
    sy = ty * tsqi
    sz = tz * tsqi
    uxp = u0 + u1 * tz - u2 * ty
    uyp = u1 + u2 * tx - u0 * tz
    uzp = u2 + u0 * ty - u1 * tx
    u0 += uyp * sz - uzp * sy
    u1 += uzp * sx - uxp * sz
    u2 += uxp * sy - uyp * sx
    return u0 + econst * E0, u1 + econst * E1, u2 + econst * E2


@njit(fastmath=True, cache=True, nogil=True, parallel=True)
def push_fields_1d(z, u0, u1, u2, n, P, qm, dt, hand, ic2, pusher, move):
    for p in prange(n):
        a0, a1, a2 = _momentum(u0[p], u1[p], u2[p], P[0, p], P[1, p], P[2, p], P[3, p], P[4, p],
                               P[5, p], qm, dt, hand, ic2, pusher)
        u0[p] = a0
        u1[p] = a1
        u2[p] = a2
        if move:
            g = math.sqrt(1.0 + (a0 * a0 + a1 * a1 + a2 * a2) * ic2)
            z[p] += a0 / g * dt


@njit(fastmath=True, cache=True, nogil=True, parallel=True)
def push_fields_2d(z, y, u0, u1, u2, n, P, qm, dt, hand, rz, ic2, pusher, move):
    for p in prange(n):
        a0, a1, a2 = _momentum(u0[p], u1[p], u2[p], P[0, p], P[1, p], P[2, p], P[3, p], P[4, p],
                               P[5, p], qm, dt, hand, ic2, pusher)
        if move:
            g = math.sqrt(1.0 + (a0 * a0 + a1 * a1 + a2 * a2) * ic2)
            z[p] += a0 / g * dt
            if rz:
                # RZ: move in the local Cartesian frame, then rotate (u_r, u_theta) to the new position
                xx = y[p] + a1 / g * dt
                yy = a2 / g * dt
                rn = math.sqrt(xx * xx + yy * yy)
                if rn > 0.0:
                    cs = xx / rn
                    sn = yy / rn
                    ur = cs * a1 + sn * a2
                    ut = -sn * a1 + cs * a2
                    a1 = ur
                    a2 = ut
                y[p] = rn
            else:
                y[p] += a1 / g * dt
        u0[p] = a0
        u1[p] = a1
        u2[p] = a2
