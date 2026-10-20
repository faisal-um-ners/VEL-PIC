"""Applied (external) electric and magnetic fields evaluated on mesh nodes."""
from __future__ import annotations

import numpy as np
from scipy.special import ellipe, ellipk

from ..constants import mu0


def loop_field(z, r, zc, a, current):
    """Bz, Br of a circular current loop of radius a at z = zc (exact, elliptic integrals)."""
    z = np.asarray(z, float)
    r = np.abs(np.asarray(r, float))
    dz = z - zc
    den = (a + r) ** 2 + dz ** 2
    m = np.clip(4 * a * r / den, 0.0, 1 - 1e-12)
    K, E = ellipk(m), ellipe(m)
    q = (a - r) ** 2 + dz ** 2
    q = np.maximum(q, 1e-30)
    pre = mu0 * current / (2 * np.pi * np.sqrt(den))
    Bz = pre * (K + (a * a - r * r - dz * dz) / q * E)
    with np.errstate(divide="ignore", invalid="ignore"):
        Br = np.where(r > 1e-14 * a, pre * dz / r * (-K + (a * a + r * r + dz * dz) / q * E), 0.0)
    return Bz, Br


class ExternalFields:
    """Sum of all applied fields.  ``nodal(t)`` returns (6, *grid.shape) or None."""

    def __init__(self, items, grid, ev):
        self.grid = grid
        self.ev = ev
        self.static = np.zeros((6,) + grid.shape)
        self.dynamic = []        # (spatial array (6,...), f(t)) or expression evaluators
        self.expr_items = []
        Z, Y = grid.nodes()
        if Y is None:
            Y = np.zeros_like(Z)
        self.Z, self.Y = Z, Y
        self.any = False
        self.uniform_B = np.zeros(3)
        for it in items:
            typ = it.get("type", "uniform")
            f_t = ev.func_t(it.get("time_factor", "1"), 1.0)
            td = ev.is_time_dependent(it.get("time_factor", "1"))
            arr = np.zeros((6,) + grid.shape)
            if typ == "uniform":
                E = [ev.num(v) for v in it.get("E", [0, 0, 0])]
                B = [ev.num(v) for v in it.get("B", [0, 0, 0])]
                for c in range(3):
                    arr[c] = E[c]
                    arr[3 + c] = B[c]
            elif typ in ("coil", "solenoid"):
                a = ev.num(it.get("radius"))
                NI = ev.num(it.get("ampere_turns"))
                z0 = ev.num(it.get("zc"))
                if typ == "coil":
                    zs = [z0]
                else:
                    z1 = ev.num(it.get("z_end"))
                    nloop = max(10, int(abs(z1 - z0) / (0.05 * a)) + 1)
                    zs = np.linspace(z0, z1, nloop)
                Ieach = NI / len(zs)
                for zc in zs:
                    Bz, Br = loop_field(Z, Y, zc, a, Ieach)
                    arr[3] += Bz
                    arr[4] += Br
            elif typ == "expression":
                keys = ("Ez", "Ey", "Ex", "Bz", "By", "Bx")
                srcs = [str(it.get(k, "0") or "0") for k in keys]
                if any(ev.is_time_dependent(s) for s in srcs):
                    self.expr_items.append((srcs, f_t))
                    self.any = True
                    continue
                for c, s in enumerate(srcs):
                    arr[c] = ev.func_space(s)(Z, Y, 0.0)
            self.any = True
            if td:
                self.dynamic.append((arr, f_t))
            else:
                self.static += arr * f_t(0.0)

    def nodal(self, t):
        if not self.any:
            return None
        out = self.static.copy()
        for arr, f in self.dynamic:
            out += arr * f(t)
        for srcs, f in self.expr_items:
            ft = f(t)
            for c, s in enumerate(srcs):
                if s.strip() not in ("0", ""):
                    out[c] += ft * self.ev.func_space(s)(self.Z, self.Y, t)
        return out

    @property
    def time_dependent(self):
        return bool(self.dynamic or self.expr_items)

    def max_B(self):
        f = self.nodal(0.0)
        if f is None:
            return 0.0
        return float(np.sqrt((f[3:] ** 2).sum(axis=0)).max())
