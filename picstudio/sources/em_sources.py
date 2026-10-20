"""Electromagnetic wave sources: current-sheet antennas (laser, plane wave, TEM port).

A sheet of surface current K placed at z = z_s radiates waves E = -eta0*K/2 to
both sides.  We impose  K = -2*eps0*c*E_target(y, t)  so the forward wave has
the requested field; the backward wave should be absorbed by an absorbing
boundary behind the antenna (place the antenna a few cells inside it).
"""
from __future__ import annotations

import numpy as np

from .. import constants as C


class WaveSource:
    def __init__(self, spec, grid, yee, ev):
        self.spec = spec
        self.name = spec["name"]
        self.g = g = grid
        self.type = spec["type"]
        self.ev = ev
        self.i_s = yee.source_plane(ev.num(spec["z_position"]))
        self.sign = 1.0 if spec.get("direction", "+z") == "+z" else -1.0
        pol = spec.get("polarization", "x")
        if self.type == "tem_port" or (g.rz and self.type != "tem_port"):
            pol = "r" if g.rz else "y"
        self.comp = 2 if pol == "x" else 1
        if g.dim == 1:
            self.ycoord = None
        elif self.comp == 2:
            self.ycoord = g.y.copy()
        else:
            self.ycoord = g.y_min + (np.arange(g.ny) + 0.5) * g.dy
        if self.type in ("laser", "plane_wave"):
            lam = ev.num(spec["wavelength"])
            self.omega = 2 * np.pi * C.c / lam
            self.k = 2 * np.pi / lam
            self.E0 = ev.num(spec["a0"]) * C.me * C.c * self.omega / C.e
        if self.type == "laser":
            self.tau = ev.num(spec["duration"])
            self.tpk = ev.num(spec["t_peak"])
            self.w0 = ev.num(spec.get("waist", 0))
            self.yc = ev.num(spec.get("y_center", 0))
            self.zf = ev.num(spec.get("focus_z", 0))
        elif self.type == "plane_wave":
            self.ramp = ev.num(spec.get("ramp", 0))
        else:
            self.V = ev.func_t(spec.get("voltage", "0"))
            self.a = ev.num(spec["inner"])
            self.b = ev.num(spec["outer"])

    def target(self, t):
        """Target field of the forward wave at the antenna plane, time t."""
        g = self.g
        if self.type == "tem_port":
            V = self.V(t)
            y = self.ycoord
            if y is None:
                return V / max(self.b - self.a, 1e-30)
            inside = (y >= self.a) & (y <= self.b)
            if g.rz:
                return np.where(inside, V / (np.maximum(y, 1e-30) * np.log(self.b / self.a)), 0.0)
            return np.where(inside, V / (self.b - self.a), 0.0)
        if self.type == "plane_wave":
            env = min(t / self.ramp, 1.0) if self.ramp > 0 else 1.0
            val = self.E0 * env * np.sin(self.omega * t)
            return val if self.ycoord is None else np.full(self.ycoord.shape, val)
        # laser
        env = np.exp(-2 * np.log(2) * ((t - self.tpk) / self.tau) ** 2)
        if self.ycoord is None or self.w0 <= 0:
            val = self.E0 * env * np.sin(self.omega * (t - self.tpk))
            return val if self.ycoord is None else np.full(self.ycoord.shape, val)
        y = self.ycoord - self.yc
        zR = self.k * self.w0 ** 2 / 2
        zeta = self.sign * (self.g.z[self.i_s] - self.zf)
        w = self.w0 * np.sqrt(1 + (zeta / zR) ** 2)
        Rinv = zeta / (zeta ** 2 + zR ** 2)
        gouy = np.arctan(zeta / zR)
        if g.rz:
            amp = self.E0 * (self.w0 / w) ** 2 * (np.sqrt(2) * self.ycoord / w)
            gouy = 2 * gouy
        else:
            amp = self.E0 * np.sqrt(self.w0 / w)
            gouy = 0.5 * gouy
        phase = self.omega * (t - self.tpk) - 0.5 * self.k * y ** 2 * Rinv + gouy
        return amp * np.exp(-(y / w) ** 2) * env * np.sin(phase)

    def apply(self, yee, t_half):
        Et = self.target(t_half)
        J = -2 * C.eps0 * C.c * Et / self.g.dz
        if self.g.dim == 1:
            if self.comp == 2:
                yee.J2[self.i_s] += J
            else:
                yee.J1[self.i_s] += J
        else:
            if self.comp == 2:
                yee.J2[self.i_s, :] += J
            else:
                yee.J1[self.i_s, :] += J
