"""Particle species container (structure of arrays with amortised growth)."""
from __future__ import annotations

import numpy as np

from .. import constants as C

_FIELDS = ("z", "y", "u0", "u1", "u2", "w")


class Species:
    def __init__(self, spec: dict, ev, index: int = 0, capacity: int = 4096):
        self.spec = spec
        self.name = spec["name"]
        self.index = index
        preset = spec.get("preset", "custom")
        if preset in C.SPECIES_PRESETS and preset != "custom":
            qe, m = C.SPECIES_PRESETS[preset]
        else:
            qe, m = ev.num(spec.get("charge", -1)), ev.num(spec.get("mass", C.me))
        self.q = qe * C.e
        self.m = m
        self.qm = self.q / self.m
        self.beam_self_B = bool(spec.get("beam_self_B", False))
        self.z_only = spec.get("motion", "full") == "z_only"
        self.y_only = spec.get("motion", "full") == "y_only"     # infinite B along y: motion along y only
        self.n = 0
        self._cap = 0
        for f in _FIELDS:
            setattr(self, "_" + f, np.zeros(0))
        self._grow(capacity)

    # ------------------------------------------------------------------
    def _grow(self, need):
        cap = max(need, int(self._cap * 1.5) + 16)
        for f in _FIELDS:
            old = getattr(self, "_" + f)
            new = np.zeros(cap)
            new[: self.n] = old[: self.n]
            setattr(self, "_" + f, new)
        self._cap = cap

    def add(self, z, y, u0, u1, u2, w):
        k = len(z)
        if k == 0:
            return
        if self.n + k > self._cap:
            self._grow(self.n + k)
        s = slice(self.n, self.n + k)
        self._z[s] = z
        self._y[s] = y
        self._u0[s] = u0
        self._u1[s] = u1
        self._u2[s] = u2
        self._w[s] = w
        self.n += k

    def keep(self, mask):
        """Keep particles where mask (length n) is True (stable compaction)."""
        n = self.n
        if mask.all():
            return
        idx = np.nonzero(mask)[0]
        k = len(idx)
        for f in _FIELDS:
            a = getattr(self, "_" + f)
            a[:k] = a[:n][idx]
        self.n = k

    # views on the active particles --------------------------------------
    @property
    def z(self):
        return self._z[: self.n]

    @property
    def y(self):
        return self._y[: self.n]

    @property
    def u0(self):
        return self._u0[: self.n]

    @property
    def u1(self):
        return self._u1[: self.n]

    @property
    def u2(self):
        return self._u2[: self.n]

    @property
    def w(self):
        return self._w[: self.n]

    def arrays(self):
        return self._z, self._y, self._u0, self._u1, self._u2, self._w

    def gamma(self, relativistic=True):
        if not relativistic:
            return np.ones(self.n)
        uu = self.u0 ** 2 + self.u1 ** 2 + self.u2 ** 2
        return np.sqrt(1.0 + uu / C.c ** 2)

    def snapshot(self, nmax, rng, relativistic=True):
        n = self.n
        if n > nmax:
            idx = np.sort(rng.choice(n, nmax, replace=False))
        else:
            idx = slice(None)
        return {f: getattr(self, f)[idx].copy() for f in _FIELDS}
