"""Geometric primitives used to paint conductors / dielectrics on the mesh."""
from __future__ import annotations

import numpy as np

from ..expr import Evaluator


def parse_points(text: str, ev: Evaluator):
    pts = []
    for chunk in str(text).replace("\n", ";").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        a, b = chunk.split(",")
        pts.append((ev.num(a), ev.num(b)))
    return np.array(pts, dtype=float)


def point_in_polygon(z, y, poly):
    """Vectorized even-odd rule."""
    z = np.asarray(z, float)
    y = np.asarray(y, float)
    inside = np.zeros(np.broadcast(z, y).shape, dtype=bool)
    n = len(poly)
    for i in range(n):
        z1, y1 = poly[i]
        z2, y2 = poly[(i + 1) % n]
        cond = (y1 > y) != (y2 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            zint = (z2 - z1) * (y - y1) / (y2 - y1 + 1e-300) + z1
        inside ^= cond & (z < zint)
    return inside


class Shape:
    def __init__(self, obj: dict, ev: Evaluator, tol: float = 0.0):
        self.obj = obj
        self.kind = obj.get("shape", "rectangle")
        self.tol = tol
        g = lambda k: ev.num(obj.get(k))  # noqa: E731
        k = self.kind
        if k == "rectangle":
            z0, z1, y0, y1 = g("z0"), g("z1"), g("y0"), g("y1")
            self.p = (min(z0, z1), max(z0, z1), min(y0, y1), max(y0, y1))
        elif k == "circle":
            self.p = (g("zc"), g("yc"), g("radius"))
        elif k == "annulus":
            self.p = (g("zc"), g("yc"), g("radius_inner"), g("radius"))
        elif k == "ellipse":
            self.p = (g("zc"), g("yc"), g("a"), g("b"))
        elif k == "polygon":
            self.p = parse_points(obj.get("points", ""), ev)
        elif k == "expression":
            self.f = ev.func_space(obj.get("expression", "0"))
        else:
            raise ValueError(f"Unknown shape '{k}'")

    def contains(self, z, y):
        z = np.asarray(z, float)
        y = np.asarray(y, float)
        t = self.tol
        k = self.kind
        if k == "rectangle":
            z0, z1, y0, y1 = self.p
            return (z >= z0 - t) & (z <= z1 + t) & (y >= y0 - t) & (y <= y1 + t)
        if k == "circle":
            zc, yc, r = self.p
            return (z - zc) ** 2 + (y - yc) ** 2 <= (r + t) ** 2
        if k == "annulus":
            zc, yc, ri, ro = self.p
            d2 = (z - zc) ** 2 + (y - yc) ** 2
            return (d2 <= (ro + t) ** 2) & (d2 >= max(ri - t, 0.0) ** 2)
        if k == "ellipse":
            zc, yc, a, b = self.p
            return ((z - zc) / (a + t)) ** 2 + ((y - yc) / (b + t)) ** 2 <= 1.0
        if k == "polygon":
            return point_in_polygon(z, y, self.p)
        if k == "expression":
            return self.f(z, y).astype(bool)
        return np.zeros(np.broadcast(z, y).shape, bool)

    def outline(self, n=200):
        """Return a list of (z, y) polylines for drawing."""
        k = self.kind
        th = np.linspace(0, 2 * np.pi, n)
        if k == "rectangle":
            z0, z1, y0, y1 = self.p
            return [np.array([[z0, y0], [z1, y0], [z1, y1], [z0, y1], [z0, y0]])]
        if k == "circle":
            zc, yc, r = self.p
            return [np.c_[zc + r * np.cos(th), yc + r * np.sin(th)]]
        if k == "annulus":
            zc, yc, ri, ro = self.p
            return [np.c_[zc + ro * np.cos(th), yc + ro * np.sin(th)],
                    np.c_[zc + ri * np.cos(th), yc + ri * np.sin(th)]]
        if k == "ellipse":
            zc, yc, a, b = self.p
            return [np.c_[zc + a * np.cos(th), yc + b * np.sin(th)]]
        if k == "polygon":
            return [np.vstack([self.p, self.p[:1]])]
        return []
