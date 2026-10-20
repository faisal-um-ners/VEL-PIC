"""1D node distributions for tensor-product meshes with local refinement.

A mesh axis is described by a background cell size h0 = L / n and a list of refinement regions
[a, b] with a target cell size h_r.  The local size is

    h(x) = min( h0,  min_r [ h_r + (growth - 1) * dist(x, [a_r, b_r]) ] ),

i.e. the cells grow by roughly the factor ``growth`` per cell when leaving a refined region (the usual
"geometric grading" of structured-mesh tools).  Region edges and the domain ends are always nodes, so
an emitter edge or an electrode surface can be placed exactly on the mesh.  Inside every interval
between such break points the nodes follow the equidistribution of 1/h(x).
"""
from __future__ import annotations

import numpy as np


def size_function(x, h0, regions, growth=1.2):
    h = np.full_like(np.asarray(x, float), float(h0))
    for (a, b, hr, g) in regions:
        g = growth if g is None else g
        dist = np.maximum(0.0, np.maximum(a - x, x - b))
        h = np.minimum(h, hr + (g - 1.0) * dist)
    return np.maximum(h, 1e-300)


def build_nodes(lo, hi, n, regions=(), growth=1.2, breaks=()):
    """Node coordinates on [lo, hi].

    n        number of cells of the background (uniform) mesh, h0 = (hi - lo) / n
    regions  iterable of (a, b, h, growth_or_None): refine [a, b] to cell size h
    """
    lo, hi = float(lo), float(hi)
    n = max(1, int(round(n)))
    regs = []
    for (a, b, h, g) in regions:
        a, b = sorted((float(a), float(b)))
        a, b = max(a, lo), min(b, hi)
        if b < a or h <= 0:
            continue
        regs.append((a, b, float(h), g))
    if not regs:
        return np.linspace(lo, hi, n + 1)
    h0 = (hi - lo) / n
    extra = [float(b) for b in breaks if lo < float(b) < hi]
    brk = sorted({lo, hi, *[r[0] for r in regs], *[r[1] for r in regs], *extra})
    # merge break points closer than a tenth of the local size (avoid sliver cells)
    merged = [brk[0]]
    for b in brk[1:]:
        if b - merged[-1] < 0.1 * float(size_function(np.array([b]), h0, regs, growth)[0]):
            if b == hi:
                merged[-1] = hi
            continue
        merged.append(b)
    brk = merged
    nodes = [lo]
    for x0, x1 in zip(brk[:-1], brk[1:]):
        if x1 - x0 <= 1e-12 * (hi - lo):
            continue
        xs = np.linspace(x0, x1, 4001)
        inv = 1.0 / size_function(xs, h0, regs, growth)
        F = np.concatenate([[0.0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(xs))])
        m = max(1, int(round(F[-1])))
        targets = np.linspace(0.0, F[-1], m + 1)
        seg = np.interp(targets, F, xs)
        seg[0], seg[-1] = x0, x1
        nodes.extend(seg[1:].tolist())
    return np.asarray(nodes)


def axis_nodes(ev, dom, refinements, axis, breaks=()):
    """Nodes for axis 'z' or 'y' from the model's domain + mesh_refinements entries."""
    if axis == "z":
        lo, hi, n = ev.num(dom["z_min"]), ev.num(dom["z_max"]), ev.num(dom["nz"])
    else:
        lo, hi, n = ev.num(dom["y_min"]), ev.num(dom["y_max"]), ev.num(dom["ny"])
    regs = []
    for r in refinements or []:
        if not r.get("enabled", True) or r.get("axis", "z") != axis:
            continue
        regs.append((ev.num(r["start"]), ev.num(r["end"]), ev.num(r["size"]), ev.num(r.get("growth", 1.2))))
    return build_nodes(lo, hi, n, regs, breaks=breaks)


def geometry_breaks(m, ev, axis):
    """Coordinates that should be mesh nodes on a refined axis: emitter segment ends and the edges of
    rectangular objects (so emitter edges and electrode surfaces conform to the mesh)."""
    out = []
    keys = ("z0", "z1") if axis == "z" else ("y0", "y1")
    for e in m.get("emitters", []):
        if e.get("surface", "segment") == "segment":
            for k in keys:
                try:
                    out.append(ev.num(e.get(k, "0")))
                except Exception:  # noqa: BLE001
                    pass
    for o in m.get("objects", []):
        if o.get("shape") == "rectangle":
            for k in keys:
                try:
                    out.append(ev.num(o.get(k, "0")))
                except Exception:  # noqa: BLE001
                    pass
    return out
