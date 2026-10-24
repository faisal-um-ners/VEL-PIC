"""Read simulation results written by the engine."""
from __future__ import annotations

import glob
import io
import json
import os
import re

import numpy as np


def load_npz(path):
    """Read an NPZ file into a dict and close it at once.

    The whole file is read into memory first, so the handle is held only for a moment.  On Windows a
    file that is open in another process cannot be replaced; this keeps the GUI from blocking the
    solver when it rewrites live.npz."""
    with open(path, "rb") as fh:
        buf = fh.read()
    with np.load(io.BytesIO(buf), allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


class Results:
    def __init__(self, out_dir):
        self.dir = out_dir
        self.info = {}
        p = os.path.join(out_dir, "run_info.json")
        if os.path.exists(p):
            with open(p) as fh:
                self.info = json.load(fh)
        self.refresh()

    def refresh(self):
        # Atomic writes use e.g. field_0000001.npz.tmp.npz until complete.
        # A broad *.npz glob also sees those files, which disappear on rename.
        self.field_files = self._snapshot_files("fields", "field")
        self.particle_files = self._snapshot_files("particles", "part")

    def _snapshot_files(self, subdir, prefix):
        pattern = re.compile(rf"{prefix}_[0-9]+\.npz")
        return sorted(path for path in glob.glob(os.path.join(self.dir, subdir, f"{prefix}_*.npz"))
                      if pattern.fullmatch(os.path.basename(path)))

    @property
    def geometry(self):
        return self.info.get("geometry", "2D-ZY")

    def history(self):
        p = os.path.join(self.dir, "history.csv")
        if not os.path.exists(p):
            return None, []
        with open(p) as fh:
            cols = fh.readline().strip().split(",")
        try:
            data = np.loadtxt(p, delimiter=",", skiprows=1, ndmin=2)
        except ValueError:
            data = np.zeros((0, len(cols)))
        return data, cols

    def geometry_data(self):
        p = os.path.join(self.dir, "geometry.npz")
        return load_npz(p) if os.path.exists(p) else None

    def field(self, i=-1):
        if not self.field_files:
            return None
        d = load_npz(self.field_files[i])
        if "cell_mat" not in d:                  # runs made before v1.2.1: material map from geometry.npz
            if not hasattr(self, "_cell_mat"):
                g = self.geometry_data()
                self._cell_mat = None if g is None or "cell_mat" not in g else np.asarray(g["cell_mat"])
            if self._cell_mat is not None:
                d["cell_mat"] = self._cell_mat
        return d

    def particles(self, i=-1):
        if not self.particle_files:
            return None
        return load_npz(self.particle_files[i])

    def live(self):
        p = os.path.join(self.dir, "live.npz")
        if not os.path.exists(p):
            return None
        try:
            return load_npz(p)
        except Exception:  # noqa: BLE001  (file being replaced or still locked)
            return None


def species_names(pdata):
    names = []
    for k in pdata:
        if "__" in k:
            s = k.split("__")[0]
            if s not in names:
                names.append(s)
    return names


def beam_moments(pdata, species, geometry, nbins=50, zrange=None):
    """Slice-wise beam diagnostics along z: current-weighted rms size, rms emittance
    (normalized, m rad) and mean kinetic energy per particle (eV)."""
    from .. import constants as C
    z = pdata[f"{species}__z"]
    if len(z) == 0:
        return None
    y = pdata[f"{species}__y"]
    u0, u1, u2 = pdata[f"{species}__u0"], pdata[f"{species}__u1"], pdata[f"{species}__u2"]
    w = pdata[f"{species}__w"]
    zr = zrange or (z.min(), z.max())
    edges = np.linspace(zr[0], zr[1], nbins + 1)
    idx = np.clip(np.digitize(z, edges) - 1, 0, nbins - 1)
    out = {"z": 0.5 * (edges[1:] + edges[:-1]), "rms": np.zeros(nbins), "emit_n": np.zeros(nbins),
           "count": np.zeros(nbins)}
    for b in range(nbins):
        s = idx == b
        if s.sum() < 5:
            continue
        ww = w[s]
        if geometry == "2D-RZ":
            # x = r (theta=0 plane projection): <x^2> = <r^2>/2 ; x' from u_r, u_theta
            r = y[s]
            x2 = np.average(r ** 2, weights=ww) / 2
            ux2 = np.average(u1[s] ** 2 + u2[s] ** 2, weights=ww) / 2
            xux = np.average(r * u1[s], weights=ww) / 2
        else:
            x = y[s] - np.average(y[s], weights=ww)
            ux = u1[s] - np.average(u1[s], weights=ww)
            x2 = np.average(x ** 2, weights=ww)
            ux2 = np.average(ux ** 2, weights=ww)
            xux = np.average(x * ux, weights=ww)
        out["rms"][b] = np.sqrt(x2)
        out["emit_n"][b] = np.sqrt(max(x2 * ux2 - xux ** 2, 0.0)) / C.c
        out["count"][b] = s.sum()
    return out
