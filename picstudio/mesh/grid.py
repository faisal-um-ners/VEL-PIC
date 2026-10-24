"""Structured tensor-product mesh and material painting.

Nodes are indexed [iz] (1D) or [iz, iy] (2D).  For RZ, y is the radius.
The same nodes are the degrees of freedom of the bilinear (Q1) finite
elements used by the scikit-fem Poisson solver, so particle shape
functions, the FEM basis and the Yee-grid nodes all coincide.
"""
from __future__ import annotations

import numpy as np

from ..geometry.shapes import Shape

VACUUM, CONDUCTOR, DIELECTRIC, ABSORBER = 0, 1, 2, 3
MATERIAL_CODES = {"vacuum": VACUUM, "lossy": VACUUM, "conductor": CONDUCTOR, "dielectric": DIELECTRIC,
                  "absorber": ABSORBER}


def conductor_interior_nodes(cell_mat):
    """Nodes strictly inside conductors: every cell touching the node is a conductor cell.

    Fields are not defined there (PEC), so diagnostics hide them.  Nodes on a conductor surface
    (touching at least one non-conductor cell) carry the physical surface field and are kept."""
    c = np.asarray(cell_mat) == CONDUCTOR
    if c.ndim == 1:
        p = np.pad(c, 1, constant_values=True)          # outside the domain does not count
        return p[:-1] & p[1:]
    p = np.pad(c, 1, constant_values=True)
    return p[:-1, :-1] & p[1:, :-1] & p[:-1, 1:] & p[1:, 1:]


class Grid:
    """Rectilinear (tensor-product) grid.  Uniform by default; ``z_nodes`` / ``y_nodes`` give a
    graded / locally refined grid (electrostatic solver).  Particle kernels work in *logical*
    coordinates (fractional node index), in which every grid is uniform with spacing 1: the bilinear
    weights computed there are exactly the Q1 shape functions of the physical rectilinear cell."""

    def __init__(self, geometry, z_min, z_max, nz, y_min=0.0, y_max=1.0, ny=1,
                 axis_correction=True, z_nodes=None, y_nodes=None):
        self.geometry = geometry
        self.dim = 1 if geometry == "1D-Z" else 2
        self.rz = geometry == "2D-RZ"
        if z_nodes is not None:
            self.z = np.asarray(z_nodes, float)
            self.nz = len(self.z) - 1
        else:
            self.nz = int(round(nz))
            self.z = np.linspace(float(z_min), float(z_max), self.nz + 1)
        self.z_min, self.z_max = float(self.z[0]), float(self.z[-1])
        self.hz = np.diff(self.z)
        self.dz = (self.z_max - self.z_min) / self.nz
        if self.dim == 2:
            if y_nodes is not None:
                self.y = np.asarray(y_nodes, float)
                self.ny = len(self.y) - 1
            else:
                self.ny = int(round(ny))
                self.y = np.linspace(float(y_min), float(y_max), self.ny + 1)
            self.y_min, self.y_max = float(self.y[0]), float(self.y[-1])
            self.hy = np.diff(self.y)
            self.dy = (self.y_max - self.y_min) / self.ny
            self.shape = (self.nz + 1, self.ny + 1)
        else:
            self.ny = 0
            self.y_min, self.y_max = 0.0, 1.0
            self.y = np.array([0.0])
            self.hy = np.array([1.0])
            self.dy = 1.0
            self.shape = (self.nz + 1,)
        tolz = 1e-9 * self.dz
        self.uniform_z = bool(np.all(np.abs(self.hz - self.dz) < tolz))
        self.uniform_y = bool(np.all(np.abs(self.hy - self.dy) < 1e-9 * self.dy))
        self.uniform = self.uniform_z and self.uniform_y
        self.hmin = float(min(self.hz.min(), self.hy.min() if self.dim == 2 else np.inf))
        self.on_axis = self.rz and abs(self.y_min) < 1e-14
        self.axis_correction = axis_correction
        self.nnodes = int(np.prod(self.shape))
        self._iz = np.arange(self.nz + 1, dtype=float)
        self._iy = np.arange(self.ny + 1, dtype=float) if self.dim == 2 else np.zeros(1)
        self._volumes()

    # ------------------------------------------------------------------
    # logical <-> physical coordinates (linear extrapolation outside the domain)
    @staticmethod
    def _to_l(x, nodes, h, idx, uniform, x0, dx):
        x = np.asarray(x, float)
        if uniform:
            return (x - x0) / dx
        out = np.interp(x, nodes, idx)
        lo, hi = x < nodes[0], x > nodes[-1]
        if np.any(lo):
            out = np.where(lo, (x - nodes[0]) / h[0], out)
        if np.any(hi):
            out = np.where(hi, idx[-1] + (x - nodes[-1]) / h[-1], out)
        return out

    @staticmethod
    def _to_p(l, nodes, h, idx, uniform, x0, dx):
        l = np.asarray(l, float)
        if uniform:
            return x0 + l * dx
        out = np.interp(l, idx, nodes)
        lo, hi = l < 0, l > idx[-1]
        if np.any(lo):
            out = np.where(lo, nodes[0] + l * h[0], out)
        if np.any(hi):
            out = np.where(hi, nodes[-1] + (l - idx[-1]) * h[-1], out)
        return out

    def lz(self, z):
        return self._to_l(z, self.z, self.hz, self._iz, self.uniform_z, self.z_min, self.dz)

    def ly(self, y):
        return self._to_l(y, self.y, self.hy, self._iy, self.uniform_y, self.y_min, self.dy)

    def pz(self, l):
        return self._to_p(l, self.z, self.hz, self._iz, self.uniform_z, self.z_min, self.dz)

    def py(self, l):
        return self._to_p(l, self.y, self.hy, self._iy, self.uniform_y, self.y_min, self.dy)

    def cell_iz(self, z):
        return np.clip(np.floor(self.lz(z)).astype(int), 0, self.nz - 1)

    def cell_iy(self, y):
        return np.clip(np.floor(self.ly(y)).astype(int), 0, self.ny - 1)

    def node_iz(self, z):
        return np.rint(self.lz(z)).astype(int)

    def node_iy(self, y):
        return np.rint(self.ly(y)).astype(int)

    def interp(self, A, z, y=None):
        """Bilinear (linear in 1D) interpolation of a nodal array at physical points."""
        from ..particles.kernels import interp_1d, interp_2d
        z = np.ascontiguousarray(np.atleast_1d(z), dtype=float)
        if self.dim == 1:
            if self.uniform:
                return interp_1d(A, z, self.z_min, self.dz, self.nz)
            return interp_1d(A, np.ascontiguousarray(self.lz(z)), 0.0, 1.0, self.nz)
        y = np.ascontiguousarray(np.atleast_1d(y), dtype=float)
        if self.uniform:
            return interp_2d(A, z, y, self.z_min, self.y_min, self.dz, self.dy, self.nz, self.ny)
        return interp_2d(A, np.ascontiguousarray(self.lz(z)), np.ascontiguousarray(self.ly(y)), 0.0, 0.0,
                         1.0, 1.0, self.nz, self.ny)

    # ------------------------------------------------------------------
    def _volumes(self):
        """Lumped node volumes  V_i = integral of N_i dV  (2*pi*r included for RZ)."""
        hz = self.hz
        wz = np.zeros(self.nz + 1)
        wz[1:] += 0.5 * hz
        wz[:-1] += 0.5 * hz
        if self.dim == 1:
            self.wz = wz
            self.wr = np.ones(1)
            self.node_volume = wz
            return
        h = self.hy
        r = self.y
        if self.rz:
            wr = np.zeros(self.ny + 1)
            # integral of the hat function times r over the left and right elements
            wr[1:] += h * (r[:-1] / 2 + h / 3)
            wr[:-1] += h * (r[:-1] / 2 + h / 6)
            if self.on_axis and not self.axis_correction:
                wr[0] = h[0] * h[0] / 8.0      # pi*(dr/2)^2 / (2 pi)
            wr *= 2 * np.pi
        else:
            wr = np.zeros(self.ny + 1)
            wr[1:] += 0.5 * h
            wr[:-1] += 0.5 * h
        self.wz, self.wr = wz, wr
        self.node_volume = np.outer(wz, wr)

    def cell_centers(self):
        zc = 0.5 * (self.z[1:] + self.z[:-1])
        if self.dim == 1:
            return zc, None
        yc = 0.5 * (self.y[1:] + self.y[:-1])
        return np.meshgrid(zc, yc, indexing="ij")

    def nodes(self):
        if self.dim == 1:
            return self.z, None
        return np.meshgrid(self.z, self.y, indexing="ij")

    def cell_volume(self):
        if self.dim == 1:
            return self.hz.copy()
        if self.rz:
            yc = 0.5 * (self.y[1:] + self.y[:-1])
            return np.outer(self.hz, 2 * np.pi * yc * self.hy)
        return np.outer(self.hz, self.hy)

    def info(self):
        def rng(h, name):
            if np.allclose(h, h[0], rtol=1e-9):
                return f"d{name} = {h[0]:.4g} m"
            return f"d{name} = {h.min():.4g} ... {h.max():.4g} m (refined)"
        if self.dim == 1:
            return f"1D-Z grid: {self.nz} cells, {rng(self.hz, 'z')}"
        return (f"{self.geometry} grid: {self.nz} x {self.ny} cells, {rng(self.hz, 'z')}, "
                f"{rng(self.hy, 'r' if self.rz else 'y')}")


def grid_from_model(m, ev, axis_correction=True, _local=True):
    """Build the Grid of a (completed) model, including its mesh refinement regions."""
    from .refine import axis_nodes, geometry_breaks
    dom = m["domain"]
    geom = m["simulation"]["geometry"]
    refs = [r for r in m.get("mesh_refinements", []) if r.get("enabled", True)]
    zn = (axis_nodes(ev, dom, refs, "z", geometry_breaks(m, ev, "z"))
          if any(r.get("axis", "z") == "z" for r in refs) else None)
    yn = (axis_nodes(ev, dom, refs, "y", geometry_breaks(m, ev, "y"))
          if geom != "1D-Z" and any(r.get("axis") == "y" for r in refs) else None)
    grid = Grid(geom, ev.num(dom["z_min"]), ev.num(dom["z_max"]), ev.num(dom["nz"]),
                ev.num(dom["y_min"]) if geom != "1D-Z" else 0.0,
                ev.num(dom["y_max"]) if geom != "1D-Z" else 1.0,
                ev.num(dom["ny"]) if geom != "1D-Z" else 1,
                axis_correction=axis_correction, z_nodes=zn, y_nodes=yn)
    if _local:
        from .local import local_regions, mesh_from_model
        if local_regions(m):
            grid.local_mesh = mesh_from_model(m)
            grid.hmin = grid.local_mesh.hmin
    return grid


def lossy_sigma(obj, ev):
    """Conductivity of a lossy object: given directly, or from a target loaded Q:
    Q = w eps0 / (sigma F)  ->  sigma = 2 pi f eps0 / (Q F),  F = electric-energy fill factor."""
    from ..constants import eps0
    if obj.get("loss_spec", "Q") == "conductivity":
        return ev.num(obj.get("sigma", 0))
    Q = ev.num(obj.get("Q", 250))
    f = ev.num(obj.get("frequency", 1e9))
    F = ev.num(obj.get("fill", 1))
    return 2 * np.pi * f * eps0 / (Q * max(F, 1e-12))


class Materials:
    """Painted material maps on a Grid.

    Attributes
    ----------
    node_cond : int array (node shape), -1 = free node, k >= 0 = conductor index
    cell_mat  : material code per cell
    cell_obj  : index of object occupying each cell (-1 none)
    eps_cell  : relative permittivity per cell
    conductors: list of dicts {name, voltage (expr), source: 'object'|'boundary', index}
    """

    def __init__(self, grid: Grid, objects: list, boundaries: dict, ev):
        self.grid = grid
        g = grid
        ncell = (g.nz,) if g.dim == 1 else (g.nz, g.ny)
        self.node_cond = -np.ones(g.shape, dtype=np.int32)
        self.cell_mat = np.zeros(ncell, dtype=np.int32)
        self.cell_obj = -np.ones(ncell, dtype=np.int32)
        self.eps_cell = np.ones(ncell)
        self.sigma_cell = np.zeros(ncell)         # EM conductivity of lossy loads [S/m]
        self.conductors = []
        self.shapes = []
        tol = 1e-6 * g.hmin

        # --- boundary conductors (Dirichlet sides) ---------------------------
        sides = ("z_min", "z_max") if g.dim == 1 else ("z_min", "z_max", "y_min", "y_max")
        for side in sides:
            b = boundaries[side]
            if b["field"] != "conductor":
                continue
            if g.rz and side == "y_min" and g.on_axis:
                continue
            k = len(self.conductors)
            self.conductors.append({"name": (b.get("name") or "").strip() or f"boundary {side}",
                                    "voltage": b.get("voltage", "0"),
                                    "source": "boundary", "side": side})
            if g.dim == 1:
                self.node_cond[0 if side == "z_min" else -1] = k
            else:
                if side == "z_min":
                    self.node_cond[0, :] = k
                elif side == "z_max":
                    self.node_cond[-1, :] = k
                elif side == "y_min":
                    self.node_cond[:, 0] = k
                else:
                    self.node_cond[:, -1] = k

        # --- objects (painted in order) ---------------------------------------
        nz_, ny_ = g.nodes()
        cz, cy = g.cell_centers()
        if g.dim == 1:
            ny_ = np.zeros_like(nz_)
            cy = np.zeros_like(cz)
        for oi, obj in enumerate(objects):
            shp = Shape(obj, ev, tol=tol)
            self.shapes.append(shp)
            mat = obj.get("material", "conductor")
            code = MATERIAL_CODES[mat]
            node_in = shp.contains(nz_, ny_)
            cell_in = shp.contains(cz, cy)
            if mat == "conductor":
                k = len(self.conductors)
                self.conductors.append({"name": obj.get("name", f"object{oi}"),
                                        "voltage": obj.get("voltage", "0"),
                                        "source": "object", "object": oi})
                self.node_cond[node_in] = k
                # a cell is solid if its centre or all of its corner nodes are inside
                if g.dim == 2:
                    allc = node_in[:-1, :-1] & node_in[1:, :-1] & node_in[:-1, 1:] & node_in[1:, 1:]
                else:
                    allc = node_in[:-1] & node_in[1:]
                cell_in = cell_in | allc
            else:
                # vacuum / dielectric / absorber remove conductor nodes strictly inside
                if mat == "vacuum":
                    strict = Shape(obj, ev, tol=-tol).contains(nz_, ny_)
                    self.node_cond[strict & (self.node_cond >= 0)] = -1
            if mat == "lossy":
                self.sigma_cell[cell_in] = lossy_sigma(obj, ev)
                continue
            self.sigma_cell[cell_in] = 0.0
            self.cell_mat[cell_in] = code
            self.cell_obj[cell_in] = oi if mat != "vacuum" else -1
            if mat == "dielectric":
                self.eps_cell[cell_in] = ev.num(obj.get("eps_r", 1))
            elif mat in ("vacuum", "conductor", "absorber"):
                self.eps_cell[cell_in] = 1.0
        # restore boundary conductors (they always win)
        self._remark_boundaries(boundaries)
        self._consistent_conductor_cells()

    def _consistent_conductor_cells(self):
        """A cell is a (particle-absorbing) conductor cell iff all its corner nodes are
        conductor nodes, so the particle surface coincides with the field staircase."""
        g = self.grid
        nc = self.node_cond
        if g.dim == 1:
            allc = (nc[:-1] >= 0) & (nc[1:] >= 0)
            kc = np.maximum(nc[:-1], nc[1:])
        else:
            allc = (nc[:-1, :-1] >= 0) & (nc[1:, :-1] >= 0) & (nc[:-1, 1:] >= 0) & (nc[1:, 1:] >= 0)
            kc = np.maximum(np.maximum(nc[:-1, :-1], nc[1:, :-1]), np.maximum(nc[:-1, 1:], nc[1:, 1:]))
        wrong = (self.cell_mat == CONDUCTOR) & ~allc
        self.cell_mat[wrong] = VACUUM
        self.cell_obj[wrong] = -1
        new = allc & (self.cell_mat != CONDUCTOR)
        if new.any():
            objmap = np.array([c.get("object", -1) if c["source"] == "object" else -1
                               for c in self.conductors] + [-1])
            self.cell_mat[new] = CONDUCTOR
            self.cell_obj[new] = objmap[kc[new]]
            self.eps_cell[new] = 1.0

    def _remark_boundaries(self, boundaries):
        g = self.grid
        for k, cnd in enumerate(self.conductors):
            if cnd["source"] != "boundary":
                continue
            side = cnd["side"]
            if g.dim == 1:
                self.node_cond[0 if side == "z_min" else -1] = k
            elif side == "z_min":
                self.node_cond[0, :] = np.where(self.node_cond[0, :] < 0, k, self.node_cond[0, :])
            elif side == "z_max":
                self.node_cond[-1, :] = np.where(self.node_cond[-1, :] < 0, k, self.node_cond[-1, :])
            elif side == "y_min":
                self.node_cond[:, 0] = np.where(self.node_cond[:, 0] < 0, k, self.node_cond[:, 0])
            else:
                self.node_cond[:, -1] = np.where(self.node_cond[:, -1] < 0, k, self.node_cond[:, -1])

    @property
    def solid_cells(self):
        """Cells in which particles are absorbed (conductors, dielectrics, absorbers)."""
        return self.cell_mat != VACUUM
