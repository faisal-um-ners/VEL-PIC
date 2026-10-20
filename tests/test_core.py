"""Fast unit tests:  python -m pytest -q tests"""
import glob
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from picstudio import constants as C  # noqa: E402
from picstudio.expr import Evaluator, ExpressionError  # noqa: E402
from picstudio.model import model as M  # noqa: E402


def test_expressions():
    ev = Evaluator({"b": "2*a", "a": "3"})
    assert ev.num("a*b") == 18
    f = ev.func_t("a*min(t/2, 1)")
    assert f(1.0) == 1.5 and f(5.0) == 3.0
    assert abs(ev.num("e/me") - C.e / C.me) < 1
    with pytest.raises(ExpressionError):
        ev.num("__import__('os')")


def test_examples_validate():
    files = glob.glob(os.path.join(ROOT, "examples", "*.pic.json"))
    assert len(files) >= 15
    for f in files:
        msgs = M.validate(M.load(f))
        assert not [m for lvl, m in msgs if lvl == "error"], f


def test_model_roundtrip(tmp_path):
    m = M.new_model("2D-RZ", "electromagnetic", "x")
    p = tmp_path / "a.pic.json"
    M.save(m, str(p))
    m2 = M.load(str(p))
    assert m2["simulation"]["geometry"] == "2D-RZ"
    assert m2["boundaries"]["y_min"]["field"] == "axis"


def test_poisson_rz_second_order():
    from picstudio.fields.poisson import PoissonFEM
    from picstudio.mesh.grid import Grid, Materials
    ev = Evaluator()
    errs = []
    for n in (16, 32):
        g = Grid("2D-RZ", 0, 1, n, 0, 1, n)
        P = PoissonFEM(g, Materials(g, [], M.default_boundaries("2D-RZ"), ev))
        Z, R = g.nodes()
        ex = (1 - R ** 2) * np.sin(np.pi * Z)
        rho = C.eps0 * (4 + np.pi ** 2 * (1 - R ** 2)) * np.sin(np.pi * Z)
        errs.append(np.abs(P.solve(rho * g.node_volume / (2 * np.pi), [0] * 3) - ex).max())
    assert 3.3 < errs[0] / errs[1] < 4.5


def test_loop_field_on_axis():
    from picstudio.fields.external import loop_field
    a, I, z = 0.05, 100.0, 0.02
    Bz, Br = loop_field(np.array([z]), np.array([0.0]), 0.0, a, I)
    assert abs(Bz[0] / (C.mu0 * I * a ** 2 / (2 * (a ** 2 + z ** 2) ** 1.5)) - 1) < 1e-6
    assert Br[0] == 0


def test_fowler_nordheim_value():
    from picstudio.sources.emitters import fowler_nordheim_J
    # Murphy-Good / Forbes reference value for phi = 4.5 eV, F = 5 GV/m  (~1.0e9 A/m^2 order)
    J = fowler_nordheim_J(5e9, 4.5)
    assert 1e8 < J < 1e10


def _run(m, steps):
    from picstudio.core.simulation import Simulation
    m["simulation"]["n_steps"] = str(steps)
    m["diagnostics"].update(history_every="10", field_every="100000", particle_every="100000",
                            live_every="100000")
    sim = Simulation(m, write_output=False, out_dir=os.devnull, log=lambda x: None)
    sim.run()
    return sim


def test_child_langmuir_1d_short():
    m = M.new_model("1D-Z", "electrostatic")
    m["domain"].update(z_min="0", z_max="1e-3", nz="64")
    m["boundaries"]["z_max"]["voltage"] = "1000"
    m["species"] = [{"name": "e", "preset": "electron"}]
    m["emitters"] = [{"name": "k", "species": "e", "model": "scl", "z0": "0", "particles_per_cell": "8"}]
    m["simulation"]["t_end"] = "1.2e-9"
    sim = _run(m, None or 4000)
    H, c = sim.history_array()
    J = -H[len(H) // 2:, c.index("I_z_max")].mean()
    JCL = 4 * C.eps0 / 9 * np.sqrt(2 * C.e / C.me) * 1000 ** 1.5 / 1e-6
    assert abs(J / JCL - 1) < 0.05


def test_em_charge_conservation_rz():
    from picstudio.core.simulation import Simulation
    m = M.new_model("2D-RZ", "electromagnetic")
    m["domain"].update(z_min="0", z_max="0.01", nz="16", y_min="0", y_max="0.01", ny="16")
    for s in ("z_min", "z_max"):
        m["boundaries"][s]["field"] = "periodic"
    m["species"] = [{"name": "e", "preset": "electron"}]
    m["loaders"] = [{"name": "p", "species": "e", "z0": "0", "z1": "0.006", "y0": "0", "y1": "0.006",
                     "density": "1e15", "ppc": "4", "temperature": "1000", "drift": ["3e7", "1e7", "2e7"]}]
    m["simulation"]["n_steps"] = "20"
    sim = Simulation(m, write_output=False, out_dir=os.devnull, log=lambda x: None)
    g, y = sim.grid, sim.yee

    def resid():
        Q, _ = sim._deposit_charge()
        rho = Q / g.node_volume
        d = ((y.E0[1:, 1:-1] - y.E0[:-1, 1:-1]) * g.wr[None, 1:-1] + 2 * np.pi * (
            y.E1[1:-1, 1:] * y.rh[None, 1:] - y.E1[1:-1, :-1] * y.rh[None, :-1]) * g.dz) / (g.dz * g.wr[None, 1:-1])
        return rho[1:-1, 1:-1] / C.eps0 - d, d
    r0, _ = resid()
    for _ in range(20):
        sim.step()
    r1, d = resid()
    assert np.abs(r1 - r0).max() < 1e-9 * np.abs(d).max()


def test_refined_mesh_nodes_and_volumes():
    import numpy as np
    from picstudio.mesh.grid import Grid
    from picstudio.mesh.refine import build_nodes
    z = build_nodes(0.0, 1.0, 10, [(0.0, 0.2, 0.01, 1.2)], breaks=[0.37])
    assert z[0] == 0.0 and z[-1] == 1.0 and np.all(np.diff(z) > 0)
    assert np.isclose(np.diff(z)[:20].max(), 0.01, rtol=0.05) and np.any(np.isclose(z, 0.37))
    for geom in ("2D-ZY", "2D-RZ"):
        g = Grid(geom, 0, 1, 1, 0, 1, 1, z_nodes=z, y_nodes=build_nodes(0, 1, 8, [(0.4, 0.6, 0.02, 1.3)]))
        assert not g.uniform
        assert np.isclose(g.node_volume.sum(), g.cell_volume().sum())
        x = np.array([0.0, 0.123, 0.5, 1.0])
        assert np.allclose(g.pz(g.lz(x)), x) and np.allclose(g.py(g.ly(x)), x)


def test_confined_flow_lau2026_E1():
    from picstudio.core.confined_flow import ConfinedFlow
    J = ConfinedFlow("strip", w=0.1, p=5.12, delta=0.01, nout=32, nz=160).jmax(nbisect=8)
    assert abs(J / 5.191 - 1) < 0.01



def test_em_current_filter_conserves_filtered_charge():
    """With the WarpX-style J filter the discrete Gauss law holds for the equally filtered charge
    (the 1-2-1 stencil commutes with the Yee divergence on a periodic uniform grid)."""
    from picstudio.core.simulation import Simulation
    m = M.new_model("2D-ZY", "electromagnetic")
    m["domain"].update(z_min="0", z_max="0.01", nz="16", y_min="0", y_max="0.01", ny="16")
    for s in ("z_min", "z_max", "y_min", "y_max"):
        m["boundaries"][s].update(field="periodic", particles="periodic")
    m["species"] = [{"name": "e", "preset": "electron"}]
    m["loaders"] = [{"name": "p", "species": "e", "z0": "0", "z1": "0.006", "y0": "0", "y1": "0.006",
                     "density": "1e15", "ppc": "4", "temperature": "1000", "drift": ["3e7", "1e7", "2e7"]}]
    m["simulation"].update(n_steps="20", current_filter="1")
    sim = Simulation(m, write_output=False, out_dir=os.devnull, log=lambda x: None)
    g, y = sim.grid, sim.yee

    def filt(A):                      # same 1-2-1 per axis, periodic (drop duplicated end node)
        B = A[:-1, :-1].copy()
        for ax in (0, 1):
            B = 0.25 * (np.roll(B, 1, ax) + 2 * B + np.roll(B, -1, ax))
        return B

    def resid():
        Q, _ = sim._deposit_charge()
        Q = Q.copy()
        Q[0, :] += Q[-1, :]                   # periodic: fold the duplicated end nodes
        Q[:, 0] += Q[:, -1]
        rho = Q / (g.dz * g.dy)
        div = ((y.E0 - np.roll(y.E0, 1, 0))[:, :-1] / g.dz + (y.E1 - np.roll(y.E1, 1, 1))[:-1, :] / g.dy)
        return filt(rho) / C.eps0 - div, div
    r0, _ = resid()
    for _ in range(20):
        sim.step()
    r1, d = resid()
    assert np.abs(r1 - r0).max() < 1e-6 * np.abs(d).max()
