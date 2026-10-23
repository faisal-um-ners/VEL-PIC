"""Numerical regression tests for independent AC/DC conduction solver."""
import numpy as np
from picstudio.conduction import ContactConfig, ContactFEM


def _tiny(**kw):
    settings=dict(n_contact=10,n_exposed=8,n_metal_depth=7,n_sem_depth=7)
    settings.update(kw)
    return ContactFEM(ContactConfig(**settings))


def test_dc_is_independent_of_interface_capacitance():
    a=_tiny(c_i=0).solve(0)
    b=_tiny(c_i=1).solve(0)
    assert np.isclose(a["Z_contact_W"],b["Z_contact_W"],rtol=1e-11)
    assert np.isclose(a["interface_specific_impedance"],1e-12)
    assert a["relative_residual"]<1e-8


def test_parallel_rc_specific_impedance():
    s=_tiny(c_i=0.1).solve(1e9)
    expected=1/(1/1e-12 + 1j*2*np.pi*1e9*0.1)
    assert np.allclose(s["interface_specific_impedance"],expected)
    assert np.isfinite(s["Z_total_W"])
    assert s["relative_residual"]<1e-7


def test_mesh_clusters_at_contact_edge():
    m=_tiny()
    assert np.diff(m.y_contact)[-1]<np.diff(m.y_contact)[0]
    assert np.diff(m.y_sem)[len(m.y_contact)-1]<np.diff(m.y_sem)[-1]
    assert len(m.i_top)==len(m.i_bottom)
    assert not np.array_equal(m.i_top,m.i_bottom)


def test_zero_capacitance_ac_and_dc_are_distinct_due_to_bulk_dielectric():
    m=_tiny(c_i=0.0)
    dc=m.solve(0)
    ac=m.solve(1e12)
    assert abs(ac["Z_total_W"]-dc["Z_total_W"])>0
