"""Generate the example projects (*.pic.json) shipped with VEL-PIC.

    python examples/make_examples.py

Every example is also a template: open it in the GUI, change parameters, save as.
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from picstudio.model.model import new_model, save  # noqa: E402


def out(m, fname):
    m["diagnostics"]["output_dir"] = "output_" + fname.replace(".pic.json", "")
    save(m, os.path.join(HERE, fname))
    print("wrote", fname)


def bset(m, side, **kw):
    m["boundaries"][side].update(kw)


# ---------------------------------------------------------------------------
def child_langmuir_1d():
    m = new_model("1D-Z", "electrostatic", "Child-Langmuir planar diode (1D)")
    m["description"] = ("Space-charge-limited (Gauss-law) emission in a planar vacuum diode. "
                        "Compare the anode current density with J_CL = 4 eps0/9 sqrt(2e/m) V^1.5/d^2.")
    m["parameters"] = {"V0": "1000", "d": "1e-3", "JCL": "4*eps0/9*sqrt(2*e/me)*V0**1.5/d**2"}
    m["domain"].update(z_min="0", z_max="d", nz="100")
    add_vacuum_gap(m, y0="0", y1="1")
    bset(m, "z_min", field="conductor", voltage="0", name="cathode")
    bset(m, "z_max", field="conductor", voltage="V0", name="anode")
    m["species"] = [{"name": "electrons", "preset": "electron"}]
    m["emitters"] = [{"name": "cathode", "species": "electrons", "model": "scl", "z0": "0",
                      "particles_per_cell": "10"}]
    m["simulation"].update(t_end="2e-9")
    m["diagnostics"].update(history_every="20", field_every="500", particle_every="500")
    m["sweep"].update(parameter="V0", values="[250, 500, 1000, 2000]", workers=2)
    add_phimin_probe(m, "d")
    out(m, "child_langmuir_1d.pic.json")


def virtual_cathode_1d():
    m = new_model("1D-Z", "electrostatic", "Over-injected diode: virtual cathode and reflected current (1D)")
    m["description"] = ("A beam injected at 2x the Child-Langmuir limit (1 eV injection energy) forms a virtual "
                        "cathode; the transmitted current saturates near J_CL and the excess is reflected back "
                        "to the cathode (Birdsall & Bridges). History: Iemit_cathode = I_anode + I_cathode "
                        "(time averaged).")
    m["parameters"] = {"V0": "1000", "d": "1e-3", "JCL": "4*eps0/9*sqrt(2*e/me)*V0**1.5/d**2",
                       "Jinj": "2*JCL"}
    m["domain"].update(z_min="0", z_max="d", nz="200")
    add_vacuum_gap(m, y0="0", y1="1")
    bset(m, "z_min", field="conductor", voltage="0", name="cathode")
    bset(m, "z_max", field="conductor", voltage="V0", name="anode")
    m["species"] = [{"name": "electrons", "preset": "electron"}]
    m["emitters"] = [{"name": "cathode", "species": "electrons", "model": "beam", "z0": "0",
                      "current_density": "Jinj", "energy": "1", "temperature": "0.1",
                      "particles_per_cell": "20"}]
    m["simulation"].update(t_end="3e-9")
    m["diagnostics"].update(history_every="10", field_every="500", particle_every="500")
    add_phimin_probe(m, "d")
    out(m, "virtual_cathode_1d.pic.json")


def two_stream_1d():
    m = new_model("1D-Z", "electrostatic", "Two-stream instability (1D)")
    m["description"] = ("Two counter-streaming cold electron beams on a neutralizing background; "
                        "mode 1 is at the maximum growth k v0 = sqrt(3)/2 w_pb, gamma_max = w_pb/2.")
    m["parameters"] = {"n0": "1e14", "v0": "1e6", "wpb": "sqrt(n0/2*e**2/(eps0*me))",
                       "L": "2*pi*v0/(sqrt(3)/2*wpb)"}
    m["domain"].update(z_min="0", z_max="L", nz="64")
    for s in ("z_min", "z_max"):
        bset(m, s, field="periodic", particles="periodic")
    m["species"] = [{"name": "beam_right", "preset": "electron"}, {"name": "beam_left", "preset": "electron"}]
    common = dict(z0="0", z1="L", y0="0", y1="1", density="n0/2", ppc="400", quiet_start=True,
                  perturbation="velocity", pert_amplitude="1e-4*v0", pert_mode="1")
    m["loaders"] = [dict(name="right", species="beam_right", drift=["v0", "0", "0"], **common),
                    dict(name="left", species="beam_left", drift=["-v0", "0", "0"], **common)]
    m["loaders"][1]["pert_amplitude"] = "-1e-4*v0"
    m["simulation"].update(t_end="45/wpb", dt="0.05/wpb")
    m["diagnostics"].update(history_every="5", field_every="100", particle_every="100")
    out(m, "two_stream_1d.pic.json")


def plasma_oscillation_1d():
    m = new_model("1D-Z", "electrostatic", "Cold plasma oscillation (1D)")
    m["description"] = "Langmuir oscillation: field energy oscillates at 2*w_p."
    m["parameters"] = {"n0": "1e15", "wp": "sqrt(n0*e**2/(eps0*me))", "L": "0.01"}
    m["domain"].update(z_min="0", z_max="L", nz="64")
    for s in ("z_min", "z_max"):
        bset(m, s, field="periodic", particles="periodic")
    m["species"] = [{"name": "electrons", "preset": "electron"}]
    m["loaders"] = [dict(name="plasma", species="electrons", z0="0", z1="L", y0="0", y1="1", density="n0",
                         ppc="100", quiet_start=True, perturbation="velocity", pert_amplitude="1e4",
                         pert_mode="1")]
    m["simulation"].update(t_end="20*2*pi/wp", dt="0.02/wp")
    m["diagnostics"].update(history_every="2", field_every="200", particle_every="200")
    out(m, "plasma_oscillation_1d.pic.json")


B_NOTE = (" Magnetic field: Species > motion = z_only is the infinite axial field (electrons move only along "
          "z, the assumption of Luginsland et al. 1996 and Lau 2001); motion = full uses the applied field "
          "'axial B' = [Bz, 0, 0] T with the parameter Bz (0 = no field, or any finite value, e.g. 0.1, 1, 10).")


def add_phimin_probe(m, z_end, y="0"):
    """Line probe: potential minimum (virtual-cathode depth) and its position along the emission axis."""
    m["line_probes"] = list(m.get("line_probes", [])) + [{"name": "phi", "quantity": "phi", "extremum": "min",
                                                          "along": "z", "y": y, "start": "0", "end": z_end}]


def add_vacuum_gap(m, z0="0", z1="d", y0="-Ly", y1="Ly"):
    """Explicit 'vacuum gap' geometry object covering the diode gap.  The domain is vacuum by default
    (eps0) and the electrodes are the conductor boundaries z = 0 (cathode) and z = d (anode), so this
    object changes nothing in the solution; it only shows the gap region in the Geometry Objects list
    (boundary conductors always take precedence over painted objects)."""
    m["objects"] = [{"name": "vacuum gap", "shape": "rectangle", "material": "vacuum",
                     "z0": z0, "z1": z1, "y0": y0, "y1": y1}] + list(m.get("objects", []))


def add_axial_B(m, motion="z_only", Bz="0"):
    """Axial magnetic field option for diode examples: parameter Bz [T] + uniform applied field; the
    species motion selects finite B ('full') or the infinite-B limit ('z_only')."""
    m["parameters"]["Bz"] = Bz
    m["external_fields"] = [{"name": "axial B", "type": "uniform", "E": ["0", "0", "0"], "B": ["Bz", "0", "0"],
                             "time_factor": "1"}]
    for sp in m["species"]:
        sp["motion"] = motion


def child_langmuir_strip_zy():
    m = new_model("2D-ZY", "electrostatic", "2D Child-Langmuir: finite strip emitter")
    m["description"] = ("SCL emission from a strip of width W centred at y = 0 in a planar diode (gap d). "
                        "The side walls y = +-(W/2 + 3d) are far from the strip: Neumann for the field (no "
                        "normal E, i.e. an open side) and absorbing for particles. Lau (2001): J/J_CL = 1 + "
                        "0.3145 d/W - 0.0004 (d/W)^2 for electrons confined to z (infinite B)." + B_NOTE)
    m["parameters"] = {"V0": "1000", "d": "1e-3", "W": "1e-3", "Ly": "W/2+3*d"}
    m["domain"].update(z_min="0", z_max="d", nz="48", y_min="-Ly", y_max="Ly", ny="336")
    add_vacuum_gap(m)
    bset(m, "z_min", field="conductor", voltage="0", name="cathode")
    bset(m, "z_max", field="conductor", voltage="V0", name="anode")
    bset(m, "y_min", field="neumann", particles="absorb")
    bset(m, "y_max", field="neumann", particles="absorb")
    m["species"] = [{"name": "electrons", "preset": "electron"}]
    m["emitters"] = [{"name": "cathode", "species": "electrons", "model": "scl", "z0": "0", "y0": "-W/2",
                      "z1": "0", "y1": "W/2", "particles_per_cell": "4"}]
    add_axial_B(m)
    m["simulation"].update(t_end="1.5e-9")
    m["diagnostics"].update(history_every="20", field_every="400", particle_every="400")
    add_phimin_probe(m, "d")
    out(m, "child_langmuir_strip_zy.pic.json")


def child_langmuir_periodic_zy():
    m = new_model("2D-ZY", "electrostatic", "Child-Langmuir diode, 2D periodic (1D-equivalent)")
    m["description"] = ("Planar SCL diode in 2D with periodic side boundaries and an emitter covering the whole "
                        "cathode: no edge effects, so the anode current density must equal the 1D Child-Langmuir "
                        "law. History: I_anode (collected), I_cathode (electrons returned to the cathode), "
                        "Iemit_cathode (emitted).")
    m["parameters"] = {"V0": "1000", "d": "1e-3", "Ly": "0.25e-3",
                       "JCL": "4*eps0/9*sqrt(2*e/me)*V0**1.5/d**2", "ICL": "JCL*Ly"}
    m["domain"].update(z_min="0", z_max="d", nz="64", y_min="0", y_max="Ly", ny="16")
    add_vacuum_gap(m, y0="0", y1="Ly")
    bset(m, "z_min", field="conductor", voltage="0", name="cathode")
    bset(m, "z_max", field="conductor", voltage="V0", name="anode")
    for s_ in ("y_min", "y_max"):
        bset(m, s_, field="periodic", particles="periodic")
    m["species"] = [{"name": "electrons", "preset": "electron"}]
    m["emitters"] = [{"name": "cathode", "species": "electrons", "model": "scl", "z0": "0", "y0": "0",
                      "z1": "0", "y1": "Ly", "particles_per_cell": "4"}]
    add_axial_B(m, motion="full")
    m["description"] += B_NOTE
    m["simulation"].update(t_end="2e-9")
    m["diagnostics"].update(history_every="10", field_every="400", particle_every="400")
    add_phimin_probe(m, "d")
    out(m, "child_langmuir_periodic_zy.pic.json")


def child_langmuir_disk_rz():
    m = new_model("2D-RZ", "electrostatic", "2D Child-Langmuir: circular emitter (RZ)")
    m["description"] = ("SCL emission from a disk of radius R on the axis (centre of the cathode); outer wall "
                        "r = R + 3d: Neumann field (open side), absorbing for particles. Lau (2001): J/J_CL ~ "
                        "1 + d/(4R) for electrons confined to z (infinite B)." + B_NOTE)
    m["parameters"] = {"V0": "1000", "d": "1e-3", "R": "1e-3", "Lr": "R+3*d"}
    m["domain"].update(z_min="0", z_max="d", nz="48", y_min="0", y_max="Lr", ny="192")
    add_vacuum_gap(m, y0="0", y1="Lr")
    bset(m, "z_min", field="conductor", voltage="0", name="cathode")
    bset(m, "z_max", field="conductor", voltage="V0", name="anode")
    bset(m, "y_max", field="neumann", particles="absorb")
    m["species"] = [{"name": "electrons", "preset": "electron"}]
    m["emitters"] = [{"name": "cathode", "species": "electrons", "model": "scl", "z0": "0", "y0": "0",
                      "z1": "0", "y1": "R", "particles_per_cell": "4"}]
    add_axial_B(m)
    m["simulation"].update(t_end="1.5e-9")
    m["diagnostics"].update(history_every="20", field_every="400", particle_every="400")
    add_phimin_probe(m, "d")
    out(m, "child_langmuir_disk_rz.pic.json")


def coaxial_diode_rz():
    m = new_model("2D-RZ", "electrostatic", "Coaxial SCL diode (Langmuir-Blodgett)")
    m["description"] = ("Radial space-charge-limited flow from an inner cylindrical cathode (radius a) to an "
                        "outer anode (radius b), periodic in z.  Compare I/L with Langmuir-Blodgett.")
    m["parameters"] = {"V0": "1000", "a": "1e-3", "b": "2e-3", "L": "0.5e-3"}
    m["domain"].update(z_min="0", z_max="L", nz="8", y_min="0", y_max="b", ny="96")
    for s in ("z_min", "z_max"):
        bset(m, s, field="periodic", particles="periodic")
    bset(m, "y_max", field="conductor", voltage="V0", name="anode")
    add_vacuum_gap(m, z0="0", z1="L", y0="a", y1="b")
    m["objects"] = m["objects"] + [{"name": "cathode", "shape": "rectangle", "z0": "-1", "z1": "1", "y0": "0", "y1": "a",
                     "material": "conductor", "voltage": "0"}]
    m["species"] = [{"name": "electrons", "preset": "electron"}]
    m["emitters"] = [{"name": "cathode", "species": "electrons", "model": "scl", "surface": "segment",
                      "z0": "0", "y0": "a", "z1": "L", "y1": "a", "normal": "+y", "particles_per_cell": "8"}]
    m["simulation"].update(t_end="1e-9")
    m["diagnostics"].update(history_every="20", field_every="400", particle_every="400")
    out(m, "coaxial_diode_rz.pic.json")


def beam_drift_tube_rz():
    m = new_model("2D-RZ", "electrostatic", "Space-charge beam expansion in a drift tube (RZ)")
    m["description"] = ("10 keV, 50 mA uniform electron beam injected into a grounded drift tube. "
                        "Compare the beam edge radius with the envelope equation r'' = K/r.")
    m["parameters"] = {"Eb": "10e3", "I0": "0.05", "a": "2e-3", "Rt": "10e-3", "Lz": "0.2"}
    m["domain"].update(z_min="0", z_max="Lz", nz="400", y_min="0", y_max="Rt", ny="40")
    bset(m, "z_min", field="conductor", voltage="0")
    bset(m, "z_max", field="conductor", voltage="0")
    bset(m, "y_max", field="conductor", voltage="0")
    m["species"] = [{"name": "beam", "preset": "electron"}]
    m["emitters"] = [{"name": "gun", "species": "beam", "model": "beam", "z0": "0", "y0": "0", "z1": "0",
                      "y1": "a", "current_density": "I0/(pi*a**2)", "energy": "Eb", "particles_per_cell": "4"}]
    m["simulation"].update(t_end="12e-9")
    m["diagnostics"].update(history_every="20", field_every="500", particle_every="500")
    out(m, "beam_drift_tube_rz.pic.json")


def beam_solenoid_rz():
    m = new_model("2D-RZ", "electrostatic", "Beam transport in a solenoid (Brillouin focusing)")
    m["description"] = ("10 keV, 50 mA beam injected from a field-free gun into a solenoid set to the "
                        "Brillouin field B_B = sqrt(2)*me*wp/e (fringe fields give the Busch rotation).")
    m["parameters"] = {"Eb": "10e3", "I0": "0.05", "a": "2e-3", "Rt": "10e-3", "Lz": "0.2",
                       "gam": "1+Eb*e/(me*c**2)", "v": "c*sqrt(1-1/gam**2)", "nb": "I0/(e*v*pi*a**2)",
                       "BB": "sqrt(2*gam)*me*sqrt(nb*e**2/(eps0*me*gam))/e", "Rs": "0.03"}
    m["domain"].update(z_min="0", z_max="Lz", nz="400", y_min="0", y_max="Rt", ny="40")
    for s in ("z_min", "z_max", "y_max"):
        bset(m, s, field="conductor", voltage="0")
    m["species"] = [{"name": "beam", "preset": "electron"}]
    m["emitters"] = [{"name": "gun", "species": "beam", "model": "beam", "z0": "0", "y0": "0", "z1": "0",
                      "y1": "a", "current_density": "I0/(pi*a**2)", "energy": "Eb", "particles_per_cell": "4"}]
    m["external_fields"] = [{"name": "solenoid", "type": "solenoid", "zc": "0.02", "z_end": "0.5",
                             "radius": "Rs", "ampere_turns": "BB*0.48/mu0"}]
    m["simulation"].update(t_end="12e-9")
    m["diagnostics"].update(history_every="20", field_every="500", particle_every="500")
    out(m, "beam_solenoid_rz.pic.json")


def thermionic_diode_1d():
    m = new_model("1D-Z", "electrostatic", "Thermionic diode: SCL to temperature-limited")
    m["description"] = ("Richardson-Dushman-Schottky emission (W = 4.5 eV, 2500 K) in a 1 mm gap. Sweep V0 "
                        "to see the 3/2-power (space-charge) and saturated (temperature-limited) regimes.")
    m["parameters"] = {"V0": "300", "d": "1e-3"}
    m["domain"].update(z_min="0", z_max="d", nz="200")
    add_vacuum_gap(m, y0="0", y1="1")
    bset(m, "z_min", field="conductor", voltage="0", name="cathode")
    bset(m, "z_max", field="conductor", voltage="V0", name="anode")
    m["species"] = [{"name": "electrons", "preset": "electron"}]
    m["emitters"] = [{"name": "cathode", "species": "electrons", "model": "thermionic", "z0": "0",
                      "work_function": "4.5", "cathode_temperature": "2500", "schottky": True,
                      "particles_per_cell": "40"}]
    m["simulation"].update(t_end="5e-9")
    m["diagnostics"].update(history_every="20", field_every="1000", particle_every="1000")
    m["sweep"].update(parameter="V0", values="[30, 100, 300, 1000, 3000]", workers=2)
    out(m, "thermionic_diode_1d.pic.json")


def field_emission_1d():
    m = new_model("1D-Z", "electrostatic", "Fowler-Nordheim field emission diode")
    m["description"] = "Murphy-Good FN emission with space charge in a 1 um vacuum gap."
    m["parameters"] = {"V0": "6000", "d": "1e-6"}
    m["domain"].update(z_min="0", z_max="d", nz="100")
    add_vacuum_gap(m, y0="0", y1="1")
    bset(m, "z_min", field="conductor", voltage="0", name="cathode")
    bset(m, "z_max", field="conductor", voltage="V0", name="anode")
    m["species"] = [{"name": "electrons", "preset": "electron"}]
    m["emitters"] = [{"name": "tip", "species": "electrons", "model": "fowler_nordheim", "z0": "0",
                      "work_function": "4.5", "beta": "1", "particles_per_cell": "20"}]
    m["simulation"].update(t_end="2e-13")
    m["diagnostics"].update(history_every="10", field_every="500", particle_every="500")
    out(m, "field_emission_1d.pic.json")


def planar_magnetron_es():
    m = new_model("2D-ZY", "electrostatic", "Crossed-field (planar magnetron) diode - Hull cutoff")
    m["description"] = ("Planar diode with a magnetic field B_x parallel to the electrodes: cathode along y = 0 "
                        "(emitting from z = 0 to Ly), anode at y = d, periodic in z. Anode current is cut "
                        "off for B > B_H = (me c/(e d)) sqrt(gamma^2-1). Sweep Bratio.")
    m["parameters"] = {"V0": "10e3", "d": "5e-3", "gam": "1+e*V0/(me*c**2)",
                       "BH": "me*c/(e*d)*sqrt(gam**2-1)", "Bratio": "0.8", "Ly": "2e-3"}
    # cathode along y = 0 (emits in +y), anode at y = d, periodic along z
    m["domain"].update(z_min="0", z_max="Ly", nz="16", y_min="0", y_max="d", ny="64")
    add_vacuum_gap(m, z0="0", z1="Ly", y0="0", y1="d")
    bset(m, "y_min", field="conductor", voltage="0", name="cathode")
    bset(m, "y_max", field="conductor", voltage="V0", name="anode")
    for s in ("z_min", "z_max"):
        bset(m, s, field="periodic", particles="periodic")
    m["species"] = [{"name": "electrons", "preset": "electron"}]
    m["emitters"] = [{"name": "cathode", "species": "electrons", "model": "scl", "z0": "0", "y0": "0", "z1": "Ly",
                      "y1": "0", "normal": "+y", "particles_per_cell": "4"}]
    m["external_fields"] = [{"name": "B", "type": "uniform", "B": ["0", "0", "Bratio*BH"]}]
    m["simulation"].update(t_end="4e-9")
    m["diagnostics"].update(history_every="20", field_every="500", particle_every="500")
    m["sweep"].update(parameter="Bratio", values="[0.6, 0.8, 0.9, 1.0, 1.1, 1.3]", workers=2)
    out(m, "planar_magnetron_es.pic.json")


def a6_magnetron_em():
    m = new_model("2D-ZY", "electromagnetic", "A6 relativistic magnetron with loss port (2D EM)")
    m["description"] = ("Bekefi A6 relativistic magnetron cross-section: cathode r=1.58 cm, anode r=2.11 cm, "
                        "6 vanes to r=4.11 cm (20 deg cavities), axial B (out of plane), explosive (SCL) "
                        "emission from the whole cathode surface. The outer half of cavity 1 is a lossy load "
                        "(loss port) giving a defined loaded Q = Qload (250) for the cold-test pi-mode at "
                        "2.313 GHz (fill calibrated by ring-down, Model > Calibrate loss Q; re-run it after "
                        "changing the geometry or Qload). "
                        "Operating point of the experiment (Bekefi & Orzechowski 1976): V0 = 360 kV, B0 = 0.8 T, "
                        "inside the window between the smooth-bore Hull field (0.51 T, Lovelace-Ott) and the "
                        "pi-mode Buneman-Hartree field (0.82 T). The pi mode starts at ~15 ns and oscillates "
                        "at ~2.16 GHz (measured ~2.3 GHz); see benchmark 21.")
    p = {"V0": "360e3", "B0": "0.8", "rc": "1.58e-2", "ra": "2.11e-2", "rv": "4.11e-2", "hw": "10*pi/180",
         "Lb": "5.6e-2", "Qload": "250", "f0": "2.313e9"}
    m["parameters"] = p
    m["domain"].update(z_min="-Lb", z_max="Lb", nz="224", y_min="-Lb", y_max="Lb", ny="224")
    for s in ("z_min", "y_min", "y_max"):
        bset(m, s, field="conductor", voltage="0")
    bset(m, "z_max", field="conductor", voltage="0")
    objs = [{"name": "anode block", "shape": "rectangle", "z0": "-2*Lb", "z1": "2*Lb", "y0": "-2*Lb",
             "y1": "2*Lb", "material": "conductor", "voltage": "0"},
            {"name": "interaction space", "shape": "circle", "zc": "0", "yc": "0", "radius": "ra",
             "material": "vacuum"}]
    for k in range(6):
        th = k * 60.0
        pts = []
        for r_, a_ in (("0.9*ra", -1), ("rv", -1), ("rv", 0), ("rv", 1), ("0.9*ra", 1)):
            ang = f"({th}*pi/180 + {a_}*hw)" if a_ != 0 else f"({th}*pi/180)"
            rr = r_ if a_ != 0 else f"rv/cos(hw)"
            pts.append(f"{rr}*cos{ang},{rr}*sin{ang}")
        objs.append({"name": f"cavity {k + 1}", "shape": "polygon", "points": "; ".join(pts),
                     "material": "vacuum"})
    objs.append({"name": "loss port", "shape": "rectangle", "z0": "0.5*(ra+rv)", "z1": "rv+1e-3",
                 "y0": "-0.5*(ra+rv)*tan(hw)", "y1": "0.5*(ra+rv)*tan(hw)", "material": "lossy",
                 "loss_spec": "Q", "Q": "Qload", "frequency": "f0", "fill": "0.01136"})
    objs.append({"name": "cathode", "shape": "circle", "zc": "0", "yc": "0", "radius": "rc",
                 "material": "conductor", "voltage": "-V0*min(t/2e-9,1)"})
    m["objects"] = objs
    m["species"] = [{"name": "electrons", "preset": "electron"}]
    m["emitters"] = [{"name": "explosive", "species": "electrons", "model": "scl", "surface": "object",
                      "object": "cathode", "particles_per_cell": "2", "t_start": "0.5e-9"}]
    m["external_fields"] = [{"name": "axial B", "type": "uniform", "B": ["0", "0", "B0"]}]
    m["probes"] = [{"name": "cavity2_Ey", "quantity": "Ey", "z": "0.5*(ra+rv)*cos(pi/3)",
                    "y": "0.5*(ra+rv)*sin(pi/3)"},
                   {"name": "gap_Ez", "quantity": "Ez", "z": "0.0", "y": "0.0195"}]
    m["power_lines"] = [{"name": "port", "orientation": "z", "position": "0.5*(ra+rv)",
                         "start": "-0.5*(ra+rv)*tan(hw)", "end": "0.5*(ra+rv)*tan(hw)"}]
    m["simulation"].update(t_end="40e-9", cfl="0.9", smoothing=1)
    m["diagnostics"].update(history_every="5", field_every="1000", particle_every="1000", live_every="100")
    out(m, "a6_magnetron_em.pic.json")


def planar_magnetron_em():
    m = new_model("2D-ZY", "electromagnetic", "Planar magnetron with vane slow-wave structure (2D EM)")
    m["description"] = ("Planar (linear) magnetron: smooth cathode along y = 0 (SCL emission from z = 0 to N*p), "
                        "anode with N vane cavities above it, periodic in z (equivalent to an infinitely long / "
                        "recirculating planar magnetron), magnetic field B_x out of the plane. The back half of "
                        "cavity 1 is a lossy load (loss port) whose conductivity gives a defined loaded Q = Qload "
                        "(250) of the operating mode at 5.85 GHz; 'fill' was set by the cold-test ring-down "
                        "calibration (Model > Calibrate loss Q; re-run it after changing the geometry or Qload). "
                        "Operating point: B0 = 0.16 T > B_H (insulated) and V0 above the Buneman-Hartree voltage "
                        "of the pi-mode (~5.7 GHz); tune V0 and B0 (or the cavity depth h) to change the mode.")
    m["parameters"] = {"V0": "170e3", "B0": "0.16", "d": "1e-2", "p": "1e-2", "wc": "0.5e-2", "h": "1e-2",
                       "N": "6", "Ly": "N*p", "Qload": "250", "f0": "5.85e9",
                       "gam": "1+e*V0/(me*c**2)", "BH": "me*c/(e*d)*sqrt(gam**2-1)"}
    m["domain"].update(z_min="0", z_max="Ly", nz="240", y_min="0", y_max="d+h+0.25e-2", ny="90")
    bset(m, "y_min", field="conductor", voltage="-V0*min(t/1e-9,1)")
    bset(m, "y_max", field="conductor", voltage="0")
    for s_ in ("z_min", "z_max"):
        bset(m, s_, field="periodic", particles="periodic")
    objs = [{"name": "anode", "shape": "rectangle", "z0": "-1", "z1": "1", "y0": "d", "y1": "1",
             "material": "conductor", "voltage": "0"}]
    for k in range(6):
        objs.append({"name": f"cavity {k + 1}", "shape": "rectangle", "y0": "d - 1e-4", "y1": "d+h",
                     "z0": f"{k}*p + (p-wc)/2", "z1": f"{k}*p + (p+wc)/2", "material": "vacuum"})
    objs.append({"name": "loss port", "shape": "rectangle", "y0": "d+0.5*h", "y1": "d+h",
                 "z0": "(p-wc)/2", "z1": "(p+wc)/2", "material": "lossy", "loss_spec": "Q",
                 "Q": "Qload", "frequency": "f0", "fill": "0.01475", "cal_period": "p", "cal_phase": "pi"})
    m["objects"] = objs
    m["species"] = [{"name": "electrons", "preset": "electron"}]
    m["emitters"] = [{"name": "cathode", "species": "electrons", "model": "scl", "z0": "0", "y0": "0",
                      "z1": "Ly", "y1": "0", "normal": "+y", "particles_per_cell": "2", "t_start": "0.3e-9"}]
    m["external_fields"] = [{"name": "B", "type": "uniform", "B": ["0", "0", "B0"]}]
    m["probes"] = [{"name": "cav2_Ez", "quantity": "Ez", "z": "1.5*p", "y": "d+h/2"},
                   {"name": "gap_Ey", "quantity": "Ey", "z": "1.5*p", "y": "0.8*d"}]
    m["power_lines"] = [{"name": "port", "orientation": "y", "position": "d+0.5*h",
                         "start": "(p-wc)/2", "end": "(p+wc)/2"}]
    m["simulation"].update(t_end="25e-9", cfl="0.9", smoothing=1)
    m["diagnostics"].update(history_every="5", field_every="1000", particle_every="1000", live_every="100")
    out(m, "planar_magnetron_em.pic.json")

def laser_plasma_1d():
    m = new_model("1D-Z", "electromagnetic", "Laser - plasma slab interaction (1D)")
    m["description"] = ("800 nm laser pulse (a0 = 0.05) incident on a plasma slab. With nfrac > 1 (overdense) "
                        "the pulse is reflected; with nfrac < 1 it propagates with v_g = c sqrt(1 - n/n_c).")
    m["parameters"] = {"lam": "0.8e-6", "w": "2*pi*c/lam", "nc": "eps0*me*w**2/e**2", "nfrac": "2.0",
                       "Lx": "60*lam"}
    m["domain"].update(z_min="0", z_max="Lx", nz="2400")
    for s in ("z_min", "z_max"):
        bset(m, s, field="absorbing", absorber_cells=40, particles="absorb")
    m["species"] = [{"name": "electrons", "preset": "electron"}]
    m["loaders"] = [dict(name="slab", species="electrons", z0="30*lam", z1="50*lam", y0="0", y1="1",
                         density="nfrac*nc", ppc="50", temperature="10")]
    m["species"].append({"name": "ions", "preset": "proton"})
    m["loaders"].append(dict(name="slab ions", species="ions", z0="30*lam", z1="50*lam", y0="0", y1="1",
                             density="nfrac*nc", ppc="50", temperature="0"))
    m["sources"] = [{"name": "laser", "type": "laser", "z_position": "2*lam", "wavelength": "lam", "a0": "0.05",
                     "duration": "15e-15", "t_peak": "30e-15", "polarization": "x"}]
    m["probes"] = [{"name": "Ex_front", "quantity": "Ex", "z": "10*lam", "y": "0"}]
    m["power_lines"] = [{"name": "S_front", "orientation": "z", "position": "10*lam", "start": "0", "end": "1"}]
    m["simulation"].update(t_end="300e-15", cfl="0.95")
    m["diagnostics"].update(history_every="2", field_every="500", particle_every="2000")
    out(m, "laser_plasma_1d.pic.json")


def laser_wakefield_1d():
    m = new_model("1D-Z", "electromagnetic", "Laser wakefield (1D)")
    m["description"] = "a0 = 0.5 Gaussian pulse in an underdense plasma (n = 0.01 n_c) drives a wake at k_p."
    m["parameters"] = {"lam": "0.8e-6", "w": "2*pi*c/lam", "nc": "eps0*me*w**2/e**2", "n0": "0.01*nc",
                       "wp": "sqrt(n0*e**2/(eps0*me))", "lp": "2*pi*c/wp", "Lx": "6*lp"}
    m["domain"].update(z_min="0", z_max="Lx", nz="6000")
    for s in ("z_min", "z_max"):
        bset(m, s, field="absorbing", absorber_cells=40)
    m["species"] = [{"name": "electrons", "preset": "electron"}]
    m["species"].append({"name": "ions", "preset": "proton"})
    m["loaders"] = [dict(name="plasma", species="electrons", z0="0.5*lp", z1="Lx", y0="0", y1="1",
                         density="n0*min((z-0.5*lp)/(0.5*lp),1)", ppc="20"),
                    dict(name="ions", species="ions", z0="0.5*lp", z1="Lx", y0="0", y1="1",
                         density="n0*min((z-0.5*lp)/(0.5*lp),1)", ppc="20")]
    m["sources"] = [{"name": "laser", "type": "laser", "z_position": "4*lam", "wavelength": "lam", "a0": "0.5",
                     "duration": "lp/c/2/sqrt(2)*1.17741", "t_peak": "lp/c", "polarization": "x"}]
    m["simulation"].update(t_end="5.2*lp/c", cfl="0.98", smoothing=1)
    m["diagnostics"].update(history_every="10", field_every="500", particle_every="2000")
    out(m, "laser_wakefield_1d.pic.json")


def beam_plasma_1d():
    m = new_model("1D-Z", "electrostatic", "Weak beam - plasma instability (1D)")
    m["description"] = ("Cold beam (alpha = n_b/n_p = 1e-3) through a cold plasma; growth rate "
                        "gamma = (sqrt(3)/2^(4/3)) alpha^(1/3) w_p at k = w_p/v_b.")
    m["parameters"] = {"np0": "1e14", "alpha": "1e-3", "vb": "1e7", "wp": "sqrt(np0*e**2/(eps0*me))",
                       "L": "2*pi*vb/wp"}
    m["domain"].update(z_min="0", z_max="L", nz="64")
    for s in ("z_min", "z_max"):
        bset(m, s, field="periodic", particles="periodic")
    m["species"] = [{"name": "plasma", "preset": "electron"}, {"name": "beam", "preset": "electron"}]
    m["loaders"] = [dict(name="plasma", species="plasma", z0="0", z1="L", y0="0", y1="1", density="np0",
                         ppc="200", quiet_start=True),
                    dict(name="beam", species="beam", z0="0", z1="L", y0="0", y1="1", density="alpha*np0",
                         ppc="200", quiet_start=True, drift=["vb", "0", "0"], perturbation="velocity",
                         pert_amplitude="1e-5*vb", pert_mode="1")]
    m["simulation"].update(t_end="150/wp", dt="0.05/wp")
    m["diagnostics"].update(history_every="5", field_every="200", particle_every="200")
    out(m, "beam_plasma_1d.pic.json")


def beam_plasma_wakefield_1d():
    m = new_model("1D-Z", "electromagnetic", "Relativistic beam-driven plasma wakefield (1D)")
    m["description"] = ("A short 50 MeV electron bunch (n_b = 0.1 n_p) excites a linear plasma wake; "
                        "E_max = (e n_b/(eps0 k_p)) 2 sin(k_p L/2).")
    m["parameters"] = {"np0": "1e22", "wp": "sqrt(np0*e**2/(eps0*me))", "kp": "wp/c", "lp": "2*pi/kp",
                       "Lb": "lp/2", "Lx": "8*lp"}
    m["domain"].update(z_min="0", z_max="Lx", nz="2400")
    for s in ("z_min", "z_max"):
        bset(m, s, field="absorbing", absorber_cells=40)
    m["species"] = [{"name": "plasma", "preset": "electron"}, {"name": "bunch", "preset": "electron"}]
    m["loaders"] = [dict(name="plasma", species="plasma", z0="0", z1="Lx", y0="0", y1="1", density="np0",
                         ppc="20"),
                    dict(name="bunch", species="bunch", z0="1.0*lp", z1="1.0*lp+Lb", y0="0", y1="1",
                         density="0.1*np0", ppc="50", drift=["c*sqrt((1+50e6*e/(me*c**2))**2-1)", "0", "0"])]
    m["simulation"].update(t_end="6*lp/c", cfl="0.98", smoothing=1, initial_field="none")
    m["diagnostics"].update(history_every="10", field_every="300", particle_every="1200")
    out(m, "beam_plasma_wakefield_1d.pic.json")


def laser_focus_zy():
    m = new_model("2D-ZY", "electromagnetic", "Gaussian laser beam focusing in vacuum (2D)")
    m["description"] = "Focused 1 um Gaussian pulse (w0 = 3 um) launched from an antenna; focus at z = 15 um."
    m["parameters"] = {"lam": "1e-6"}
    m["domain"].update(z_min="0", z_max="30*lam", nz="600", y_min="-15*lam", y_max="15*lam", ny="300")
    for s in ("z_min", "z_max", "y_min", "y_max"):
        bset(m, s, field="absorbing", absorber_cells=20)
    m["sources"] = [{"name": "laser", "type": "laser", "z_position": "1.5*lam", "wavelength": "lam", "a0": "0.1",
                     "duration": "15e-15", "t_peak": "30e-15", "polarization": "x", "waist": "3*lam",
                     "focus_z": "15*lam"}]
    m["probes"] = [{"name": "Ex_focus", "quantity": "Ex", "z": "15*lam", "y": "0"}]
    m["simulation"].update(t_end="110e-15", cfl="0.95")
    m["diagnostics"].update(history_every="2", field_every="100", particle_every="100000")
    out(m, "laser_focus_zy.pic.json")


def laser_plasma_zy():
    m = new_model("2D-ZY", "electromagnetic", "2D laser - plasma interaction (ZY)")
    m["description"] = ("Focused 0.8 um laser pulse (a0 = 1, w0 = 4 um) incident on a near-critical plasma "
                        "slab (n = 0.5 n_c) with an up-ramp: relativistic self-focusing, filamentation, "
                        "reflection and hot-electron generation in 2D.  Ions are mobile protons.")
    m["parameters"] = {"lam": "0.8e-6", "w": "2*pi*c/lam", "nc": "eps0*me*w**2/e**2", "nfrac": "0.5",
                       "Lz": "40*lam", "Ly": "24*lam"}
    m["domain"].update(z_min="0", z_max="Lz", nz="800", y_min="-Ly/2", y_max="Ly/2", ny="480")
    for s_ in ("z_min", "z_max", "y_min", "y_max"):
        bset(m, s_, field="absorbing", absorber_cells=20, particles="absorb")
    m["species"] = [{"name": "electrons", "preset": "electron"}, {"name": "ions", "preset": "proton"}]
    dens = "nfrac*nc*min(max((z-12*lam)/(4*lam),0),1)"
    m["loaders"] = [dict(name="plasma e", species="electrons", z0="12*lam", z1="36*lam", y0="-Ly/2+2*lam",
                         y1="Ly/2-2*lam", density=dens, ppc="2", temperature="10"),
                    dict(name="plasma i", species="ions", z0="12*lam", z1="36*lam", y0="-Ly/2+2*lam",
                         y1="Ly/2-2*lam", density=dens, ppc="2")]
    m["sources"] = [{"name": "laser", "type": "laser", "z_position": "1.5*lam", "wavelength": "lam", "a0": "1.0",
                     "duration": "20e-15", "t_peak": "40e-15", "polarization": "y", "waist": "4*lam",
                     "y_center": "0", "focus_z": "16*lam"}]
    m["probes"] = [{"name": "Ey_front", "quantity": "Ey", "z": "8*lam", "y": "0"}]
    m["power_lines"] = [{"name": "reflected", "orientation": "z", "position": "4*lam", "start": "-Ly/2+2*lam",
                         "end": "Ly/2-2*lam"},
                        {"name": "transmitted", "orientation": "z", "position": "37*lam",
                         "start": "-Ly/2+2*lam", "end": "Ly/2-2*lam"}]
    m["simulation"].update(t_end="200e-15", cfl="0.95", smoothing=1, initial_field="none")
    m["diagnostics"].update(history_every="4", field_every="300", particle_every="600", live_every="100")
    out(m, "laser_plasma_zy.pic.json")


def beam_plasma_zy():
    m = new_model("2D-ZY", "electromagnetic", "2D beam - plasma interaction (ZY)")
    m["description"] = ("A relativistic electron beam (gamma = 3, n_b = 0.1 n_p) streams through a uniform plasma "
                        "with a neutralizing return current: two-stream (longitudinal) and filamentation/Weibel "
                        "(transverse) instabilities grow together in the (z,y) plane.  Periodic box.")
    m["parameters"] = {"np0": "1e18", "wp": "sqrt(np0*e**2/(eps0*me))", "kp": "wp/c", "L": "16*2*pi/kp",
                       "nb": "0.1*np0", "ub": "c*sqrt(3**2-1)", "vr": "-0.1*c*sqrt(1-1/3**2)"}
    m["domain"].update(z_min="0", z_max="L", nz="256", y_min="0", y_max="L/2", ny="128")
    for s_ in ("z_min", "z_max", "y_min", "y_max"):
        bset(m, s_, field="periodic", particles="periodic")
    m["species"] = [{"name": "plasma", "preset": "electron"}, {"name": "beam", "preset": "electron"},
                    {"name": "ions", "preset": "proton"}]
    m["loaders"] = [dict(name="plasma", species="plasma", z0="0", z1="L", y0="0", y1="L/2", density="np0",
                         ppc="8", temperature="50", drift=["vr", "0", "0"]),
                    dict(name="beam", species="beam", z0="0", z1="L", y0="0", y1="L/2", density="nb",
                         ppc="8", temperature="500", drift=["ub", "0", "0"]),
                    dict(name="ions", species="ions", z0="0", z1="L", y0="0", y1="L/2", density="np0+nb",
                         ppc="4")]
    m["simulation"].update(t_end="120/wp", cfl="0.95", smoothing=1)
    m["diagnostics"].update(history_every="5", field_every="300", particle_every="300", live_every="100")
    out(m, "beam_plasma_zy.pic.json")


def plasma_wakefield_rz():
    m = new_model("2D-RZ", "electromagnetic", "Beam-driven plasma wakefield (RZ, PWFA)")
    m["description"] = ("A relativistic Gaussian electron bunch (1 GeV, sigma_z = sigma_r = 1/k_p, "
                        "n_b = 0.3 n_p) drives a plasma wake in cylindrical geometry: radial focusing "
                        "and longitudinal acceleration behind the driver.")
    m["parameters"] = {"np0": "1e22", "wp": "sqrt(np0*e**2/(eps0*me))", "kp": "wp/c", "lp": "2*pi/kp",
                       "sz": "1/kp", "sr": "1/kp", "z0": "1.5*lp", "Lz": "7*lp", "Lr": "2.5*lp",
                       "ub": "c*sqrt((1+1e9*e/(me*c**2))**2-1)"}
    m["domain"].update(z_min="0", z_max="Lz", nz="560", y_min="0", y_max="Lr", ny="200")
    for s_ in ("z_min", "z_max"):
        bset(m, s_, field="absorbing", absorber_cells=20, particles="absorb")
    bset(m, "y_max", field="absorbing", absorber_cells=20, particles="absorb")
    m["species"] = [{"name": "plasma", "preset": "electron"}, {"name": "bunch", "preset": "electron"},
                    {"name": "ions", "preset": "proton"}]
    m["loaders"] = [dict(name="plasma", species="plasma", z0="0", z1="Lz", y0="0", y1="Lr-0.3*lp",
                         density="np0", ppc="4"),
                    dict(name="ions", species="ions", z0="0", z1="Lz", y0="0", y1="Lr-0.3*lp",
                         density="np0", ppc="4"),
                    dict(name="bunch", species="bunch", z0="z0-4*sz", z1="z0+4*sz", y0="0", y1="4*sr",
                         density="0.3*np0*exp(-0.5*((z-z0)/sz)**2-0.5*(r/sr)**2)", ppc="16",
                         drift=["ub", "0", "0"])]
    m["simulation"].update(t_end="4*lp/c", cfl="0.95", smoothing=1, initial_field="none")
    m["probes"] = [{"name": "Ez_axis", "quantity": "Ez", "z": "1.5*lp", "y": "0"}]
    m["diagnostics"].update(history_every="5", field_every="200", particle_every="400", live_every="50")
    out(m, "plasma_wakefield_rz.pic.json")


def relativistic_diode_rz_em():
    m = new_model("2D-RZ", "electromagnetic", "Coax-fed relativistic diode (RZ EM, TEM port)")
    m["description"] = ("A TEM voltage pulse launched in a coaxial line drives a cathode stalk; explosive SCL "
                        "emission from the cathode tip; electrons cross the A-K gap to the anode plate.")
    m["parameters"] = {"V0": "300e3", "rin": "1e-2", "rout": "3e-2", "gap": "1e-2", "Lz": "0.1"}
    m["domain"].update(z_min="0", z_max="Lz", nz="200", y_min="0", y_max="rout", ny="60")
    bset(m, "z_min", field="absorbing", absorber_cells=12)
    bset(m, "z_max", field="conductor")
    bset(m, "y_max", field="conductor")
    m["objects"] = [{"name": "cathode stalk", "shape": "rectangle", "z0": "-1", "z1": "Lz-gap", "y0": "0",
                     "y1": "rin", "material": "conductor"}]
    m["species"] = [{"name": "electrons", "preset": "electron"}]
    m["emitters"] = [{"name": "cathode tip", "species": "electrons", "model": "scl", "z0": "Lz-gap", "y0": "0",
                      "z1": "Lz-gap", "y1": "rin", "normal": "+z", "particles_per_cell": "4"}]
    m["sources"] = [{"name": "TEM drive", "type": "tem_port", "z_position": "0.012", "voltage":
                     "-V0*min(t/1e-9,1)", "inner": "rin", "outer": "rout"}]
    m["power_lines"] = [{"name": "P_in", "orientation": "z", "position": "0.03", "start": "rin", "end": "rout"}]
    m["simulation"].update(t_end="5e-9", cfl="0.9")
    m["diagnostics"].update(history_every="5", field_every="300", particle_every="300")
    out(m, "relativistic_diode_rz_em.pic.json")


def dielectric_charging_zy():
    m = new_model("2D-ZY", "electrostatic", "Dielectric surface charging by an electron beam")
    m["description"] = ("A 1 keV electron beam hits a dielectric slab (eps_r = 4) on a grounded electrode; the "
                        "deposited surface charge builds up a retarding potential that deflects the beam.")
    m["parameters"] = {"Lz": "1e-2", "Ly": "1e-2"}
    m["domain"].update(z_min="0", z_max="Lz", nz="100", y_min="0", y_max="Ly", ny="100")
    for s in ("z_min", "z_max", "y_min", "y_max"):
        bset(m, s, field="conductor", voltage="0")
    m["objects"] = [{"name": "slab", "shape": "rectangle", "z0": "0.7*Lz", "z1": "0.85*Lz", "y0": "0.2*Ly",
                     "y1": "0.8*Ly", "material": "dielectric", "eps_r": "4", "collect_charge": True}]
    m["species"] = [{"name": "electrons", "preset": "electron"}]
    m["emitters"] = [{"name": "gun", "species": "electrons", "model": "beam", "z0": "0", "y0": "0.45*Ly",
                      "z1": "0", "y1": "0.55*Ly", "current_density": "20", "energy": "1000",
                      "particles_per_cell": "4"}]
    m["simulation"].update(t_end="40e-9")
    m["diagnostics"].update(history_every="20", field_every="500", particle_every="500")
    out(m, "dielectric_charging_zy.pic.json")


# ---------------------------------------------------------------------------
# Field-free electrostatic problems (no applied E or B; periodic or grounded walls)
def landau_damping_1d():
    m = new_model("1D-Z", "electrostatic", "Landau damping (1D)")
    m["description"] = ("Collisionless damping of a Langmuir wave in a Maxwellian electron plasma on a fixed "
                        "neutralizing background, k lambda_D = 0.5. Exact Vlasov root: w = 1.4156 - 0.1533i w_p; the "
                        "field energy decays as exp(-0.307 w_p t) while oscillating at 2w. Quiet start in position "
                        "and velocity, 4000 particles per cell.")
    m["parameters"] = {"n0": "1e15", "Te": "1", "kld": "0.5", "wp": "sqrt(n0*e**2/(eps0*me))",
                       "lD": "sqrt(eps0*Te/(n0*e))", "L": "2*pi*lD/kld"}
    m["domain"].update(z_min="0", z_max="L", nz="64")
    for s_ in ("z_min", "z_max"):
        bset(m, s_, field="periodic", particles="periodic")
    m["species"] = [{"name": "electrons", "preset": "electron"}]
    m["loaders"] = [dict(name="plasma", species="electrons", z0="0", z1="L", y0="0", y1="1", density="n0",
                         ppc="4000", quiet_start=True, temperature="Te", perturbation="density",
                         pert_amplitude="0.05", pert_mode="1")]
    m["simulation"].update(t_end="22/wp", dt="0.05/wp")
    m["diagnostics"].update(history_every="1", field_every="40", particle_every="40")
    out(m, "landau_damping_1d.pic.json")


def pierce_diode_1d():
    m = new_model("1D-Z", "electrostatic", "Pierce diode instability (1D)")
    m["description"] = ("Cold electron beam injected through a grounded gap with a neutralizing ion background "
                        "(Pierce diode). For a Pierce parameter alpha = w_p L / v0 = 2.5 pi the equilibrium is "
                        "unstable; the exact dispersion root is w = (1.429 + 0.589i) v0/L. Compare with "
                        "benchmarks/bench_published.py (pierce).")
    m["parameters"] = {"n0": "1e13", "v0": "1e6", "alpha": "2.5*pi", "wp": "sqrt(n0*e**2/(eps0*me))",
                       "L": "alpha*v0/wp"}
    m["domain"].update(z_min="0", z_max="L", nz="200")
    for s_ in ("z_min", "z_max"):
        bset(m, s_, field="conductor", voltage="0", particles="absorb")
    m["species"] = [{"name": "electrons", "preset": "electron"},
                    {"name": "ions", "preset": "custom", "charge": "1", "mass": "1e6*mp"}]
    m["loaders"] = [dict(name="beam", species="electrons", z0="0", z1="L", y0="0", y1="1", density="n0",
                         ppc="64", quiet_start=True, drift=["v0", "0", "0"], perturbation="velocity",
                         pert_amplitude="1e-3*v0", pert_mode="0.5"),
                    dict(name="background", species="ions", z0="0", z1="L", y0="0", y1="1", density="n0",
                         ppc="64", quiet_start=True)]
    m["emitters"] = [{"name": "gun", "species": "electrons", "model": "beam", "z0": "0", "y0": "0", "z1": "0",
                      "y1": "1", "normal": "+z", "current_density": "e*n0*v0", "energy": "0.5*me*v0**2/e",
                      "temperature": "0", "particles_per_cell": "64"}]
    m["simulation"].update(dt="0.25*L/200/v0", t_end="12*L/v0", initial_field="poisson")
    m["diagnostics"].update(history_every="4", field_every="400", particle_every="400")
    out(m, "pierce_diode_1d.pic.json")


def beam_expansion_zy():
    m = new_model("2D-ZY", "electrostatic", "Space-charge expansion of a sheet beam (ZY)")
    m["description"] = ("A 10 keV, 1 A/m electron sheet beam (half-width 0.5 mm) enters a field-free, grounded drift "
                        "region. With no focusing field it spreads under its own space charge. The edge of a uniform sheet "
                        "feels a constant field, so y_edge = a0 + K z^2/2 with K = e I/(2 eps0 m gamma^3 v^3) = 0.047 /m: "
                        "the rms half-width grows from 0.289 mm to 0.398 mm at z = 9 cm (VEL-PIC: 0.397 mm). Use "
                        "Results > Beam envelope.")
    m["parameters"] = {"V0": "10e3", "I": "1", "a0": "0.5e-3", "Lz": "0.1", "Ly": "6e-3"}
    m["domain"].update(z_min="0", z_max="Lz", nz="200", y_min="-Ly", y_max="Ly", ny="96")
    for s_ in ("z_min", "z_max", "y_min", "y_max"):
        bset(m, s_, field="conductor", voltage="0", particles="absorb")
    m["species"] = [{"name": "electrons", "preset": "electron"}]
    m["emitters"] = [{"name": "gun", "species": "electrons", "model": "beam", "z0": "0", "y0": "-a0", "z1": "0",
                      "y1": "a0", "normal": "+z", "current_density": "I/(2*a0)", "energy": "V0",
                      "temperature": "0", "particles_per_cell": "8"}]
    m["simulation"].update(t_end="4e-9")
    m["diagnostics"].update(history_every="10", field_every="500", particle_every="500")
    out(m, "beam_expansion_zy.pic.json")


# ---------------------------------------------------------------------------
# Lau, Chernin, Heri & Zhang (2026), arXiv:2607.24590: limiting current for small emission area and
# low emission energy.  Confined flow (motion along z only = infinite axial B), monoenergetic
# injection E_in = Delta*e*V, V = 2.5 kV, d = 1 cm.  Reference J_max from the steady solver
# picstudio.core.confined_flow (Eq. 7 of the paper), see benchmarks/bench_lau2026.py.
LAU_REF = "Lau, Chernin, Heri & Zhang, arXiv:2607.24590 (2026)"


def _lau_common(m, Jn, Delta, nz, dzf):
    m["parameters"].update({"V0": "2500", "d": "1e-2", "Delta": repr(Delta), "Jn": repr(Jn),
                            "JCL": "4*eps0/9*sqrt(2*e/me)*V0**1.5/d**2", "v0": "sqrt(2*e*V0/me)",
                            "Nz": str(nz), "dz_fine": dzf})
    m["domain"].update(z_min="0", z_max="d", nz="Nz")
    bset(m, "z_min", field="conductor", voltage="0", name="cathode")
    bset(m, "z_max", field="conductor", voltage="V0", name="anode")
    m["species"] = [{"name": "electrons", "preset": "electron"}]
    add_axial_B(m, motion="z_only")
    # z-only motion: the time step is set by the smallest z cell (electrons cross < 0.4 cell per step)
    m["simulation"].update(t_end="8e-9", dt="0.4*dz_fine/v0")
    m["diagnostics"].update(history_every="20", field_every="2000", particle_every="2000", live_every="200")
    # refined mesh in front of the cathode, where the potential minimum sits (z_min ~ 0.01-0.02 d)
    m["mesh_refinements"] = [{"name": "cathode region", "axis": "z", "start": "0", "end": "0.05*d",
                              "size": "dz_fine", "growth": "1.1"}]


def _lau_strip(fname, title, wbar, pbar, Delta, Jn, Jref, ny, nz, dzf, dyf, sweep, ref_note):
    m = new_model("2D-ZY", "electrostatic", title)
    m["description"] = (
        f"{LAU_REF}: periodic array of emitting strips (width w = {wbar:g} d, period p = {pbar:g} d), "
        f"confined flow (z-only motion, infinite axial B), injection energy Delta = E_in/eV = {Delta:g}, "
        "V0 = 2.5 kV, d = 1 cm. One period of the array is modelled exactly as in the paper (Fig. 1): "
        "the strip is centred at y = 0 and the side boundaries y = -p/2, +p/2 are periodic for fields "
        "and particles (an isolated strip is the limit p >> d; p/d = 2 is already close to it). The mesh "
        "is refined across the strip and in front of the cathode (Mesh Refinement). "
        f"Steady-state limit, Eq. (7): J_max = {Jref:.4g} J_CL ({ref_note}). Below J_max: I_anode = "
        "Iemit, I_cathode = 0 and the potential minimum (line probe 'phi' along the centre line: depth "
        "probe_phi_min and position probe_phi_zmin) stays above -Delta*V0. Above "
        "J_max: virtual cathode, phi_min -> -Delta*V0 and part of the current returns to the cathode "
        "(I_cathode < 0). Use the sweep over Jn = J/J_CL to locate the limit." + B_NOTE)
    _lau_common(m, Jn, Delta, nz, dzf)
    m["parameters"].update({"wbar": repr(wbar), "pbar": repr(pbar), "Ny": str(ny), "dy_fine": dyf})
    m["domain"].update(y_min="-pbar*d/2", y_max="pbar*d/2", ny="Ny")
    m["mesh_refinements"].append({"name": "strip", "axis": "y", "start": "-wbar*d/2", "end": "wbar*d/2",
                                  "size": "dy_fine", "growth": "1.1"})
    add_vacuum_gap(m, y0="-pbar*d/2", y1="pbar*d/2")
    for s_ in ("y_min", "y_max"):
        bset(m, s_, field="periodic", particles="periodic")
    m["emitters"] = [{"name": "cathode", "species": "electrons", "model": "beam", "z0": "0",
                      "y0": "-wbar*d/2", "z1": "0", "y1": "wbar*d/2", "normal": "+z",
                      "current_density": "Jn*JCL",
                      "energy": "Delta*V0", "temperature": "0", "particles_per_cell": "2"}]
    # potential minimum (depth and position) along the centre line of the strip
    m["line_probes"] = [{"name": "phi", "quantity": "phi", "extremum": "min", "along": "z", "y": "0",
                         "start": "0", "end": "0.2*d"}]
    m["sweep"].update(parameter="Jn", values=sweep, workers=2)
    out(m, fname)


def _lau_disk(fname, title, Rbar, Delta, Jn, Jref, nr, nz, dzf, drf, sweep):
    m = new_model("2D-RZ", "electrostatic", title)
    m["description"] = (
        f"{LAU_REF}: circular emitting patch of radius R = {Rbar:g} d in a planar diode (RZ), confined "
        f"flow (z-only motion), injection energy Delta = {Delta:g}, V0 = 2.5 kV, d = 1 cm; outer surface "
        "r = d with dphi/dr = 0. Mesh refined over the patch and in front of the cathode. "
        f"Steady-state limit (Eq. 7 in r-z, picstudio.core.confined_flow): J_max = {Jref:.4g} J_CL. "
        "First-order estimate, paper Eqs. (5)/(B10): 1.328 (R/d)^-1/2 [1 + 0.34 (R/d)^1.5] (valid for "
        "Delta << R/d, an upper estimate). Below J_max all current reaches the anode; above it a "
        "virtual cathode forms (phi_min -> -Delta*V0, I_cathode < 0)." + B_NOTE)
    _lau_common(m, Jn, Delta, nz, dzf)
    m["parameters"].update({"Rbar": repr(Rbar), "Nr": str(nr), "dr_fine": drf})
    m["domain"].update(y_min="0", y_max="d", ny="Nr")
    m["mesh_refinements"].append({"name": "patch", "axis": "y", "start": "0", "end": "Rbar*d",
                                  "size": "dr_fine", "growth": "1.1"})
    add_vacuum_gap(m, y0="0", y1="d")
    bset(m, "y_max", field="neumann", particles="reflect")
    m["emitters"] = [{"name": "cathode", "species": "electrons", "model": "beam", "z0": "0", "y0": "0",
                      "z1": "0", "y1": "Rbar*d", "normal": "+z", "current_density": "Jn*JCL",
                      "energy": "Delta*V0", "temperature": "0", "particles_per_cell": "2"}]
    m["line_probes"] = [{"name": "phi", "quantity": "phi", "extremum": "min", "along": "z", "y": "0",
                         "start": "0", "end": "0.2*d"}]
    m["sweep"].update(parameter="Jn", values=sweep, workers=2)
    out(m, fname)


def yylau_2026_examples():
    common_a = dict(wbar=0.1, pbar=2.0, Delta=0.01, Jref=5.197, ny=80, nz=100, dzf="25e-6",
                    dyf="31.25e-6", sweep="[4.5, 5.0, 5.4, 6.0]",
                    ref_note="5.193 in the Lau 2026 tutorial; paper test E1 with p = 5.12 d: 5.191")
    _lau_strip("yylau_2026_strip_caseA_zy.pic.json", "Lau 2026: strip array, w/d = 0.1, below J_max (ZY)",
               Jn=5.0, **common_a)
    _lau_strip("yylau_2026_strip_caseA_virtual_cathode_zy.pic.json",
               "Lau 2026: strip array, w/d = 0.1, above J_max: virtual cathode (ZY)", Jn=6.0, **common_a)
    _lau_strip("yylau_2026_strip_E3_zy.pic.json", "Lau 2026: narrow strip, test E3 (w/d = 0.01, ZY)",
               wbar=0.01, pbar=2.048, Delta=0.01, Jn=31.0, Jref=32.863, ny=80, nz=100, dzf="12.5e-6",
               dyf="6.25e-6", sweep="[31, 33, 35]",
               ref_note="paper POISSON 32.863, XOOPIC 32.75")
    common_d = dict(Rbar=0.1, Delta=0.01, Jref=5.872, nr=40, nz=100, dzf="25e-6", drf="62.5e-6",
                    sweep="[5.0, 5.5, 6.3, 7.0]")
    _lau_disk("yylau_2026_disk_rz.pic.json", "Lau 2026: circular patch R/d = 0.1, below J_max (RZ)",
              Jn=5.5, **common_d)
    _lau_disk("yylau_2026_disk_virtual_cathode_rz.pic.json",
              "Lau 2026: circular patch R/d = 0.1, above J_max: virtual cathode (RZ)", Jn=7.0, **common_d)

# ---------------------------------------------------------------------------
# Field-only electrostatics (no particles)
def two_cylinder_lens_rz():
    m = new_model("2D-RZ", "electrostatic", "Two-cylinder electrostatic lens (RZ, field only)")
    m["description"] = ("Electrostatic immersion lens: two coaxial tubes of inner radius R (diameter D = 2R) at V1 and "
                        "V2, separated by a gap g = 0.1 D, wall thickness 0.1 R. No particles: the run is a pure "
                        "Laplace solution (scikit-fem, r-weighted weak form) on a mesh refined at the gap and the "
                        "tube walls. The axial potential phi(z) and field E_z(z) are recorded by line probes. "
                        "Reference: the exact series for two coaxial cylinders with a linear potential across the "
                        "gap (Goddard 1946; Cook & Heddle 1976); VEL-PIC agrees with it to 3e-4 V2 along the axis.")
    m["parameters"] = {"V1": "0", "V2": "1e3", "R": "10e-3", "g": "0.2*R", "t": "0.1*R", "L": "6*R", "hf": "0.1e-3"}
    m["domain"].update(z_min="-L", z_max="L", nz="240", y_min="0", y_max="1.5*R", ny="60")
    bset(m, "z_min", field="conductor", voltage="V1", name="entrance")
    bset(m, "z_max", field="conductor", voltage="V2", name="exit")
    bset(m, "y_max", field="neumann")
    m["objects"] = [
        {"name": "tube 1", "shape": "rectangle", "z0": "-L", "z1": "-g/2", "y0": "R", "y1": "R+t",
         "material": "conductor", "voltage": "V1"},
        {"name": "tube 2", "shape": "rectangle", "z0": "g/2", "z1": "L", "y0": "R", "y1": "R+t",
         "material": "conductor", "voltage": "V2"}]
    m["mesh_refinements"] = [
        {"name": "gap", "axis": "z", "start": "-g", "end": "g", "size": "hf", "growth": "1.1"},
        {"name": "tube wall", "axis": "y", "start": "R-t", "end": "R+2*t", "size": "hf", "growth": "1.1"}]
    m["species"] = []
    m["line_probes"] = [
        {"name": "phi_axis", "quantity": "phi", "extremum": "max", "along": "z", "y": "0", "start": "-4*R", "end": "4*R"},
        {"name": "Ez_axis", "quantity": "Ez", "extremum": "min", "along": "z", "y": "0", "start": "-4*R", "end": "4*R"}]
    m["simulation"].update(t_end="1e-9", dt="1e-10")
    m["diagnostics"].update(history_every="1", field_every="10", particle_every="10")
    out(m, "two_cylinder_lens_rz.pic.json")


if __name__ == "__main__":
    for f in (child_langmuir_1d, virtual_cathode_1d, two_stream_1d, plasma_oscillation_1d, child_langmuir_strip_zy,
              child_langmuir_periodic_zy, laser_plasma_zy, beam_plasma_zy, plasma_wakefield_rz,
              child_langmuir_disk_rz, coaxial_diode_rz, beam_drift_tube_rz, beam_solenoid_rz,
              thermionic_diode_1d, field_emission_1d, planar_magnetron_es, planar_magnetron_em, a6_magnetron_em,
              laser_plasma_1d, laser_wakefield_1d, beam_plasma_1d, beam_plasma_wakefield_1d,
              laser_focus_zy, relativistic_diode_rz_em, dielectric_charging_zy, yylau_2026_examples,
              landau_damping_1d, pierce_diode_1d, beam_expansion_zy, two_cylinder_lens_rz):
        f()
