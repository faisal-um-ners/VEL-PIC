"""Model schema: every input of VEXPACE is described here.

The same description drives
  * default values and validation (``picstudio.model.model``),
  * the automatically generated property forms of the GUI,
  * the JSON project file format (``*.pic.json``).

Conventions
-----------
Coordinates are written (z, y) for both Cartesian ZY and cylindrical RZ
geometries; in RZ the second coordinate ``y`` is the radius ``r``.
Vector quantities are ordered (z, y, x) in ZY, (z, r, theta) in RZ and
(z, x, y) in 1D-Z.  All units are SI except energies/temperatures in eV.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Field:
    name: str
    kind: str                     # expr | int | bool | choice | str | vec3 | text
    default: Any = None
    label: str = ""
    unit: str = ""
    help: str = ""
    choices: tuple = ()
    show_if: dict = field(default_factory=dict)   # {other_field: [allowed values]}
    choice_labels: dict = field(default_factory=dict)  # display labels; stored values stay compatible

    def __post_init__(self):
        if not self.label:
            self.label = self.name.replace("_", " ").capitalize()


GEOMETRIES = ("1D-Z", "2D-ZY", "2D-RZ")
SOLVERS = ("electrostatic", "electromagnetic")

SIMULATION = [
    Field("geometry", "choice", "2D-ZY", "Geometry", choices=GEOMETRIES,
          help="1D-Z: planar 1D along z.  2D-ZY: Cartesian (z,y), per unit depth in x.  "
               "2D-RZ: axisymmetric cylinder (z,r)."),
    Field("solver", "choice", "electrostatic", "Field solver", choices=SOLVERS,
          help="Electrostatic: scikit-fem finite-element Poisson solve every step.  "
               "Electromagnetic: Yee FDTD Maxwell solver with charge-conserving current deposition."),
    Field("t_end", "expr", "1e-9", "End time", "s"),
    Field("dt", "expr", "auto", "Time step", "s",
          help="'auto' chooses dt from CFL, plasma and cyclotron frequencies."),
    Field("cfl", "expr", "0.95", "CFL number", "",
          help="EM: fraction of the Courant limit.  ES (auto dt): max cells a particle moves per step."),
    Field("n_steps", "expr", "", "Number of steps (override)", "",
          help="If given, overrides End time."),
    Field("relativistic", "bool", True, "Relativistic pusher"),
    Field("particle_pusher", "choice", "boris", "Particle pusher", choices=("boris", "vay", "higuera"),
          help="Same algorithms as WarpX (algo.particle_pusher): boris (default), vay (Vay 2008, correct "
               "E x B drift for relativistic particles) or higuera (Higuera & Cary 2017, volume preserving "
               "and correct E x B drift)."),
    Field("field_gathering", "choice", "auto", "Field gathering",
          choices=("auto", "energy-conserving", "momentum-conserving"),
          help="energy-conserving (WarpX default, Galerkin): every Yee component is interpolated from its "
               "own staggered position with the shape order lowered by one along its staggering; in ES "
               "runs this is E = -grad of the bilinear interpolant of phi.  momentum-conserving: bilinear "
               "interpolation of node-centred fields - second-order accurate at emitting surfaces, so "
               "space-charge-limited emission converges faster.  auto: energy-conserving for the EM "
               "solver (as WarpX), momentum-conserving for the ES solver.  Choose energy-conserving "
               "for runs that must reproduce WarpX exactly. Local 2D P1 mesh: auto uses the direct "
               "element gradient; momentum-conserving selects an L2-projected nodal reconstruction, "
               "which does not imply exact momentum conservation on an irregular mesh."),
    Field("smoothing", "int", 0, "Binomial smoothing passes", "",
          help="1-2-1 filter passes. ES: applied to the charge density.  EM: applied to the fields "
               "gathered by the particles (keeps the current deposition charge conserving)."),
    Field("em_conductors", "choice", "held at their potentials", "Conductors (EM)",
          choices=("held at their potentials", "floating"),
          help="held: every conductor is connected to an ideal voltage source, so its potential stays at "
               "the set voltage while current flows (diodes, magnetrons, guns).  floating: isolated PEC "
               "conductors whose charge only changes by absorbed / emitted particles (their potential "
               "drifts as the gap discharges)."),
    Field("current_filter", "int", 0, "Current filter passes (EM)", "",
          help="WarpX-style bilinear 1-2-1 filter of the deposited current J, this many passes along every "
               "axis before the field update (WarpX warpx.use_filter, default 1 pass in Cartesian WarpX).  "
               "Damps grid-scale noise and numerical Cherenkov; 0 = off (exact discrete Gauss law with the "
               "unfiltered charge)."),
    Field("axis_correction", "bool", True, "RZ axis volume correction",
          help="Use the exact linear-weighting volume pi*dr^2/3 for the on-axis node "
               "(Verboncoeur 2001). Applies to 2D-RZ only.",
          show_if={"geometry": ["2D-RZ"]}),
    Field("initial_field", "choice", "poisson", "Initial self-field (EM)", choices=("poisson", "none"),
          show_if={"solver": ["electromagnetic"]},
          help="poisson: start EM runs from the electrostatic field of the initial charge.  none: start "
               "with zero fields (like WarpX's default; use for relativistic beams injected into neutral "
               "plasma)."),
    Field("seed", "int", 12345, "Random seed"),
    Field("max_particles", "expr", "2e7", "Particle limit (safety)"),
]

DOMAIN = [
    Field("z_min", "expr", "0", "z min", "m"),
    Field("z_max", "expr", "1e-2", "z max", "m"),
    Field("nz", "expr", "128", "Cells in z"),
    Field("y_min", "expr", "0", "y / r min", "m", help="In RZ this is r_min (0 = on axis)."),
    Field("y_max", "expr", "1e-2", "y / r max", "m"),
    Field("ny", "expr", "128", "Cells in y / r"),
]

MESH_REFINE = [
    Field("name", "str", "emission region"),
    Field("enabled", "bool", True, "Enabled"),
    Field("axis", "choice", "z", "Refinement type", choices=("z", "y", "local 2D"),
          choice_labels={"z": "Along z", "y": "Along y", "local 2D": "2D"},
          help="Along z / Along y: full-domain graded bands along the chosen coordinate. "
               "2D: refine only a filled rectangle or "
               "polygon that you prescribe before the electrostatic PIC run; its conforming triangular "
               "mesh stays fixed. Particles can leave the region freely."),
    Field("start", "expr", "0", "From", "m", show_if={"axis": ["z", "y"]}),
    Field("end", "expr", "1e-3", "To", "m", show_if={"axis": ["z", "y"]}),
    Field("region_shape", "choice", "rectangle", "Region shape", choices=("rectangle", "polygon"),
          show_if={"axis": ["local 2D"]}),
    Field("z0", "expr", "0", "Region z min", "m", show_if={"axis": ["local 2D"], "region_shape": ["rectangle"]}),
    Field("z1", "expr", "1e-3", "Region z max", "m", show_if={"axis": ["local 2D"], "region_shape": ["rectangle"]}),
    Field("y0", "expr", "0", "Region y / r min", "m", show_if={"axis": ["local 2D"], "region_shape": ["rectangle"]}),
    Field("y1", "expr", "1e-3", "Region y / r max", "m", show_if={"axis": ["local 2D"], "region_shape": ["rectangle"]}),
    Field("points", "text", "0,0; 1e-2,0; 1e-2,2e-3; 0,2e-4", "Polygon points (z,y; ...)", "m",
          show_if={"axis": ["local 2D"], "region_shape": ["polygon"]},
          help="Ordered boundary vertices around the entire expected electron region, not just a trajectory. "
               "Use parameter expressions. In RZ the second coordinate is radius; in a half-domain "
               "start at y=0. Add separate regions for cathode return footprints if needed."),
    Field("buffer", "expr", "0", "Safety buffer outside", "m", show_if={"axis": ["local 2D"]},
          help="Additional refined width around the prescribed region, including possible sideways motion."),
    Field("size", "expr", "1", "Cell size inside", "m",
          help="For local 2D this is the maximum rectangular leaf-edge length in the refined region; "
               "each leaf is split into conforming P1 triangles."),
    Field("growth", "expr", "1.2", "Growth ratio outside",
          help="Smooth target size grows with distance from the region (1.05-1.2 recommended). "
               "Local 2D uses discrete subdivisions; neighboring triangle sizes therefore change in levels. "
               "The Domain & Mesh counts retain moderate background resolution elsewhere."),
]

BOUNDARY_FIELD = ("conductor", "neumann", "periodic", "absorbing", "axis")
BOUNDARY_PARTICLE = ("absorb", "reflect", "periodic")
BOUNDARY = [
    Field("name", "str", "", "Name (e.g. cathode, anode)",
          help="Optional label used for this boundary's current / voltage diagnostics "
               "(I_<name>, V_<name>); default is the side name."),
    Field("field", "choice", "conductor", "Field boundary", choices=BOUNDARY_FIELD,
          help="conductor: Dirichlet potential (ES) / perfect conductor (EM).  "
               "neumann: zero normal E (symmetry / insulating).  periodic.  "
               "absorbing: open boundary (EM graded absorbing layer, ES Neumann).  "
               "axis: r = 0 in RZ."),
    Field("voltage", "expr", "0", "Voltage V(t)", "V", show_if={"field": ["conductor"]}),
    Field("particles", "choice", "absorb", "Particle boundary", choices=BOUNDARY_PARTICLE),
    Field("absorber_cells", "int", 16, "Absorbing layer cells", show_if={"field": ["absorbing"]}),
]
SIDES = ("z_min", "z_max", "y_min", "y_max")

SHAPES = ("rectangle", "circle", "ellipse", "annulus", "polygon", "expression")
MATERIALS = ("conductor", "dielectric", "vacuum", "absorber", "lossy")
MATERIAL_SUMMARIES = {
    "conductor": (
        "Electrode: absorbs particles; emission is set separately.",
        "ES: φ = V(t). EM: perfect conductor, tangential E = 0."),
    "dielectric": (
        "Insulator (εr): absorbs particles; charging is optional.",
        "ES: continuous φ; normal εE changes with surface charge."),
    "vacuum": (
        "Empty region: particles pass through; can carve holes.",
        "No internal field boundary or prescribed potential."),
    "absorber": (
        "Particle sink: removes incoming particles.",
        "Vacuum-like fields; no EM wave-absorbing layer."),
    "lossy": (
        "Resistive EM load: damps fields; particles pass through.",
        "EM: J = σE (Q or σ input). No resistive damping in ES."),
}
OBJECT = [
    Field("name", "str", "object"),
    Field("shape", "choice", "rectangle", "Shape", choices=SHAPES),
    Field("z0", "expr", "0", "z0", "m", show_if={"shape": ["rectangle"]}),
    Field("z1", "expr", "1e-3", "z1", "m", show_if={"shape": ["rectangle"]}),
    Field("y0", "expr", "0", "y0 / r0", "m", show_if={"shape": ["rectangle"]}),
    Field("y1", "expr", "1e-3", "y1 / r1", "m", show_if={"shape": ["rectangle"]}),
    Field("zc", "expr", "0", "Center z", "m", show_if={"shape": ["circle", "ellipse", "annulus"]}),
    Field("yc", "expr", "0", "Center y", "m", show_if={"shape": ["circle", "ellipse", "annulus"]}),
    Field("radius", "expr", "1e-3", "Radius", "m", show_if={"shape": ["circle", "annulus"]}),
    Field("radius_inner", "expr", "5e-4", "Inner radius", "m", show_if={"shape": ["annulus"]}),
    Field("a", "expr", "1e-3", "Semi-axis z", "m", show_if={"shape": ["ellipse"]}),
    Field("b", "expr", "1e-3", "Semi-axis y", "m", show_if={"shape": ["ellipse"]}),
    Field("points", "text", "0,0; 1e-3,0; 0,1e-3", "Polygon points (z,y; ...)", "m",
          show_if={"shape": ["polygon"]}),
    Field("expression", "str", "z**2 + y**2 < 1e-6", "Inside if (expr of z,y)", "",
          show_if={"shape": ["expression"]}),
    Field("material", "choice", "conductor", "Material", choices=MATERIALS,
          help="Objects are painted in list order; 'vacuum' carves holes in earlier objects."),
    Field("voltage", "expr", "0", "Voltage V(t)", "V", show_if={"material": ["conductor"]}),
    Field("eps_r", "expr", "1", "Relative permittivity", "", show_if={"material": ["dielectric"]}),
    Field("loss_spec", "choice", "Q", "Loss specified by", choices=("Q", "conductivity"),
          show_if={"material": ["lossy"]},
          help="EM lossy load (e.g. a magnetron loss port).  Q: give the loaded quality factor at the "
               "mode frequency; the conductivity is sigma = 2 pi f eps0 / (Q * fill).  Use Model > "
               "'Calibrate loss Q' to adjust 'fill' with a cold-test ring-down so the structure has "
               "exactly this Q.  Particles pass through lossy regions."),
    Field("Q", "expr", "250", "Loaded Q", "", show_if={"material": ["lossy"], "loss_spec": ["Q"]}),
    Field("frequency", "expr", "5.8e9", "Mode frequency", "Hz",
          show_if={"material": ["lossy"], "loss_spec": ["Q"]}),
    Field("fill", "expr", "1", "Energy fill factor (calibrated)", "",
          show_if={"material": ["lossy"], "loss_spec": ["Q"]},
          help="Fraction of the mode's electric energy inside the lossy region (1 = region fills the "
               "mode).  Several lossy objects with the same Q and frequency form one load: give them the "
               "same fill (the total fraction in all of them).  Set automatically by 'Calibrate loss Q'."),
    Field("cal_period", "expr", "", "Calibration: mode period", "m",
          show_if={"material": ["lossy"], "loss_spec": ["Q"]},
          help="Optional, for periodic slow-wave structures (e.g. the vane period of a magnetron). If set, "
               "the cold test drives the first probe position and all its periodic images z + k*period "
               "with phase k*(mode phase), which excites the wanted mode alone. Without it a single point "
               "drive excites the neighbouring modes too, and close modes (pi and 2pi/3 in a 6-cavity "
               "magnetron) are hard to separate."),
    Field("cal_phase", "expr", "pi", "Calibration: mode phase per period", "rad",
          show_if={"material": ["lossy"], "loss_spec": ["Q"]},
          help="Phase advance of the calibrated mode per period: pi for the pi mode, 2*pi/3 for the "
               "2pi/3 mode. Used only when the mode period is set."),
    Field("sigma", "expr", "1", "Conductivity", "S/m",
          show_if={"material": ["lossy"], "loss_spec": ["conductivity"]}),
    Field("collect_charge", "bool", True, "Accumulate surface charge",
          show_if={"material": ["dielectric"]},
          help="Particles hitting a dielectric leave their charge on its surface."),
]

SPECIES_PRESETS = ("electron", "positron", "proton", "H+", "H2+", "He+", "Ar+", "Xe+", "Cs+", "custom")
SPECIES = [
    Field("name", "str", "electrons"),
    Field("preset", "choice", "electron", "Preset", choices=SPECIES_PRESETS),
    Field("charge", "expr", "-1", "Charge", "e", show_if={"preset": ["custom"]}),
    Field("mass", "expr", "me", "Mass", "kg", show_if={"preset": ["custom"]}),
    Field("beam_self_B", "bool", False, "Relativistic beam self-B (ES)",
          help="Electrostatic solver only: adds the magnetic self-field B = beta x E / c of this "
               "species (1/gamma^2 space-charge reduction for relativistic beams)."),
    Field("motion", "choice", "full", "Motion (B field model)", choices=("full", "z_only", "y_only"),
          help="full: 3V Boris motion in the applied fields - B = 0 if no Applied Field is defined, or "
               "any finite B (e.g. a uniform axial field B = [Bz, 0, 0] T in Applied Fields).  "
               "z_only: motion along z only = the limit of an INFINITE magnetic field along z (confined "
               "flow), as in Luginsland et al. (1996), Lau (2001) and Lau, Chernin, Heri & Zhang (2026).  "
               "y_only: motion along y (r in RZ) only = the limit of an INFINITE magnetic field along y, "
               "e.g. for a cathode lying along y = 0 that emits in +y (2D only)."),
]

LOADER = [
    Field("name", "str", "plasma"),
    Field("species", "str", "electrons"),
    Field("z0", "expr", "0", "Region z0", "m"),
    Field("z1", "expr", "1e-2", "Region z1", "m"),
    Field("y0", "expr", "0", "Region y0 / r0", "m"),
    Field("y1", "expr", "1e-2", "Region y1 / r1", "m"),
    Field("density", "expr", "1e15", "Density n(z,y)", "m^-3",
          help="Number or expression of z, y (or r)."),
    Field("ppc", "expr", "16", "Particles per cell"),
    Field("temperature", "expr", "0", "Temperature", "eV"),
    Field("drift", "vec3", ["0", "0", "0"], "Drift velocity", "m/s",
          help="(vz, vy/vr, vx/vtheta).  Non-relativistic value or gamma*v."),
    Field("quiet_start", "bool", False, "Quiet start (regular positions)"),
    Field("perturbation", "choice", "none", "Perturbation", choices=("none", "density", "velocity")),
    Field("pert_amplitude", "expr", "0.01", "Perturbation amplitude",
          help="density: relative amplitude;  velocity: m/s", show_if={"perturbation": ["density", "velocity"]}),
    Field("pert_mode", "expr", "1", "Mode number (along z)", show_if={"perturbation": ["density", "velocity"]}),
]

EMISSION_MODELS = ("beam", "scl", "fowler_nordheim", "thermionic", "photo")
NORMALS = ("auto", "+z", "-z", "+y", "-y")
EMITTER = [
    Field("name", "str", "cathode"),
    Field("species", "str", "electrons"),
    Field("model", "choice", "scl", "Emission model", choices=EMISSION_MODELS,
          help="beam: prescribed current & energy.  scl: space-charge-limited (Gauss law / "
               "Child-Langmuir).  fowler_nordheim: field emission (Murphy-Good).  thermionic: "
               "Richardson-Dushman + Schottky.  photo: prescribed J(t) with excess energy."),
    Field("surface", "choice", "segment", "Emitting surface", choices=("segment", "object"),
          help="segment: straight line (z0,y0)-(z1,y1) (a point z0 in 1D).  object: the whole "
               "surface of a conductor object that faces vacuum."),
    Field("object", "str", "", "Object name", show_if={"surface": ["object"]}),
    Field("z0", "expr", "0", "Segment z0", "m", show_if={"surface": ["segment"]}),
    Field("y0", "expr", "0", "Segment y0 / r0", "m", show_if={"surface": ["segment"]}),
    Field("z1", "expr", "0", "Segment z1", "m", show_if={"surface": ["segment"]}),
    Field("y1", "expr", "1e-3", "Segment y1 / r1", "m", show_if={"surface": ["segment"]}),
    Field("normal", "choice", "auto", "Emission direction", choices=NORMALS, show_if={"surface": ["segment"]}),
    Field("current_density", "expr", "1e4", "Current density J(t)", "A/m^2",
          show_if={"model": ["beam", "photo"]}),
    Field("energy", "expr", "0", "Beam / excess energy", "eV", show_if={"model": ["beam", "photo"]}),
    Field("temperature", "expr", "0", "Emission temperature", "eV",
          show_if={"model": ["beam", "photo", "scl"]}),
    Field("work_function", "expr", "4.5", "Work function", "eV",
          show_if={"model": ["fowler_nordheim", "thermionic"]}),
    Field("beta", "expr", "1", "Field enhancement factor", "", show_if={"model": ["fowler_nordheim"]}),
    Field("cathode_temperature", "expr", "2000", "Cathode temperature", "K",
          show_if={"model": ["thermionic"]}),
    Field("richardson", "expr", "1.20173e6", "Richardson constant", "A/m^2/K^2",
          show_if={"model": ["thermionic"]}),
    Field("schottky", "bool", True, "Schottky barrier lowering", show_if={"model": ["thermionic"]}),
    Field("relax", "expr", "1.0", "Gauss-law relaxation factor", show_if={"model": ["scl"]}),
    Field("profile", "expr", "1", "Spatial profile f(z,y)", "",
          help="Multiplies the emitted current (e.g. gauss(r,0,1e-3))."),
    Field("particles_per_cell", "expr", "4", "Macro-particles per cell per step"),
    Field("t_start", "expr", "0", "Start time", "s"),
    Field("t_stop", "expr", "1e9", "Stop time", "s"),
]

EXTFIELD_TYPES = ("uniform", "expression", "coil", "solenoid")
EXTFIELD = [
    Field("name", "str", "Bext"),
    Field("type", "choice", "uniform", "Type", choices=EXTFIELD_TYPES),
    Field("E", "vec3", ["0", "0", "0"], "Uniform E", "V/m", show_if={"type": ["uniform"]}),
    Field("B", "vec3", ["0", "0", "0"], "Uniform B", "T", show_if={"type": ["uniform"]}),
    Field("Ez", "str", "0", "Ez(z,y,t)", "V/m", show_if={"type": ["expression"]}),
    Field("Ey", "str", "0", "Ey / Er(z,y,t)", "V/m", show_if={"type": ["expression"]}),
    Field("Ex", "str", "0", "Ex / Etheta(z,y,t)", "V/m", show_if={"type": ["expression"]}),
    Field("Bz", "str", "0", "Bz(z,y,t)", "T", show_if={"type": ["expression"]}),
    Field("By", "str", "0", "By / Br(z,y,t)", "T", show_if={"type": ["expression"]}),
    Field("Bx", "str", "0", "Bx / Btheta(z,y,t)", "T", show_if={"type": ["expression"]}),
    Field("zc", "expr", "0", "Coil center z / solenoid start", "m", show_if={"type": ["coil", "solenoid"]}),
    Field("z_end", "expr", "1e-2", "Solenoid end z", "m", show_if={"type": ["solenoid"]}),
    Field("radius", "expr", "1e-2", "Coil radius", "m", show_if={"type": ["coil", "solenoid"]}),
    Field("ampere_turns", "expr", "1000", "Ampere-turns", "A", show_if={"type": ["coil", "solenoid"]}),
    Field("time_factor", "expr", "1", "Time factor f(t)"),
]

SOURCE_TYPES = ("laser", "tem_port", "plane_wave")
SOURCE = [
    Field("name", "str", "laser"),
    Field("type", "choice", "laser", "Type", choices=SOURCE_TYPES,
          help="EM only.  Current-sheet antenna launching a wave from plane z = z_position.  "
               "laser: Gaussian pulse/beam.  tem_port: TEM wave V(t) in a coax (RZ) or "
               "parallel-plate gap (ZY).  plane_wave: CW wave with ramp."),
    Field("z_position", "expr", "0", "Antenna plane z", "m"),
    Field("direction", "choice", "+z", "Propagation", choices=("+z", "-z")),
    Field("wavelength", "expr", "800e-9", "Wavelength", "m", show_if={"type": ["laser", "plane_wave"]}),
    Field("a0", "expr", "0.1", "Normalized amplitude a0", "", show_if={"type": ["laser", "plane_wave"]}),
    Field("polarization", "choice", "x", "Polarization", choices=("x", "y", "r"),
          show_if={"type": ["laser", "plane_wave"]},
          help="1D: x or y.  ZY: x = out of plane, y = in plane.  RZ: r (radial).")
    ,
    Field("duration", "expr", "30e-15", "Pulse duration (FWHM intensity)", "s", show_if={"type": ["laser"]}),
    Field("t_peak", "expr", "60e-15", "Peak time at antenna", "s", show_if={"type": ["laser"]}),
    Field("waist", "expr", "0", "Waist w0 (0 = plane)", "m", show_if={"type": ["laser"]}),
    Field("y_center", "expr", "0", "Transverse center", "m", show_if={"type": ["laser"]}),
    Field("focus_z", "expr", "0", "Focal plane z", "m", show_if={"type": ["laser"]}),
    Field("ramp", "expr", "10e-15", "Ramp time", "s", show_if={"type": ["plane_wave"]}),
    Field("voltage", "expr", "1e5*min(t/1e-9,1)", "Port voltage V(t)", "V", show_if={"type": ["tem_port"]}),
    Field("inner", "expr", "1e-3", "Inner radius / gap start", "m", show_if={"type": ["tem_port"]}),
    Field("outer", "expr", "2e-3", "Outer radius / gap end", "m", show_if={"type": ["tem_port"]}),
]

PROBE_QTY = ("Ez", "Ey", "Ex", "Bz", "By", "Bx", "phi", "rho")
PROBE = [
    Field("name", "str", "probe"),
    Field("quantity", "choice", "Ez", "Quantity", choices=PROBE_QTY,
          help="Value at one point vs time (history column probe_<name>).  Components: Ey=Er, "
               "Ex=Etheta in RZ.  In 1D: Ex,Ey transverse."),
    Field("z", "expr", "0", "z", "m"),
    Field("y", "expr", "0", "y / r", "m"),
]

LINE_PROBE = [
    Field("name", "str", "phi"),
    Field("quantity", "choice", "phi", "Quantity", choices=PROBE_QTY,
          help="The quantity is sampled along a straight line every step of the history; its minimum "
               "and/or maximum and the position where it occurs are recorded.  Example: phi along z on "
               "the emission axis gives the potential minimum (virtual-cathode depth) phi_min and its "
               "position z_min.  The live monitor can also show the whole profile (Field: 'line: <name>')."),
    Field("extremum", "choice", "min", "Record", choices=("min", "max", "min and max"),
          help="History columns, for a probe named 'phi' along z:  always probe_phi_avg (average along "
               "the line);  min -> probe_phi_min [V] and probe_phi_zmin [m];  max -> probe_phi_max and "
               "probe_phi_zmax."),
    Field("along", "choice", "z", "Line along", choices=("z", "y"),
          help="z: the line z_from..z_to at a fixed y (r in RZ).  y: the line y_from..y_to at a fixed z."),
    Field("y", "expr", "0", "at y / r", "m", show_if={"along": ["z"]}),
    Field("z", "expr", "0", "at z", "m", show_if={"along": ["y"]}),
    Field("start", "expr", "0", "from", "m"),
    Field("end", "expr", "1e-3", "to", "m"),
]

POWERLINE = [
    Field("name", "str", "port_power"),
    Field("orientation", "choice", "z", "Line of constant", choices=("z", "y")),
    Field("position", "expr", "0", "Position of line", "m"),
    Field("start", "expr", "0", "From", "m"),
    Field("end", "expr", "1e-2", "To", "m"),
]

DIAGNOSTICS = [
    Field("output_dir", "str", "output", "Output folder",
          help="Relative to the project file location."),
    Field("history_every", "expr", "10", "History every N steps"),
    Field("field_every", "expr", "200", "Field snapshot every N steps"),
    Field("particle_every", "expr", "200", "Particle snapshot every N steps"),
    Field("max_dump_particles", "expr", "50000", "Max particles per snapshot"),
    Field("live_every", "expr", "50", "Live plot update every N steps"),
]

PARALLEL = [
    Field("mode", "choice", "threads", "Parallel mode", choices=("serial", "threads", "mpi"),
          help="threads: shared-memory multi-core (numba), one process.  mpi: particle decomposition "
               "over several processes (mpi4py + an MPI library); Run starts 'mpiexec -n <ranks>' "
               "automatically.  The field solve is repeated on every rank, so MPI pays off for "
               "particle-dominated runs (many particles per cell)."),
    Field("ranks", "int", 4, "MPI processes (ranks)", show_if={"mode": ["mpi"]}),
    Field("threads", "int", 0, "Threads per process (0 = auto)",
          help="0 = all cores for 'threads'; with MPI, 0 = cores / ranks for each process."),
]

SWEEP = [
    Field("enabled", "bool", False, "Enable parametric sweep"),
    Field("parameter", "str", "", "Parameter name"),
    Field("values", "str", "linspace(0,1,5)", "Values (list or expression)"),
    Field("workers", "int", 2, "Parallel workers"),
]

SINGLE_SECTIONS = {
    "simulation": SIMULATION,
    "domain": DOMAIN,
    "diagnostics": DIAGNOSTICS,
    "parallel": PARALLEL,
    "sweep": SWEEP,
}
LIST_SECTIONS = {
    "mesh_refinements": MESH_REFINE,
    "objects": OBJECT,
    "species": SPECIES,
    "loaders": LOADER,
    "emitters": EMITTER,
    "external_fields": EXTFIELD,
    "sources": SOURCE,
    "probes": PROBE,
    "line_probes": LINE_PROBE,
    "power_lines": POWERLINE,
}
SECTION_TITLES = {
    "parameters": "Parameters",
    "simulation": "Study / Solver",
    "domain": "Domain & Mesh",
    "mesh_refinements": "Mesh Refinement",
    "boundaries": "Boundaries",
    "objects": "Geometry Objects",
    "species": "Species",
    "loaders": "Particle Loading",
    "emitters": "Emission",
    "external_fields": "Applied Fields",
    "sources": "EM Sources / Ports",
    "probes": "Point Probes",
    "line_probes": "Line Probes (min / max)",
    "power_lines": "Power Monitors",
    "diagnostics": "Output",
    "parallel": "Parallel",
    "sweep": "Parametric Sweep",
}


def defaults(fields):
    out = {}
    for f in fields:
        v = f.default
        out[f.name] = list(v) if isinstance(v, list) else v
    return out


def is_visible(f: Field, values: dict) -> bool:
    for k, allowed in f.show_if.items():
        if values.get(k) not in allowed:
            return False
    return True


# ----------------------------------------------------------------------------------------------
# Plain-language meaning of every input that has no help text above.  Shown as tooltips and in the
# "Meaning of the settings" paragraph at the end of every form.
DESCRIPTIONS = {
    "simulation": {
        "t_end": "Physical time to simulate. The number of steps is End time / dt.",
        "relativistic": "On: momentum u = gamma v is pushed (needed above ~10 keV). Off: classical "
                        "Newtonian push (gamma = 1).",
        "seed": "Seed of the random-number generator (particle positions, thermal velocities). The same "
                "seed reproduces the same run.",
        "max_particles": "Safety limit on the number of macro-particles; the run stops with a message when "
                         "it is exceeded (protects memory when, e.g., emission runs away).",
    },
    "domain": {
        "z_min": "Lower end of the simulation box along z (the axial direction in RZ).",
        "z_max": "Upper end of the simulation box along z.",
        "nz": "Number of cells along z of the background (uniform) mesh; dz = (z max - z min) / nz. "
              "Mesh Refinement can add finer cells locally.",
        "y_max": "Upper end of the box along y (the radius r in RZ).",
        "ny": "Number of cells along y / r of the background mesh.",
    },
    "boundaries": {
        "voltage": "Potential of this side when it is a conductor (V). May depend on time t, e.g. "
                   "V0*min(t/1e-9,1) for a 1 ns ramp.",
        "particles": "absorb: particles leaving through this side are removed (and counted as current). "
                     "reflect: specular reflection. periodic: they re-enter at the opposite side.",
        "absorber_cells": "Thickness (in cells) of the graded absorbing layer of an open EM boundary; "
                          "16-32 is typical.",
    },
    "diagnostics": {
        "history_every": "How often (in steps) currents, energies, probes and powers are written to the "
                         "history (History plot and history.csv).",
        "field_every": "How often (in steps) full field maps (phi, rho, E, B) are saved for the Results tab.",
        "particle_every": "How often (in steps) particle snapshots (positions, momenta) are saved.",
        "max_dump_particles": "Maximum number of particles stored per snapshot (a random subset is saved "
                              "when there are more).",
        "live_every": "How often (in steps) the Live monitor is refreshed while the run proceeds.",
    },
    "parallel": {
        "ranks": "Number of MPI processes started by Run (mpiexec -n ranks). Particles are divided among "
                 "them; every rank solves the fields.",
    },
    "sweep": {
        "enabled": "Run the model several times, once per value of the chosen parameter.",
        "parameter": "Name of a global parameter (from Parameters) to vary, e.g. Jn or V0.",
        "values": "Values for the sweep: a list [1, 2, 3] or an expression such as linspace(0.6, 1.3, 8).",
        "workers": "Number of sweep runs executed at the same time.",
    },
    "mesh_refinements": {
        "name": "Label of this refinement region.",
        "enabled": "Untick to keep the region in the model but not use it.",
        "start": "Where the refined band begins (z or y/r, see Refine along).",
        "end": "Where the refined band ends.",
        "size": "Target cell size inside the band (m), e.g. 1/10 of the emitter width.",
    },
    "objects": {
        "name": "Label of the object (used for current diagnostics I_<name> and in emitters on objects).",
        "shape": "Geometric shape. rectangle: z0..z1 x y0..y1. circle / annulus / ellipse: centre and "
                 "radii. polygon: list of corner points. expression: inside where the expression is true.",
        "z0": "Lower z corner of the rectangle.",
        "z1": "Upper z corner of the rectangle.",
        "y0": "Lower y (r) corner of the rectangle.",
        "y1": "Upper y (r) corner of the rectangle.",
        "zc": "Centre z of the circle / annulus / ellipse.",
        "yc": "Centre y (r) of the circle / annulus / ellipse.",
        "radius": "Radius of the circle (outer radius of the annulus).",
        "radius_inner": "Inner radius of the annulus (the hole).",
        "a": "Semi-axis of the ellipse along z.",
        "b": "Semi-axis of the ellipse along y.",
        "points": "Polygon corners as 'z1,y1; z2,y2; ...' (expressions allowed).",
        "expression": "Logical expression of z and y (r) that is true inside the object, e.g. "
                      "(z-1e-3)**2 + y**2 < (5e-4)**2.",
        "voltage": "Potential of a conductor object (V); may depend on time t.",
        "eps_r": "Relative permittivity of a dielectric object.",
        "Q": "Target loaded quality factor of the resonator due to this lossy load (Q specification).",
        "frequency": "Frequency of the mode for which the loaded Q is specified (Hz).",
        "sigma": "Electrical conductivity of the lossy load (S/m), when the loss is given directly.",
        "cal_period": "Period of the slow-wave structure for the Q calibration (optional): the cold test then "
                      "drives every period with the mode phase, so only the wanted mode is excited.",
        "cal_phase": "Phase advance per period of the calibrated mode (pi for the pi mode).",
    },
    "species": {
        "name": "Label of the species, used by Particle Loading and Emission.",
        "preset": "Particle type with known charge and mass (electron, proton, ions...). custom: give the "
                  "charge and mass yourself.",
        "charge": "Charge in units of the elementary charge e (electron = -1).",
        "mass": "Mass in kg (constants me, mp, amu may be used).",
    },
    "loaders": {
        "name": "Label of this initial particle load.",
        "species": "Which species is loaded.",
        "z0": "Lower z of the loaded region.",
        "z1": "Upper z of the loaded region.",
        "y0": "Lower y (r) of the loaded region.",
        "y1": "Upper y (r) of the loaded region.",
        "ppc": "Macro-particles per cell. More particles = less noise, slower run.",
        "temperature": "Thermal energy kT of the loaded particles (eV); 0 = cold.",
        "quiet_start": "Place particles on a regular sub-grid (and, with a temperature and >= 16 per "
                       "cell, on exact Maxwellian quantiles) for very low initial noise.",
        "perturbation": "Optional small sinusoidal perturbation to seed a wave or instability: of the "
                        "density or of the z velocity.",
        "pert_mode": "Number of wavelengths of the perturbation across the loaded region along z.",
    },
    "emitters": {
        "name": "Label of the emitter (current diagnostic Iemit_<name>).",
        "species": "Species of the emitted particles (usually electrons).",
        "object": "Name of the geometry object whose whole surface emits.",
        "z0": "z of the first end of the emitting segment.",
        "y0": "y (r) of the first end of the emitting segment.",
        "z1": "z of the second end of the emitting segment.",
        "y1": "y (r) of the second end of the emitting segment.",
        "normal": "Direction in which particles are emitted. auto: into the vacuum side; or force +z, -z, "
                  "+y, -y.",
        "current_density": "Prescribed emitted current density (A/m^2) for the beam and photo models; "
                           "may depend on time t. Above the space-charge limit a virtual cathode forms.",
        "energy": "beam: kinetic energy of every injected particle (eV). photo: maximum excess energy "
                  "(uniform 0..E). 0 = emitted at rest.",
        "temperature": "Thermal spread kT of the emission velocities (eV); 0 = cold emission.",
        "work_function": "Work function of the cathode material (eV), used by Fowler-Nordheim and "
                         "thermionic emission.",
        "beta": "Local field enhancement (Fowler-Nordheim): the surface field is multiplied by beta "
                "(tips, roughness).",
        "cathode_temperature": "Cathode temperature for thermionic (Richardson-Dushman) emission (K).",
        "richardson": "Richardson constant A_R (A/m^2/K^2); 1.2e6 is the theoretical value.",
        "schottky": "Include Schottky barrier lowering by the surface field in thermionic emission.",
        "relax": "Fraction of the Gauss-law charge emitted per step by the space-charge-limited model "
                 "(1 = full; lower values damp oscillations of the emitted current).",
        "particles_per_cell": "Macro-particles created per surface panel (cell) and per time step.",
        "t_start": "Emission starts at this time (s).",
        "t_stop": "Emission stops at this time (s).",
    },
    "external_fields": {
        "name": "Label of this applied field.",
        "type": "uniform: constant E and B. expression: E(z,y,t), B(z,y,t) formulas. coil: one current "
                "loop. solenoid: finite solenoid of current loops.",
        "E": "Constant applied electric field [z, y, x] components (V/m); in RZ [z, r, theta].",
        "B": "Constant applied magnetic field [z, y, x] components (T); in RZ [z, r, theta]. An axial "
             "field is [Bz, 0, 0].",
        "Ez": "Applied E_z as an expression of z, y (r) and t (V/m).",
        "Ey": "Applied E_y (E_r in RZ) as an expression (V/m).",
        "Ex": "Applied E_x (E_theta in RZ) as an expression (V/m).",
        "Bz": "Applied B_z as an expression of z, y (r) and t (T).",
        "By": "Applied B_y (B_r in RZ) as an expression (T).",
        "Bx": "Applied B_x (B_theta in RZ) as an expression (T).",
        "zc": "Axial position of the coil (start of the solenoid).",
        "z_end": "Axial end of the solenoid.",
        "radius": "Radius of the coil / solenoid windings.",
        "ampere_turns": "Current times number of turns (A-turns) of the coil or solenoid.",
        "time_factor": "Multiplies the whole applied field: f(t), e.g. 1 or sin(2*pi*1e9*t).",
    },
    "sources": {
        "name": "Label of the EM source.",
        "z_position": "Axial plane where the antenna radiates (a few cells inside an absorbing "
                      "boundary).",
        "direction": "Direction of propagation of the launched wave.",
        "wavelength": "Vacuum wavelength of the laser / wave (m).",
        "a0": "Normalized vector potential a0 = e E / (m c w); 1 is relativistic intensity.",
        "duration": "Pulse length, full width at half maximum of the intensity (s). For WarpX's 1/e "
                    "field duration divide by 1.177.",
        "t_peak": "Time at which the pulse peak passes the antenna (s).",
        "waist": "Gaussian beam waist w0 (m); 0 = transversely uniform plane wave.",
        "y_center": "Transverse position of the beam axis (y or r).",
        "focus_z": "Axial position of the focal plane.",
        "ramp": "Rise time of a continuous wave / port voltage (s).",
        "voltage": "TEM port driving voltage V(t) (V).",
        "inner": "Inner conductor radius (RZ coax) or start of the gap (ZY) of the TEM port.",
        "outer": "Outer conductor radius (RZ coax) or end of the gap (ZY).",
    },
    "probes": {
        "name": "Label of the probe; its history column is probe_<name>.",
        "z": "z position of the probe point.",
        "y": "y (r) position of the probe point.",
    },
    "line_probes": {
        "name": "Label; history columns probe_<name>_avg (line average), probe_<name>_min / _zmin "
                "(and _max / _zmax).",
        "y": "y (r) of the line when it runs along z.",
        "z": "z of the line when it runs along y.",
        "start": "Start of the line (z or y).",
        "end": "End of the line (z or y).",
    },
    "power_lines": {
        "name": "Label; the Poynting power through the line is the history column P_<name>.",
        "orientation": "z: a line of constant z (power flowing along z). y: a line of constant y.",
        "position": "The constant coordinate of the line.",
        "start": "Start of the line along the other coordinate.",
        "end": "End of the line along the other coordinate.",
    },
}


def _apply_descriptions():
    secs = dict(SINGLE_SECTIONS)
    secs.update(LIST_SECTIONS)
    secs["boundaries"] = BOUNDARY
    for sec, d in DESCRIPTIONS.items():
        for f in secs.get(sec, []):
            if not f.help and f.name in d:
                f.help = d[f.name]


_apply_descriptions()
