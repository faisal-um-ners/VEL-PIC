# VEL-PIC 1.3.0

First public release of VEL-PIC (Vacuum Electrodynamics and Laser Particle-in-Cell).

- Geometries 1D-Z, 2D-ZY and 2D-RZ; finite-element electrostatic solver (scikit-fem) and Yee FDTD
  electromagnetic solver.
- Space-charge-limited, Fowler–Nordheim, thermionic, photo- and beam emission; Boris, Vay and Higuera–Cary
  pushers; uniform, coil, solenoid and analytic applied fields; lasers, plane waves and TEM ports.
- Model-builder graphical interface with live monitor and results viewer; command-line runner, parametric
  sweeps, MPI particle decomposition and a Python API.
- 33 ready-to-run examples, including diodes below and above the limiting current, a planar vane magnetron
  and the A6 relativistic magnetron.
- Validation suite of 107 checks against analytic theory, published results and WarpX.

License: BSD 3-Clause. Citation: see `CITATION.cff` or the "How to cite" section of the README.
