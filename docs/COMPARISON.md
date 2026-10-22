# VEXPACE compared with other PIC codes

## Feature overview

The entries for the other codes reflect their public documentation as generally known. Check each code's own documentation for current details.

| | **VEXPACE** | WarpX | XOOPIC | MAGIC (2D) |
|---|---|---|---|---|
| Language / install | Pure Python + Numba, `pip install` | C++/AMReX (+ Python PICMI), compiled | C++, X11 GUI, compiled | Commercial, Windows |
| Geometries | 1D-Z, 2D-ZY, 2D-RZ | 1D, 2D-XZ, RZ (multi-mode), 3D | 2D XY, RZ | 2D XY / RZ / Rθ, 3D (MAGIC3D) |
| ES solver | scikit-fem FEM (Q1/P1), dielectrics, conductors | MLMG multigrid, embedded boundaries | Multigrid / DADI | Poisson (quasi-static) |
| EM solver | Yee FDTD, charge-conserving (Esirkepov) | Yee, CKC, PSATD, hybrid | Yee | Yee-type (several variants) |
| Conductors | staircase (painted shapes) | embedded boundaries (cut cells) | staircase | conformal / staircase |
| Emission models | SCL (Gauss), FN, thermionic+Schottky, photo, beam; on object surfaces | flux injection, plasma loading; emission via Python | Gauss/SCL, FN, beam, secondary | explosive, FN, thermionic, photo, secondary, beam, gyro |
| Absorbing / ports | graded matched layer, TEM port, laser antenna | PML, Silver-Müller, laser antenna | Port, absorbing | FREESPACE, PORT (TEM, voltage) |
| Collisions / MCC | not yet (roadmap) | yes (MCC, binary, DSMC) | yes (MCC) | yes (gas models) |
| Parallel | Numba threads, MPI particle decomposition, sweeps | MPI + OpenMP/GPU, AMR | serial | parallel |
| GUI / workflow | **COMSOL-style model builder, live monitor, post-processing, parametric sweeps** | input files / Python | X11 interactive | MAGIC GUI / MCL files |

## Algorithm-by-algorithm comparison with WarpX

WarpX is distributed under BSD-3-Clause-LBNL. Selected VEXPACE particle
routines are translated or adapted from WarpX; other components implement
established numerical methods with a different architecture. See
[third-party notices](../THIRD_PARTY_NOTICES.md) for provenance and licensing.
This table compares numerical formulations in the specified configurations,
not literal source identity. Benchmark agreement is not proof of identical
behavior in every configuration.

| Component | WarpX | VEXPACE | Verdict |
|---|---|---|---|
| Yee FDTD, Cartesian 1D/2D | `EvolveB/E.cpp`, `CartesianYeeAlgorithm.H` | `fields/fdtd.py`, `fdtd_kernels.py` | **same stated numerical formulation** (same staggering and stencils; ZY uses a left-handed (z, y, x) basis with matching curl signs) |
| Yee FDTD, RZ m = 0, off axis | `CylindricalYeeAlgorithm.H` (`UpwardDrr_over_r`, `DownwardDrr_over_r`) | same kernels, `rz=True` | **same stated numerical formulation** |
| RZ on-axis E_z | 4·B_θ/dr with the π·dr²/3 axis volume | 3·B_θ/dr with the same volume (Verboncoeur 2001, exact discrete Gauss law on the axis); 4 if *RZ axis volume correction* is off | **different, documented**: VEXPACE keeps the Gauss-consistent factor |
| Loop order | push, deposit, filter J, B½, E, B½ (`OneStep_nosub`) | push, boundaries, emission, deposit, (filter J), B½, E, B½ | **similar numerical method, with noted differences**. WarpX applies particle boundaries/injection after the field solve, VEXPACE before deposition, so absorbed particles deposit up to the wall. |
| Time step, EM | `ComputeDt`, RZ uses the 1.2105/dr² limit, cfl 0.999 | same formulas (RZ 1.2105/dr² term), default cfl 0.95 | **same stated numerical formulation** formula; VEXPACE's default CFL is more conservative |
| Esirkepov current deposition (1D, 2D, RZ) | `doEsirkepovDepositionShapeN` + RZ volume scaling | `kernels.esirkepov_*` | **same stated numerical formulation** for linear shape (VEXPACE: linear/CIC only) |
| J filter | bilinear 1-2-1, on by default in Cartesian EM (`warpx.use_filter`) | *Current filter passes* (`current_filter`), default 0 | **same stated numerical formulation** when enabled (benchmark 29; Gauss law exact for the filtered charge, unit test) |
| Field gather | `doGatherShapeN`, energy-conserving (Galerkin) by default | `gather_ec_yee_*` (EM default), `gather_nodal_*` | **same stated numerical formulation** in EM. ES default is momentum-conserving: the Gauss-law SCL emitter is 2–4 % more accurate with it (see THEORY §3); EC is available. |
| Pushers Boris / Vay / Higuera–Cary, −dt/2 start, RZ rotation | `UpdateMomentum*.H`, `UpdatePosition.H`, `PushP(-dt/2)` | `kernels._momentum`, `push_fields_*` | **same stated numerical formulation** (benchmark 26: 1e-8…1e-15) |
| Field boundaries | PEC (`WarpX_PEC.cpp`), periodic, split-field PML outside the domain (not in RZ FDTD), Silver–Müller | PEC, PMC, periodic; graded matched-loss absorbing layer inside the domain (also in RZ) | PEC/periodic **similar numerical method, with noted differences**; absorber **different** (more reflection at oblique incidence than a PML; benchmark 11) |
| Particle boundaries | absorb, reflect, periodic, thermal (`ParticleBoundaries_K.H`) | absorb (with the current deposited up to the wall), reflect, periodic | **similar numerical method, with noted differences** (no thermal reflection in VEXPACE) |
| Laser antenna | antenna macro-particles depositing J (`LaserParticleContainer`), Gaussian profile with 1/e field duration, cos phase | current sheet J = −2ε0·c·E at a node (`sources/em_sources.py`), FWHM-intensity duration, sin phase | **different implementation**, same emitted sheet strength. Convert τ_WarpX = τ_FWHM/√(2 ln 2) when comparing. |
| Electrostatic solver | MLMG finite differences (5-point) | scikit-fem FEM (Q1, r-weighted), direct LU, conductors/dielectrics painted | **different** (both 2nd order; benchmarks 01, 17, 18 agree with WarpX) |
| Emission | flux injection only (`AddPlasmaFlux`) | Gauss-law SCL, Fowler–Nordheim, thermionic + Schottky, photo, beam (charge-conserving) | **VEXPACE only** |
| Initial fields | zero unless `initialize_self_fields` | ES: Poisson; EM: Poisson-initialised by default (`initial_field = none` for WarpX-like start) | **different default**, selectable |

VEXPACE is not a complete translation of WarpX. The software architectures differ; reported benchmark agreement applies to the tested configurations only.

## Code-to-code results (benchmarks 16–19, 26)

Identical problems were run with WarpX 26.09, built from source with the 1D, 2D and RZ dimensions:

| Problem | Compared quantity | Difference |
|---|---|---|
| 1D ES two-stream instability | growth rate, saturation energy | 0.04 %, 4 % |
| 1D ES Pierce diode at J_CL | potential profile φ(z) | 0.05 % of V |
| RZ ES space-charge beam expansion | beam edge radius a(z) | ≤ 1.4 % |
| 1D EM beam-driven plasma wake (both with WarpX defaults: J filter + energy-conserving gather) | E_z line-out behind the bunch | 0.06 % of E_lin |
| single-particle pushers (Boris, Vay, Higuera–Cary) | trajectories in gyration, E×B, relativistic E×B (γ = 20) | 1e-8 … 1e-15 of the path |
| planar smooth-bore magnetron gap (2D, B = 0.6 and 1.3 B_H) | potential across the gap, electrons in the gap | ≤ 1.2e-4 of V, ≤ 0.02 % |
| cylindrical (coaxial) smooth-bore magnetron gap (RZ, B = 0.6 and 1.3 B*) | potential across the gap, electrons in the gap | ≤ 1e-6 of V, ≤ 0.01 % |

In the crossed-field comparisons (benchmarks 30, 31) WarpX's RZ solver needs r_min = 0, so its inner cathode is an embedded boundary held at −V; VEXPACE uses an annular domain. Both codes inject the same electron flux, use linear shape functions, the Boris pusher, energy-conserving gather and the same time step.

XOOPIC and MAGIC are compared through published laws that those codes are commonly benchmarked against:

* Child–Langmuir (1D, relativistic), Lau's 2D strip law, and Langmuir–Blodgett coaxial flow;
* Hull cutoff;
* cavity eigenfrequencies;
* the A6 magnetron: π-mode cold test 2.313 GHz. The hot test at the experimental point (360 kV, 0.8 T; Bekefi & Orzechowski 1976) oscillates in the π mode at 2.16 GHz, 6 % below the measured ~2.3 GHz;
* Hull (planar; cylindrical after Lovelace & Ott) and Buneman–Hartree thresholds: the planar vane magnetron oscillates inside the Hull–BH window and stays quiet above the BH field;
* the Pierce diode instability: exact dispersion roots, as reproduced by XPDP1/XOOPIC.

See `benchmarks/results/VALIDATION_REPORT.md`.

## Roadmap

Planned additions:

* Monte-Carlo collisions (electron–neutral, charge exchange) and secondary electron emission.
* Cut-cell conductors and refined meshes for the EM solver (graded, locally refined tensor meshes are already available for the ES solver).
* PML absorbing layers and the CKC / PSATD Maxwell solvers of WarpX.
* An Rθ geometry for cylindrical magnetrons.
* External circuits.
* GPU kernels (numba-cuda).
