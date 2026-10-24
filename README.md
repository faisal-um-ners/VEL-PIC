<p align="center"><img src="docs/images/vexpace_logo.png" alt="VEXPACE" width="420"></p>

<h1 align="center">VEXPACE</h1>
<p align="center"><b>Virtual Experiments in Plasma and Computational Electrodynamics</b><br>
An Open-Source Particle-in-Cell Simulation Framework for Plasma and Vacuum Electronics</p>

<p align="center">
<a href="LICENSE"><img alt="License: BSD-3-Clause" src="https://img.shields.io/badge/license-BSD--3--Clause-blue.svg"></a>
<img alt="Python 3.9+" src="https://img.shields.io/badge/python-3.9%2B-blue.svg">
<img alt="Validation: 107/107" src="https://img.shields.io/badge/validation-107%2F107%20passed-brightgreen.svg">
<a href="https://doi.org/10.5281/zenodo.23512471"><img alt="DOI 10.5281/zenodo.23512471" src="https://zenodo.org/badge/DOI/10.5281/zenodo.23512471.svg"></a>
</p>

VEXPACE is a particle-in-cell (PIC) code for space-charge-limited flow, electron beams, plasmas and vacuum
electron devices. It is written in Python with Numba-compiled kernels and comes with a model-builder
graphical interface, a command-line runner and a Python API. (Version 1.3.0 was released under the earlier
name VEL-PIC.)

| Geometry | Electrostatic | Electromagnetic |
|---|---|---|
| **1D-Z** (planar) | finite elements (scikit-fem [[1]](#references), P1) | Yee FDTD, charge-conserving |
| **2D-ZY** (Cartesian, per unit depth) | Q1 graded bands or P1 local triangular meshes | Yee FDTD, all six field components |
| **2D-RZ** (axisymmetric) | Q1 / local P1 finite elements with r-weighted weak form | cylindrical Yee FDTD with exact on-axis update |

<p align="center"><img src="docs/images/gui_model_builder.png" alt="VEXPACE model builder" width="900"></p>

## Features

- **Model builder.** Parameters and expressions in every field, geometry objects (rectangles, circles,
  ellipses, annuli, polygons, implicit shapes) as conductors with V(t), dielectrics, absorbers, resistive
  loads of prescribed Q, or vacuum. Every settings form explains each input.
- **Electrostatic mesh.** Graded refinement along z or y/r, plus manually prescribed 2D rectangles and
  polygons with safety buffers and graded transitions. The local triangular mesh is fixed before PIC.
- **Emission.** Space-charge-limited (Gauss's law), Fowler–Nordheim (Murphy–Good), thermionic
  (Richardson–Dushman with Schottky lowering), photoemission and prescribed beams, from segments or from
  the whole surface of a conductor.
- **Particles.** Boris, Vay and Higuera–Cary pushers; energy- or momentum-conserving gather; species motion
  in the applied fields or confined to one direction (the infinite-magnetic-field limit along z or y).
- **Fields.** Uniform, analytic, coil and solenoid fields; laser antennas, plane waves and TEM ports;
  graded absorbing layers; electrodes held at their potentials in electromagnetic runs.
- **Diagnostics.** Currents on every electrode, point and line probes (minimum, maximum and average along a
  line), Poynting power, field and phase-space snapshots, spectra, beam envelope and emittance, animations.
- **Performance.** Numba-threaded kernels, MPI particle decomposition and parallel parametric sweeps.
- **Validation.** 107 checks against analytic theory, published results and the WarpX code
  ([validation report](benchmarks/results/VALIDATION_REPORT.md)).

## Results

### Field-only electrostatics: two-cylinder lens

VEXPACE also solves pure electrostatic problems without particles. The example `two_cylinder_lens_rz` is the
classic immersion lens: two coaxial tubes of radius R at V₁ = 0 and V₂ = 1 kV with a gap of 0.1 D, on a mesh
refined at the gap and the tube walls [[2, 3]](#references). Along the axis the potential agrees with the
exact series solution to 3×10⁻⁴ V₂ and the peak field to 0.2 %.

<p align="center"><img src="docs/images/lens_validation.png" alt="Two-cylinder lens: axial potential and field versus the exact series" width="900"></p>

For a run without particles the live monitor shows the potential map, the line-probe profiles φ(z) and E_z(z)
on the axis, and a second field component.

<p align="center"><img src="docs/images/gui_lens_live.png" alt="Two-cylinder lens in the live monitor" width="900"></p>

### Limiting current of small emitters

Space-charge-limited flow from a strip (2D-ZY) and a disk (2D-RZ) emitter with injection energy
Δ·eV₀ = 25 eV at V₀ = 2.5 kV, following Lau et al. (2026) [[4]](#references). The Geometry tab previews the
mesh before the run; manual 2D polygons and rectangles can resolve the beam, the cathode and possible
return paths. The live monitor shows the axial potential and its minimum, phase space, particles and the
history of φ_min. The stationary reference solver agrees with the matched strip POISSON values within
0.25%; the [local P1 checks](docs/LOCAL_MESH_VERIFICATION.md) give errors of 0.054–0.206% for three cases.
These stationary comparisons are separate from the below/above-limit PIC behavior shown here.

**Mesh setup in VEXPACE (full domain).** Both electrodes and periodic sides are visible. A prescribed
polygon envelope and cathode rectangle provide a fine
interior, safety buffers and a coarser surrounding mesh. This is a setup preview; the two live-monitor
images below retain the original graded-mesh PIC runs.

<p align="center"><img src="docs/images/gui_diode_local_mesh.png" alt="VEXPACE Geometry view: manual 2D polygon and rectangular cathode refinement" width="900"></p>

**Below the limit (strip, J = 0.96 J_max).** The flow is steady: φ_min settles at −4.6 V, well above
−Δ·V₀ = −25 V, and the whole current reaches the anode.

<p align="center"><img src="docs/images/gui_diode_strip_zy.png" alt="Strip emitter below the limiting current" width="900"></p>

**Above the limit (disk, J = 1.04 J_max).** The potential minimum reaches −Δ·V₀, a virtual cathode forms and
oscillates around φ_min ≈ −25.5 V (FFT inset: peak at 7.8 GHz), and part of the current returns to the cathode.

<p align="center"><img src="docs/images/gui_diode_disk_rz.png" alt="Disk emitter above the limiting current" width="900"></p>

Examples: `yylau_2026_strip_caseA_zy` (below the limit) and `yylau_2026_disk_virtual_cathode_rz` (above it); geometry,
normalisation and the limiting-current equation (their Eq. 7) are those of Ref. [[4]](#references).

### Manual mesh refinement for electrostatics

Small emitters require fine resolution near injection, potential barriers and returning electrons, while
the surrounding field still needs a well-resolved background. **Mesh Refinement** provides three choices:

- **Along z / Along y:** the existing graded bands on a tensor-product Q1 mesh.
- **2D:** a filled **rectangle** or **polygon** with a chosen cell size, a **Safety buffer outside** and a
  **Growth ratio outside**. Regions combine using the finer requested size; the rest of the domain keeps
  the resolution set in **Domain & Mesh**.

In 2D mode, VEXPACE builds conforming P1 triangles, solves Poisson on those triangles, deposits particle
charge with their shape functions and gathers the electric field from the same potential. Prescribe the
expected electron region before PIC, including room for transverse spreading and cathode returns. The
mesh stays fixed during the run; a refinement outline guides resolution and does not confine particles.
Emitter endpoints are fitted locally at the boundary without adding an unwanted line across the domain.

The comparison below uses actual rectangle/polygon meshes at d = 1 cm. The strip mesh has **29,954
triangles** and the disk mesh **18,192**. A uniform P1 mesh with the same smallest leaf widths
(31.25 µm × 39.06 µm) needs **655,360** and **327,680** triangles, respectively: **21.9×** and **18.0×**
more. Panel **a** shows the full strip domain with both electrodes and periodic sides; the dashed frame
marks the near-emitter detail in **b**. Panel **c** compares full-domain element budgets. These are element
counts, not measured runtime speedups; the uniform references use the same four-triangle center-fan construction.
All three panels have equal display dimensions; panel a uses separate coordinate scales to show the full domain.

<p align="center"><img src="docs/images/mesh_refinement.png" alt="Full-domain local triangular mesh, near-emitter polygon/rectangle detail, and strip/disk element budgets" width="1000"></p>

Refinement is available for **2D-ZY and 2D-RZ electrostatics**; use a gentle transition (growth 1.05–1.2)
and check mesh convergence. Local P1 runs require **smoothing = 0**. With the electromagnetic solver,
**Check model** and **Run** reject enabled refinement: the implemented Yee update and charge-conserving
deposition require a **uniform grid**.

Ready-to-edit examples: [`local_mesh_confined_strip_zy`](examples/local_mesh_confined_strip_zy.pic.json)
and [`local_mesh_b0_envelope_zy`](examples/local_mesh_b0_envelope_zy.pic.json). See the
[mesh guide](docs/LOCAL_MESH.md), [verification records](docs/LOCAL_MESH_VERIFICATION.md) and
[figure generator](docs/make_mesh_refinement.py) for settings and reproduction.

### Planar vane magnetron

A six-cavity planar vane magnetron at 170 kV and 0.16 T. It is a generic design made for VEXPACE, not a
reproduction of a particular device: the operating point lies between the relativistic planar Hull cutoff
(0.150 T) and the π-mode Buneman–Hartree field (0.180 T) [[6, 7]](#references), and the periodic geometry is the
planar analogue of the recirculating planar magnetron [[8]](#references) (model shown at the top of this page). Space-charge-limited emission from the cathode at y = 0 forms three spokes; the cavity field
grows from noise, locks into the π mode and saturates at about 35 MV/m and 5.70 GHz, 2.4 % below the cold-test
frequency. A resistive load in one cavity sets the loaded Q to 250.

The model: cathode at y = 0 (red, emitting over the full period), anode block with six vane cavities above the
1 cm interaction gap, periodic side boundaries, the loss port in cavity 1 and two field probes, on a uniform
240 × 90 Yee mesh (0.25 mm cells).

<p align="center"><img src="docs/images/gui_magnetron_mesh.png" alt="Planar magnetron geometry and mesh" width="900"></p>

The live monitor at the end of the run (25 ns) shows E_z in the π mode, the phase space, the spokes and the
cavity-2 E_z probe.

<p align="center"><img src="docs/images/gui_live_monitor.png" alt="Planar magnetron live monitor at 25 ns" width="900"></p>

Example: `planar_magnetron_em`.

## Installation

Python 3.9 or newer is required.

```bash
git clone https://github.com/faisal-um-ners/VEXPACE.git
cd VEXPACE
pip install -e ".[gui]"          # add ",mpi" for MPI support
```

Installer scripts are also provided: `install/install_mac_linux.sh` (macOS and Linux),
`install/install_windows.bat` (Windows) and the double-click launcher `install/VEXPACE.command` (macOS).
Kernels are compiled by Numba on first use and cached.

## Usage

```bash
python -m picstudio gui                                   # graphical interface
python -m picstudio run examples/child_langmuir_1d.pic.json
python -m picstudio sweep examples/thermionic_diode_1d.pic.json
mpiexec -n 4 python -m picstudio run model.pic.json --mode mpi
python -m picstudio check model.pic.json                  # validate a model
python -m picstudio info                                  # versions, threads, MPI
```

From Python:

```python
from picstudio import load_model, Simulation

model = load_model("examples/child_langmuir_1d.pic.json")
model["parameters"]["V0"] = "2000"
sim = Simulation(model, out_dir="out")
sim.run()
history, columns = sim.history_array()
```

The Python package is named `picstudio`; model files use the extension `.pic.json`.

## Examples

All examples are in [`examples/`](examples) and under **File → Examples** in the interface.

| Example | Geometry, solver | Physics |
|---|---|---|
| `child_langmuir_1d`, `virtual_cathode_1d` | 1D ES | Child–Langmuir diode; over-injected beam and virtual cathode |
| `child_langmuir_periodic_zy`, `child_langmuir_strip_zy`, `child_langmuir_disk_rz` | ZY / RZ ES | 2D space-charge-limited flow [[9, 10]](#references) |
| `coaxial_diode_rz` | RZ ES | Langmuir–Blodgett radial flow [[11]](#references) |
| `thermionic_diode_1d`, `field_emission_1d` | 1D ES | emission laws with space charge |
| `yylau_2026_strip_caseA_zy`, `yylau_2026_strip_E3_zy`, `yylau_2026_disk_rz` (+ `_virtual_cathode` variants) | ZY / RZ ES, refined mesh | limiting current of small emitters [[4]](#references) |
| `local_mesh_confined_strip_zy`, `local_mesh_b0_envelope_zy` | ZY ES, manual local P1 triangles | rectangle/polygon refinement for confined and expanding beams |
| `plasma_oscillation_1d`, `landau_damping_1d` | 1D ES | Langmuir oscillation; Landau damping (kλ_D = 0.5) |
| `two_stream_1d`, `beam_plasma_1d`, `beam_plasma_zy` | 1D / ZY ES | streaming instabilities |
| `pierce_diode_1d` | 1D ES | Pierce diode instability [[12]](#references) |
| `beam_expansion_zy`, `beam_drift_tube_rz`, `beam_solenoid_rz` | ZY / RZ ES | space-charge beam expansion; Brillouin focusing |
| `dielectric_charging_zy` | ZY ES | beam charging of a dielectric |
| `two_cylinder_lens_rz` | RZ ES, field only | two-cylinder electrostatic lens [[2, 3]](#references) |
| `planar_magnetron_es` | ZY ES | crossed-field diode, Hull cutoff [[6]](#references) |
| `planar_magnetron_em`, `a6_magnetron_em` | ZY EM | planar vane magnetron [[6–8]](#references); A6 relativistic magnetron [[13, 14]](#references) |
| `relativistic_diode_rz_em` | RZ EM | TEM-port-driven coaxial diode |
| `laser_plasma_1d`, `laser_plasma_zy`, `laser_focus_zy`, `laser_wakefield_1d` | 1D / ZY EM | laser–plasma interaction and focusing |
| `beam_plasma_wakefield_1d`, `plasma_wakefield_rz` | 1D / RZ EM | beam-driven plasma wakefields |

## Validation

`python benchmarks/run_all.py` runs the complete suite and writes
[`benchmarks/results/VALIDATION_REPORT.md`](benchmarks/results/VALIDATION_REPORT.md). Selected results:

| Benchmark | Result |
|---|---|
| Poisson solver, 1D / ZY / RZ | second-order convergence |
| Child–Langmuir law, 1 kV / 100 kV (relativistic) | +1.3 % / +1.5 % |
| 2D strip emitter vs Lau [[10]](#references), W/d = 1–4 | +0.2 … +1.9 % |
| Langmuir–Blodgett coaxial flow [[11]](#references) | +0.6 % |
| Two-cylinder lens, axial potential / peak field [[2, 3]](#references) | 3×10⁻⁴ V₂ / −0.19 % |
| Landau damping, kλ_D = 0.5 (exact Vlasov root) | γ +3.5 %, ω −0.04 % |
| Two-stream / beam–plasma growth rates | −0.25 % / −0.3 % |
| Pierce diode instability [[12]](#references), α = 1.5π / 2.5π | γ −2.1 % / −5.4 % |
| Cavity eigenmodes (TM, TE, RZ) | −0.02 % |
| Electromagnetic charge conservation | 10⁻¹⁵ |
| Limiting current of small emitters vs POISSON [[4]](#references) | +0.06 … +0.24 % |
| Code-to-code with WarpX [[15]](#references): two-stream, Pierce diode, RZ beam, EM wake | 0.04 %, 0.05 %, 1.4 %, 0.06 % |
| Particle pushers vs WarpX (Boris, Vay, Higuera–Cary) | 10⁻⁸ … 10⁻¹⁵ |
| Crossed-field gaps vs WarpX, planar / coaxial | potential ≤ 1.2·10⁻⁴ V₀ / 10⁻⁶ V₀ |
| A6 magnetron at 360 kV, 0.8 T (measured ≈ 2.3 GHz [[13, 14]](#references)) | π mode at 2.16 GHz |

## References

1. T. Gustafsson and G. D. McBain, "scikit-fem: A Python package for finite element assembly,"
   J. Open Source Softw. **5**, 2369 (2020). [doi:10.21105/joss.02369](https://doi.org/10.21105/joss.02369)
2. L. S. Goddard, "Optical characteristics of a two-cylinder electrostatic lens," Math. Proc. Camb. Phil. Soc.
   **42**, 106 (1946). [doi:10.1017/S0305004100022829](https://doi.org/10.1017/S0305004100022829)
3. R. D. Cook and D. W. O. Heddle, "The simple accurate calculation of cylinder lens potentials and focal
   properties," J. Phys. E: Sci. Instrum. **9**, 279 (1976). [doi:10.1088/0022-3735/9/4/012](https://doi.org/10.1088/0022-3735/9/4/012)
4. Y. Y. Lau, D. P. Chernin, Y. Heri and P. Zhang, "Solution to an unsolved problem in diodes: Limiting current for
   small emission area and low emission energy," arXiv:2607.24590 (2026). [doi:10.48550/arXiv.2607.24590](https://doi.org/10.48550/arXiv.2607.24590)
5. J. P. Verboncoeur, A. B. Langdon and N. T. Gladd, "An object-oriented electromagnetic PIC code,"
   Comput. Phys. Commun. **87**, 199 (1995). [doi:10.1016/0010-4655(94)00173-Y](https://doi.org/10.1016/0010-4655%2894%2900173-Y)
6. Y. Y. Lau, J. W. Luginsland, K. L. Cartwright, D. H. Simon, W. Tang, B. W. Hoff and R. M. Gilgenbach,
   "A re-examination of the Buneman–Hartree condition in a cylindrical smooth-bore relativistic magnetron,"
   Phys. Plasmas **17**, 033102 (2010). [doi:10.1063/1.3328804](https://doi.org/10.1063/1.3328804)
7. R. V. Lovelace and E. Ott, "Theory of magnetic insulation," Phys. Fluids **17**, 1263 (1974).
   [doi:10.1063/1.1694876](https://doi.org/10.1063/1.1694876)
8. M. A. Franzi, R. M. Gilgenbach, B. W. Hoff, D. A. Chalenski, D. Simon, Y. Y. Lau and J. Luginsland,
   "Recirculating-planar-magnetron simulations and experiment," IEEE Trans. Plasma Sci. **41**, 639 (2013).
   [doi:10.1109/TPS.2013.2242493](https://doi.org/10.1109/TPS.2013.2242493)
9. J. W. Luginsland, Y. Y. Lau and R. M. Gilgenbach, "Two-dimensional Child–Langmuir law,"
   Phys. Rev. Lett. **77**, 4668 (1996). [doi:10.1103/PhysRevLett.77.4668](https://doi.org/10.1103/PhysRevLett.77.4668)
10. Y. Y. Lau, "Simple theory for the two-dimensional Child–Langmuir law," Phys. Rev. Lett. **87**, 278301 (2001).
    [doi:10.1103/PhysRevLett.87.278301](https://doi.org/10.1103/PhysRevLett.87.278301)
11. I. Langmuir and K. B. Blodgett, "Currents limited by space charge between coaxial cylinders,"
    Phys. Rev. **22**, 347 (1923). [doi:10.1103/PhysRev.22.347](https://doi.org/10.1103/PhysRev.22.347)
12. J. R. Pierce, "Limiting stable current in electron beams in the presence of ions," J. Appl. Phys. **15**, 721
    (1944). [doi:10.1063/1.1707378](https://doi.org/10.1063/1.1707378)
13. G. Bekefi and T. J. Orzechowski, "Giant microwave bursts emitted from a field-emission, relativistic-electron-beam
    magnetron," Phys. Rev. Lett. **37**, 379 (1976). [doi:10.1103/PhysRevLett.37.379](https://doi.org/10.1103/PhysRevLett.37.379)
14. A. Palevsky and G. Bekefi, "Microwave emission from pulsed, relativistic e-beam diodes. II. The
    multiresonator magnetron," Phys. Fluids **22**, 986 (1979). [doi:10.1063/1.862663](https://doi.org/10.1063/1.862663)
15. L. Fedeli, A. Huebl, F. Boillod-Cerneux, T. Clark, K. Gott, C. Hillairet, S. Jaure, A. Leblanc, R. Lehe, A. Myers,
    C. Piechurski, M. Sato, N. Zaim, W. Zhang, J.-L. Vay and H. Vincenti, "Pushing the frontier in the design of
    laser-based electron accelerators with groundbreaking mesh-refined particle-in-cell simulations on
    exascale-class supercomputers," in *SC22: International Conference for High Performance Computing,
    Networking, Storage and Analysis* (IEEE, 2022), pp. 1–12. [doi:10.1109/SC41404.2022.00008](https://doi.org/10.1109/SC41404.2022.00008)
    WarpX source code: [github.com/BLAST-WarpX/warpx](https://github.com/BLAST-WarpX/warpx).

## Documentation

- [User guide](docs/USER_GUIDE.md): building, running and post-processing a model.
- [Numerical methods](docs/THEORY.md): mesh, solvers, emission, particle push, parallelisation.
- [Comparison with other codes](docs/COMPARISON.md): feature overview and algorithm-by-algorithm comparison with WarpX.
- [Manual local mesh refinement](docs/LOCAL_MESH.md): rectangle/polygon settings and fixed P1 mesh.
- [Local mesh verification](docs/LOCAL_MESH_VERIFICATION.md): stationary Lau comparisons, PIC behavior checks and unit tests.
- [Validation report](benchmarks/results/VALIDATION_REPORT.md).

## Repository layout

```
picstudio/      Python package: model schema, mesh, field solvers, particles, emission, GUI
examples/       ready-to-run models and their generator (make_examples.py)
benchmarks/     validation suite, WarpX comparison scripts and results
docs/           user guide, numerical methods, code comparison, figures
tests/          unit tests (pytest)
install/        installers and launchers
```

## How to cite

If VEXPACE contributes to published work, please cite it. The software is archived on Zenodo with the
permanent DOI [10.5281/zenodo.23512471](https://doi.org/10.5281/zenodo.23512471), which covers all versions.

> Faisal, M. A. (2026). *VEXPACE: An Open-Source Particle-in-Cell Simulation Framework for Plasma and Vacuum Electronics* [Computer software].
> Zenodo. https://doi.org/10.5281/zenodo.23512471

```bibtex
@software{faisal_vexpace,
  author    = {Faisal, Md Arifuzzaman},
  title     = {{VEXPACE}: An Open-Source Particle-in-Cell Simulation Framework for Plasma and Vacuum Electronics},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23512471},
  url       = {https://doi.org/10.5281/zenodo.23512471}
}
```

Citation metadata are also provided in [`CITATION.cff`](CITATION.cff).

## Author

**Md Arifuzzaman Faisal**
Department of Nuclear Engineering and Radiological Sciences, University of Michigan, Ann Arbor, MI, USA

## License

Copyright © 2026 Md Arifuzzaman Faisal for original VEXPACE contributions. These contributions are distributed under the [BSD 3-Clause License](LICENSE).

Selected particle-pushing, position-update, field-gathering, and current-deposition routines in VEXPACE are translated or adapted from [WarpX](https://github.com/BLAST-WarpX/warpx), under its **BSD-3-Clause-LBNL** license. The original copyright and license notices are preserved in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md), and WarpX's supplementary [`NOTICE.txt`](third_party/warpx/NOTICE.txt) is included.

VEXPACE is an independent research project and is not affiliated with or endorsed by the WarpX team or its institutions.
