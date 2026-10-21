# VEXPACE user guide

## 1. Start

Run `python -m picstudio gui`, or use the launchers in `install/`. The window has four parts:

* **Model Builder** (left): the model tree. Work through it from top to bottom.
* **Settings** (middle): the properties of the selected tree node.
* **Graphics** (right): the Geometry, Live monitor and Results tabs.
* **Messages** (bottom): solver output and model checks.

All four panels sit in splitters: drag the dividers to resize them, both horizontally and vertically. **View → Light / Dark theme** (or **Toggle theme** in the toolbar) switches the look, including the plots.

Use **File → Examples** to open a ready-made model. **Save as…** copies it to your own folder. Results are written next to the project file, in the folder named under *Output*.

## 2. Building a model

1. **Parameters.** Add named values or expressions, for example `V0 = 1e3` and `d = 1e-3`. Every numeric field can use them, together with the constants c, e, me, mp, amu, eps0, mu0, kB, pi and functions such as sin, exp, sqrt, min, max, gauss(x, x0, s), ramp(t, tr), step(x) and rect(x, a, b).
2. **Study / Solver.** Choose:
   * the geometry: 1D-Z, 2D-ZY or 2D-RZ;
   * the solver: electrostatic or electromagnetic;
   * the end time, the time step (`auto` or a value) and the CFL number;
   * optionally, smoothing passes and the RZ axis correction;
   * the **particle pusher**: Boris (default), Vay or Higuera–Cary, the same kernels as WarpX. Use Vay or Higuera–Cary for relativistic beams in crossed E and B;
   * **conductors (EM)**: *held at their potentials* (default) connects every electrode to an ideal voltage source, so the gap voltage stays at its set value while current flows (diodes, guns, magnetrons). *floating* leaves the PEC conductors isolated;
   * **current filter passes** (EM): WarpX's bilinear J filter, which WarpX applies by default (1 pass) in Cartesian EM. Set 1 to reproduce WarpX defaults; 0 keeps the exact Gauss law with the unfiltered charge;
   * **field gathering**: auto (default: energy-conserving in EM, as in WarpX; momentum-conserving in ES, the most accurate for space-charge-limited emission), energy-conserving or momentum-conserving. See THEORY §3.
3. **Domain & Mesh.** Set the extent and number of cells (the background, uniform mesh). In RZ, y means r, and r_min = 0 places the axis on the boundary.
   **Mesh Refinement** (electrostatic solver): add regions that refine the cell size along z or along y (r) between *From* and *To*. Example: a z region [0, 0.5 mm] with 5 µm cells in front of a cathode, plus a y region over the emitter width. Away from a region the cells grow by the *growth ratio* per cell (1.05 smooth … 1.5 fast). Keep the growth ratio gentle (1.05–1.2): abrupt jumps in cell size reduce the field accuracy and add self-forces on particles. Region edges, emitter ends and rectangle edges become mesh nodes. Press **Show mesh** (or F6) to see the mesh in the Geometry tab; the refined bands are shaded, and the cell count and smallest/largest cell size are shown above the plot. On a tensor-product grid a refined band runs across the whole domain; the extra cells far from the emitter hold no particles and cost little field-solve time. With a refined mesh the time step follows the smallest cell (set *dt* by hand when particles move only along z, as in the confined-flow examples). *Electrostatic solver only:* the Geometry tab draws refinement regions for either solver, but with the electromagnetic solver **Check model** reports an error and **Run** stops before the first step. Disable the regions (untick *Enabled*) and use a uniform mesh fine enough for the emitter, or use the electrostatic solver if the problem allows it (no RF fields and no magnetic self-field of the beam).
4. **Boundaries.** Set each side, for both fields and particles:
   * conductor: a voltage V(t) in ES, PEC in EM; give it a *Name* (cathode, anode) and its collected current appears as `I_<name>`;
   * neumann: symmetry or insulating;
   * periodic;
   * absorbing: an open EM boundary;
   * axis.
5. **Geometry Objects.** The domain itself is vacuum (ε0), and the electrodes of a planar diode are its conductor boundaries, so a simple SCL diode needs no object. The diode examples still list a **vacuum gap** object covering the gap, for clarity; boundary conductors always take precedence over painted objects. Add objects for anything inside the domain. Shapes are painted in list order:
   * conductor, with a voltage that can depend on t;
   * dielectric, with εr and optional surface charging;
   * absorber;
   * lossy: a resistive EM load. With *Loss specified by = Q*, give the target loaded Q, the mode frequency and the fill factor F; σ = 2π f ε0 /(Q·F). Use **Model → Calibrate loss Q** to compute F with a cold-test ring-down (needs one probe in the resonator). The ring-down is decomposed into damped modes (harmonic inversion) and the mode nearest the given frequency is used, so a neighbouring mode cannot be measured by mistake. For a periodic slow-wave structure set *Calibration: mode period* (e.g. the vane period) and *mode phase per period* (π for the π mode): the cold test then drives every period with that phase and excites the wanted mode alone. In a 6-cavity planar magnetron the π mode and the 2π/3 pair are only about 1 % apart, closer than a loaded line width at Q = 250, and a single-point drive excites both. The calibration scales F by Q_target/Q_measured (Q ∝ F while σ ≪ ωε0) with steps limited to a factor 5, and stops if Q moves the wrong way: past σ ~ ωε0 the load behaves like a metal, expels the field and Q rises again. Alternatively give σ directly;
   * vacuum, which carves holes in earlier objects (cavities, ports).

   Tick *Show painted cells* to see the staircase the solver uses.
6. **Species.** Pick a preset (electron, proton, Ar+, …) or define charge and mass yourself. **Motion (B field model):** *full* = normal 3V motion in the applied fields (B = 0 if there is no Applied Field, or a finite B such as a uniform axial field [Bz, 0, 0] T). *z_only* = motion along z only, the limit of an **infinite** axial magnetic field (confined flow). *y_only* = motion along y (r) only, the infinite-field limit along y, for an emitter lying along y = 0 that emits in +y (2D only). The diode examples (Child–Langmuir strip, disk and periodic, and `yylau_2026_*`) have a parameter **Bz** and an applied field *axial B*: Bz = 0 is no field, a finite value (for example 0.1, 1 or 10 T) is a finite field with motion = full, and motion = z_only is the limit Bz → ∞. *Relativistic beam self-B* adds the β×E/c magnetic self-field in ES runs.
7. **Particle Loading.** Set a region, a density n(z, y) (an expression is allowed), particles per cell, temperature, drift, a quiet start and a perturbation. *Quiet start* puts the particles on a regular sub-grid in every cell. With a temperature and at least 16 particles per cell, it also gives every cell the exact Maxwellian quantiles, shuffled independently per velocity component (Birdsall & Langdon). This is the standard low-noise start for Landau damping and dispersion tests.
8. **Emission.** Choose the model (scl, fowler_nordheim, thermionic, beam, photo) and the surface: a segment, or the whole surface of an object. Axis-aligned segments that lie on conductor nodes use the exact "reaction" surface field.
9. **Applied Fields.** Uniform, expression, coil or solenoid fields, each with a time factor f(t).
10. **EM Sources / Ports** (EM only). A laser (Gaussian pulse and beam, polarization, waist, focus), a plane wave, or a TEM port with V(t). Place the antenna a few cells inside an absorbing boundary.
11. **Point Probes / Line Probes / Power Monitors.**
   * **Point Probes:** the value of a field component, φ or ρ at one point (z, y), recorded as `probe_<name>`.
   * **Line Probes (min / max):** sample a quantity along a line every history step and record its average (`probe_<name>_avg`), its minimum and/or maximum, *and where they occur*. *Line along z*, *at y / r*, *from*, *to* defines the line (or along y at a fixed z). Example: name `phi`, quantity phi, record min, along z at y = 0 from 0 to d. This gives `probe_phi_min` (the virtual-cathode depth, V) and `probe_phi_zmin` (its position, m); *max* adds `probe_phi_max` and `probe_phi_zmax`. The extremum is refined between samples by a parabola. In the live monitor, choose **Field: line: phi** to watch the whole profile φ(z) with the minimum marked. The line is drawn dashed on the 2D field map. Parts of the line inside a conductor are skipped (no field exists in a PEC): the average is taken over the vacuum part only and the min/max are searched there; the conductor surface is included.
   * **Power Monitors:** the Poynting power through a line, for example a magnetron loss port.
12. **Output.** Set the history, field and particle snapshot intervals and the live-plot interval.
13. **Parallel.** Choose serial, threads (the default: all cores in one process) or **mpi**. With mpi, set *MPI processes (ranks)*; **Run** then starts `mpiexec -n <ranks>` itself. MPI needs `mpi4py` and an MPI library in VEXPACE's Python environment: `python -m pip install mpi4py mpich`, or Homebrew `open-mpi` on macOS, or Microsoft MPI on Windows. The particles are divided among the ranks, but every rank repeats the field solve, so MPI pays off for runs with many particles per cell. For small meshes with few particles, threads is usually as fast.
14. **Parametric Sweep.** Give a parameter name and its values, for example `linspace(0.6, 1.3, 8)`, and the number of workers.

Every settings form ends with a **Meaning of the settings** paragraph. It explains each input that is currently shown, and it updates when a choice reveals new inputs. The same text appears as a tooltip on each field.

Press **F7** to check the model and **F5** to run. The solver runs as a separate process, so the GUI stays responsive. **Pause** (F8) holds the run at the current step (the live plots show that state) and the same button becomes **Resume**. **Stop** (Shift+F5) ends the run cleanly and keeps everything computed so far.

## 3. Results

*Conductors show nothing.* A PEC carries no field, so the inside of every conductor is left blank (axes background) in field maps, line cuts and 1D plots, in the live monitor and in Results, and it is excluded from the colour scale. Conductor surfaces keep their physical value (surface E, φ = V).

* **Live monitor.** A field map (choose *phi*, **phi_sc** = space-charge potential = phi minus the applied electrode (Laplace) potential, *rho*, *Ez*, *Ey*, …), a phase-space plot, particle positions and history signals, updated while the run proceeds. The axes do not jump: the time axis is fixed to [0, t_end] and the other ranges only grow to cover the minimum and maximum seen so far, with a fixed number of decimals in the tick labels. Pick the history signals with the **Signals** menu (probes, `I_anode`, `I_cathode`, `Iemit_*`, energies, port power). With **FFT inset** ticked, the spectrum of the first probe (or signal) over the second half of the record is shown inside the history plot, with the peak frequency. In a field-only run (no particle species) the live monitor shows the chosen field map, the first two line-probe profiles and a second field component (colour scale from the 99th area-weighted percentile, so that singular fields at conductor corners do not wash out the map).
* **Results tab.** Field maps (optionally mirrored about the RZ axis), line cuts, phase space (z, y, vz, vy, vx, uz, …, γ, kinetic energy), particles, histories (several signals, log scale), probe spectra (FFT) and beam envelope/emittance. Use the snapshot slider to move through time. Figures export to PNG, PDF or SVG, animations to GIF or MP4, and data to CSV.
* **Files in the output folder:**
  * `history.csv`: all time series;
  * `fields/field_*.npz`: φ, E, B and ρ for each species;
  * `particles/part_*.npz`: phase space;
  * `geometry.npz`, `run_info.json`, and `model.pic.json` (a copy of the model).

## 4. Typical workflows

* **Diode or gun design (ES, RZ).** Model the conductors as objects at their voltages and use SCL emission from the cathode surface. Read the collected current from `I_anode`, the current returned to the cathode from `I_cathode` and the emitted current from `Iemit_*`, then sweep the voltage to get the perveance. For a 1D-equivalent planar diode in 2D, use periodic side boundaries with the emitter across the whole cathode (`child_langmuir_periodic_zy`). For edge effects, make the emitter narrower than the domain (`child_langmuir_strip_zy`, `child_langmuir_disk_rz`).
* **Beam transport.** Inject a beam emitter, add solenoids or coils, and use **Beam envelope** in Results to see the rms size and emittance along z.
* **Magnetron (EM, ZY).** Set up the anode block with vacuum cavities and a port slot leading to an absorbing side. Give the cathode a negative voltage with a ramp, add uniform B_x and SCL emission from the cathode object. Load the output with a `lossy` object of the desired loaded Q (for example 250) and calibrate it with **Model → Calibrate loss Q**. Probes plus FFT give the frequency, and the power monitor gives the output power. Open `planar_magnetron_em` (linear or recirculating type; cathode along y = 0 emitting from z = 0 to N·p, periodic in z) or `a6_magnetron_em` (cylindrical A6 cross-section) as a starting point.
* **Laser–plasma (EM, 1D/ZY).** Put a laser antenna near an absorbing boundary and load the plasma with a density expression (ramps). Use the Poynting monitors for reflection and transmission. See `laser_plasma_zy` for a 2D example.
* **Beam–plasma (ES 1D/ZY, EM RZ).** `beam_plasma_zy` shows a finite-width beam in a plasma, and `plasma_wakefield_rz` an axisymmetric beam-driven wakefield.
* **Field-free plasma and beam problems (ES).** `landau_damping_1d` (collisionless damping, kλ_D = 0.5), `pierce_diode_1d` (Pierce instability, α = 2.5π) and `beam_expansion_zy` (space-charge spreading of a sheet beam in a grounded drift region) need no applied field; each description gives the analytic result to compare with.
* **Field-only electrostatics.** `two_cylinder_lens_rz` has no particle species: the run is a pure Laplace solution for two coaxial tubes at V1 and V2 (gap 0.1 D) on a mesh refined at the gap. Line probes record φ(z) and E_z(z) on the axis; they agree with the exact series (Goddard 1946; Cook & Heddle 1976) to 3×10⁻⁴ V2 and 0.2 % in the peak field.
* **Small emitters, low emission energy (Lau et al. 2026).** Open a `yylau_2026_*` example. Below the limiting current the whole injected current reaches the anode; above it a virtual cathode forms (the line probe `phi` gives `probe_phi_min` → −Δ·V0 at `probe_phi_zmin`, and `I_cathode` becomes negative). Run the parametric sweep over `Jn` to bracket the limit. The steady value comes from `picstudio.core.confined_flow.ConfinedFlow(...).jmax()`.

## 5. Accuracy tips

* Resolve the Debye length (ES, warm plasmas) and the skin depth c/ω_p (EM). Keep ω_p·dt ≲ 0.2.
* SCL emission converges like (cells across the gap)^−1. About 50–100 cells gives roughly 1–2 %.
* Use smoothing = 1 for relativistic beams in EM (it reduces numerical Cherenkov radiation).
* Make conductors at least one cell thick so they absorb particles.
* Resolve the potential minimum in front of an emitter with at least 3–4 cells; use a Mesh Refinement region there instead of refining the whole domain.
* **Check model** (F7) warns when a prescribed beam current is far above the space-charge limit of the gap. Small emitters are taken into account, but such a beam would only build a virtual cathode at the emitter.
