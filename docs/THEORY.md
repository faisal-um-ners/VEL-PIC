# VEXPACE: numerical methods

## 1. Mesh and coordinates

The default mesh is a structured tensor-product grid, uniform (dz, dy) or graded/refined along z and y (electrostatic solver). Particles are pushed and deposited in logical coordinates (the fractional node index), where every cell has unit size, so the bilinear weights are the Q1 shape functions of the physical cell.

**Why abrupt cell-size jumps are avoided.** The electromagnetic (Yee) solver runs on a uniform grid; a refined mesh with the electromagnetic solver is a model-check error. Three reasons: (i) the Courant limit of the explicit Yee update is set by the smallest cell and applies to the whole domain, so a 10 µm band on a 78 µm grid would shorten every step about eightfold; (ii) the staggered update is second-order only for equal cells: on a stretched grid it drops to first order, and a change of cell size reflects waves and changes the numerical dispersion; (iii) the Esirkepov deposition and the held-electrode correction (discrete harmonic functions of the Yee operator) are written for equal cells. Codes that refine electromagnetic grids (e.g. WarpX) use separate refinement patches with buffer and absorbing layers, which VEXPACE does not provide. In the electrostatic solver a jump lowers the local accuracy of the finite-element solution, adds a self-force on particles crossing it and changes the particle statistics per cell. The tensor axes are therefore graded geometrically (growth ratio 1.05–1.2 recommended; the model check warns above 1.3), with grid lines continuous across the domain.

**Manual local 2D mesh.** Rectangle/polygon regions optionally replace the 2D electrostatic Q1 mesh with conforming P1 triangles. Filled regions and safety buffers prescribe a target size; the mesh is generated before PIC and remains fixed. The actual triangles provide barycentric charge deposition, the Poisson weak form and direct element-gradient field gathering (`auto`). The optional momentum-conserving choice uses a consistent-mass L2 nodal projection; the name does not guarantee exact momentum conservation on irregular triangles. RZ includes the radial weight. Emitter endpoints on domain edges are inserted locally; no emitter-aligned full-domain band is added. Rectangular material interfaces and explicit line refinements retain their alignment. The rectangular arrays saved for compatibility are field samples, not the particle source mesh. See [LOCAL_MESH.md](LOCAL_MESH.md) for controls and restrictions. The tensor-grid descriptions below apply to the original Q1 path unless stated otherwise.

Nodes are indexed `[iz]` in 1D and `[iz, iy]` in 2D. In RZ the second coordinate is the radius r, and either r_min = 0 (on axis) or r_min > 0 (annular domain).

Vector components are stored as:

| Geometry | component 0 | component 1 | component 2 | handedness |
|---|---|---|---|---|
| 1D-Z | z | y | x | left (v×B sign flip handled internally) |
| 2D-ZY | z | y | x (out of plane) | left |
| 2D-RZ | z | r | θ | right |

Shapes are painted in list order. A node is a Dirichlet (conductor) node if it lies inside a conductor shape. A cell absorbs particles when it is inside a dielectric or absorber, or when all of its corner nodes are conductor nodes. This makes the particle surface coincide with the electrostatic staircase.

## 2. Electrostatic solver (scikit-fem)

The solver works on the weak form of −∇·(ε0 εr ∇φ) = ρ, using bilinear Q1 elements on the default 2D tensor grid, linear P1 elements in 1D, or P1 triangles in the optional local 2D mode:

* Cartesian: ∫ ε ∇φ·∇v dA = ∫ ρ v dA
* RZ: ∫ ε ∇φ·∇v r dr dz = ∫ ρ v r dr dz. The factor 2π cancels, and the axis needs no special treatment.

The particle right-hand side is assembled directly from the shape functions: b_i = Σ_p q_p w_p N_i(x_p), divided by 2π in RZ. This is exactly linear (cloud-in-cell) weighting, so the deposit and the discretisation are consistent. The charge density for diagnostics is b_i / V_i, where V_i = ∫ N_i dV is the lumped node volume. On the axis this gives V_0 = π dr² dz / 3, which is Verboncoeur's correction.

The stiffness matrix is assembled once (with εr per element) and reordered to grid ordering. Periodic directions are handled by a node-identification matrix R, so K_red = Rᵀ K R. The matrix is factorised once with SuperLU. Conductors with time-dependent voltages use superposition, φ = φ_ρ + Σ_k V_k(t) φ_k, with pre-computed Laplace solutions φ_k. A fully periodic or Neumann problem is pinned, and a uniform neutralising background is subtracted.

Nodal E = −∇φ uses second-order centred differences (one-sided at boundaries), followed by bilinear interpolation to the particles. This is the momentum-conserving scheme, with no self-force on a uniform grid.

Convergence is second order in 1D, ZY and RZ (benchmark 01).

## 3. Particle push

The particle kernels follow WarpX (BSD-3-Clause-LBNL; the notice is kept in `picstudio/particles/kernels.py`). They are line-by-line Python/Numba translations of WarpX's `UpdateMomentumBoris`, `UpdateMomentumVay`, `UpdateMomentumHigueraCary` and `UpdatePosition`, chosen with *Study / Solver → Particle pusher*:

* **Boris** (default): u = γv at half steps, the standard relativistic scheme;
* **Vay** (2008): exact E×B drift for relativistic particles in crossed fields;
* **Higuera–Cary** (2017): volume-preserving and correct E×B drift.

At t = 0 the momenta are pushed back by −Δt/2 with the full Lorentz force (WarpX `PushP(-dt/2)`). In RZ, particles move in a local Cartesian frame and are rotated back to (z, r), as in WarpX and XOOPIC.

**Field gathering** (*Study / Solver → Field gathering*):

* *energy-conserving*: WarpX's default, `doGatherShapeN` with the Galerkin choice. Each field component is interpolated with the shape order lowered by one along its own staggered direction. With the Yee staggering this is exactly WarpX's: E_z at (i+½, j), E_y at (i, j+½), E_x at (i, j), B_z at (i, j+½), B_y at (i+½, j), B_x at (i+½, j+½). In ES the cell-centred differences of φ are gathered the same way. This removes the self-force of the gather–deposit pair and suppresses grid heating (benchmark 29: thermal plasma with Δz = λ_D, total energy constant to 2·10⁻⁵ versus +0.14 % with momentum-conserving).
* *momentum-conserving*: nodal fields with linear weights in all directions. No self-force on average, and the most accurate choice for space-charge-limited emission: the Gauss-law emitter needs the nodal surface field.
* *auto* (default): energy-conserving for EM (like WarpX) and momentum-conserving for ES.

Benchmark 26 runs the three pushers against WarpX 26.09 on single-particle trajectories (gyration, E×B, relativistic E×B with γ = 20). The paths agree to 10⁻⁸–10⁻¹⁵ of the path size, i.e. round-off.

## 4. Emission

Each emitting surface is split into panels, one per cell face. These are either the panels of a straight segment or every face where a conductor object meets vacuum.

* **Surface field (ES).** The FEM reaction R = Kφ − b on Dirichlet nodes is the induced surface charge. E_n = R / (ε0 A_node) is exact in the discrete sense and includes the charge of freshly emitted particles in the first cell. This makes Gauss-law emission free of oscillation.
* **SCL (Gauss's law).** Q = ε0 E_n A per step whenever the field accelerates the particles away from the surface. In steady state this converges to the Child–Langmuir law (benchmarks 02–04).
* **Fowler–Nordheim.** J = a F² / (φ t²) exp(−v b φ^{3/2} / F), with F = β|E_n| and Forbes' v(f) = 1 − f + (f/6) ln f.
* **Thermionic.** J = A T² exp(−(W − ΔW)/kT), with Schottky lowering ΔW = sqrt(e³ E / 4πε0). Electrons are emitted with flux-weighted half-Maxwellian velocities.
* **Beam / photo.** Prescribed J(t), energy and temperature (photo uses a Lambertian angular distribution).
* **Fractional-time placement.** A random emission time τ ∈ (0, dt) and constant-acceleration kinematics place each particle at x = ½ a τ² + v τ, with u_{n+1/2} = v + a (τ − dt/2). This removes the emission "sheets".

## 5. Electromagnetic solver (Yee FDTD)

The staggered grid is:

```
E_z (i+½, j)   E_y|E_r (i, j+½)   E_x|E_θ (i, j)
B_z (i, j+½)   B_y|B_r (i+½, j)   B_x|B_θ (i+½, j+½)
```

The update is leapfrog: B^{n+½} → E^{n+1} → B^{n+1}. In RZ, the curls use the r-weighted finite-volume forms. On the axis, E_z is updated from Ampère's law in integral form over the axis cell, with an area consistent with the charge deposit, and E_θ = B_r = 0.

The current deposition is Esirkepov's charge-conserving scheme in index space. It produces charge fluxes through cell faces, which are divided by the r-weighted face areas. **Gauss's law is conserved to round-off in 1D, ZY and RZ, including the axis** (benchmark 12).

Boundaries:

* PEC (conductors, staircased).
* PMC, for symmetry and Neumann sides.
* Periodic.
* Absorbing: a graded matched loss layer (σ_E = σ_B, cubic profile, PEC behind it), equivalent to MAGIC's FREESPACE.

Dielectrics enter through a per-component εr.

Sources are current-sheet antennas with K = −2 ε0 c E_target(y, t), which give the requested forward-wave field. The available sources are lasers (Gaussian pulse and beam with focus and Gouy phase), plane waves, and TEM ports (coax E_r = V / (r ln(b/a)) in RZ, parallel-plate in ZY).

Static electrode voltages in EM runs are added as the Laplace field of the conductors, which is an exact static Maxwell solution with PEC boundaries, superposed on the self-consistent fields.

**Electrode voltages in EM runs (ideal voltage sources).** In a Yee solver a PEC conductor is isolated: its total charge changes only when particles are absorbed on it or emitted from it. As current flows from cathode to anode the gap capacitor would discharge and the electrode potentials would drift.

VEXPACE connects every conductor to an ideal (zero-impedance) voltage source (`em_conductors = held`, the default):

* The applied voltages are carried by the electrostatic Laplace solution, superposed on the fields seen by the particles.
* After every field update, the Yee field is corrected to keep each electrode at zero potential (`fields/circuit.py`). The steps are:
  * With the Yee grid's own operators (M = flux of ε_r E through the node control volume, applied to E = −∇φ on the Yee edges), discrete harmonic functions φ_k are precomputed: 1 on electrode k, 0 on the others.
  * The charge implied by the Yee field at every node is q = ε0·flux(E).
  * The excess electrode charge is ΔQ_k = Σ φ_k q (Green's reciprocity), and with the capacitance matrix C_kj = ε0 φ_kᵀMφ_j the floating potentials are p = C⁻¹ΔQ.
  * Finally E −= Σ p_k(−∇φ_k).
* The correction is curl-free and divergence-free at every vacuum node, so B and the discrete Gauss law are untouched (the Gauss benchmark stays at 1e-15).
* Conductors that touch and have the same voltage (the walls of a cavity, an anode block touching the outer wall) form one electrode, so cavity modes are not affected.
* The EM initial self-field is computed with the same Yee-consistent operator, so Gauss's law is exact from t = 0.

**Space-charge-limited emission in EM runs.** The emitter uses the staggered Yee E half a cell in front of the surface. It corrects that field with the free charge sitting on the surface nodes (Gauss's law over the half cell), E_s = E_½ − Q_node/(ε0 A). This is the counterpart of the finite-element reaction used in ES; without it, freshly emitted charge near the cathode would be invisible to the surface field. The 1D EM Child–Langmuir current converges to J_CL: +4.7 %, +2.9 %, +1.7 % at 50, 100, 200 cells (first order, like the ES Gauss-law emitter), with no returned current.

**Current filter (WarpX `use_filter`).** *Current filter passes* > 0 applies the bilinear 1-2-1 filter to the deposited J along every axis before the field update, exactly as WarpX does by default in Cartesian EM runs. The stencil commutes with the Yee divergence, so the discrete Gauss law holds for the equally filtered charge (unit test `test_em_current_filter_conserves_filtered_charge`).

**EM time step.** dt = CFL/(c·√(1/dz² + 1/dy²)) in ZY and CFL/(c·√(1.2105/dr² + 1/dz²)) in RZ. The RZ form is the cylindrical Yee limit used by WarpX; the on-axis update tightens the radial term.

Optional 1-2-1 binomial filtering is also available. In ES it is applied to ρ. In EM it is applied to the fields gathered by the particles rather than to J, so charge conservation is not affected; conductor nodes are excluded from the filter.

## 6. Parallelisation

* **Threads.** All particle kernels are compiled with Numba in two variants, serial and `prange`-parallel. Deposits use private per-thread grids that are then reduced. Small particle counts automatically use the serial kernels.
* **MPI.** Particle decomposition: each rank owns a share of the particles (loaders split by index, emitters by panel), charge and current are summed with Allreduce, and the 2D field solve is replicated on every rank.
* **Sweeps.** A process pool runs one simulation per parameter value.

## 7. Time step

ES `auto` uses dt = min(0.5·cfl·h / v_max, 0.2/ω_p, 0.25/ω_c). EM uses cfl × the Courant limit.
