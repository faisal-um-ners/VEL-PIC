# Manual local mesh refinement (electrostatic)

The existing **Along z / Along y** refinement bands are unchanged. The **2D** mode refines a filled region in both coordinates without extending its fine cells across the full domain. The prescribed mesh is generated before PIC and remains fixed throughout the run. Particles may leave the region: its outline is not a wall or an imposed trajectory.

## In the interface

1. Select **Mesh Refinement** in the Model Builder.
2. Click **Add refinement region** and choose **Refinement type**: **Along z**, **Along y**, or **2D**. Existing saved models keep their original settings; these are display labels for the compatible stored values `z`, `y`, and `local 2D`.
3. For **2D**, choose **Region shape = rectangle** or **polygon** in the dropdown. Enter rectangle limits or ordered polygon corner coordinates, using parameter expressions if desired. Set **Cell size inside**, **Safety buffer outside**, and **Growth ratio outside**. The two separate drawing buttons have been removed.
4. Use a filled envelope covering the expected beam, with extra space for lateral movement and cathode returns. Add another local region for the return footprint if needed. Regions combine by taking the finer requested size.
5. **Domain & Mesh** sets the background resolution. Avoid an excessively coarse background: the surrounding potential still affects the beam. Along z / Along y bands can coexist with local regions; disable them if broad refinement bands are unwanted.
6. Click **Show mesh** to preview the actual triangles and node/element counts. Check the model, save it, and run normally.

Emitter segment endpoints on an outer boundary are exact local boundary vertices. They do not insert an emitter-aligned row or column across the whole triangular domain. Rectangular material interfaces still align with the mesh; explicitly chosen line refinements retain their full-domain bands. Disabling a polygon removes its size constraint, while any other enabled rectangle, buffer, and graded transition remain active.

The size is the maximum leaf-edge length in the prescribed region. Leaves split in discrete levels and are triangulated with all shared edge vertices included, so the mesh is conforming. The safety buffer extends the fine area around the region; beyond it the target size grows with distance. Resource limits reject oversized meshes with an explanation rather than silently dropping cells.

## Examples

- `examples/local_mesh_confined_strip_zy.pic.json`: confined narrow strip, cathode/barrier refinement and emitting-column refinement; D=2 cm, V=100 V, full eta=0.1, Delta=0.01, p/D=5.12.
- `examples/local_mesh_b0_envelope_zy.pic.json`: B=0, manually assumed expanding polygon and a cathode return footprint; D=2 cm, V=100 V, full eta=1, Delta=0.5, p/D=9.

Both appear in **File > Examples**. These are editable run setups, not claimed measurements of their SCL limits.

## Solver and compatibility

- Available for **2D-ZY and 2D-RZ electrostatics**. The Poisson operator and factorization are prepared once, with the correct radial weak-form weight in RZ.
- Particles deposit charge with the actual triangles' P1 barycentric basis. Source loading and emitter panels use the local elements/surfaces; existing emission laws and particle pushers are reused.
- **Field gathering = auto** on this new P1 mesh uses the direct constant element gradient, consistent with the P1 potential. Explicit **momentum-conserving** selects a consistent-mass L2 nodal reconstruction; this option's existing name does not guarantee exact momentum conservation on an irregular mesh. It can introduce a different energy error. Check timestep/mesh convergence for the intended problem.
- Symmetry/periodicity, conductor Dirichlet conditions, material painting, dielectric charge and reaction-based surface fields are handled on the actual local mesh. Native mesh/field arrays are saved in output snapshots and used for field maps. Rectangular arrays remain as compatibility samples, not the computational source mesh.
- The rectangular 1-2-1 smoothing filter is undefined on triangles. **Smoothing must be 0** for this new mode; Check model and Run explain this requirement.
- **EM is excluded.** Check model and Run reject enabled local regions: “The EM Yee solver requires a uniform grid. Disable the local refinement regions.” The original EM/Yee solver is unchanged.
- Existing models without an enabled local region retain their original structured Q1/EM paths and settings. No live particle-following mesh adaptation was added.

See [the verification report](LOCAL_MESH_VERIFICATION.md) for the three Lau et al. (2026) stationary comparisons, six PIC trials, test coverage and numerical limits. The source uses the FEM-PIC principles described in [Particle In Cell Consulting's tutorial](https://www.particleincell.com/2015/fem-pic/); no Boltzmann-electron closure is added to the diode model.
