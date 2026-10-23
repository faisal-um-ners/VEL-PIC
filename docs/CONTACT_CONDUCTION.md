# AC/DC Contact Conduction (FEM)

This is an **independent** finite-element study module. Existing electrostatic and electromagnetic particle-in-cell solvers, their input models and algorithms are unchanged.

## Launch

GUI: **Model → AC/DC Contact Conduction (FEM)...** in VEXPACE.

CLI:

```bash
python -m picstudio.conduction examples/contact_acdc_reference.json --out contact_results --fmin 1e3 --fmax 1e12 --points 31
```

For an explicit list of AC frequencies (Hz), with optional DC:

```bash
python -m picstudio.conduction examples/contact_acdc_reference.json --frequency 0 --frequency 1e9 --frequency 1e11 --out contact_results
```

Install the existing NumPy/SciPy/Matplotlib dependencies used by VEXPACE.

## Physics

The **2D anisotropic, quasi-static** potential equation in metal (region 1) and semiconductor (region 2), with exp(+i omega t) phasors, is

```
div[ diag(1/rho_jy + i omega eps0 epsr_jy,
          1/rho_jz + i omega eps0 epsr_jz) grad(phi_j) ] = 0
```

The interface is **parallel RC**, with specific resistance `rho_i` (Ohm m²) and specific capacitance `c_i` (F/m²):

```
J_i = (1/rho_i + i omega c_i) (phi_1 - phi_2)
z_i = rho_i / (1 + i omega rho_i c_i)  [Ohm m²]
```

Setting `frequency=0` returns DC, and `c_i=0` yields a purely resistive interface. The interface is modeled using separately numbered P1 nodes at the two faces and a consistent 4x4 coupling matrix, based on the supplied Figure_3 and Figure_S1_S2 notebook formulation.

Material bulk parameters are currently frequency-independent resistivity and permittivity inputs. **No Drude, dispersive ZnO dielectric function, magnetic-wave propagation, oxide/vacuum exterior, PEC-gate full-wave physics, or inductive impedance is included.** Those need a separately verified full-wave extension corresponding to the supplied Al–ZnO notebooks.

## Meshing

The mesh is refined near the metal edge y=Lc, from the covered semiconductor toward the exposed channel, and near the metal–semiconductor interface z=0:

```
y_contact = Lc [1-(1-t)^2]
y_exposed = Lc + (L/2) t^2
z_metal = h1 t^1.8
z_sem = -h2 + h2 [1-(1-t)^1.8]
```

As in Figure_3.ipynb, region 1 is modeled for y in [0,Lc], z in [0,h1]; region 2 for y in [0,Lc+L/2], z in [-h2,0]. Terminal potentials are fixed at the metal top and channel right end. Each layer has insulating natural boundaries elsewhere. This is the **top-contact half-device** model, not the vertical five-region full-wave model.

## Outputs

- `impedance_sweep.csv`: complex total/contact/access impedance times width, magnitude and phase, interface impedance
- `mesh.png`: P1 triangular mesh
- `impedance_vs_frequency.png`, `phase_vs_frequency.png`, `nyquist.png`
- DC/AC field images: potential real and magnitude, current density real and magnitude, contact current-crowding profile
- `dc_*.npy`, `ac_*.npy`: raw complex potential, current density, electric field, interface jump and current
- `*_summary.json`: terminal impedance, FEM residual, conductor and interface loss, frequency

**Impedance normalization:** `Z_total_W=(Vmetal-Vchannel)/I_per_width`. The contact contribution is `Z_contact_W = Z_total_W - Z_access_W`, where `Z_access_W=(L/2)/(h2 eta_2y)` is a *uniform-access-sheet approximation*. In strongly 2D or dispersive regimes, regard this as a defined subtraction, not a unique physical contact decomposition. Negative or unusual components after subtraction require review.

**Scope:** For frequencies where wavelengths, inductance, skin effect, external gate coupling, or displacement fields outside the two conductor regions matter, do not treat this quasi-static impedance as a validated full-wave result.

The JSON example uses demonstration parameters, **not manuscript-validated material data**.

## Validation

Run `python -m pytest tests/test_conduction.py -q`. Check mesh convergence, DC versus your exact contact-field model, terminal flux balance, passivity and high-frequency behavior before using numerical values in a publication. The existing PIC validation suite is separate.
