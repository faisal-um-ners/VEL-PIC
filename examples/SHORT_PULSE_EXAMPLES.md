# Six short-pulse PIC examples

Open these models in VEXPACE with the **tfgpu** environment. They appear under
**File > Examples** after restarting the application. Each is a self-contained
`.pic.json` model and runs with the existing solver; no helper script is required
to run it. All six use **1D electrostatic PIC**, with an initially empty gap and
absorbing conductor plates.

| Example | Source | Pulse | Gap | Voltage / injection | k | f |
|---|---|---|---|---|---:|---:|
| `short_pulse_diode_2025_square_1d.pic.json` | Heri and Zhang (2025) | Square | 1.5 mm | 30 kV / cold | 0.1 | 1 |
| `short_pulse_diode_2025_trapezoid_1d.pic.json` | Heri and Zhang (2025) | Trapezoid | 1.5 mm | 30 kV / cold | 0.1 | 1 |
| `short_pulse_diode_2025_gaussian_1d.pic.json` | Heri and Zhang (2025) | Gaussian | 1.5 mm | 30 kV / cold | 0.1 | 1 |
| `short_pulse_drift_2026_square_1d.pic.json` | Heri and Zhang (2026) | Square | 1 cm | grounded plates / 1 keV | 0.1 | 0.95 |
| `short_pulse_drift_2026_trapezoid_1d.pic.json` | Heri and Zhang (2026) | Trapezoid | 1 cm | grounded plates / 1 keV | 0.1 | 0.95 |
| `short_pulse_drift_2026_gaussian_1d.pic.json` | Heri and Zhang (2026) | Gaussian | 1 cm | grounded plates / 1 keV | 0.1 | 0.95 |

These are runnable paper-parameter calculation examples, **not six claimed exact
paper reproductions or measured limiting currents**. The authors' original input
files were unavailable. The three shapes in each group have equal prescribed
injected charge per unit area. `SHORT_PULSE_VERIFICATION.json` records actual
full-run checks of the saved input files.

## Edit and run

Change `D`, `V0`, `k`, `f`, and `dt_fraction` under **Parameters**. `T`, pulse FWHM
`tau`, full support, current amplitude, and timestep update through expressions.
Change the axial cell count under **Domain & Mesh**. The timestep is aligned to
an integer number of steps across the pulse support. The emission expression
samples current at each step's midpoint. `end_over_T` controls the observation
window; extend it if particles remain in the gap.

The trapezoid uses an inferred ramp fraction `0.15` of its full support, so its
support is `tau/(1-ramp_fraction)`. The truncated Gaussian has inferred cutoff
+/-2.5 standard deviations (diode) or +/-2 (drift); `tau` denotes its FWHM.
VEXPACE's evaluator has no `erf`, so `gaussian_retained_fraction` is the exact
numerical value generated with that cutoff. If changing the cutoff, update both
values consistently or regenerate using `make_short_pulse_examples.py`.

With tfgpu active, for example:

```bash
python -m picstudio gui examples/short_pulse_diode_2025_gaussian_1d.pic.json
python -m picstudio check examples/short_pulse_drift_2026_square_1d.pic.json
python -m picstudio run examples/short_pulse_drift_2026_square_1d.pic.json
```

Each model has its own output folder. Potential probes, field snapshots, particle
snapshots and current history are enabled. History, field snapshots, particle snapshots and the live monitor update every
5 steps. The maximum particles per snapshot remains 10000.

## Charge, current and interpretation

Let `vstar=sqrt(2*e*V0/me)`. The diode has `T=3*D/vstar` and
`sigma_ref=eps0*V0/D`. Drift has `T=1.5*D/vstar` and the stated paper charge unit
`sigma_ref=4*eps0*V0/D`. Here the explicit initial trial prescription is
`Q/A=f*sigma_ref`, with `tau=k*T`. Thus **f is a pulse charge ratio, not Jem/JCL**.
In drift, `V0` denotes injection energy in eV, not a voltage difference across the
grounded gap.

Diode critical pulse charge can depend on duration and profile. These examples
retain the verified initial fixed-total-charge inputs; the separate research
study also audits a cold-cathode-field normalization. The drift paper's Eq.(5)
amplitude initialization and Eq.(11) field normalization remain unresolved.
Consequently, the literal f=0.95 drift examples can return electrons; they are
not advertised as reproducing the paper's fully transmitting f=0.95 caption.
The installed emitter also has a small relativistic-momentum offset when the
pusher is nonrelativistic at 1 keV; this is not silently compensated.

The electrode history reports **collected current**. The papers' induced
Ramo-Shockley current is instead `I_ind=sum(abs(q)*w*vz)/D`. These observables are
not interchangeable. Likewise, full-support arrival span and FWHM-label arrival
span differ for shaped pulses. Incomplete or reflecting runs do not establish a
fully transmitted pulse width; report returned and remaining charge separately.
Neither solver failure nor an unfinished pulse proves a physical limiting current.

## Sources

- Y. Heri and P. Zhang, "Space Charge Effects on Short-Pulse Electron Beam Dynamics
  in a Classical Vacuum Diode," IEEE TED 72, 2591-2596 (2025),
  [DOI: 10.1109/TED.2025.3552061](https://doi.org/10.1109/TED.2025.3552061).
  Parameters and shapes follow the Fig.2/3 setup, with inferred profile settings.
- Y. Heri and P. Zhang, "Space Charge Effects on Short-Pulse Beam Dynamics in a
  Drift Space," IEEE TED 73, 3010-3016 (2026),
  [DOI: 10.1109/TED.2026.3670804](https://doi.org/10.1109/TED.2026.3670804).
  Parameters and shapes follow the Fig.2/5 setup, subject to the stated
  normalization ambiguity.

`make_short_pulse_examples.py` regenerates only these six files. It does not
regenerate unrelated examples or modify any VEXPACE solver or interface code.
