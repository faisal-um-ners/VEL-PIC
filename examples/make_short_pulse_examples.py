"""Generate only the six Heri-Zhang short-pulse examples.

python examples/make_short_pulse_examples.py

This does not regenerate or overwrite any unrelated VEXPACE examples.
"""
from pathlib import Path
import argparse
import json
import math
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from picstudio.model.model import new_model, complete, validate

SHAPES = ('square', 'trapezoid', 'gaussian')
PAPERS = {
    'diode': dict(year=2025, D='1.5e-3', V0='30000', f='1', s='3',
                  charge_scale='1', cutoff=2.5, doi='10.1109/TED.2025.3552061'),
    'drift': dict(year=2026, D='0.01', V0='1000', f='0.95', s='1.5',
                  charge_scale='4', cutoff=2.0, doi='10.1109/TED.2026.3670804'),
}


def build_model(paper, shape):
    p = PAPERS[paper]
    stem = f'short_pulse_{paper}_{p["year"]}_{shape}_1d'
    m = new_model('1D-Z', 'electrostatic')
    m['name'] = f'Heri-Zhang {p["year"]}: {paper}, {shape} short pulse (1D)'
    note = (
        'Paper-parameter PIC example; exact reproduction is not claimed. '
        'Nominal FWHM tau=k*T; equal total injected charge for all three profiles. '
        'Trapezoid ramp fraction 0.15 and Gaussian cutoff are inferred, not reported inputs. '
        'The paper FWHM-label and full-support arrival-span metrics must be distinguished. '
        'History electrode currents are collected currents; the paper induced current requires sum(|q|*w*vz)/D. '
    )
    if paper == 'drift':
        note += (
            'Drift charge uses the explicitly documented small-spacing interpretation '
            'Q/A=4*f*eps0*V0/D. The paper Eq.(5) amplitude initialization and '
            'Eq.(11) field normalization are unresolved: f=0.95 may reflect in this physical model. '
            'The installed emitter has a small relativistic-momentum offset with a nonrelativistic pusher at 1 keV. '
        )
    else:
        note += (
            'This is the verified initial fixed-total-charge trial Q/A=f*eps0*V0/D, '
            'not the separately audited duration-dependent cold-cathode critical-charge normalization. '
        )
    m['description'] = note + f'See SHORT_PULSE_EXAMPLES.md. Source DOI {p["doi"]}.'
    m['parameters'] = {
        'D': p['D'], 'V0': p['V0'], 'k': '0.1', 'f': p['f'],
        'vstar': 'sqrt(2*e*V0/me)', 'T': f'{p["s"]}*D/vstar',
        'tau': 'k*T', 'charge_scale': p['charge_scale'],
        'sigma_ref': 'charge_scale*eps0*V0/D',
        'pulse_charge_per_area': 'f*sigma_ref',
        'ramp_fraction': '0.15',
        'gaussian_cutoff_sigma': str(p['cutoff']),
        # VEXPACE's expression evaluator has no erf. This exact retained fraction
        # is generated together with the cutoff; regenerate if changing cutoff.
        'gaussian_retained_fraction': repr(math.erf(p['cutoff']/math.sqrt(2))),
        'support': 'tau' if shape == 'square' else
                   'tau/(1-ramp_fraction)' if shape == 'trapezoid' else
                   '2*gaussian_cutoff_sigma*tau/(2*sqrt(2*log(2)))',
        'sigma_t': 'support/(2*gaussian_cutoff_sigma)',
        'pulse_area_time': 'support' if shape == 'square' else
                           'support*(1-ramp_fraction)' if shape == 'trapezoid' else
                           'sigma_t*sqrt(2*pi)*gaussian_retained_fraction',
        'Jamp': 'pulse_charge_per_area/pulse_area_time',
        'dt_fraction': '0.0002',
        'sample_dt': 'support/ceil(support/(dt_fraction*T))',
        'end_over_T': 'max(4,support/T+3.5)',
    }
    m['domain'].update(z_min='0', z_max='D', nz='256')
    m['boundaries']['z_min'].update(name='cathode', field='conductor', voltage='0', particles='absorb')
    m['boundaries']['z_max'].update(name='anode', field='conductor',
                                   voltage='V0' if paper == 'diode' else '0', particles='absorb')
    m['species'] = [dict(name='electrons', preset='electron', motion='full')]
    clock = '(t+sample_dt/2)'
    if shape == 'square':
        profile = '1'
    elif shape == 'trapezoid':
        profile = f'max(0,min(1,min({clock}/support,1-{clock}/support)/ramp_fraction))'
    else:
        profile = f'exp(-0.5*(({clock}-support/2)/sigma_t)**2)'
    # Sample prescribed current at the centre of each step. Alignment makes the
    # injection support an integer number of steps and avoids an extra final step.
    current = f'Jamp*({profile})*where(({clock}>=0)&({clock}<support),1,0)'
    m['emitters'] = [dict(name='pulse', species='electrons', model='beam', surface='segment',
                         z0='0', z1='0', y0='0', y1='0', normal='+z', current_density=current,
                         energy='0' if paper == 'diode' else 'V0', temperature='0',
                         particles_per_cell='8', t_start='0', t_stop='support')]
    m['simulation'].update(relativistic=False, dt='sample_dt', t_end='end_over_T*T', n_steps='',
                           field_gathering='momentum-conserving', smoothing=0, seed=12345)
    m['parallel'].update(mode='serial', threads=1)
    m['diagnostics'].update(output_dir='output_'+stem, history_every='5', field_every='5',
                            particle_every='5', max_dump_particles='10000', live_every='5')
    m['line_probes'] = [dict(name='phi_axis', quantity='phi', extremum='min and max',
                             along='z', y='0', z='0', start='0', end='D')]
    m['probes'] = [dict(name='phi_midgap', quantity='phi', z='D/2', y='0')]
    m = complete(m)
    errors = [message for level, message in validate(m) if level == 'error']
    if errors:
        raise ValueError(errors)
    return stem+'.pic.json', m


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for paper in PAPERS:
        for shape in SHAPES:
            name, model = build_model(paper, shape)
            (args.out_dir/name).write_text(json.dumps(model, indent=2)+'\n')
            print(name)

if __name__ == '__main__':
    main()
