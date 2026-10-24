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


def pulse_expression(shape):
    clock = '(t+sample_dt/2)'
    if shape == 'square':
        profile = '1'
    elif shape == 'trapezoid':
        profile = f'max(0,min(1,min({clock}/support,1-{clock}/support)/ramp_fraction))'
    elif shape == 'gaussian':
        profile = f'exp(-0.5*(({clock}-support/2)/sigma_t)**2)'
    else:
        raise ValueError(f'Unknown pulse shape: {shape}')
    # Current is sampled at each step midpoint; support spans whole steps.
    return f'Jamp*({profile})*where(({clock}>=0)&({clock}<support),1,0)'


def pulse_description(shape):
    detail = {
        'square': 'Square: shape=1 inside support; support=tau; pulse_area_time=support.',
        'trapezoid': (
            'Trapezoid: with u=t+sample_dt/2, rise linearly from 0 to Jamp over '
            '0 <= u < ramp_fraction*support; stay at Jamp until '
            'u=(1-ramp_fraction)*support; then fall linearly to zero at u=support. '
            'ramp_fraction is EACH ramp duration / full support (0 < ramp_fraction <= 0.5; '
            'default 0.15). The min/max terms clip the shape between 0 and 1. '
            'support=tau/(1-ramp_fraction); pulse_area_time=support*(1-ramp_fraction).'
        ),
        'gaussian': (
            'Gaussian: center=support/2, shape=exp(-0.5*((u-support/2)/sigma_t)**2), '
            'where u=t+sample_dt/2. sigma_t=tau/(2*sqrt(2*log(2))) is the time standard deviation; '
            'support=2*gaussian_cutoff_sigma*sigma_t. Cutoff is 2.5 sigma (diode) or 2 sigma (drift). '
            'pulse_area_time=sigma_t*sqrt(2*pi)*gaussian_retained_fraction; '
            'gaussian_retained_fraction=erf(gaussian_cutoff_sigma/sqrt(2)). '
            'The evaluator has no erf: update this numerical fraction together with the cutoff. '
            'Use cutoff > sqrt(2*log(2)) so tau remains the retained pulse FWHM.'
        ),
    }[shape]
    return (
        '\n\nPulse setup: edit named values under Parameters, and the expression under '
        'Emission > pulse > Current density [A/m^2]. t is solver time [s]; '
        'u=t+sample_dt/2 samples the midpoint of the current step. '
        'The where((u>=0)&(u<support),1,0) gate emits only for 0 <= u < support; '
        '& means both comparisons must hold. Keep emitter t_start=0, t_stop=support '
        'and simulation dt=sample_dt. support is full pulse duration [s]; tau=k*T is FWHM [s]. '
        'Jamp=pulse_charge_per_area/pulse_area_time is peak current density [A/m^2], '
        'not total current. pulse_charge_per_area=f*sigma_ref [C/m^2]; '
        'pulse_area_time is the integral of the dimensionless pulse shape [s]. '
        'sample_dt=support/ceil(support/(dt_fraction*T)) aligns support to whole steps; '
        'dt_fraction=0.0002 is the requested maximum step / T. '
        + detail + '\nCurrent density expression: ' + pulse_expression(shape)
    )


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
    current = pulse_expression(shape)
    m['description'] += pulse_description(shape)
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
