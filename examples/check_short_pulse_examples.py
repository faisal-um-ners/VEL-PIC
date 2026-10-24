"""Small in-memory checks of the six saved example models; no large result dumps."""
from pathlib import Path
import hashlib
import json
import math
import time
import numpy as np
from picstudio import constants as C
from picstudio.expr import Evaluator
from picstudio.model.model import load, validate
from picstudio.core.simulation import Simulation


def check(root):
    rows = []
    files = sorted(root.glob('short_pulse_*_1d.pic.json'))
    assert len(files) == 6, files
    for path in files:
        m = load(str(path))
        errors = [message for level, message in validate(m) if level == 'error']
        assert not errors, (path.name, errors)
        ev = Evaluator(m['parameters'])
        Jamp, support, tau, dt = [ev.num(k) for k in ('Jamp','support','tau','sample_dt')]
        current = m['emitters'][0]['current_density']
        def value(physical_t):
            return ev.num(current, t=physical_t-dt/2)
        shape = path.stem.split('_')[-2]
        peak_t = support/2
        assert abs(value(peak_t)/Jamp-1) < 1e-12
        assert value(-dt) == 0 and value(support+dt) == 0
        if shape == 'trapezoid':
            left = .5*ev.num('ramp_fraction')*support
            right = support-left
        elif shape == 'gaussian':
            left, right = support/2-tau/2, support/2+tau/2
        else:
            left, right = 0., support
        if shape != 'square':
            assert abs(value(left)/Jamp-.5) < 1e-12
            assert abs(value(right)/Jamp-.5) < 1e-12
        assert abs((right-left)/tau-1) < 1e-12
        # Main editable parameters remain coupled to pulse duration and charge.
        altered = dict(m['parameters']);altered.update(D='2*'+m['parameters']['D'],k='0.2',f='0.4')
        alt = Evaluator(altered)
        assert math.isclose(alt.num('tau')/alt.num('T'), .2, rel_tol=1e-12)
        assert math.isclose(alt.num('pulse_charge_per_area')/alt.num('sigma_ref'), .4, rel_tol=1e-12)
        start=time.time()
        sim = Simulation(m, write_output=False, log=lambda _: None)
        min_phi=math.inf
        for step in range(sim.n_steps):
            sim.step()
            if step % 250 == 0:
                assert np.isfinite(sim.phi).all()
                sp=sim.species[0]
                assert np.isfinite(sp.z).all() and np.isfinite(sp.u0).all()
                min_phi=min(min_phi,float(sim.phi.min()))
        sp=sim.species[0]
        emitted=abs(sim.emitters[0].emitted_charge)
        returned=abs(sim.coll_side[1]);collected=abs(sim.coll_side[2])
        remaining=abs(sp.q)*sp.w.sum()
        balance=abs(emitted-collected-returned-remaining)/emitted
        wanted=ev.num('pulse_charge_per_area')
        charge_error=abs(emitted/wanted-1)
        assert balance < 1e-10, balance
        assert charge_error < 2e-5, charge_error
        # Check the ordinary GUI/CLI diagnostic path too, after accounting totals.
        assert np.isfinite(np.asarray(sim.history_row(),dtype=float)).all()
        field=sim.field_snapshot();particles=sim.particle_snapshot(300)
        assert 'phi' in field and particles
        row=dict(file=path.name,input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 geometry=sim.geometry,steps=sim.n_steps,elapsed_seconds=time.time()-start,
                 D_m=ev.num('D'),V0_V=ev.num('V0'),k=ev.num('k'),f=ev.num('f'),
                 FWHM_s=tau,support_s=support,dt_s=sim.dt,observation_time_over_T=sim.t/ev.num('T'),
                 emitted_charge_over_requested=emitted/wanted,
                 transmitted_fraction_at_end=collected/emitted,returned_fraction_at_end=returned/emitted,
                 remaining_fraction_at_end=remaining/emitted,charge_balance_relative_error=balance,
                 minimum_sampled_potential_V=min_phi,normal_model_validation_passed=True,
                 pulse_FWHM_and_charge_checks_passed=True,standard_diagnostics_passed=True)
        rows.append(row)
        print(json.dumps(row),flush=True)
    result=dict(scope='Actual complete VEXPACE 1D PIC runs with output disabled; input/physics/diagnostic functionality checks, not proof of exact paper reproduction.',
                executable='tfgpu Python',cases=rows)
    return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--examples-dir',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--report',type=Path)
    a=p.parse_args();result=check(a.examples_dir)
    if a.report:a.report.write_text(json.dumps(result,indent=2)+'\n')
