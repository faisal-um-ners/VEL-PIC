"""PIC checks of prescribed local meshes against three Lau et al. (2026) cases.

Runs below/above Table 1 reference currents. This checks field/particle coupling,
not a sub-percent extraction of a PIC limiting current. Results retain time
histories, native fields, meshes, model inputs and numerical settings.
"""
import argparse,copy,json,time
from pathlib import Path
import numpy as np
from picstudio.model.model import load,complete
from picstudio.core.simulation import Simulation

REFERENCES=[('E1',.1,.01,5.12,5.191),('E2',.05,.001,2.56,4.829),('E3',.01,.01,2.048,32.863)]
ROOT=Path(__file__).resolve().parents[1]

def make_case(eta,delta,period,j,dtau=.004,tau=8):
 m=load(str(ROOT/'examples/local_mesh_confined_strip_zy.pic.json'))
 m['parameters'].update(eta=str(eta),Delta=str(delta),pbar=str(period),Jn=str(j))
 m['domain'].update(nz='28',ny='28')
 # Fine only near the cathode/virtual-cathode layer; preserve moderate
 # background and adequate axial columns farther into the gap.
 h=min(.002,eta/12)
 layer=min(.05,max(.012,3*delta))
 m['mesh_refinements']=[dict(name='confined columns',axis='local 2D',region_shape='rectangle',
    z0='0',z1='d',y0='0',y1='eta*d/2',size='.025*d',buffer='.005*d',growth='1.2'),
  dict(name='cathode barrier layer',axis='local 2D',region_shape='rectangle',
    z0='0',z1=f'{layer}*d',y0='0',y1='eta*d/2',size=f'{h}*d',buffer=f'{h}*d',growth='1.2')]
 m['emitters'][0]['particles_per_cell']='2'
 m['simulation'].update(dt=f'{dtau}*d/v0',t_end=f'{tau}*d/v0',n_steps=str(round(tau/dtau)),field_gathering='energy-conserving')
 m['diagnostics'].update(history_every='10',field_every='1000000',particle_every='1000000',live_every='1000000')
 m['parallel'].update(mode='serial',threads=1)
 return complete(m)

def run_case(label,eta,delta,period,jref,fraction,out,dtau=.004,tau=8):
 name=f'{label}_{fraction:g}';dest=out/name
 if (dest/'summary.json').exists():return json.loads((dest/'summary.json').read_text())
 model=make_case(eta,delta,period,jref*fraction,dtau,tau)
 started=time.time()
 print(f'START {name}',flush=True)
 sim=Simulation(model,out_dir=str(dest),log=lambda x:None)
 t=sim.local_es.mesh;initial_hash=hash(t.triangles.tobytes()+t.points.tobytes())
 rows=[];cols=sim.history_columns();sim.out.start_history(cols)
 sim.out.write_json('model.pic.json',model)
 sim.out.save_npz('','geometry.npz',sim.geometry_data())
 for k in range(sim.n_steps):
  sim.step()
  if (k+1)%10==0:
   row=sim.history_row();rows.append(row);sim.out.history(row)
  if (k+1)%500==0:print(f'{name}: {k+1}/{sim.n_steps}, particles={sim.species[0].n}, elapsed={time.time()-started:.1f}s',flush=True)
 H=np.array(rows);tail=H[len(H)//2:]
 # A half-strip PIC current is charge/time per invariant depth.
 emitted=np.abs(tail[:,cols.index('Iemit_cathode')]).mean()
 transmitted=np.abs(tail[:,cols.index('I_anode')]).mean()
 returned=np.abs(tail[:,cols.index('I_cathode')]).mean()
 # Axis profile is evaluated on actual triangles, not the display grid.
 z=np.linspace(0,sim.grid.z_max,4001);profile=sim.local_es.mesh.interpolate(sim.local_es.phi,z,np.zeros_like(z))
 rhs=sim.local_es.P.rhs
 residual=sim.local_es.P.R.T@(sim.local_es.P.K@sim.local_es.phi-rhs)
 free=sim.local_es.P.free
 defect=float(np.max(abs(residual[free]))/max(np.max(abs((sim.local_es.P.R.T@rhs)[free])),1e-30))
 qparticle=sum(s.q*s.w.sum() for s in sim.species)
 qerror=abs(sim.local_es.q.sum()-qparticle)/max(abs(qparticle),1e-30)
 summary=dict(case=label,reference='Lau et al. (2026), Table 1, POISSON column',eta=eta,Delta=delta,p_over_D=period,
   reference_J_over_JCL=jref,injected_fraction_of_reference=fraction,injected_J_over_JCL=jref*fraction,
   measured_transmitted_fraction=float(transmitted/max(emitted,1e-30)),measured_returned_fraction=float(returned/max(emitted,1e-30)),
   min_axis_phi_over_V=float(profile.min()/float(sim.ev.num('V0'))),poisson_relative_residual=defect,
   relative_charge_error=float(qerror),mesh_fixed=initial_hash==hash(t.triangles.tobytes()+t.points.tobytes()),
   nodes=len(t.points),triangles=len(t.triangles),dtau=dtau,simulated_tau=tau,steps=sim.n_steps,
   elapsed_seconds=time.time()-started,
   interpretation='PIC below/above-reference behavior check; not a measured SCL fold or a sub-percent limiting-current benchmark.')
 sim.out.save_npz('fields',f'field_{sim.step_count:07d}.npz',sim.field_snapshot())
 sim.out.save_npz('particles',f'part_{sim.step_count:07d}.npz',sim.particle_snapshot(10000))
 sim.out.write_json('summary.json',summary);sim.out.close()
 print('RESULT '+json.dumps(summary),flush=True)
 return summary

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'benchmarks/local_mesh_results');p.add_argument('--cases',nargs='*',default=['E1','E2','E3']);p.add_argument('--tau',type=float,default=8);p.add_argument('--dtau',type=float,default=.004);p.add_argument('--fractions',nargs='*',type=float,default=[.85,1.15]);a=p.parse_args()
 a.out.mkdir(parents=True,exist_ok=True);results=[]
 for name,eta,delta,period,jref in REFERENCES:
  if name not in a.cases:continue
  for fraction in a.fractions:
   results.append(run_case(name,eta,delta,period,jref,fraction,a.out,a.dtau,a.tau))
 (a.out/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
