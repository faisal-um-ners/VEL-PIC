"""Reflection-free stationary check of the actual new P1 mesh against Lau Table 1.

Independent of PIC time stepping. Potential-amplitude bordered Newton continuation
must traverse a current maximum; solver failure never establishes a limit.
"""
from pathlib import Path
import json,time
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from scipy.optimize import minimize_scalar
from check_local_mesh_lau2026 import make_case,REFERENCES
from picstudio.core.simulation import Simulation
from picstudio import constants as C


def strip_quadrature(mesh,edge,order,D):
 """Integrate only the strip portion of each P1 element, including cut cells.

 The computational mesh need not extend the emitter endpoint into a full
 column edge. Clipping changes quadrature, not the mesh or its unknowns.
 """
 t=mesh.points[mesh.triangles];y=t[:,:,1]
 inside=np.all(y<=edge,axis=1)
 parents=list(np.flatnonzero(inside))
 vertices=[np.eye(3) for _ in parents]
 for k in np.flatnonzero(np.any(y<edge,axis=1)&np.any(y>edge,axis=1)):
  polygon=[];eye=np.eye(3)
  for i in range(3):
   a,b=eye[i-1],eye[i];ya,yb=y[k,i-1],y[k,i]
   if (ya<=edge)!=(yb<=edge):
    polygon.append(a+(edge-ya)/(yb-ya)*(b-a))
   if yb<=edge:polygon.append(b)
  unique=[]
  for p in polygon:
   if not unique or not np.array_equal(p,unique[-1]):unique.append(p)
  if len(unique)>1 and np.array_equal(unique[0],unique[-1]):unique.pop()
  for i in range(1,len(unique)-1):
   v=np.array([unique[0],unique[i],unique[i+1]])
   coords=v@t[k];u,w=coords[1:]-coords[0]
   if u[0]*w[1]-u[1]*w[0]>0:
    parents.append(k);vertices.append(v)
 parents=np.asarray(parents,int);vertices=np.asarray(vertices)
 q,w=np.polynomial.legendre.leggauss(order);q=(q+1)/2;w=w/2
 a,b=np.meshgrid(q,q,indexing='ij');wa,wb=np.meshgrid(w,w,indexing='ij')
 duffy=np.c_[(1-a).ravel(),(a*(1-b)).ravel(),(a*b).ravel()]
 N=np.einsum('qi,eij->eqj',duffy,vertices)
 coords=np.einsum('eij,ejk->eik',vertices,t[parents])
 u,v=coords[:,1]-coords[:,0],coords[:,2]-coords[:,0]
 det=u[:,0]*v[:,1]-u[:,1]*v[:,0]
 weight=(wa*wb*a).ravel()[None,:]*(det[:,None]/D**2)
 return mesh.triangles[parents],N,weight,vertices


class FoldCheck:
 def __init__(self,model,order=5):
  self.sim=Simulation(model,write_output=False,log=lambda _:None)
  self.backend=self.sim.local_es;self.P=self.backend.P;self.mesh=self.backend.mesh
  self.D=float(self.sim.ev.num('d'));self.delta=float(self.sim.ev.num('Delta'));self.width=float(self.sim.ev.num('eta'))
  self.Z=self.mesh.points[:,0]/self.D;self.scale=max(1.,np.sqrt(self.delta)/self.width)
  self.K=(self.P.K/C.eps0).tocsr();self.free=self.P.free;self.R=self.P.R
  self.Kf=(self.R.T@self.K@self.R)[self.free][:,self.free].tocsr()
  self.tri,self.N,self.weight,self.source_vertices=strip_quadrature(self.mesh,self.width*self.D/2,order,self.D)
  self.ii=np.repeat(self.tri,3,axis=1).ravel();self.jj=np.tile(self.tri,(1,3)).ravel()
  probe_z=min(.15,max(2*self.delta,min(self.width/2,.015)))
  axis=self.mesh.boundary_nodes['y_min'];axis=axis[(self.Z[axis]>0)&(self.Z[axis]<1)]
  self.probe=int(axis[np.argmin(abs(self.Z[axis]-probe_z))])
  self.probe_free=int(np.flatnonzero(self.free==self.P.red[self.probe])[0])
  self.n=len(self.mesh.points)

 def unpack(self,v):
  x=np.zeros(self.R.shape[1]);x[self.free]=v[:-1]
  return np.asarray(self.R@x)

 def residual(self,v,a,jac=True):
  psi=self.unpack(v);alpha=v[-1]*self.scale;phi=self.Z+psi
  if alpha<0 or np.min(self.delta+np.einsum('ei,eji->ej',phi[self.tri],self.source_vertices))<=0:raise ValueError('inadmissible transmitting source')
  kin=self.delta+np.einsum('ei,eqi->eq',phi[self.tri],self.N)
  q=np.einsum('eq,eqi->ei',self.weight/np.sqrt(kin),self.N)
  Q=np.bincount(self.tri.ravel(),weights=q.ravel(),minlength=self.n)
  Qr=np.asarray(self.R.T@Q).ravel()[self.free]
  F=np.r_[self.Kf@v[:-1]+4/9*alpha*Qr,v[self.probe_free]+a]
  if not jac:return F,phi,alpha,float(kin.min())
  vals=np.einsum('eq,eqi,eqj->eij',self.weight*kin**-1.5,self.N,self.N)
  W=sparse.coo_matrix((vals.ravel(),(self.ii,self.jj)),shape=(self.n,self.n)).tocsr()
  J=self.Kf-(2/9)*alpha*(self.R.T@W@self.R)[self.free][:,self.free]
  row=sparse.csr_matrix(([1.],([0],[self.probe_free])),shape=(1,len(self.free)))
  matrix=sparse.bmat([[J,sparse.csr_matrix((4/9*self.scale*Qr)[:,None])],[row,None]],format='csc')
  return F,matrix,phi,alpha,float(kin.min())

 def solve(self,a,initial=None):
  v=np.zeros(len(self.free)+1) if initial is None else initial.copy()
  for it in range(35):
   try:F,J,phi,alpha,kmin=self.residual(v,a)
   except ValueError:return None,dict(converged=False,reason='inadmissible')
   error=float(np.max(abs(F)))
   if error<2e-11:return v,dict(converged=True,amplitude=float(a),current=float(alpha),residual=error,iterations=it,minimum_kinetic=kmin)
   step=spsolve(J,-F);accepted=False
   for h in range(24):
    trial=v+step*2**(-h)
    try:newF,*_=self.residual(trial,a,False)
    except ValueError:continue
    if np.max(abs(newF))<error*(1-1e-4*2**(-h)):
     v=trial;accepted=True;break
   if not accepted:return None,dict(converged=False,reason='backtrack')
  return None,dict(converged=False,reason='iteration_budget')

 def fold(self):
  a=0.;da=max(1e-5,min(.001,self.delta/25));v=None;points=[];states=[];fails=[];best=0
  for k in range(200):
   nv,info=self.solve(a+da,v)
   if nv is None:
    fails.append(dict(amplitude=a+da,**info));da/=2
    if da<1e-8:break
    continue
   a+=da;v=nv;points.append(info);states.append(v.copy())
   if info['current']>points[best]['current']:best=len(points)-1
   if len(points)-best>4 and info['current']<points[best]['current']*(1-3e-4):break
   da=min(da*1.24,.002)
  result=dict(state='lower_bound_no_bracket',points=points,failed=fails,nodes=self.n,triangles=len(self.mesh.triangles),probe_node=self.probe)
  if not points:return dict(result,state='unresolved'),None
  peak=points[best];state=states[best]
  if 0<best<len(points)-1:
   def fun(a):
    _,info=self.solve(a,states[best]);return -info['current'] if info['converged'] else 1e50
   lo,hi=points[best-1]['amplitude'],points[best+1]['amplitude']
   opt=minimize_scalar(fun,bounds=(lo,hi),method='bounded',options={'xatol':2e-9})
   candidate,info=self.solve(float(opt.x),states[best])
   if opt.success and info['converged'] and info['current']>=max(points[best-1]['current'],points[best+1]['current']):
    result.update(state='converged_bracketed_stationary_fold',peak=info,bracket=[points[best-1],points[best+1]])
    state=candidate;peak=info
  result['attained_current']=peak['current']
  return result,self.Z+self.unpack(state)

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path(__file__).parent/'local_mesh_results/stationary');p.add_argument('--fine',action='store_true');p.add_argument('--cases',nargs='*',default=['E1','E2','E3']);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
 for name,eta,delta,period,ref in REFERENCES:
  if name not in args.cases:continue
  dest=args.out/f'{name}.json'
  if dest.exists():continue
  m=make_case(eta,delta,period,ref)
  if args.fine:
   m['domain'].update(nz='48',ny='48')
   m['mesh_refinements'][0].update(size='.0125*d',growth='1.15')
   m['mesh_refinements'][1].update(size=f'{min(.001,delta/2,eta/24)}*d',growth='1.15')
  started=time.time();solver=FoldCheck(m);print(f'START stationary {name}: {solver.n} nodes',flush=True)
  r,phi=solver.fold();r.update(case=name,reference_J_over_JCL=ref,relative_error_percent=100*(r.get('attained_current',0)/ref-1),elapsed_seconds=time.time()-started,
    scope='Stationary transmitting P1 fold on the actual new mesh; separate from PIC below/above behavior checks.',fine=args.fine,model=m)
  dest.write_text(json.dumps(r,indent=2)+'\n')
  if phi is not None:np.savez_compressed(args.out/f'{name}.npz',points=solver.mesh.points,triangles=solver.mesh.triangles,phi=phi)
  print('RESULT '+json.dumps({k:v for k,v in r.items() if k not in ('points','failed','model','bracket')}),flush=True)
