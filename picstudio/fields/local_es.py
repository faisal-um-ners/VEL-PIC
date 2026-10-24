"""Electrostatic-only P1 backend for the optional prescribed local mesh.

The ordinary tensor Q1 / Yee paths are untouched. This backend deposits on
actual triangles, solves their weak Poisson problem, and gathers their fields.
The rectangular arrays retained by Simulation are display/compatibility samples.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from .. import constants as C
from ..geometry.shapes import Shape
from ..mesh.grid import VACUUM,CONDUCTOR,DIELECTRIC,ABSORBER
from ..sources.emitters import Emitter


class LocalPoisson:
    def __init__(self,mesh,model,ev,mats,rz):
        self.mesh=mesh;self.rz=rz;self.model=model
        p=mesh.points;t=mesh.triangles;n=len(p)
        self.node_cond=np.full(n,-1,int)
        self.cell_obj=np.full(len(t),-1,int);self.cell_mat=np.zeros(len(t),int)
        self.eps_cell=np.ones(len(t));cent=p[t].mean(1)
        # Match the existing object's paint order, on the actual new elements.
        obj_conductor={c['object']:i for i,c in enumerate(mats.conductors) if c['source']=='object'}
        for k,obj in enumerate(model['objects']):
            mat=obj.get('material','conductor');shape=Shape(obj,ev)
            ni=shape.contains(*p.T);ci=shape.contains(*cent.T)
            if mat=='conductor':
                self.node_cond[ni]=obj_conductor[k]
            elif mat=='vacuum':self.node_cond[Shape(obj,ev,tol=-1e-9*mesh.hmin).contains(*p.T)]=-1
            if mat=='lossy':continue
            code={'vacuum':VACUUM,'conductor':CONDUCTOR,'dielectric':DIELECTRIC,'absorber':ABSORBER}[mat]
            self.cell_mat[ci]=code;self.cell_obj[ci]=k if mat!='vacuum' else -1
            self.eps_cell[ci]=ev.num(obj.get('eps_r',1)) if mat=='dielectric' else 1
        self.boundary_cond={c['side']:i for i,c in enumerate(mats.conductors) if c['source']=='boundary'}
        for side,k in self.boundary_cond.items():self.node_cond[mesh.boundary_nodes[side]]=k
        # A particle-absorbing metal triangle must have all three conductor DOFs.
        solid=np.all(self.node_cond[t]>=0,axis=1)
        wrong=(self.cell_mat==CONDUCTOR)&~solid
        self.cell_mat[wrong]=VACUUM;self.cell_obj[wrong]=-1
        for k in obj_conductor:
            sel=solid&np.all(self.node_cond[t]==obj_conductor[k],axis=1)
            self.cell_mat[sel]=CONDUCTOR;self.cell_obj[sel]=k
        self.solid=solid
        area=mesh.area;r=p[t,1]
        weight=area*r.mean(1) if rz else area
        ke=C.eps0*self.eps_cell[:,None,None]*weight[:,None,None]*np.einsum('eia,eja->eij',mesh.grad,mesh.grad)
        ii=np.repeat(t,3,axis=1).ravel();jj=np.tile(t,(1,3)).ravel()
        self.K=sp.coo_matrix((ke.ravel(),(ii,jj)),shape=(n,n)).tocsr()
        mass=np.empty((len(t),3,3))
        if rz:
            for i in range(3):
                for j in range(3):
                    mass[:,i,j]=area*((r.sum(1)+2*r[:,i])/30 if i==j else (r.sum(1)+r[:,i]+r[:,j])/60)
        else:mass[:]=area[:,None,None]*(np.ones((3,3))+np.eye(3))[None,:,:]/12
        self.M=sp.coo_matrix((mass.ravel(),(ii,jj)),shape=(n,n)).tocsc()
        self.volume=np.asarray(self.M.sum(1)).ravel()
        self.physical_volume=self.volume*(2*np.pi if rz else 1)
        self.mass_lu=splu(self.M)
        # Exact pairwise identification of matching boundary tick coordinates.
        parent=np.arange(n)
        def root(i):
            while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
            return i
        def union(a,b):
            a,b=root(a),root(b)
            if a!=b:parent[b]=a
        for axis in range(2):
            if not mesh.periodic[axis]:continue
            a,b=mesh.boundary_nodes[('z','y')[axis]+'_min'],mesh.boundary_nodes[('z','y')[axis]+'_max']
            if len(a)!=len(b) or not np.array_equal(mesh.keys[a,1-axis],mesh.keys[b,1-axis]):
                raise ArithmeticError('Periodic local mesh boundaries do not match.')
            for x,y in zip(a,b):union(int(x),int(y))
        reps=np.array([root(i) for i in range(n)]);_,red=np.unique(reps,return_inverse=True)
        nr=red.max()+1
        self.R=sp.csr_matrix((np.ones(n),(np.arange(n),red)),shape=(n,nr));self.red=red
        self.Kr=(self.R.T@self.K@self.R).tocsr()
        cond=np.full(nr,-1,int)
        for i in np.flatnonzero(self.node_cond>=0):
            # Existing corner precedence is retained; do not invent a voltage average.
            cond[red[i]]=self.node_cond[i]
        self.cond=cond;dm=cond>=0;self.singular=not dm.any()
        if self.singular:dm[0]=True
        self.free=np.flatnonzero(~dm);self.dir=np.flatnonzero(dm)
        self.Kfd=self.Kr[self.free][:,self.dir].tocsr()
        self.lu=splu(self.Kr[self.free][:,self.free].tocsc())
        self.vol_red=np.asarray(self.R.T@self.volume).ravel()
        self.laplace=[]
        for k in range(len(mats.conductors)):
            d=(cond[self.dir]==k).astype(float);v=np.zeros(nr)
            v[self.dir]=d;v[self.free]=self.lu.solve(-self.Kfd@d)
            self.laplace.append(np.asarray(self.R@v))
        self.phi=np.zeros(n);self.rhs=np.zeros(n)
        self.Mr=(self.R.T@self.M@self.R).tocsc()
        self._projectors={}

    def solve(self,rhs,voltages=None):
        b=np.asarray(self.R.T@rhs).ravel()
        if self.singular:b-=self.vol_red*b.sum()/self.vol_red.sum()
        v=np.zeros(len(b));v[self.free]=self.lu.solve(b[self.free])
        if self.singular:v-=np.dot(v,self.vol_red)/self.vol_red.sum()
        phi=np.asarray(self.R@v)
        if voltages is not None:
            for a,w in zip(voltages,self.laplace):phi+=a*w
        return phi

    def nodal_field(self,phi):
        grad=np.einsum('ei,eia->ea',phi[self.mesh.triangles],self.mesh.grad)
        # Consistent L2 projection; use a single periodic node space for fields.
        mr=self.Mr;out=[]
        for axis in range(2):
            rhs=np.bincount(self.mesh.triangles.ravel(),weights=(-grad[:,axis,None]*
                self._element_volume()).ravel(),minlength=len(phi))
            b=np.asarray(self.R.T@rhs).ravel()
            fixed=[]
            for side,ids in self.mesh.boundary_nodes.items():
                if side.startswith(('z','y')[axis]) and self.model['boundaries'][side]['field'] in ('neumann','axis'):
                    fixed.extend(self.red[ids])
            mask=np.ones(mr.shape[0],bool);mask[np.unique(fixed).astype(int)]=False
            if axis not in self._projectors:self._projectors[axis]=(mask,splu(mr[mask][:,mask].tocsc()))
            mask,lu=self._projectors[axis];v=np.zeros(mr.shape[0]);v[mask]=lu.solve(b[mask])
            out.append(np.asarray(self.R@v))
        return np.array(out)

    def _element_volume(self):
        if self.rz:
            r=self.mesh.points[self.mesh.triangles,1]
            return self.mesh.area[:,None]*(r.sum(1)[:,None]+r)/12
        return np.broadcast_to(self.mesh.area[:,None]/3,(len(self.mesh.triangles),3))


class LocalElectrostatics:
    def __init__(self,grid,mats,model,ev):
        self.mesh=grid.local_mesh;self.grid=grid;self.model=model
        self.P=LocalPoisson(self.mesh,model,ev,mats,grid.rz)
        self.phi=np.zeros(len(self.mesh.points));self.q=np.zeros_like(self.phi)
        self.stuck=np.zeros_like(self.phi);self.field=np.zeros((6,len(self.phi)))
        self.per={};self.external=None
        Z,Y=grid.nodes();self.sample_z=Z.ravel();self.sample_y=Y.ravel()
        self._sample_ids,self._sample_N=self.mesh.locate(self.sample_z,self.sample_y)
        self.surface_areas={}
        for side,ids in self.mesh.boundary_nodes.items():
            edges=np.c_[ids[:-1],ids[1:]]
            self.surface_areas[side]=self._edge_areas(edges)

    def _edge_areas(self,edges):
        p=self.mesh.points[edges];length=np.linalg.norm(p[:,1]-p[:,0],axis=1)
        if self.grid.rz:
            wa=2*np.pi*length*(p[:,0,1]/3+p[:,1,1]/6)
            wb=2*np.pi*length*(p[:,1,1]/3+p[:,0,1]/6)
        else:wa=wb=length/2
        return np.bincount(edges.ravel(),weights=np.c_[wa,wb].ravel(),minlength=len(self.phi))

    def sample(self,v):
        a=np.asarray(v)[self.mesh.triangles[self._sample_ids]]
        return np.einsum('ij,ij->i',a,self._sample_N).reshape(self.grid.shape)

    def laplace_native(self,V):
        out=np.zeros_like(self.phi)
        for a,v in zip(V,self.P.laplace):out+=a*v
        return out

    def laplace_field(self,V):return self.sample(self.laplace_native(V))
    def energy(self,phi=None):return .5*float(self.phi@(self.P.K@self.phi))*(2*np.pi if self.grid.rz else 1)

    def update(self,sim,t):
        self.per={}
        for species in sim.species:
            q=self.mesh.deposit(species.z,species.y,species.q*species.w) if species.n else np.zeros_like(self.q)
            self.per[species.name]=q
        q=sum(self.per.values(),np.zeros_like(self.q))
        if sim.size>1:q=sim.comm.allreduce(q)
        self.q=q+self.stuck;rhs=self.q/(2*np.pi) if self.grid.rz else self.q.copy()
        self.phi=self.P.solve(rhs,sim.voltages(t));self.P.phi=self.phi;self.P.rhs=rhs
        self.reaction=self.P.K@self.phi-rhs
        self.field[:]=0;self.field[:2]=self.P.nodal_field(self.phi)
        for species in sim.species:
            if not species.beam_self_B:continue
            weights=sim.comm.allsum(float(species.w.sum())) if species.n or sim.size>1 else 0
            if not weights:continue
            beta=sim.comm.allsum(float((species.w*species.u0/species.gamma(sim.rel)).sum()))/weights/C.c
            qs=self.per[species.name]
            if sim.size>1:qs=sim.comm.allreduce(qs)
            phis=self.P.solve(qs/(2*np.pi) if self.grid.rz else qs)
            Es=self.P.nodal_field(phis)
            self.field[5]+=(1 if self.grid.rz else -1)*beta*Es[1]/C.c
        sim.phi=self.sample(self.phi)
        self.rho_native=self._density(self.q)
        sim.rho=self.sample(self.rho_native);sim.rhs_fem=rhs
        sim.F[:]=np.array([self.sample(v) for v in self.field])
        self.external=sim.ext.nodal(t)
        if self.external is not None:sim.F+=self.external
        sim.F_gather[:]=sim.F

    def _density(self,q):
        # Periodic duplicates share both charge and control volume.
        redq=np.asarray(self.P.R.T@q).ravel()
        volume=np.asarray(self.P.R.T@self.P.physical_volume).ravel()
        return np.asarray(self.P.R@(redq/volume))

    def gather(self,sim,species):
        z,y=species.z,species.y;ids,N=self.mesh.locate(z,y)
        P=np.einsum('cni,ni->cn',self.field[:,self.mesh.triangles[ids]],N)
        if sim.gather_ec:
            P[:2]=-np.einsum('ni,nia->na',self.phi[self.mesh.triangles[ids]],self.mesh.grad[ids]).T
        if self.external is not None:
            for c in range(6):P[c]+=self.grid.interp(self.external[c],z,y)
        return np.ascontiguousarray(P)

    def interp(self,sim,A,z,y):
        if A is sim.phi:return self.mesh.interpolate(self.phi,z,y)
        if A is sim.rho:return self.mesh.interpolate(self.rho_native,z,y)
        if np.shares_memory(A,sim.F):
            offset=(A.__array_interface__['data'][0]-sim.F.__array_interface__['data'][0])//sim.F.strides[0]
            v=self.mesh.interpolate(self.field[offset],z,y)
            if self.external is not None:v+=self.grid.interp(self.external[offset],z,y)
            return v
        return self.grid.interp(A,z,y)

    def boundary_objects(self,z,y,flag):
        inside=flag==0
        if not inside.any():return flag
        ids,_=self.mesh.locate(z[inside],y[inside]);obj=self.P.cell_obj[ids]
        hit=obj>=0
        at=np.flatnonzero(inside)[hit];flag[at]=10+obj[hit]
        return flag

    def snapshot(self,sim):
        d={'local_points':self.mesh.points,'local_triangles':self.mesh.triangles,
           'local_phi':self.phi,'local_phi_sc':self.phi-self.laplace_native(sim.voltages(sim.t)),
           'local_rho':self.rho_native,'local_cell_mat':self.P.cell_mat,
           'mesh_type':'local P1 triangles (fixed)'}
        for k,name in enumerate(('E'+sim.comp[0],'E'+sim.comp[1],'E'+sim.comp[2],
                                 'B'+sim.comp[0],'B'+sim.comp[1],'B'+sim.comp[2])):
            a=self.field[k].copy()
            if self.external is not None:a+=self.grid.interp(self.external[k],*self.mesh.points.T)
            d['local_'+name]=a
        for name,q in self.per.items():
            if sim.size>1:q=sim.comm.allreduce(q)
            d['local_rho_'+name]=self._density(q)
            d['rho_'+name]=self.sample(d['local_rho_'+name])
        return d


class LocalEmitter(Emitter):
    """Existing emission laws, with panels and reaction fields on the real mesh."""
    def __init__(self,backend,*args,**kwargs):
        self.backend=backend
        super().__init__(*args,**kwargs)

    def _set_panels(self,A,B,n,area_nodes=None):
        mesh=self.backend.mesh
        self.start=A;self.centers=(A+B)/2;self.ds=np.linalg.norm(B-A,axis=1)
        if not len(A) or np.any(self.ds<=0):raise ValueError('Emitter has no nonzero local mesh panels.')
        self.tvec=(B-A)/self.ds[:,None];self.nvec=np.broadcast_to(n,A.shape).copy()
        self.area=self.ds*(2*np.pi*np.abs(self.centers[:,1]) if self.g.rz else 1)
        ids,_=mesh.locate(*(self.centers+self.eps*self.nvec).T)
        self.h=.25*mesh.altitude[ids]
        self.local_node_ids=area_nodes
        self.react_idx='local'

    def _panels_segment(self,spec):
        ev=self.ev;mesh=self.backend.mesh
        a=np.array([ev.num(spec['z0']),ev.num(spec['y0'])]);b=np.array([ev.num(spec['z1']),ev.num(spec['y1'])])
        length=np.linalg.norm(b-a)
        if length<=0:raise ValueError('Emitter has zero length.')
        direction=(b-a)/length;normal=spec.get('normal','auto')
        n=np.array([-direction[1],direction[0]]) if normal=='auto' else np.array({'+z':(1,0),'-z':(-1,0),'+y':(0,1),'-y':(0,-1)}[normal])
        self.local_side=None
        for side,ids in mesh.boundary_nodes.items():
            axis=0 if side.startswith('z') else 1
            if abs(direction[axis])<1e-12 and abs(a[axis]-mesh.points[ids[0],axis])<self.eps:
                transverse=1-axis
                coord=mesh.points[ids,transverse]
                lo,hi=sorted((a[transverse],b[transverse]))
                selected=ids[(coord>=lo-self.eps)&(coord<=hi+self.eps)]
                if len(selected)<2:raise ValueError('Emitter endpoints must be local boundary nodes.')
                if direction[transverse]<0:selected=selected[::-1]
                ends=mesh.points[selected]
                if max(np.linalg.norm(ends[0]-a),np.linalg.norm(ends[-1]-b))>self.eps:
                    raise ValueError('Emitter ends are not resolved by the local mesh.')
                if normal=='auto':n=np.eye(2)[axis]*(1 if side.endswith('min') else -1)
                self.local_side=side
                self._set_panels(ends[:-1],ends[1:],n,np.c_[selected[:-1],selected[1:]])
                return
        # An internal prescribed source is not a conductor wall. Split it by
        # its crossed elements rather than using the background display grid.
        edges=mesh.points[mesh.edge_nodes];v=edges[:,1]-edges[:,0]
        cross=lambda a,b:a[...,0]*b[...,1]-a[...,1]*b[...,0]
        denom=cross(direction,v);ok=np.abs(denom)>1e-30
        t=np.zeros(len(v));u=np.zeros(len(v))
        t[ok]=cross(edges[ok,0]-a,v[ok])/denom[ok]
        u[ok]=cross(edges[ok,0]-a,direction)/denom[ok]
        sk=np.unique(np.r_[0,t[ok&(t>0)&(t<length)&(u>=0)&(u<=1)],length])
        ends=a+sk[:,None]*direction
        self._set_panels(ends[:-1],ends[1:],n)
        self.react_idx=None

    def _panels_object(self,name,objects):
        names=[o.get('name','') for o in objects]
        if name not in names:raise ValueError(f'Unknown emitting object {name}')
        k=names.index(name);mesh=self.backend.mesh;P=self.backend.P
        edges=mesh.edge_nodes;owners=mesh.edge_cells
        a=owners[:,0];b=owners[:,1];valid=b>=0
        sa=(P.cell_obj[a]==k)&(P.cell_mat[a]==CONDUCTOR)
        sb=np.zeros(len(edges),bool);sb[valid]=(P.cell_obj[b[valid]]==k)&(P.cell_mat[b[valid]]==CONDUCTOR)
        va=P.cell_mat[a]==VACUUM;vb=np.zeros(len(edges),bool);vb[valid]=P.cell_mat[b[valid]]==VACUUM
        hit=valid&((sa&vb)|(sb&va));idx=edges[hit]
        if not len(idx):raise ValueError('Emitting object has no local surface facing vacuum.')
        A,B=mesh.points[idx[:,0]],mesh.points[idx[:,1]];v=B-A
        n=np.c_[v[:,1],-v[:,0]];n/=np.linalg.norm(n,axis=1)[:,None]
        centers=mesh.points[mesh.triangles].mean(1)
        vacuum=np.where(sa[hit],b[hit],a[hit]);toward=centers[vacuum]-(A+B)/2
        n*=np.where(np.einsum('ij,ij->i',n,toward)>0,1,-1)[:,None]
        self.local_side=None;self.object_node_area=self.backend._edge_areas(idx)
        self._set_panels(A,B,n,idx)

    def normal_field(self,phi=None,efield=None,reaction=None,yee=None,static_E=None,qnodes=None):
        backend=self.backend;P=backend.P
        if self.local_node_ids is not None:
            area=backend.surface_areas[self.local_side] if self.local_side else self.object_node_area
            idx=self.local_node_ids
            if np.all(P.node_cond[idx]>=0):
                values=backend.reaction[idx]*(2*np.pi if self.g.rz else 1)/(C.eps0*np.maximum(area[idx],1e-300))
                return values.mean(1)
        ids,N=backend.mesh.locate(*(self.centers+self.eps*self.nvec).T)
        E=np.einsum('cni,ni->nc',backend.field[:2,backend.mesh.triangles[ids]],N)
        return np.einsum('ij,ij->i',E,self.nvec)


def load_local_particles(spec,species,backend,ev,rng,rank=0,size=1):
    """Same loading law; particles-per-cell now refers to actual triangles."""
    mesh=backend.mesh;g=backend.grid
    z0,z1=max(ev.num(spec['z0']),g.z_min),min(ev.num(spec['z1']),g.z_max)
    y0,y1=max(ev.num(spec['y0']),g.y_min),min(ev.num(spec['y1']),g.y_max)
    if z1<=z0 or y1<=y0:return 0
    ppc=max(1,int(round(ev.num(spec['ppc']))));quiet=bool(spec.get('quiet_start',False))
    pts=mesh.points[mesh.triangles]
    active=(pts[:,:,0].max(1)>z0)&(pts[:,:,0].min(1)<z1)&(pts[:,:,1].max(1)>y0)&(pts[:,:,1].min(1)<y1)&(backend.P.cell_mat==VACUUM)
    cells=np.flatnonzero(active)
    if len(cells)*ppc>20_000_000:raise ValueError('Local particle loading exceeds 20 million initial particles; reduce particles per cell or the loaded region.')
    if quiet:
        k=np.arange(ppc);a=np.sqrt((k+.5)/ppc);b=(.5+k*((np.sqrt(5)-1)/2))%1
        bary=np.c_[1-a,a*(1-b),a*b]
        pos=np.einsum('ki,eij->ekj',bary,pts[cells]).reshape(-1,2)
    else:
        a=np.sqrt(rng.random((len(cells),ppc)));b=rng.random((len(cells),ppc))
        bary=np.stack((1-a,a*(1-b),a*b),axis=-1)
        pos=np.einsum('eki,eij->ekj',bary,pts[cells]).reshape(-1,2)
    cell=np.repeat(cells,ppc);slot=np.tile(np.arange(ppc),len(cells))
    Z,Y=pos.T
    keep=(Z>=z0)&(Z<z1)&(Y>=y0)&(Y<y1)
    Z,Y,cell,slot=Z[keep],Y[keep],cell[keep],slot[keep]
    density=np.broadcast_to(np.asarray(ev.value(spec['density'],z=Z,y=Y,r=Y,t=0),float),Z.shape)
    W=density*mesh.area[cell]/ppc*(2*np.pi*Y if g.rz else 1)
    keep=W>0;Z,Y,W,cell,slot=Z[keep],Y[keep],W[keep],cell[keep],slot[keep]
    n=len(Z);temperature=ev.num(spec.get('temperature',0));vt=np.sqrt(max(temperature,0)*C.e/species.m)
    drift=[ev.num(v) for v in spec.get('drift',[0,0,0])];U=np.empty((3,n))
    if quiet and temperature>0 and ppc>=16:
        from scipy.special import erfinv
        quant=np.sqrt(2)*erfinv(2*(np.arange(ppc)+.5)/ppc-1)
        _,block=np.unique(cell,return_inverse=True)
        nb=int(block.max())+1 if n else 0
        for c in range(3):
            perm=np.argsort(rng.random((nb,ppc)),axis=1)
            U[c]=drift[c]+vt*quant[perm[block,slot]]
    else:
        for c in range(3):U[c]=drift[c]+(vt*rng.standard_normal(n) if vt>0 else 0)
    perturb=spec.get('perturbation','none')
    if perturb!='none':
        A=ev.num(spec.get('pert_amplitude',0));mode=ev.num(spec.get('pert_mode',1));k=2*np.pi*mode/(z1-z0)
        if perturb=='density':W*=1+A*np.cos(k*(Z-z0))
        else:U[0]+=A*np.sin(k*(Z-z0))
    if size>1:
        part=slice(rank,None,size);Z,Y,U,W=Z[part],Y[part],U[:,part],W[part]
    species.add(Z,Y,U[0],U[1],U[2],W)
    return len(Z)
