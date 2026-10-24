"""Static, user-prescribed local 2D refinement. No particle-following remeshing.

Axis-aligned background leaves split only where the prescribed size requires it.
Every leaf edge includes its neighbors' hanging vertices; center fans therefore
produce conforming P1 triangles, without Delaunay jitter or discarded elements.
"""
from __future__ import annotations
from collections import defaultdict
from functools import lru_cache
import numpy as np
from matplotlib.tri import Triangulation
from ..geometry.shapes import parse_points, point_in_polygon

LOCAL = 'local 2D'
MAX_LEAVES = 150000
MAX_DEPTH = 20
TICKS = 1 << MAX_DEPTH


def local_regions(model):
    return [r for r in model.get('mesh_refinements', [])
            if r.get('enabled', True) and r.get('axis') == LOCAL]


class Region:
    def __init__(self, spec, ev):
        self.spec = spec
        self.h = float(ev.num(spec['size']))
        self.growth = float(ev.num(spec.get('growth', 1.2)))
        self.buffer = float(ev.num(spec.get('buffer', 0)))
        shape = spec.get('region_shape', 'rectangle')
        if shape == 'rectangle':
            a,b = sorted((ev.num(spec['z0']), ev.num(spec['z1'])))
            c,d = sorted((ev.num(spec['y0']), ev.num(spec['y1'])))
            self.points = np.array([[a,c],[b,c],[b,d],[a,d]],float)
        elif shape == 'polygon':
            self.points = parse_points(spec.get('points',''), ev)
        else:
            raise ValueError(f'Unknown local refinement shape: {shape}')
        p = self.points
        if len(p)>3 and np.array_equal(p[0],p[-1]): self.points=p=p[:-1]
        if (p.ndim!=2 or p.shape[1]!=2 or len(p)<3 or not np.isfinite(p).all()
                or not np.isfinite([self.h,self.growth,self.buffer]).all()
                or self.h<=0 or self.growth<=1 or self.buffer<0):
            raise ValueError('Local refinement needs a finite polygon, positive size, growth > 1 and buffer >= 0.')
        cross = p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1)
        if abs(cross.sum()) <= np.finfo(float).eps*max(np.ptp(p,axis=0).prod(),1e-300):
            raise ValueError('Local refinement polygon has zero area.')
        # Reject self-intersections; the boundary is a simple filled polygon.
        for i in range(len(p)):
            for j in range(i+1,len(p)):
                if j==i+1 or (i==0 and j==len(p)-1): continue
                a,b,c,d=p[i],p[(i+1)%len(p)],p[j],p[(j+1)%len(p)]
                def orient(a,b,c):
                    u,v=b-a,c-a
                    return u[0]*v[1]-u[1]*v[0]
                if orient(a,b,c)*orient(a,b,d)<0 and orient(c,d,a)*orient(c,d,b)<0:
                    raise ValueError('Local refinement polygon must not intersect itself.')
        self.a=p;self.v=np.roll(p,-1,axis=0)-p
        self.vv=np.einsum('ij,ij->i',self.v,self.v)
        if np.any(self.vv==0):raise ValueError('Local polygon has a repeated consecutive point.')

    def distance(self, z, y):
        z,y=np.broadcast_arrays(np.asarray(z,float),np.asarray(y,float))
        q=np.stack((z,y),axis=-1)
        t=np.clip(np.sum((q[...,None,:]-self.a)*self.v,axis=-1)/self.vv,0,1)
        d=np.sqrt(np.min(np.sum((q[...,None,:]-(self.a+t[...,None]*self.v))**2,axis=-1),axis=-1))
        return np.where(point_in_polygon(z,y,self.points),0,d)


def validate_local(model,ev):
    regs=local_regions(model)
    if not regs:return
    if model['simulation']['solver']!='electrostatic':
        raise ValueError('Local 2D refinement is electrostatic only. The EM Yee solver requires a uniform grid. Disable the local refinement regions.')
    if model['simulation']['geometry']=='1D-Z':
        raise ValueError('Local 2D refinement requires 2D-ZY or 2D-RZ electrostatics.')
    if int(model['simulation'].get('smoothing',0) or 0):
        raise ValueError('Local 2D mesh: set smoothing to 0; the rectangular 1-2-1 filter is not defined on triangles.')
    d=model['domain'];lo=np.array([ev.num(d['z_min']),ev.num(d['y_min'])]);hi=np.array([ev.num(d['z_max']),ev.num(d['y_max'])])
    if np.any(hi<=lo):raise ValueError('Local mesh needs a positive 2D domain.')
    for r in regs:
        R=Region(r,ev)
        if np.any(R.points.max(axis=0)<lo) or np.any(R.points.min(axis=0)>hi):
            raise ValueError(f"Local region '{r.get('name','')}' does not intersect the domain.")


class LocalMesh:
    def __init__(self, model,ev,grid):
        validate_local(model,ev)
        from .refine import geometry_breaks
        # Solid rectangular interfaces need axis alignment. Emitter endpoints
        # only need boundary vertices, not a grid line through empty space.
        solids = dict(model, emitters=[])
        axes=[]
        for name,nodes in [('z',grid.z),('y',grid.y)]:
            extra=[v for v in geometry_breaks(solids,ev,name) if nodes[0]<v<nodes[-1]]
            axes.append(np.unique(np.r_[nodes,extra]))
        za,ya=axes;self.axes=axes
        regs=[Region(r,ev) for r in local_regions(model)]
        periodic=[model['boundaries'][a+'_min']['field']=='periodic' for a in ('z','y')]
        if grid.rz and grid.on_axis: periodic[1]=False
        self.periodic=periodic
        spans=np.array([za[-1]-za[0],ya[-1]-ya[0]])
        shifts=[(dz,dy) for dz in (-spans[0],0,spans[0]) if periodic[0] or dz==0
                         for dy in (-spans[1],0,spans[1]) if periodic[1] or dy==0]
        exact_coordinates=[{},{}]
        def physical(t,axis):
            g=axes[axis];i=np.minimum(np.asarray(t)//TICKS,len(g)-2).astype(int)
            f=(np.asarray(t)-i*TICKS)/TICKS
            result=g[i]+f*(g[i+1]-g[i])
            for tick,value in exact_coordinates[axis].items():
                result=np.where(np.asarray(t)==tick,value,result)
            return result
        self._physical=physical
        pending=[(i*TICKS,(i+1)*TICKS,j*TICKS,(j+1)*TICKS)
                 for i in range(len(za)-1) for j in range(len(ya)-1)]
        if len(pending)>MAX_LEAVES:raise ValueError('Local mesh background exceeds the leaf limit; reduce domain cell counts.')
        leaves=[]
        while pending:
            batch=pending;pending=[]
            a=np.array(batch,dtype=np.int64)
            z0,z1=physical(a[:,0],0),physical(a[:,1],0)
            y0,y1=physical(a[:,2],1),physical(a[:,3],1)
            dz,dy=z1-z0,y1-y0;cz,cy=(z0+z1)/2,(y0+y1)/2
            radius=.5*np.hypot(dz,dy)
            target=np.full(len(a),np.inf)
            for r in regs:
                dist=np.full(len(a),np.inf)
                for sz,sy in shifts:dist=np.minimum(dist,r.distance(cz+sz,cy+sy))
                target=np.minimum(target,r.h+(r.growth-1)*np.maximum(0,dist-radius-r.buffer))
            sz=dz>target*(1+1e-10);sy=dy>target*(1+1e-10)
            for k,(iz0,iz1,iy0,iy1) in enumerate(batch):
                if not (sz[k] or sy[k]):leaves.append((iz0,iz1,iy0,iy1));continue
                if (sz[k] and iz1-iz0<=1) or (sy[k] and iy1-iy0<=1):
                    raise ValueError('Local refinement exceeds 20 subdivisions of a background cell; increase requested cell size.')
                xs=[iz0,(iz0+iz1)//2,iz1] if sz[k] else [iz0,iz1]
                ys=[iy0,(iy0+iy1)//2,iy1] if sy[k] else [iy0,iy1]
                pending.extend((xa,xb,ya_,yb) for xa,xb in zip(xs[:-1],xs[1:]) for ya_,yb in zip(ys[:-1],ys[1:]))
            if len(pending)+len(leaves)>MAX_LEAVES:
                raise ValueError(f'Local mesh exceeds {MAX_LEAVES:,} leaves; increase cell size or reduce the region.')
        self.leaves=np.array(leaves,np.int64)
        horizontal=defaultdict(set);vertical=defaultdict(set)
        for x0,x1,y0,y1 in leaves:
            horizontal[y0].update((x0,x1));horizontal[y1].update((x0,x1))
            vertical[x0].update((y0,y1));vertical[x1].update((y0,y1))
        zmax=(len(za)-1)*TICKS;ymax=(len(ya)-1)*TICKS
        # Add exact segment endpoints only to their outer boundary edge. The
        # adjacent leaf's center fan accommodates them without refining a row.
        # Dyadic vertices retain exact integer ticks; these prescribed vertices
        # use distinct floating tick coordinates and their original SI values.
        def boundary_tick(value,axis,existing):
            ticks=np.array(sorted(existing),float)
            coordinates=physical(ticks,axis)
            hit=np.flatnonzero(coordinates==value)
            if len(hit):return float(ticks[hit[0]])
            g=axes[axis]
            i=int(np.clip(np.searchsorted(g,value,side='right')-1,0,len(g)-2))
            tick=(i+(value-g[i])/(g[i+1]-g[i]))*TICKS
            if tick in exact_coordinates[axis] and exact_coordinates[axis][tick]!=value:
                raise ValueError('Emitter endpoints are closer than floating-point mesh resolution.')
            if tick in existing:
                raise ValueError('Emitter endpoint cannot be resolved without moving an existing mesh node.')
            exact_coordinates[axis][tick]=value
            return float(tick)
        for emitter in model.get('emitters',[]):
            if emitter.get('surface','segment')!='segment':continue
            ends=np.array([[ev.num(emitter.get('z'+str(k),'0')),
                            ev.num(emitter.get('y'+str(k),'0'))] for k in (0,1)])
            for axis,edges,maximum in ((0,vertical,zmax),(1,horizontal,ymax)):
                other=1-axis
                for side,value in ((0,axes[axis][0]),(maximum,axes[axis][-1])):
                    if not np.all(ends[:,axis]==value):continue
                    for point in ends:
                        if not axes[other][0]<=point[other]<=axes[other][-1]:
                            raise ValueError('Emitter endpoint lies outside the local mesh boundary.')
                        edges[side].add(boundary_tick(point[other],other,edges[side]))
        if periodic[0]:
            s=vertical[0]|vertical[zmax];vertical[0]=s.copy();vertical[zmax]=s.copy()
        if periodic[1]:
            s=horizontal[0]|horizontal[ymax];horizontal[0]=s.copy();horizontal[ymax]=s.copy()
        horizontal={k:np.array(sorted(v)) for k,v in horizontal.items()}
        vertical={k:np.array(sorted(v)) for k,v in vertical.items()}
        nodes=[];index={};tri=[]
        # Exact dyadic keys plus prescribed boundary keys; no tolerance merge.
        def node(x,y):
            key=(2*x,2*y)
            if key not in index:index[key]=len(nodes);nodes.append((x,y))
            return index[key]
        def between(vals,lo,hi):return vals[np.searchsorted(vals,lo):np.searchsorted(vals,hi,'right')]
        for x0,x1,y0,y1 in leaves:
            edge=[(x,y0) for x in between(horizontal[y0],x0,x1)]
            edge += [(x1,y) for y in between(vertical[x1],y0,y1)[1:]]
            edge += [(x,y1) for x in between(horizontal[y1],x0,x1)[-2::-1]]
            edge += [(x0,y) for y in between(vertical[x0],y0,y1)[-2:0:-1]]
            ids=[node(x,y) for x,y in edge];center=node((x0+x1)/2,(y0+y1)/2)
            tri.extend((center,ids[k],ids[(k+1)%len(ids)]) for k in range(len(ids)))
        logical=np.array(nodes,float)
        self.points=np.c_[physical(logical[:,0],0),physical(logical[:,1],1)]
        self.triangles=np.array(tri,np.int32)
        self.keys=2*logical
        t=self.points[self.triangles];u=t[:,1]-t[:,0];v=t[:,2]-t[:,0]
        det=u[:,0]*v[:,1]-u[:,1]*v[:,0]
        if np.any(det<=0):raise ArithmeticError('Nonpositive local mesh triangle; no element was silently discarded.')
        self.area=det/2
        if not np.isclose(self.area.sum(),spans.prod(),rtol=2e-12,atol=0):raise ArithmeticError('Local mesh does not cover the full domain.')
        self.grad=np.stack((np.c_[t[:,1,1]-t[:,2,1],t[:,2,0]-t[:,1,0]],
                            np.c_[t[:,2,1]-t[:,0,1],t[:,0,0]-t[:,2,0]],
                            np.c_[t[:,0,1]-t[:,1,1],t[:,1,0]-t[:,0,0]]),axis=1)/det[:,None,None]
        self.triangulation=Triangulation(*self.points.T,triangles=self.triangles)
        self.finder=self.triangulation.get_trifinder()
        lengths=np.linalg.norm(t-np.roll(t,1,axis=1),axis=2)
        self.altitude=det/lengths.max(axis=1)
        self.hmin=float(self.altitude.min())
        self.boundary_nodes={
            'z_min':np.flatnonzero(self.keys[:,0]==0),'z_max':np.flatnonzero(self.keys[:,0]==2*zmax),
            'y_min':np.flatnonzero(self.keys[:,1]==0),'y_max':np.flatnonzero(self.keys[:,1]==2*ymax)}
        for side,ids in self.boundary_nodes.items():
            axis=1 if side.startswith('z') else 0
            self.boundary_nodes[side]=ids[np.argsort(self.points[ids,axis])]
        self.regions=regs
        self.edge_nodes,self.edge_cells=self._edges()

    def _edges(self):
        raw=np.concatenate([self.triangles[:,[0,1]],self.triangles[:,[1,2]],self.triangles[:,[2,0]]])
        raw.sort(axis=1)
        edges,inv=np.unique(raw,axis=0,return_inverse=True)
        owners=np.tile(np.arange(len(self.triangles)),3)
        cells=np.full((len(edges),2),-1,np.int32)
        order=np.argsort(inv,kind='stable');group=inv[order]
        starts=np.r_[0,1+np.flatnonzero(np.diff(group))];counts=np.diff(np.r_[starts,len(group)])
        if np.any(counts>2):raise ArithmeticError('Non-manifold local mesh edge.')
        cells[group[starts],0]=owners[order[starts]]
        both=counts==2;cells[group[starts[both]],1]=owners[order[starts[both]+1]]
        # All singly-owned edges must be on the outer rectangle; an interior
        # crack is not accepted as an accidental Neumann wall.
        p=self.points[edges[counts==1]]
        lo=self.points.min(0);hi=self.points.max(0)
        # Check same side, rather than merely two different outer corners.
        outer=np.zeros(len(p),bool)
        for k in range(2):outer |= np.all(p[:,:,k]==lo[k],axis=1)|np.all(p[:,:,k]==hi[k],axis=1)
        if not outer.all():raise ArithmeticError('Interior crack in local triangular mesh.')
        return edges,cells

    def locate(self,z,y):
        z,y=np.broadcast_arrays(np.asarray(z,float),np.asarray(y,float))
        ids=np.asarray(self.finder(z,y),int)
        if np.any(ids<0):raise ValueError('Point outside the fixed local mesh.')
        p=np.stack((z,y),axis=-1)
        N=np.einsum('...ij,...j->...i',self.grad[ids],p-self.points[self.triangles[ids,0]])
        N[...,0]+=1
        if np.min(N)<-1e-8:raise ArithmeticError('Invalid triangular interpolation coordinates.')
        N=np.maximum(N,0);N/=N.sum(axis=-1,keepdims=True)
        return ids,N

    def interpolate(self,values,z,y):
        ids,N=self.locate(z,y)
        return np.einsum('...i,...i->...',np.asarray(values)[self.triangles[ids]],N)

    def deposit(self,z,y,charges):
        ids,N=self.locate(z,y)
        return np.bincount(self.triangles[ids].ravel(),weights=(np.asarray(charges)[...,None]*N).ravel(),minlength=len(self.points))


@lru_cache(maxsize=8)
def _cached_mesh(serialized):
    import json
    from ..model.model import evaluator
    from .grid import grid_from_model
    m=json.loads(serialized);ev=evaluator(m)
    return LocalMesh(m,ev,grid_from_model(m,ev,_local=False))


def mesh_from_model(model):
    import json
    # Only geometry/mesh data affects the static topology. Never cache particles.
    keys=('simulation','domain','parameters','boundaries','objects','emitters','mesh_refinements')
    serial=json.dumps({k:model.get(k,{} if k in ('simulation','domain','parameters','boundaries') else []) for k in keys},sort_keys=True)
    return _cached_mesh(serial)
