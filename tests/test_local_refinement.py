"""Local mesh: topology, electrostatics, particle coupling, GUI and PIC regressions."""
import copy
import numpy as np
import pytest
from picstudio.model import model as M
from picstudio.mesh.grid import grid_from_model
from picstudio.fields.local_es import LocalPoisson
from picstudio.mesh.grid import Materials
from picstudio.core.simulation import Simulation
from picstudio import constants as C


def model(geometry='2D-ZY',n=8,h=.035):
    m=M.new_model(geometry)
    m['domain'].update(z_min='0',z_max='1',nz=str(n),y_min='0',y_max='1',ny=str(n))
    m['mesh_refinements']=[dict(axis='local 2D',region_shape='polygon',
        points='0,0;1,0;1,0.3;0,0.06',size=str(h),growth='1.2',buffer='.02')]
    m['boundaries']['z_max']['voltage']='100'
    for side in ('y_min','y_max'):m['boundaries'][side].update(field='neumann',particles='reflect')
    if geometry=='2D-RZ':m['boundaries']['y_min']['field']='axis'
    m['parallel'].update(mode='serial',threads=1)
    return M.complete(m)


def test_topology_locality_and_partition():
    m=model();g=grid_from_model(m,M.evaluator(m));t=g.local_mesh
    assert np.isclose(t.area.sum(),1)
    assert t.area.min()>0
    assert (t.edge_cells[:,1]<0).sum()==sum(len(a)-1 for a in t.boundary_nodes.values())
    cent=t.points[t.triangles].mean(1)
    assert np.median(t.area[cent[:,1]>.85])>6*np.median(t.area[cent[:,1]<.04])
    rng=np.random.default_rng(7);z,y=rng.random((2,300))
    affine=3*t.points[:,0]-2*t.points[:,1]+1
    assert np.max(abs(t.interpolate(affine,z,y)-(3*z-2*y+1)))<1e-12
    charges=rng.normal(size=300);q=t.deposit(z,y,charges)
    assert abs(q.sum()-charges.sum())<1e-12
    v=rng.normal(size=len(q))
    assert abs(q@v-np.dot(charges,t.interpolate(v,z,y)))<1e-12


@pytest.mark.parametrize('geom',['2D-ZY','2D-RZ'])
def test_vacuum_fields_and_volumes(geom):
    s=Simulation(model(geom),write_output=False,log=lambda _:None)
    b=s.local_es;t=b.mesh
    assert np.max(abs(b.phi-100*t.points[:,0]))<1e-8
    assert np.max(abs(b.field[0]+100))<1e-8
    assert np.max(abs(b.field[1]))<1e-8
    expected=np.pi if geom=='2D-RZ' else 1
    assert abs(b.P.physical_volume.sum()-expected)<1e-12
    cathode=b.P.node_cond==b.P.boundary_cond['z_min']
    anode=b.P.node_cond==b.P.boundary_cond['z_max']
    factor=2*np.pi if geom=='2D-RZ' else 1
    assert abs(b.reaction[cathode].sum()*factor+C.eps0*100*expected)<1e-18
    assert abs(b.reaction[anode].sum()*factor-C.eps0*100*expected)<1e-18


def test_periodic_asymmetric_region():
    m=model();m['mesh_refinements'][0].update(points='0,0;1,0;1,.1;0,.04')
    for side in ('y_min','y_max'):m['boundaries'][side].update(field='periodic',particles='periodic')
    s=Simulation(m,write_output=False,log=lambda _:None);b=s.local_es
    a,c=(b.mesh.boundary_nodes[x] for x in ('y_min','y_max'))
    assert np.array_equal(b.mesh.keys[a,0],b.mesh.keys[c,0])
    assert np.max(abs(b.phi[a]-b.phi[c]))<1e-12
    assert np.max(abs(b.phi-100*b.mesh.points[:,0]))<1e-8


def test_manufactured_solution_converges():
    errors=[];l2=[]
    for n in (6,12):
        m=model(n=n,h=.1/n)
        m['mesh_refinements'][0]['growth']=str(1+.2*6/n)
        for side in m['boundaries']:m['boundaries'][side]['field']='conductor'
        m['boundaries']['z_max']['voltage']='0'
        s=Simulation(m,write_output=False,log=lambda _:None);b=s.local_es;p=b.mesh.points
        exact=np.sin(np.pi*p[:,0])*np.sin(np.pi*p[:,1])
        rhs=b.P.M@(2*np.pi**2*C.eps0*exact)
        phi=b.P.solve(rhs,[0]*len(b.P.laplace))
        error=phi-exact
        errors.append(np.max(abs(error)));l2.append(float(np.sqrt(error@(b.P.M@error))))
    assert errors[1]<errors[0]/2
    assert l2[1]<l2[0]/3


def test_fixed_mesh_PIC_and_both_gathers():
    for gather in ('energy-conserving','momentum-conserving'):
        m=model();m['species']=[dict(name='e',preset='electron',motion='full')]
        m['emitters']=[dict(name='beam',species='e',model='beam',z0='0',z1='0',y0='0',y1='.08',normal='+z',current_density='1e-7',energy='1',particles_per_cell='1')]
        m['simulation'].update(dt='1e-9',n_steps='4',field_gathering=gather)
        s=Simulation(m,write_output=False,log=lambda _:None)
        t=s.local_es.mesh;points=t.points.copy();tri=t.triangles.copy()
        for _ in range(4):s.step()
        assert s.species[0].n>0 and np.isfinite(s.local_es.phi).all()
        assert np.array_equal(t.points,points) and np.array_equal(t.triangles,tri)
        assert abs(s.local_es.q.sum()-s.species[0].q*s.species[0].w.sum())<1e-25
        assert s._gather(s.species[0]).shape==(6,s.species[0].n)
        assert 'local_phi' in s.field_snapshot()


def test_unsupported_modes_and_invalid_region_rejected():
    m=model();m['simulation']['solver']='electromagnetic'
    assert any(k=='error' and 'electrostatic' in v for k,v in M.validate(m))
    with pytest.raises(ValueError):Simulation(m,write_output=False)
    m=model();m['simulation']['smoothing']=1
    assert any(k=='error' and 'smoothing' in v for k,v in M.validate(m))
    m=model();m['mesh_refinements'][0]['points']='0,0;1,1;0,1;1,0'
    assert any(k=='error' for k,v in M.validate(m))


def test_disabled_local_and_old_bands_unchanged():
    m=model();m['mesh_refinements'][0]['enabled']=False
    g=grid_from_model(m,M.evaluator(m));assert not hasattr(g,'local_mesh') and g.uniform
    m['mesh_refinements'].append(dict(axis='z',start='0',end='.1',size='.01',growth='1.2'))
    g=grid_from_model(m,M.evaluator(m));assert not hasattr(g,'local_mesh') and not g.uniform


def test_dielectric_object_and_object_emitter():
    m=model();m['objects']=[dict(name='metal',shape='rectangle',material='conductor',z0='0',z1='.25',y0='0',y1='.25',voltage='0')]
    m['species']=[dict(name='e',preset='electron')]
    m['emitters']=[dict(name='surface',species='e',model='scl',surface='object',object='metal',particles_per_cell='1')]
    s=Simulation(m,write_output=False,log=lambda _:None)
    assert s.emitters[0].npanel>0
    assert np.isfinite(s.emitters[0].normal_field()).all()
    m=model();m['objects']=[dict(name='glass',shape='rectangle',material='dielectric',z0='.4',z1='.6',y0='0',y1='.5',eps_r='4')]
    s=Simulation(m,write_output=False,log=lambda _:None)
    assert s.local_es.P.eps_cell.max()==4
    assert np.all(s.local_es.P.node_cond[s.local_es.P.node_cond<0]==-1)


def test_loading_uses_actual_triangles():
    m=model(n=5,h=.06)
    m['species']=[dict(name='e',preset='electron')]
    m['loaders']=[dict(name='plasma',species='e',z0='0',z1='1',y0='0',y1='1',density='1e8',ppc='4',quiet_start=False)]
    m['simulation']['dt']='1e-10'
    s=Simulation(m,write_output=False,log=lambda _:None)
    assert s.species[0].n==4*len(s.local_es.mesh.triangles)
    assert np.isclose(s.species[0].w.sum(),1e8)
    assert np.isclose(s.local_es.q.sum(),s.species[0].q*1e8)


def test_direct_gradient_is_particle_potential_derivative():
    s=Simulation(model(),write_output=False,log=lambda _:None);b=s.local_es
    rng=np.random.default_rng(7);ids=rng.integers(len(b.mesh.triangles),size=20)
    p=b.mesh.points[b.mesh.triangles[ids]].mean(1);values=rng.normal(size=len(b.phi))
    grad=np.einsum('ni,nia->na',values[b.mesh.triangles[ids]],b.mesh.grad[ids])
    eps=1e-8
    for k in range(2):
        shift=np.eye(2)[k]*eps
        numeric=(b.mesh.interpolate(values,*(p+shift).T)-b.mesh.interpolate(values,*(p-shift).T))/(2*eps)
        assert np.max(abs(numeric-grad[:,k]))<2e-6


@pytest.mark.parametrize('side',['z_min','z_max','y_min','y_max'])
def test_emitter_endpoint_is_local_not_a_full_domain_band(side):
    m=model();base=grid_from_model(m,M.evaluator(m)).local_mesh
    axis=0 if side.startswith('z') else 1
    other=1-axis;ends=np.zeros((2,2));ends[:,axis]=int(side.endswith('max'))
    ends[1,other]=.371
    m['species']=[dict(name='e',preset='electron')]
    m['emitters']=[dict(name='beam',species='e',model='beam',surface='segment',
        z0=str(ends[0,0]),z1=str(ends[1,0]),y0=str(ends[0,1]),y1=str(ends[1,1]),
        normal=('+' if side.endswith('min') else '-')+side[0],energy='1',current_density='1e-7')]
    s=Simulation(m,write_output=False,log=lambda _:None);mesh=s.local_es.mesh
    # Exactly one endpoint/triangle is added; the refined leaf distribution is
    # identical. The far boundary gets no extra emitter-aligned row or column.
    assert np.array_equal(mesh.leaves,base.leaves)
    assert all(np.array_equal(a,b) for a,b in zip(mesh.axes,base.axes))
    assert len(mesh.points)==len(base.points)+1
    assert len(mesh.triangles)==len(base.triangles)+1
    assert np.sum(mesh.points[:,other]==.371)==1
    assert np.array_equal(s.emitters[0].start[0],ends[0])
    assert np.array_equal(s.emitters[0].start[-1]+s.emitters[0].ds[-1]*s.emitters[0].tvec[-1],ends[1])
    assert np.isclose(s.emitters[0].ds.sum(),.371)
    p=mesh.points
    assert np.max(abs(s.local_es.phi-100*p[:,0]))<1e-8


def test_periodic_emitter_endpoint_has_matching_boundary_vertex():
    m=model()
    for side in ('y_min','y_max'):m['boundaries'][side].update(field='periodic',particles='periodic')
    m['species']=[dict(name='e',preset='electron')]
    m['emitters']=[dict(name='beam',species='e',model='beam',surface='segment',
        z0='0',z1='.371',y0='0',y1='0',normal='+y',current_density='1e-7',energy='1')]
    s=Simulation(m,write_output=False,log=lambda _:None);mesh=s.local_es.mesh
    a,b=(mesh.boundary_nodes[x] for x in ('y_min','y_max'))
    assert np.array_equal(mesh.points[a,0],mesh.points[b,0])
    assert .371 in mesh.points[a,0]
    assert np.max(abs(s.local_es.phi[a]-s.local_es.phi[b]))<1e-12


def test_stationary_strip_quadrature_clips_crossing_elements(monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'benchmarks'))
    from local_mesh_stationary import strip_quadrature
    m=model(n=5,h=.08);mesh=grid_from_model(m,M.evaluator(m)).local_mesh
    edge=.371
    tri,N,w,vertices=strip_quadrature(mesh,edge,5,1)
    coords=np.einsum('eqi,eij->eqj',N,mesh.points[tri])
    assert np.any(np.max(mesh.points[tri,1],axis=1)>edge)
    assert np.max(coords[:,:,1])<edge
    assert abs(w.sum()-edge)<1e-13
    assert abs(np.sum(w*coords[:,:,0])-edge/2)<1e-13
    assert abs(np.sum(w*coords[:,:,1])-edge**2/2)<1e-13
    assert np.max(abs(N.sum(axis=2)-1))<1e-14
    assert np.min(N)>-1e-14
