"""Reproduce the README local-P1 / uniform-P1 mesh comparison from actual meshes.

Run with tfgpu Python: python docs/make_mesh_refinement.py
The saved models describe mesh demonstrations, not new PIC limit measurements.
"""
from pathlib import Path
import argparse
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
from picstudio.model.model import load, evaluator, validate
from picstudio.mesh.grid import grid_from_model


def demonstration(root, geometry):
    name = 'strip_caseA_zy' if geometry == '2D-ZY' else 'disk_rz'
    model = load(str(root / f'examples/yylau_2026_{name}.pic.json'))
    model['name'] = ('Manual 2D mesh: strip, polygon + cathode rectangle' if geometry == '2D-ZY'
                     else 'Manual 2D mesh: disk, polygon + cathode rectangle')
    model['description'] = ('Mesh preview only; derived from the Lau et al. (2026) example geometry. '
        'The polygon is an assumed resolution envelope, not a computed trajectory or an imposed wall. '
        'The README live-monitor images retain the original graded-mesh PIC runs.')
    model['domain'].update(nz='40', ny='64' if geometry == '2D-ZY' else '32')
    polygon = ('0,-wbar*d/2; d,-.17*d; d,.17*d; 0,wbar*d/2' if geometry == '2D-ZY'
               else '0,0; d,0; d,.24*d; 0,Rbar*d')
    lo = '-.12*d' if geometry == '2D-ZY' else '0'
    hi = '.12*d' if geometry == '2D-ZY' else '.17*d'
    model['mesh_refinements'] = [
        dict(name='assumed beam envelope',enabled=True,axis='local 2D',region_shape='polygon',
             points=polygon,size='.0125*d',buffer='.01*d',growth='1.2'),
        dict(name='cathode return region',enabled=True,axis='local 2D',region_shape='rectangle',
             z0='0',z1='.05*d',y0=lo,y1=hi,size='.005*d',buffer='.005*d',growth='1.2')]
    errors=[s for level,s in validate(model) if level=='error']
    if errors:raise ValueError(errors)
    return model


def mesh_metrics(model):
    mesh=grid_from_model(model,evaluator(model)).local_mesh
    leaf=mesh.leaves
    hz=mesh._physical(leaf[:,1],0)-mesh._physical(leaf[:,0],0)
    hy=mesh._physical(leaf[:,3],1)-mesh._physical(leaf[:,2],1)
    lengths=np.ptp(mesh.points,axis=0)
    nz,ny=np.ceil(lengths/np.array([hz.min(),hy.min()])-1e-10).astype(int)
    metrics=dict(nodes=len(mesh.points),triangles=len(mesh.triangles),leaves=len(leaf),
        minimum_leaf_dz_m=float(hz.min()),minimum_leaf_dy_m=float(hy.min()),
        uniform_finest_nz=int(nz),uniform_finest_ny=int(ny),uniform_finest_triangles=int(4*nz*ny),
        uniform_finest_nodes=int((nz+1)*(ny+1)+nz*ny),
        comparison='Same full domain and smallest leaf widths; four P1 center-fan triangles per uniform rectangle. Counts are not timings.')
    return mesh,metrics


def uniform_triangles(bounds,nz,ny):
    z=np.linspace(bounds[0,0],bounds[1,0],nz+1)
    y=np.linspace(bounds[0,1],bounds[1,1],ny+1)
    za,ya=np.meshgrid(z[:-1],y[:-1],indexing='ij')
    zb,yb=np.meshgrid(z[1:],y[1:],indexing='ij')
    a=np.c_[za.ravel(),ya.ravel()];b=np.c_[zb.ravel(),ya.ravel()]
    c=np.c_[zb.ravel(),yb.ravel()];d=np.c_[za.ravel(),yb.ravel()]
    center=(a+c)/2
    return np.concatenate([np.stack((center,p,q),axis=1) for p,q in ((a,b),(b,c),(c,d),(d,a))])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out',type=Path)
    args=parser.parse_args();out=args.out or args.root/'docs/images';out.mkdir(parents=True,exist_ok=True)
    models={g:demonstration(args.root,g) for g in ('2D-ZY','2D-RZ')}
    pairs={g:mesh_metrics(m) for g,m in models.items()}
    for geometry,m in models.items():
        (out/f'mesh_demo_{geometry[-2:].lower()}.pic.json').write_text(json.dumps(m,indent=2)+'\n')
    mesh,metrics=pairs['2D-ZY'];bounds=np.array([mesh.points.min(0),mesh.points.max(0)])
    aspect=np.ptp(bounds[:,0])/np.ptp(bounds[:,1]);budget=metrics['triangles']/4
    nz=max(1,round(math.sqrt(budget*aspect)));ny=max(1,round(budget/nz))
    uniform=uniform_triangles(bounds,nz,ny);uniform_area=np.prod(np.ptp(bounds,axis=0))/len(uniform)
    cmap=LinearSegmentedColormap.from_list('mesh_blue',['#f3f7fd','#c9ddf6','#6ca1e1','#1c5eb2'])
    norm=LogNorm(.5,32)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelcolor':'#52514e',
        'xtick.color':'#52514e','ytick.color':'#52514e','axes.edgecolor':'#52514e',
        'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    fig=plt.figure(figsize=(15,5.9),layout=None)
    axes=[fig.add_axes([.055,.22,.255,.65]),fig.add_axes([.37,.22,.255,.65]),fig.add_axes([.735,.22,.25,.65])]
    collection=None
    for i,(ax,triangles) in enumerate(zip(axes[:2],(mesh.points[mesh.triangles],uniform))):
        ratio=uniform_area/mesh.area if i==0 else np.ones(len(uniform))
        collection=PolyCollection(triangles*1000,array=ratio,cmap=cmap,norm=norm,
            edgecolors='white',linewidths=.22,rasterized=True)
        ax.add_collection(collection);ax.set(xlim=(0,3),ylim=(-1.5,1.5),xlabel='z (mm)')
        ax.set_aspect('equal');ax.spines['left'].set_color('black');ax.spines['left'].set_linewidth(1.7)
        ax.plot([0,0],[-.5,.5],color='#ef5350',lw=3,clip_on=False,zorder=10)
        ax.tick_params(direction='out',length=3)
    axes[0].set_ylabel('y (mm)')
    axes[0].annotate('emitting strip\nw = 1 mm',xy=(.025,.32),xytext=(.54,.98),fontsize=9,
        bbox=dict(boxstyle='round,pad=.25',fc='white',ec='none',alpha=.88),arrowprops=dict(arrowstyle='-',color='.35'))
    ev=evaluator(models['2D-ZY'])
    from picstudio.mesh.local import Region
    for spec,style,color in zip(models['2D-ZY']['mesh_refinements'],('--',':'),('#1a568a','#2f526b')):
        points=Region(spec,ev).points*1000;points=np.vstack((points,points[0]))
        axes[0].plot(*points.T,style,color=color,lw=1.2)
    axes[0].text(1.28,1.23,'polygon',color='#1a568a',fontsize=9)
    axes[0].text(.58,-1.33,'rectangle',color='#2f526b',fontsize=9)
    titles=['Local 2D mesh (VEXPACE)','Uniform mesh, similar element count','Element budget']
    subtitles=[f"{metrics['triangles']:,} triangles · polygon + rectangle",
        f'{len(uniform):,} triangles · {10000/nz:.0f} µm × {20000/ny:.0f} µm leaves',
        'Same domain and finest leaf widths']
    for i,ax in enumerate(axes):
        ax.text(-.17 if i<2 else -.31,1.10,'abc'[i],transform=ax.transAxes,fontsize=13,fontweight='bold')
        ax.text(0,1.09,titles[i],transform=ax.transAxes,fontsize=12,fontweight='bold')
        ax.text(0,1.027,subtitles[i],transform=ax.transAxes,fontsize=9,color='#62615c')
    ax=axes[2];labels=['Strip emitter\n(2D-ZY)','Disk emitter\n(2D-RZ)']
    uniform_counts=[pairs[g][1]['uniform_finest_triangles'] for g in models]
    local_counts=[pairs[g][1]['triangles'] for g in models]
    y=np.array([1.,0.]);height=.32
    ax.barh(y+.18,uniform_counts,height,color='#c6c5ba',label='uniform P1 mesh')
    ax.barh(y-.18,local_counts,height,color='#2a78d6',label='local 2D P1 mesh')
    maximum=max(uniform_counts)
    for i,(u,l) in enumerate(zip(uniform_counts,local_counts)):
        ax.text(u+maximum*.015,y[i]+.18,f'{u:,}',va='center',fontsize=9,color='#62615c')
        ax.text(l+maximum*.015,y[i]-.18,f'{l:,} ({u/l:.1f}× fewer)',va='center',fontsize=9,fontweight='bold')
    ax.set_yticks(y,labels);ax.tick_params(axis='y',length=0);ax.spines['left'].set_visible(False)
    ax.set_xlim(0,maximum*1.24);ax.set_ylim(-.55,1.58);ax.set_xlabel('P1 triangles over the full domain')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v/1000:g}k' if v else '0'))
    ax.grid(axis='x',color='#dededb',lw=.8);ax.set_axisbelow(True)
    ax.legend(loc='lower right',fontsize=9,frameon=False)
    cax=fig.add_axes([.20,.095,.28,.022]);cbar=fig.colorbar(collection,cax=cax,orientation='horizontal')
    cbar.set_ticks([.5,1,2,4,8,16,32],labels=['½×','1×','2×','4×','8×','16×','32×'])
    cbar.outline.set_visible(False);cbar.ax.tick_params(length=0,labelsize=9)
    cbar.minorticks_off()
    cbar.set_label('resolution relative to uniform mesh (triangle-area ratio)',fontsize=9)
    fig.savefig(out/'mesh_refinement.png',dpi=160)
    allmetrics={g:v[1] for g,v in pairs.items()}
    allmetrics['same_budget_uniform_triangles']=len(uniform)
    (out/'mesh_refinement_counts.json').write_text(json.dumps(allmetrics,indent=2)+'\n')
    print(json.dumps(allmetrics,indent=2))


if __name__=='__main__':main()
