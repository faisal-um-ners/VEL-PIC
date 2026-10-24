"""Reproduce the README full-domain / detail / element-budget figure from actual meshes.

Run with tfgpu Python: python docs/make_mesh_refinement.py
The saved models describe mesh demonstrations, not new PIC limit measurements.
"""
from pathlib import Path
import argparse
import json
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
    mean_area=np.prod(np.ptp(bounds,axis=0))/metrics['triangles']
    ratio=mean_area/mesh.area
    cmap=LinearSegmentedColormap.from_list('mesh_blue',['#f3f7fd','#c9ddf6','#6ca1e1','#1c5eb2'])
    norm=LogNorm(.5,32,clip=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelcolor':'#52514e',
        'xtick.color':'#52514e','ytick.color':'#52514e','axes.edgecolor':'#52514e',
        'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    fig=plt.figure(figsize=(15,6.25),layout=None)
    axes=[fig.add_axes([x,.25,.25,.60]) for x in (.055,.37,.735)]
    collection=None
    ev=evaluator(models['2D-ZY'])
    from picstudio.mesh.local import Region
    from matplotlib.patches import Rectangle
    for i,ax in enumerate(axes[:2]):
        if i==0:
            # Show every edge, but lighten dense fine cells so coarse cells and
            # the zoom frame remain readable at full-domain scale.
            edges=np.tile([.36,.47,.61,1.],(len(ratio),1))
            edges[:,3]=np.where(ratio<1.5,.82,.12)
            linewidths=np.where(ratio<1.5,.29,.10)
        else:
            edges='white';linewidths=.22
        collection=PolyCollection(mesh.points[mesh.triangles]*1000,array=ratio,cmap=cmap,norm=norm,
            edgecolors=edges,linewidths=linewidths,rasterized=True)
        ax.add_collection(collection)
        if i==0:
            ax.set(xlim=(0,10),ylim=(-10,10),xticks=[0,5,10],yticks=[-10,-5,0,5,10])
            # Show both physical electrodes and the periodic side boundaries.
            for z in (0,10):ax.plot([z,z],[-10,10],color='black',lw=1.8,zorder=7)
            for y in (-10,10):ax.plot([0,10],[y,y],color='#4d846c',ls='-.',lw=1.4,zorder=7)
            ax.add_patch(Rectangle((0,-1.5),3,3,fill=False,ec='#383838',ls=(0,(4,3)),lw=1.5,zorder=8))
            ax.text(3.3,1.55,'b',fontsize=10,fontweight='bold',color='#383838',bbox=dict(fc='white',ec='none',alpha=.85,pad=1))
            ax.text(.38,6.5,'cathode',rotation=90,fontsize=8,color='#343434',bbox=dict(fc='white',ec='none',alpha=.85,pad=1))
            ax.text(9.5,6.5,'anode',rotation=90,fontsize=8,color='#343434',ha='right',bbox=dict(fc='white',ec='none',alpha=.85,pad=1))
        else:
            ax.set(xlim=(0,3),ylim=(-1.5,1.5))
            ax.spines['left'].set_color('black');ax.spines['left'].set_linewidth(1.7)
        ax.set(xlabel='z (mm)',ylabel='y (mm)');ax.set_aspect('auto')
        ax.plot([0,0],[-.5,.5],color='#ef5350',lw=3,clip_on=False,zorder=10)
        ax.tick_params(direction='out',length=3)
        for spec,style,color in zip(models['2D-ZY']['mesh_refinements'],('--',':'),('#1a568a','#2f526b')):
            points=Region(spec,ev).points*1000;points=np.vstack((points,points[0]))
            ax.plot(*points.T,style,color=color,lw=1 if i==0 else 1.2,zorder=6)
    axes[1].annotate('emitting strip\nw = 1 mm',xy=(.025,.32),xytext=(.54,.98),fontsize=9,
        bbox=dict(boxstyle='round,pad=.25',fc='white',ec='none',alpha=.88),arrowprops=dict(arrowstyle='-',color='.35'))
    axes[1].text(1.28,1.23,'polygon',color='#1a568a',fontsize=9)
    axes[1].text(.58,-1.33,'rectangle',color='#2f526b',fontsize=9,
        bbox=dict(boxstyle='round,pad=.15',fc='white',ec='none',alpha=.78))
    titles=['Full-domain local mesh','Near-emitter detail','Element budget']
    subtitles=[f"{metrics['triangles']:,} triangles · {metrics['nodes']:,} nodes",
        'Same mesh · polygon + rectangle','Same domain and finest leaf widths']
    # Keep panel headers aligned even though panel a has a taller physical aspect.
    for i,x in enumerate((.055,.37,.735)):
        fig.text(x-.04,.942,'abc'[i],fontsize=13,fontweight='bold')
        fig.text(x,.94,titles[i],fontsize=12,fontweight='bold')
        fig.text(x,.902,subtitles[i],fontsize=9,color='#62615c')
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
    cax=fig.add_axes([.20,.105,.28,.022]);cbar=fig.colorbar(collection,cax=cax,orientation='horizontal')
    cbar.set_ticks([.5,1,2,4,8,16,32],labels=['≤½×','1×','2×','4×','8×','16×','≥32×'])
    cbar.outline.set_visible(False);cbar.ax.tick_params(length=0,labelsize=9)
    cbar.minorticks_off()
    cbar.set_label('relative resolution (mean triangle area / local triangle area)',fontsize=9)
    fig.savefig(out/'mesh_refinement.png',dpi=160)
    allmetrics={g:v[1] for g,v in pairs.items()}
    allmetrics['figure_panels']={'a':'Full strip domain: z = 0..10 mm, y = -10..10 mm; both electrodes and periodic sides',
        'b':'Same mesh near the emitter: z = 0..3 mm, y = -1.5..1.5 mm',
        'c':'Local / uniform P1 triangle counts over each full domain, with matching finest leaf widths'}
    (out/'mesh_refinement_counts.json').write_text(json.dumps(allmetrics,indent=2)+'\n')
    print(json.dumps(allmetrics,indent=2))


if __name__=='__main__':main()
