#!/usr/bin/env python3
"""CPU dense preview and shared fixed-orientation comparison helpers.

The dense preview is not an analytically averaged T result or a certified mesh.
Geometry and polarization frames are matched directly to native MBS phi=0.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from bounded_bem_run import atomic_json

ROOT = Path(__file__).resolve().parents[1]
BETAS = np.array([0., 45., 75., 85.])
THETA = np.linspace(0, 180, 721)


def frames():
    t = np.radians(THETA)
    outgoing = np.stack([np.sin(t), np.zeros_like(t), -np.cos(t)], axis=-1)
    parallel = np.stack([np.cos(t), np.zeros_like(t), np.sin(t)], axis=-1)
    incoming, directions, projectors = [], [], []
    for beta in np.radians(BETAS):
        c, s = np.cos(beta), np.sin(beta)
        rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        # BEM base vertices have azimuth pi/6; MBS uses zero. Native
        # gamma=30 degrees cancels this difference, so total R is Ry(beta).
        incoming.append(np.array([[0., 0., -1.], [1., 0., 0.], [0., 1., 0.]]) @ rotation)
        directions.append(outgoing @ rotation)
        projectors.append(np.stack([parallel @ rotation,
            np.broadcast_to(np.array([0., 1., 0.]) @ rotation, parallel.shape)], axis=1))
    return np.array(incoming), np.array(directions), np.array(projectors)


def native_mueller(jones):
    """Native MBS Stokes convention (src/common/MullerMatrix.cpp)."""
    a, b, c, d = [jones[..., i, j] for i, j in [(0,0),(0,1),(1,0),(1,1)]]
    aa, bb, cc, dd = [abs(x)**2 for x in [a,b,c,d]]
    m = np.zeros(jones.shape[:-2]+(4,4))
    m[...,0,0]=.5*(aa+bb+cc+dd);m[...,0,1]=.5*(aa-bb+cc-dd)
    m[...,1,0]=.5*(aa+bb-cc-dd);m[...,1,1]=.5*(aa-bb-cc+dd)
    x,y=a*b.conj(),d*c.conj()
    m[...,0,2]=-x.real-y.real;m[...,0,3]=y.imag-x.imag
    m[...,1,2]=y.real-x.real;m[...,1,3]=-x.imag-y.imag
    x,y=a*c.conj(),d*b.conj()
    m[...,2,0]=-x.real-y.real;m[...,2,1]=y.real-x.real
    m[...,3,0]=x.imag-y.imag;m[...,3,1]=y.imag+x.imag
    x,y=a*d.conj(),b*c.conj()
    m[...,2,2]=x.real+y.real;m[...,2,3]=x.imag-y.imag
    m[...,3,2]=-x.imag-y.imag;m[...,3,3]=x.real-y.real
    return m


def far_to_mueller(far, plan):
    _, _, projectors = frames()
    jones = np.einsum('bptc,btsc->btsp', far, projectors)
    radius = plan['ka']*plan['wavelength']/(2*np.pi)
    return native_mueller(radius*jones), radius*jones


def difference(candidate, reference):
    denominator = np.maximum(reference[...,0,0], np.max(reference[...,0,0], axis=-1, keepdims=True)*1e-12)
    error = abs(candidate-reference)/denominator[...,None,None]
    return dict(max_all16_over_reference_M11=float(error.max()),
        per_orientation=[float(e.max()) for e in error],
        relative_L2=float(np.linalg.norm(candidate-reference)/np.linalg.norm(reference)))


def mbs_fixed(out, plan):
    means = {}
    for label, key in [('original','mbs_binary'),('corrected','mbs_corrected_binary')]:
        arrays=[]
        for beta in BETAS:
            case=out/'mbs_fixed'/label/f'beta{beta:g}'
            case.mkdir(parents=True,exist_ok=True)
            output=case/'result/result.dat'
            if not output.exists():
                command=[plan[key],'--method','po','--backend','cpu','--fixed-orientation',str(beta),'30',
                    '--particle','1',str(plan['particle_L_over_D_vertex']),'1','--k-eq',str(plan['ka']),
                    '--refractive-index',str(plan['ri']),str(plan['ri_imag']),
                    '--wavelength-um',str(plan['wavelength']),'--max-reflections','6',
                    '--scattering-grid','0','180','1','720','--threads','2','--output',str(case/'result'),'--close']
                atomic_json(case/'command.json',dict(command=command,scope='Fixed orientation; depth6; no fullauto certificate'))
                with (case/'worker.log').open('a') as log:
                    subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,
                        env=dict(os.environ,OMP_NUM_THREADS='2'))
            table=np.loadtxt(output,skiprows=1)
            if table.shape!=(len(THETA),19) or not np.isfinite(table).all():raise ValueError('invalid MBS fixed output')
            np.testing.assert_allclose(table[:,0],THETA,rtol=0,atol=1e-12)
            arrays.append(table[:,3:].reshape(-1,4,4))
        means[label]=np.array(arrays)
    return means


def plot(out, bem, mbs, label, certified=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out.mkdir(parents=True,exist_ok=True)
    report=dict(BEM_label=label,mesh_accuracy_certified=certified,
        convention='MBS native Stokes, same lab frames, dimensional um^2/sr; no fitted scale',
        scope='Fixed orientations, not orientation average',
        original=difference(mbs['original'],bem),corrected=difference(mbs['corrected'],bem))
    atomic_json(out/'comparison.json',report)
    np.savez_compressed(out/'comparison.npz',theta_deg=THETA,beta_deg=BETAS,bem=bem,**mbs)
    for b,beta in enumerate(BETAS):
        fig,axes=plt.subplots(4,4,figsize=(15,11),layout='constrained')
        for i in range(4):
            for j in range(4):
                ax=axes[i,j]
                for values,color,style,name in [(bem,'#1765a3','-',label),
                    (mbs['original'],'#d75c22','--','MBS original'),
                    (mbs['corrected'],'#21864b',':','MBS corrected external reflection')]:
                    y=values[b,:,i,j]
                    if i==0 and j==0:ax.semilogy(THETA,np.maximum(y,1e-300),style,color=color,label=name)
                    else:ax.plot(THETA,y/bem[b,:,0,0],style,color=color,label=name)
                ax.axvline(180-2*beta,color='.5',lw=.7,alpha=.5)
                ax.set_title(f'M{i+1}{j+1}' + (' / BEM M11' if (i,j)!=(0,0) else ' [um^2/sr]'))
                ax.set_xlim(0,180);ax.grid(alpha=.2)
                if i==3:ax.set_xlabel('Scattering angle, degrees')
        axes[0,0].legend(fontsize=7)
        fig.suptitle(f'ka=20, m=1.3116+10i, lambda=0.532 um, H/width=10/7\n'
            f'Fixed beta={beta:g}, gamma=30 deg. '+('Mesh control passed' if certified else 'PRELIMINARY: mesh convergence not established'))
        for ext in ['png','pdf']:fig.savefig(out/f'all16_beta{beta:g}.{ext}',dpi=150)
        plt.close(fig)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--binary',type=Path,required=True)
    args=parser.parse_args();out=args.out.resolve();plan=json.loads((out/'plan.json').read_text())
    for key in ['mbs_binary','mbs_corrected_binary']:
        if hashlib.sha256(Path(plan[key]).read_bytes()).hexdigest()!=plan[key+'_sha256']:
            raise RuntimeError(f'{key} changed since preparation')
    incoming,directions,_=frames()
    incoming.tofile(out/'incoming.bin');directions.tofile(out/'directions.bin')
    mbs=mbs_fixed(out,plan)
    previous=None
    for ref,quad,duffy in [(1,7,8),(2,7,8),(2,13,12)]:
        case=out/'cpu_preview'/f'ref{ref}_q{quad}_d{duffy}';case.mkdir(parents=True,exist_ok=True)
        prefix=case/'system'
        atomic_json(out/'cpu_status.json',dict(state='running',pid=os.getpid(),phase='dense_assembly',ref=ref,quad=quad,duffy=duffy,updated_unix=time.time()))
        command=[str(args.binary.resolve()),'assemble',str(ref),str(plan['ka']),str(plan['ri']),str(plan['ri_imag']),
            str(plan['prism_aspect']),str(quad),str(duffy),str(out/'incoming.bin'),str(prefix)]
        if not prefix.with_suffix('.json').exists():
            atomic_json(case/'command.json',dict(command=command))
            with (case/'worker.log').open('a') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        meta=json.loads(prefix.with_suffix('.json').read_text());n=meta['dofs'];cols=meta['rhs_count']
        matrix=np.memmap(str(prefix)+'.A.bin',dtype=np.complex128,mode='r',shape=(n,n))
        rhs=np.fromfile(str(prefix)+'.B.bin',dtype=np.complex128).reshape(n,cols)
        atomic_json(out/'cpu_status.json',dict(state='running',pid=os.getpid(),phase='dense_LU',ref=ref,updated_unix=time.time()))
        start=time.monotonic()
        if not Path(str(prefix)+'.X.bin').exists():
            factors,pivots=lu_factor(np.array(matrix,order='F'),overwrite_a=True,check_finite=True)
            solution=lu_solve((factors,pivots),rhs,check_finite=True)
            solution.tofile(str(prefix)+'.X.bin');del factors,pivots
        else:solution=np.fromfile(str(prefix)+'.X.bin',dtype=np.complex128).reshape(n,cols)
        residual=np.linalg.norm(matrix@solution-rhs,axis=0)/np.linalg.norm(rhs,axis=0)
        if not np.isfinite(solution).all() or residual.max()>2e-6:raise RuntimeError('CPU BEM residual failed')
        atomic_json(case/'solve.json',dict(residuals=residual.tolist(),max_residual=float(residual.max()),solve_s=time.monotonic()-start))
        del matrix,solution
        farcommand=command.copy();farcommand[1]='far';farcommand[-2]=str(out/'directions.bin')
        with (case/'worker.log').open('a') as log:subprocess.run(farcommand,stdout=log,stderr=subprocess.STDOUT,check=True)
        far=np.fromfile(str(prefix)+'.far.bin',dtype=np.complex128).reshape(len(BETAS),2,len(THETA),3)
        bem,jones=far_to_mueller(far,plan)
        if not np.isfinite(bem).all() or np.any(bem[:,:,0,0]<=0):raise RuntimeError('invalid BEM far field')
        np.savez_compressed(case/'fields.npz',far=far,jones_um=jones,mueller_um2_per_sr=bem)
        report=plot(case/'comparison',bem,mbs,f'BEM CPU ref{ref}, q{quad}/d{duffy}')
        if previous is not None:atomic_json(case/'convergence.json',difference(bem,previous))
        previous=bem
        atomic_json(out/'cpu_latest.json',dict(case=str(case),ref=ref,quad=quad,duffy=duffy,comparison=report,mesh_accuracy_certified=False))
    atomic_json(out/'cpu_status.json',dict(state='complete_preview',pid=os.getpid(),mesh_accuracy_certified=False,
        note='Dense ref<=2 preview only; fine GPU mesh and analytic T average remain pending.'))


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        import sys
        if '--out' in sys.argv:
            destination=Path(sys.argv[sys.argv.index('--out')+1])
            atomic_json(destination/'cpu_status.json',dict(state='failed',pid=os.getpid(),
                error=repr(error),updated_unix=time.time()))
        raise
