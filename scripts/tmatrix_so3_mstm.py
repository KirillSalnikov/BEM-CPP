#!/usr/bin/env python3
"""Full analytic SO(3) Mueller average of a BEM T matrix.

Uses Mackowski's generalized-spherical-function contractions, not Euler
quadrature. All 16 Mueller entries are retained, including optical activity.
Input convention is the regular/outgoing-wave T from build_fmm_tmatrix.py.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import numpy as np
from tmatrix_reference import modes

ROOT=Path(__file__).resolve().parents[1]


def analytic_mueller_tmatrix(t, order, theta_degrees, *, precision='auto', threads=4,
                            cyclic_order=1, reciprocal_fast_tol=1e-12, symmetry_tol=1e-3):
    start=time.monotonic()
    selected=modes(order);q=len(selected);half=q//2
    if order<1 or order>256 or t.shape!=(q,q) or threads<1 or cyclic_order<1:
        raise ValueError('invalid T dimensions, multipole order, threads, or symmetry')
    if precision not in ('auto','fp32','fp64'):raise ValueError('invalid T storage precision')
    precision=('fp64' if order<=32 else 'fp32') if precision=='auto' else precision
    dtype=np.complex128 if precision=='fp64' else np.complex64
    binary=Path(os.environ.get('BEM_SO3_BINARY_'+precision.upper(),str(ROOT/'bin'/('tmatrix_so3_mstm' if precision=='fp64' else 'tmatrix_so3_mstm_fp32'))))
    if not binary.exists(): raise RuntimeError(f'build {binary.name} before averaging')
    # Complete the two entries for which the original backend uses reciprocity
    # with a second contraction of the reciprocal partner. Do not symmetrize T.
    lookup={tuple(mode):i for i,mode in enumerate(selected)}
    flipped=np.array([lookup[p,l,-m] for p,l,m in selected])
    signs=np.array([(-1.)**m for p,l,m in selected])
    azimuth=np.array([m for p,l,m in selected])
    power=defect=forbidden=0.
    disk_bytes=q*q*np.dtype(dtype).itemsize
    temporary_root=os.environ.get('BEM_SO3_TMPDIR',str(ROOT/'runs'))
    Path(temporary_root).mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(temporary_root).free<disk_bytes+2**30:
        raise RuntimeError('insufficient temporary storage for analytic T contractions')
    # Shared T + shared and thread-local contraction tensors; budget before
    # starting the large process rather than allocating until the OS fails.
    dm_bytes=(2*order+3)*3*order*2*order*2*16
    estimated_ram=disk_bytes+2*dm_bytes*(threads+1)
    import psutil
    if estimated_ram>psutil.virtual_memory().available-4*2**30:
        raise RuntimeError(f'analytic contractions need approximately {estimated_ram/2**30:.1f} GiB RAM; reduce threads/storage precision')
    with tempfile.TemporaryDirectory(prefix='bem-so3-',dir=temporary_root) as directory:
        directory=Path(directory)
        # MSTM's p=1 is our electric N and p=2 our magnetic M. Its mode
        # address n*(n+1)+m is the same radial/azimuthal ordering, one-based.
        def export(partner=False):
            nonlocal power,defect,forbidden
            with (directory/'t.bin').open('wb') as stream:
                for radial in range(half):
                    for polarization in (1,0):
                        column=polarization*half+radial
                        values=np.array(t[:,column],dtype=np.complex128)
                        reciprocal=np.asarray(t[flipped[column],flipped])*signs[column]*signs
                        if not partner:
                            power+=float(np.vdot(values,values).real)
                            difference=values-reciprocal
                            defect+=float(np.vdot(difference,difference).real)
                        else:
                            values=reciprocal
                        if cyclic_order>1:
                            mask=(azimuth-azimuth[column])%cyclic_order!=0
                            if not partner: forbidden+=float(np.vdot(values[mask],values[mask]).real)
                            values[mask]=0
                        packed=np.stack((values[half:],values[:half]),axis=1)
                        packed.astype(dtype).tofile(stream)
        export()
        reciprocity_error=float(np.sqrt(defect/max(power,1e-300)))
        symmetry_error=float(np.sqrt(forbidden/max(power,1e-300)))
        if not np.isfinite(reciprocity_error):raise RuntimeError('nonfinite T matrix')
        if not np.isfinite(symmetry_error) or symmetry_error>symmetry_tol:
            raise RuntimeError(f'T cyclic-symmetry defect {symmetry_error:.3e} exceeds {symmetry_tol:.3e}')
        theta=np.radians(np.asarray(theta_degrees,dtype=float))
        theta.tofile(directory/'theta.bin')
        env=os.environ.copy();env['OMP_NUM_THREADS']=str(threads)
        command=[str(binary),str(order),str(len(theta)),str(directory/'t.bin'),
                 str(directory/'theta.bin'),str(directory/'mueller.bin'),str(cyclic_order)]
        contractions_seconds=0.
        def contract():
            nonlocal contractions_seconds
            if isinstance(t,np.memmap):
                import mmap
                t._mmap.madvise(mmap.MADV_DONTNEED)
            begin=time.monotonic()
            process=subprocess.run(command,env=env,text=True,capture_output=True)
            contractions_seconds+=time.monotonic()-begin
            if process.returncode: raise RuntimeError(f'analytic SO3 backend failed: {process.stdout}\n{process.stderr}')
            values=np.fromfile(directory/'mueller.bin',dtype=np.float64)
            if values.size!=16*len(theta):raise RuntimeError('incomplete analytic Mueller output')
            # MSTM uses a factor two in its dimensionless scattering matrix.
            return .5*values.reshape(4,4,len(theta),order='F').transpose(2,0,1)
        mean=contract()
        passes=1
        if reciprocity_error>reciprocal_fast_tol:
            export(partner=True)
            other=contract()
            mean[:,0,1]=other[:,1,0]
            mean[:,3,2]=-other[:,2,3]
            passes=2
        if not np.isfinite(mean).all(): raise RuntimeError('nonfinite analytic Mueller output')
        metadata={'method':'analytic_full_SO3_Mackowski_GSF','Euler_quadrature_nodes':0,
                  'all_16_mueller_entries':True,'order':order,'mode_count':q,
                  'reciprocity_relative_defect':reciprocity_error,'cyclic_relative_defect':symmetry_error,
                  'reciprocity_projection_applied':False,'reciprocal_partner_passes':passes,'cyclic_order':cyclic_order,
                  'T_storage_precision':precision,'contraction_precision':'fp64','threads':threads,
                  'estimated_peak_ram_gib':estimated_ram/2**30,'contractions_s':contractions_seconds,
                  'end_to_end_s':time.monotonic()-start,
                  'source':'https://github.com/dmckwski/MSTM/tree/a0c982121cf9ac352531f4816639a07d814385bd'}
        return mean,metadata


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tmatrix',type=Path,required=True)
    parser.add_argument('--order',type=int,required=True)
    parser.add_argument('--ntheta',type=int,default=181)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--precision',choices=['auto','fp32','fp64'],default='auto')
    parser.add_argument('--cyclic-order',type=int,default=1)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    t=np.load(args.tmatrix,mmap_mode='r');theta=np.linspace(0,180,args.ntheta)
    mean,metadata=analytic_mueller_tmatrix(t,args.order,theta,precision=args.precision,
                                         threads=args.threads,cyclic_order=args.cyclic_order)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps({**metadata,'theta':theta.tolist(),
                                  'mueller':np.moveaxis(mean,0,-1).tolist()},indent=2)+'\n')
    print(json.dumps(metadata,indent=2))
