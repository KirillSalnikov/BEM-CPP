"""Assemble Fortran-order scattering matrices and holdout fields with bounded RAM."""
from pathlib import Path
import numpy as np


def assemble(out, columns, phase, ka, check_basis, observed, block_columns=32):
    n=len(phase)
    target=Path(out)
    names=('kernel.npy','tmatrix.npy')
    paths=[target/(name+'.tmp') for name in names]
    header={'descr':np.lib.format.dtype_to_descr(np.dtype(np.complex128)),
            'fortran_order':True,'shape':(n,n)}
    predictions=np.zeros((len(check_basis),2*len(observed)),np.complex128)
    power=extinction=0.
    try:
        with paths[0].open('wb') as kernel, paths[1].open('wb') as tmatrix:
            for stream in (kernel,tmatrix):
                np.lib.format.write_array_header_2_0(stream,header)
            for first in range(0,n,block_columns):
                stop=min(n,first+block_columns)
                block=np.empty((n,stop-first),dtype=np.complex128,order='F')
                for j in range(first,stop):
                    value=np.load(columns/f'{j:06d}.npy',allow_pickle=False)
                    if value.shape!=(n,):raise ValueError('invalid T checkpoint shape')
                    block[:,j-first]=value
                block.T.tofile(kernel)
                predictions+=(check_basis@block)@check_basis[:predictions.shape[1],first:stop].conj().T
                for offset,j in enumerate(range(first,stop)):
                    block[:,offset]=(-1j*ka/(4*np.pi))*phase*block[:,offset]*phase[j].conj()
                    power+=np.vdot(block[:,offset],block[:,offset]).real
                    extinction-=block[j,offset].real
                block.T.tofile(tmatrix)
        for path,name in zip(paths,names):path.replace(target/name)
    finally:
        for path in paths:path.unlink(missing_ok=True)
    norm=sum(np.linalg.norm(field)**2 for field in observed)
    error=sum(np.linalg.norm(predictions[:,2*i:2*i+2]-field)**2 for i,field in enumerate(observed))
    return float(np.sqrt(error/norm)),float(power),float(extinction)
