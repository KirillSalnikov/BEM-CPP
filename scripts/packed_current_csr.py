"""Shared-index near/mass operator [[A,B],[C,-A]] for the two surface currents.

Preparation is chunked and uses the original complex128 correction and mass.
The GPU action stores three complex64 coefficients and one column per entry.
"""
import json
from pathlib import Path
import numpy as np
from scipy import sparse


def prepare(directory, output, *, chunk_rows=8192):
    directory, output=Path(directory),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    meta=json.loads((directory/'geometry.json').read_text());n=meta['current_dofs']
    rows=np.memmap(directory/'near_rows.bin',dtype=np.int32,mode='r')
    cols=np.memmap(directory/'near_cols.bin',dtype=np.int32,mode='r')
    values=np.memmap(directory/'near_values.bin',dtype=np.complex128,mode='r').reshape(-1,3)
    if len(rows)!=n+1 or rows[0]!=0 or rows[-1]!=len(cols) or len(values)!=len(cols):
        raise ValueError('Invalid native correction dimensions')
    if np.any(np.diff(rows)<0):raise ValueError('Nonmonotone row offsets')
    mass=sparse.coo_matrix((np.memmap(directory/'mass_values.bin',dtype=np.float64,mode='r'),
        (np.memmap(directory/'mass_rows.bin',dtype=np.int32,mode='r'),
         np.memmap(directory/'mass_cols.bin',dtype=np.int32,mode='r'))),shape=(n,n)).tocsr()
    mass.sum_duplicates();mass.sort_indices()
    target=np.lib.format.open_memmap(output/'coefficients.npy',mode='w+',dtype=np.complex64,shape=(len(cols),3))
    ri=complex(*meta['ri']);k=meta['ka']
    for first in range(0,n,chunk_rows):
        last=min(n,first+chunk_rows);a,b=int(rows[first]),int(rows[last])
        row=np.repeat(np.arange(first,last,dtype=np.int64),np.diff(rows[first:last+1]))
        keys=(row<<32)+cols[a:b]
        if np.any(np.diff(keys)<=0):raise ValueError('Native near rows must have unique sorted columns')
        ma,mb=mass.indptr[first],mass.indptr[last]
        mass_row=np.repeat(np.arange(first,last,dtype=np.int64),np.diff(mass.indptr[first:last+1]))
        mass_keys=(mass_row<<32)+mass.indices[ma:mb];positions=np.searchsorted(keys,mass_keys)
        if np.any(positions>=len(keys)) or not np.array_equal(keys[positions],mass_keys):
            raise ValueError('Mass support is not contained in the correction pattern')
        data=np.array(values[a:b],copy=True);data[:,0]*=1j/k
        data[positions,1]+=.5*(ri*ri+1)*mass.data[ma:mb]
        data[positions,2]+=mass.data[ma:mb]
        target[a:b]=data
    target.flush();del target
    np.save(output/'rows.npy',rows);np.save(output/'columns.npy',cols)
    report=dict(format=1,current_dofs=n,system_dofs=2*n,nnz=len(cols),
        max_row_entries=int(np.max(np.diff(rows))),dtype='complex64',
        device_bytes=int((n+1)*4+len(cols)*28),source_geometry=meta)
    (output/'metadata.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


class PackedCurrentCSR:
    def __init__(self,directory,device='cuda'):
        import torch
        self.meta=json.loads((Path(directory)/'metadata.json').read_text())
        self.n=self.meta['current_dofs'];self.device=device
        # Clone CPU mappings on upload; source files remain read-only thereafter.
        self.rows=torch.as_tensor(np.load(Path(directory)/'rows.npy',mmap_mode='c'),device=device)
        self.columns=torch.as_tensor(np.load(Path(directory)/'columns.npy',mmap_mode='c'),device=device)
        self.values=torch.as_tensor(np.load(Path(directory)/'coefficients.npy',mmap_mode='c'),device=device)
        self.bytes=sum(v.numel()*v.element_size() for v in (self.rows,self.columns,self.values))
        if self.bytes!=self.meta['device_bytes']:raise ValueError('Packed CSR byte count mismatch')

    def __call__(self,x):
        import torch
        if x.shape!=(2*self.n,) or x.dtype!=torch.complex64 or not x.is_contiguous():
            raise ValueError('Packed CSR requires one contiguous complex64 pair of currents')
        if x.device!=self.values.device:raise ValueError('Packed CSR input is on another device')
        if x.is_cuda:
            from packed_current_csr_kernel import action
            return action(self,x)
        a,b,c=[torch.sparse_csr_tensor(self.rows,self.columns,self.values[:,j],size=(self.n,self.n)) for j in range(3)]
        return torch.cat((a@x[:self.n]+b@x[self.n:],c@x[:self.n]-a@x[self.n:]))
