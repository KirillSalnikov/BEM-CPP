import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from scipy import sparse
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from packed_current_csr import prepare,PackedCurrentCSR

class PackedCurrentTests(unittest.TestCase):
    def test_complex_material_duplicates_zero_and_two_currents(self):
        rng=np.random.default_rng(78);n=37;k=13.;ri=1.31+.2j
        rows=np.arange(n+1,dtype=np.int32)*n;cols=np.tile(np.arange(n,dtype=np.int32),n)
        values=rng.normal(size=(n*n,3))+1j*rng.normal(size=(n*n,3))
        # Duplicated mass entries must sum before applying either coefficient.
        mr=np.repeat(np.arange(n,dtype=np.int32),2);mc=mr.copy();mv=rng.normal(size=len(mr))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'packed'
            (root/'geometry.json').write_text(json.dumps(dict(current_dofs=n,ka=k,ri=[ri.real,ri.imag])))
            for name,array in [('near_rows',rows),('near_cols',cols),('near_values',values),
                               ('mass_rows',mr),('mass_cols',mc),('mass_values',mv)]:array.tofile(root/(name+'.bin'))
            meta=prepare(root,out,chunk_rows=8)
            a,b,c=[sparse.csr_matrix((values[:,j],cols,rows),shape=(n,n)) for j in range(3)]
            mass=sparse.coo_matrix((mv,(mr,mc)),shape=(n,n)).tocsr()
            reference=sparse.bmat([[1j/k*a,b+.5*(ri*ri+1)*mass],[c+mass,-1j/k*a]],format='csr')
            devices=['cpu']+(['cuda'] if torch.cuda.is_available() else [])
            for device in devices:
                op=PackedCurrentCSR(out,device)
                for vector in [np.zeros(2*n),rng.normal(size=2*n)+1j*rng.normal(size=2*n)]:
                    x=np.asarray(vector,np.complex64);got=op(torch.as_tensor(x,device=device)).cpu().numpy()
                    expected=reference@x
                    self.assertLess(np.linalg.norm(got-expected)/max(np.linalg.norm(expected),1.),3e-7)
                self.assertEqual(op.bytes,meta['device_bytes'])
                if device=='cuda':
                    from packed_current_host import PackedHostCurrentCSR
                    host=PackedHostCurrentCSR(out,device,threads=2)
                    for _ in range(4):
                        x=np.asarray(rng.normal(size=2*n)+1j*rng.normal(size=2*n),np.complex64)
                        host.begin(torch.as_tensor(x,device=device));actual=host.finish().cpu().numpy()
                        expected=reference@x
                        self.assertLess(np.linalg.norm(actual-expected)/np.linalg.norm(expected),3e-7)
                    host.pool.shutdown()

if __name__=='__main__':unittest.main()
