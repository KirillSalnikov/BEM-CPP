import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from bounded_spectral_storage import StreamedSpectralBlock,half_factors
from symmetric_galerkin_fft import TorchSymmetricAction


class HalfSymbolTests(unittest.TestCase):
    def test_resident_half_symbol_across_batches(self):
        rng=np.random.default_rng(77)
        symbol=np.asarray(rng.normal(size=(519,7,5))+1j*rng.normal(size=(519,7,5)),np.complex64)
        symbol[4]=0;symbol[5]*=1e10;symbol[6]*=1e-10
        x=np.asarray(rng.normal(size=(519,5,3))+1j*rng.normal(size=(519,5,3)),np.complex64)
        with tempfile.TemporaryDirectory() as tmp:
            encoded,scale=half_factors(symbol,tmp,'symbol')
            decoded=(encoded[...,0].astype(np.float32)+1j*encoded[...,1].astype(np.float32))*scale
            group=dict(symbol_half=torch.from_numpy(encoded),symbol_scale=torch.from_numpy(scale))
            actual=TorchSymmetricAction.symbol_product(group,torch.from_numpy(x),7).numpy()
            expected=decoded@x
            np.testing.assert_allclose(actual,expected,rtol=3e-6,atol=1e-5)
            self.assertLess(np.linalg.norm(decoded-symbol)/np.linalg.norm(symbol),5e-4)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required for asynchronous storage')
class BoundedStorageTests(unittest.TestCase):
    def test_tail_batch_repeat_and_changed_inputs(self):
        rng=np.random.default_rng(710)
        complex_array=lambda shape:np.asarray(rng.normal(size=shape)+1j*rng.normal(size=shape),np.complex64)
        nf,nr,nc,rank,count=35,37,29,7,3
        left,right=complex_array((nf,nr,rank)),complex_array((nf,rank,nc))
        left[2]=0;right[6]=0
        with tempfile.TemporaryDirectory() as tmp:
            u,us=half_factors(left,tmp,'left');v,vs=half_factors(right,tmp,'right')
            op=StreamedSpectralBlock((u,v,us*vs),device='cuda',batch=8,half=True)
            decoded=lambda a:np.asarray(a[...,0],np.float32)+1j*np.asarray(a[...,1],np.float32)
            ul,vr=decoded(u),decoded(v)
            symbol=complex_array((nf,nr,nc))
            full=StreamedSpectralBlock((symbol,),device='cuda',batch=8)
            half_symbol,scale=half_factors(symbol,tmp,'symbol')
            dense_half=StreamedSpectralBlock((half_symbol,scale),device='cuda',batch=8,half_matrix=True)
            for _ in range(4):
                x=complex_array((nf,nc,count));gpu=torch.as_tensor(x,device='cuda')
                for action,expected in ((op,(ul@(vr@x))*(us*vs)),(full,symbol@x),
                                        (dense_half,(decoded(half_symbol)@x)*scale)):
                    actual=action(gpu,nr).cpu().numpy()
                    self.assertLess(np.linalg.norm(actual-expected)/np.linalg.norm(expected),3e-7)
            self.assertEqual(op.bytes,op.host_bytes)
            self.assertLess(op.bytes,sum(a.nbytes for a in (u,v,us*vs)))

if __name__=='__main__':unittest.main()
