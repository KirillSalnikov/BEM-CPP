import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from recompress_spectral_factors import recompress_factors


class RecompressionTests(unittest.TestCase):
    def test_dense_error_variable_rank_padding_zero_and_cache(self):
        rng=np.random.default_rng(702);nf,nr,nc,rank=3,37,31,12
        complex_array=lambda shape:rng.normal(size=shape)+1j*rng.normal(size=shape)
        left=[];right=[]
        for frequency in range(nf):
            u=np.linalg.qr(complex_array((nr,rank)))[0]
            v=np.linalg.qr(complex_array((nc,rank)))[0]
            mixing=np.eye(rank)+.03*complex_array((rank,rank))
            spectrum=np.geomspace(1,1e-10,rank)
            if frequency==0:spectrum[4:]*=1e-3
            if frequency==2:spectrum[:]=0
            left.append(u@mixing);right.append(np.linalg.solve(mixing,spectrum[:,None]*v.conj().T))
        left=np.asarray(left,np.complex64);right=np.asarray(right,np.complex64)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);np.save(root/'input_left.npy',left);np.save(root/'input_right.npy',right)
            source=(np.load(root/'input_left.npy',mmap_mode='r'),np.load(root/'input_right.npy',mmap_mode='r'))
            (a,b),report=recompress_factors(*source,root/'packed',tolerance=1e-5,rank_multiple=2)
            self.assertLess(report['rank'],rank)
            self.assertEqual(report['rank']%2,0)
            self.assertLess(report['new_bytes'],report['old_bytes'])
            for frequency in range(nf):
                expected=left[frequency].astype(np.complex128)@right[frequency].astype(np.complex128)
                actual=a[frequency].astype(np.complex128)@b[frequency].astype(np.complex128)
                norm=np.linalg.norm(expected)
                error=np.linalg.norm(actual-expected)/norm if norm else np.linalg.norm(actual)
                self.assertLessEqual(error,report['records'][frequency]['combined_relative_error_bound']+1e-13)
                self.assertLess(error,1e-5)
            (_, _),cached=recompress_factors(*source,root/'packed',tolerance=1e-5,rank_multiple=2)
            self.assertEqual(cached,report)
            with self.assertRaisesRegex(ValueError,'Nonfinite'):
                invalid=left.copy();invalid[0,0,0]=np.nan
                recompress_factors(invalid,right,root/'bad')

    def test_rejects_invalid_settings_and_shapes(self):
        a=np.ones((2,7,3),np.complex64);b=np.ones((2,3,5),np.complex64)
        with tempfile.TemporaryDirectory() as tmp:
            for tol in (0,1,np.nan):
                with self.assertRaises(ValueError):recompress_factors(a,b,tmp,tolerance=tol)
            with self.assertRaises(ValueError):recompress_factors(a,b[:1],tmp)
            with self.assertRaises(ValueError):recompress_factors(a,b,tmp,rank_multiple=0)

if __name__=='__main__':unittest.main()
