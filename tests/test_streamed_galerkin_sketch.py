"""Independent matrix checks for the one-pass spectral sketch and its transpose."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from streamed_galerkin_fft import compress_streamed


class MatrixTiles:
    """Supply arbitrary complex matrices, with the two-current index layout."""
    def __init__(self,block,device,**options):
        self.matrix=torch.as_tensor(block.matrix,device=device,dtype=torch.complex64)
    def symbol_tile(self,first_t,last_t,first_s,last_s):
        nr,nc=self.matrix.shape[1:]
        r=np.arange(12*first_t,12*last_t);r=np.r_[r,nr//2+r]
        c=np.arange(12*first_s,12*last_s);c=np.r_[c,nc//2+c]
        return self.matrix[:,r[:,None],c],r,c


class StreamedSketchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(2)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)
    def run_sketch(self,matrix,rank,transpose):
        nf,nr,nc=matrix.shape
        block=SimpleNamespace(nfft=nf,tc=nr//2,sc=nc//2,
            target=SimpleNamespace(nx=nr//24),source=SimpleNamespace(nx=nc//24),matrix=matrix)
        with tempfile.TemporaryDirectory() as directory,patch('streamed_galerkin_fft.GPUCrossTiles',MatrixTiles):
            (a,b),report=compress_streamed(block,directory,rank,tile=2,device='cpu',
                reference=matrix,transpose_sketch=transpose,tolerance=1e-4)
            result=np.array(a,dtype=np.complex128)@np.array(b,dtype=np.complex128)
            shapes=(a.shape,b.shape)
            del a,b
        return result,shapes,report

    def test_rectangular_complex_operator_both_orientations(self):
        rng=np.random.default_rng(54391)
        random=lambda shape:rng.normal(size=shape)+1j*rng.normal(size=shape)
        for nr,nc in ((72,120),(120,72)):
            matrix=random((5,nr,8))@random((5,8,nc))
            for transpose in (False,True):
                with self.subTest(shape=(nr,nc),transpose=transpose):
                    found,shapes,report=self.run_sketch(matrix,16,transpose)
                    self.assertEqual(shapes,((5,nr,16),(5,16,nc)))
                    self.assertEqual(report['status'],'complete')
                    self.assertLess(np.linalg.norm(found-matrix)/np.linalg.norm(matrix),3e-5)
                    self.assertLess(report['reference_max'],3e-5)
                    self.assertLess(report['holdout_max'],3e-5)

    def test_insufficient_rank_rejected_by_original_action(self):
        rng=np.random.default_rng(731)
        matrix=rng.normal(size=(3,48,96))+1j*rng.normal(size=(3,48,96))
        for transpose in (False,True):
            _,_,report=self.run_sketch(matrix,3,transpose)
            self.assertEqual(report['status'],'failed_accuracy')
            self.assertGreater(report['holdout_max'],.1)

    def test_zero_operator_has_finite_zero_error(self):
        for transpose in (False,True):
            found,_,report=self.run_sketch(np.zeros((3,48,96),complex),4,transpose)
            self.assertEqual(report['status'],'complete')
            self.assertEqual(report['holdout_max'],0)
            self.assertEqual(report['reference_max'],0)
            self.assertTrue(np.array_equal(found,np.zeros_like(found)))

if __name__=='__main__':unittest.main()
