"""Condition the sketch while preserving complex physical-coordinate actions."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from galerkin_basis_metric import FaceBasisMetric
from streamed_galerkin_fft import compress_streamed
from test_streamed_galerkin_sketch import MatrixTiles


def fixture():
    rng=np.random.default_rng(38291)
    points=rng.normal(size=(7,2));p=np.c_[np.ones(7),points]
    basis=np.zeros((7,6,3));basis[:,:3,0]=p;basis[:,3:,1]=p
    q,_=np.linalg.qr(rng.normal(size=(6,6)))
    basis=np.einsum('qia,ij->qja',basis,q@np.diag([.5,.8,1,1.5,2,3]))
    face=SimpleNamespace(nx=3,patterns=(0,1),normal=np.array([0.,0.,1.]))
    block=SimpleNamespace(pattern=lambda face,kind:(points,basis))
    return block,face,basis


class MetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(2)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_local_rotation_becomes_orthogonal_and_maps_roundtrip(self):
        block,face,basis=fixture()
        metric=FaceBasisMetric(block,face,dtype=torch.complex128)
        b=basis.transpose(0,2,1).reshape(21,6)
        turned=np.cross(face.normal,basis).transpose(0,2,1).reshape(21,6)
        r=np.linalg.lstsq(b,turned,rcond=None)[0]
        s=metric.root[0].numpy();inverse=metric.inverse[0].numpy()
        white=s@r@inverse
        self.assertLess(np.linalg.norm(white.T@white-np.eye(6)),1e-12)
        x=torch.randn((3,72,120),dtype=torch.complex128)
        restored=metric.rows(metric.rows(x,inverse=True))
        self.assertLess(float(torch.linalg.vector_norm(restored-x)/torch.linalg.vector_norm(x)),1e-13)
        restored=metric.columns(metric.columns(x,inverse=True))
        self.assertLess(float(torch.linalg.vector_norm(restored-x)/torch.linalg.vector_norm(x)),1e-13)

    def test_balanced_sketch_validates_and_returns_original_operator(self):
        block,face,_=fixture();rng=np.random.default_rng(47381)
        random=lambda shape:rng.normal(size=shape)+1j*rng.normal(size=shape)
        matrix=random((3,72,8))@random((3,8,120))
        block.nfft=3;block.tc=36;block.sc=60;block.target=face
        block.source=SimpleNamespace(nx=5,patterns=(0,1));block.matrix=matrix
        for transpose in (False,True):
            with self.subTest(transpose=transpose),tempfile.TemporaryDirectory() as directory,patch('streamed_galerkin_fft.GPUCrossTiles',MatrixTiles):
                (u,v),report=compress_streamed(block,directory,16,device='cpu',
                    metric_balance=True,transpose_sketch=transpose,reference=matrix)
                actual=np.array(u,dtype=complex)@np.array(v,dtype=complex)
                self.assertEqual(report['status'],'complete')
                self.assertLess(np.linalg.norm(actual-matrix)/np.linalg.norm(matrix),3e-5)
                self.assertLess(report['reference_max'],3e-5)
                del u,v

    def test_singular_basis_is_rejected(self):
        block,face,basis=fixture();basis[:,0,:]=0
        with self.assertRaises(ValueError):FaceBasisMetric(block,face)


if __name__=='__main__':unittest.main()
