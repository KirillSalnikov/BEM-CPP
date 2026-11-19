"""Boundary panels, order of currents/components, and right-factor transpose."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from bounded_spectral_storage import StreamedSpectralBlock, SharedStreamBuffers
from transverse_fft_interpolation import TransverseLines


def lines(nx,channels):
    return TransverseLines(nx,2*nx*channels,stride=2,degree=3,edge=2)


def fixture():
    target,source=lines(13,2),lines(16,3)
    rng=np.random.default_rng(817)
    arrays=[rng.normal(size=shape).astype(t) for shape,t in [
        ((11,52,5,2),'f2'),((11,5,96,2),'f2'),((11,1,1),'f4')]]
    return arrays,target,source


class TransverseTests(unittest.TestCase):
    def test_adjoint_and_polynomial_reproduction(self):
        op=lines(17,3)
        p=torch.from_numpy(op.matrix)
        for degree in range(4):
            a=(op.nodes/(op.nx-1))**degree
            np.testing.assert_allclose(op.matrix@a,(np.arange(op.nx)/(op.nx-1))**degree,atol=3e-7)
        x=torch.randn(3,2*len(op.nodes)*3,4,dtype=torch.complex64)
        y=torch.randn(3,2*op.nx*3,4,dtype=torch.complex64)
        lhs=torch.vdot(op.apply(x,p,3).flatten(),y.flatten())
        rhs=torch.vdot(x.flatten(),op.apply(y,p.T,3).flatten())
        torch.testing.assert_close(lhs,rhs,atol=2e-5,rtol=2e-6)

    def test_guarded_packing_layout(self):
        arrays,a,b=fixture()
        block=StreamedSpectralBlock.__new__(StreamedSpectralBlock)
        block.arrays=arrays
        block.half=True
        block.nf,block.batch=11,2
        block.pinned_batches=None
        block.host_bytes=block.bytes=0
        block.device='cpu'
        ranks=[1,3,5,2,4,1]
        block.set_batch_ranks(ranks)
        block.set_transverse_lines(a,b,guard=2)
        self.assertEqual(block.compact_panels,[False,True,True,True,False,False])
        allocate=torch.empty
        with patch('torch.empty',side_effect=lambda *a,**k:allocate(*a,**{n:v for n,v in k.items() if n!='pin_memory'})):
            block.pin_coefficients()
        total=0
        for j,(left,right,scale) in enumerate(block.pinned_batches):
            sl=slice(2*j,2*j+2)
            l,r=arrays[0][sl,:,:ranks[j]],arrays[1][sl,:ranks[j]]
            if block.compact_panels[j]:
                l,r=np.take(l,a.indices,axis=1),np.take(r,b.indices,axis=2)
            for actual,expected in zip((left,right,scale),(l,r,arrays[2][sl])):
                np.testing.assert_array_equal(actual.numpy(),expected)
                total+=expected.nbytes
        self.assertEqual(total,block.transfer_bytes)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA device required')
    def test_guarded_gpu_action(self):
        arrays,target,source=fixture()
        pool=SharedStreamBuffers('cuda')
        block=StreamedSpectralBlock(arrays,device='cuda',batch=2,half=True,pool=pool)
        ranks=[1,3,5,2,4,1]
        block.set_batch_ranks(ranks)
        block.set_transverse_lines(target,source,guard=2)
        block.pin_coefficients()
        def complex_matrix(a):
            return a[...,0].astype('f')+1j*a[...,1].astype('f')
        def interpolator(op):
            return np.kron(np.eye(2),np.kron(op.matrix,np.eye(op.channels))).astype('f')
        pt,ps=interpolator(target),interpolator(source)
        for count in (1,3):
            x=torch.randn(11,96,count,dtype=torch.complex64)
            expected=[]
            for f in range(11):
                rank=ranks[f//2]
                l,r=complex_matrix(arrays[0][f,:,:rank]),complex_matrix(arrays[1][f,:rank])
                if block.compact_panels[f//2]:
                    l,r=pt@l[target.indices],r[:,source.indices]@ps.T
                expected.append((l@(r@x[f].numpy()))*arrays[2][f])
            result=block(x.cuda(),52).cpu().numpy()
            np.testing.assert_allclose(result,np.asarray(expected),atol=4e-4,rtol=2e-5)
