"""Numerical checks for the direct GPU kernels, also runnable in Triton's CPU interpreter."""
import importlib.util
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from bounded_spectral_storage import decode_half_symbol
from transverse_fft_interpolation import TransverseLines

interpreted=os.environ.get("TRITON_INTERPRET")=="1"
available=importlib.util.find_spec('triton') is not None and (interpreted or torch.cuda.is_available())


@unittest.skipUnless(available,'CUDA or the Triton interpreter required')
class DirectPanelTests(unittest.TestCase):
    def test_thin_panels_and_strided_fft(self):
        from spectral_panel_kernel import half_product,current_product,stencil_table,interpolate
        device='cpu' if interpreted else 'cuda'
        generator=torch.Generator().manual_seed(1453)
        def sample(shape,dtype):
            return torch.randn(shape,dtype=dtype,generator=generator).to(device)
        def compare(y,reference):
            relative=torch.linalg.vector_norm(y-reference)/torch.linalg.vector_norm(reference)
            self.assertLess(float(relative),2e-6)
        for rhs in (1,2):
            for size in (13,513):
                a=sample((5,11,size,2),torch.float16)
                x=sample((rhs,size,5),torch.complex64).permute(2,1,0)
                compare(half_product(a,x),torch.bmm(torch.view_as_complex(a.float()),x))
            a=sample((5,3,11,13,2),torch.float16)
            scale=sample((5,1,1),torch.float32)
            x=sample((rhs,26,5),torch.complex64).permute(2,1,0)
            compare(current_product(a,scale,x),torch.bmm(decode_half_symbol(a),x)*scale)
            lines=TransverseLines(31,124,stride=2,degree=7,edge=3)
            for matrix in (lines.matrix,lines.matrix.T):
                x=sample((rhs,4*matrix.shape[1],5),torch.complex64).permute(2,1,0)
                compare(interpolate(x,stencil_table(matrix,device),2),
                        lines.apply(x,torch.as_tensor(matrix,device=device),2))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA streams required')
    def test_streamed_and_resident_routes(self):
        from bounded_spectral_storage import StreamedSpectralBlock,SharedStreamBuffers
        from symmetric_galerkin_fft import TorchSymmetricAction
        rng=np.random.default_rng(562)
        arrays=[rng.standard_normal(shape).astype(dtype) for shape,dtype in
                [((9,3,7,5,2),'f2'),((9,1,1),'f4')]]
        pool=SharedStreamBuffers('cuda')
        streamed=StreamedSpectralBlock(arrays,device='cuda',batch=4,half_matrix=True,pool=pool)
        streamed.enable_direct_panels()
        resident=dict(symbol_half=torch.as_tensor(arrays[0],device='cuda'),
                      symbol_scale=torch.as_tensor(arrays[1],device='cuda'),direct_currents=True)
        for rhs in (1,2,6):
            x=torch.randn(9,10,rhs,device='cuda',dtype=torch.complex64)
            ref=torch.bmm(decode_half_symbol(resident['symbol_half']),x)*resident['symbol_scale']
            for y in (streamed(x,14),TorchSymmetricAction.symbol_product(resident,x,14)):
                self.assertLess(float(torch.linalg.vector_norm(y-ref)/torch.linalg.vector_norm(ref)),2e-6)

        # Vary the retained rank and test both compact and protected bands
        # with the same reused asynchronous coefficient buffers.
        a,b=(rng.standard_normal(shape).astype('f2') for shape in ((9,52,5,2),(9,5,48,2)))
        pool=SharedStreamBuffers('cuda')
        blocks=[StreamedSpectralBlock((a,b,arrays[1]),device='cuda',batch=2,half=True,pool=pool)
                for _ in range(2)]
        for block in blocks:
            block.set_batch_ranks([1,3,5,2,4])
            opts=dict(degree=3,edge=2,stride=2)
            block.set_transverse_lines(TransverseLines(13,52,**opts),TransverseLines(12,48,**opts),guard=2)
            block.pin_coefficients()
        blocks[1].enable_direct_panels()
        for rhs in (1,2,6):
            x=torch.randn(9,48,rhs,device='cuda',dtype=torch.complex64)
            ref,y=(block(x,52) for block in blocks)
            self.assertLess(float(torch.linalg.vector_norm(y-ref)/torch.linalg.vector_norm(ref)),2e-6)
