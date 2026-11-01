import sys
import unittest
from pathlib import Path

import numpy as np
try:
    import torch
except ImportError:
    torch=None

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
if torch is not None:
    from probe_solver_tolerance import BatchedGMRES


@unittest.skipUnless(torch is not None and torch.cuda.is_available(),'optional PyTorch/CUDA diagnostic')
class BatchedToleranceGMRES(unittest.TestCase):
    def test_original_equations_independent_rhs_scales_and_order(self):
        rng=np.random.default_rng(442)
        n=32
        a=4*np.eye(n)+(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)))/np.sqrt(n)
        expected=rng.normal(size=(n,4))+1j*rng.normal(size=(n,4))
        expected*=np.array([1e-9,1.,1e4,1e-4])
        b=a@expected
        solver=BatchedGMRES(a,group_edges=2)
        for tolerance in [1e-3,1e-6]:
            x,info=solver.packet(b,tolerance,restart=8,limit=200)
            true=np.linalg.norm(a@x-b,axis=0)/np.linalg.norm(b,axis=0)
            self.assertTrue(np.all(true<=tolerance))
            np.testing.assert_allclose(true,info['residuals'],atol=2e-14,rtol=1e-6)
            self.assertLess(np.max(np.linalg.norm(x-expected,axis=0)/np.linalg.norm(expected,axis=0)),3*tolerance)
            self.assertGreater(min(info['iterations']),1)

    def test_zero_rhs_rejected(self):
        solver=BatchedGMRES(np.eye(16,dtype=complex),group_edges=2)
        with self.assertRaisesRegex(ValueError,'zero modal RHS'):
            solver.packet(np.zeros((16,1),complex),1e-3)


if __name__=='__main__':unittest.main()
