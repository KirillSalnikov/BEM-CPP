"""Forward/adjoint inverse versus the independent full-coordinate SVD."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from constrained_current_inverse import ConstrainedCurrentInverse
from constrained_current_actions import ActionConstrainedCurrentInverse


class CurrentActionTests(unittest.TestCase):
    def setUp(self): self.rng=np.random.default_rng(713)
    def complex(self,*shape): return self.rng.normal(size=shape)+1j*self.rng.normal(size=shape)
    def make(self,r,c,x,**options):
        return ActionConstrainedCurrentInverse(r.shape[1],r.shape[0],c.shape[0],
            lambda v:r@v,lambda v:r.conj().T@v,lambda v:c@v,lambda v:c.conj().T@v,x,**options)

    def test_weighted_affine_prior_and_complement_match_full_inverse(self):
        r,c,x=self.complex(12,80),self.complex(5,80),self.complex(80,4)
        g=np.exp(self.rng.uniform(-2,2,80));w=np.exp(self.rng.uniform(-1,1,12))
        field,rhs=self.complex(12,3),self.complex(5,3)
        for block in (1,3,16):
            options=dict(current_metric=g,field_weights=w,radiation_rcond=1e-8,prior_rcond=1e-10)
            full=ConstrainedCurrentInverse(r,c,x,**options)
            new=self.make(r,c,x,block_size=block,**options)
            a,_=new.solve(field,rhs);b,_=full.solve(field,rhs)
            np.testing.assert_allclose(a,b,rtol=2e-10,atol=2e-11)
            np.testing.assert_allclose(c@a,rhs,rtol=2e-11,atol=2e-11)
            np.testing.assert_allclose(r@a,field,rtol=2e-10,atol=2e-11)
            self.assertLessEqual(new.report['current_basis_rank'],21)
            self.assertFalse(new.report['full_radiation_or_constraint_matrices_assembled'])
            self.assertLess(new.report['basis_allocated_bytes'],80*80*16)
            truth=x@self.complex(4,2)
            restored,_=new.solve(r@truth,c@truth)
            np.testing.assert_allclose(restored,truth,rtol=3e-10,atol=3e-10)

    def test_regularization_complex_phase_and_nonradiating_prior(self):
        r=np.zeros((3,12),complex);r[:,:3]=np.diag([1.,.1,1e-8])
        c=np.zeros((1,12),complex);c[0,-1]=2+3j
        x=np.zeros((12,2),complex);x[2,0]=1;x[3,0]=1j;x[5,1]=1
        new=self.make(r,c,x,radiation_rcond=1e-3,prior_rcond=1e-12)
        old=ConstrainedCurrentInverse(r,c,x,radiation_rcond=1e-3,prior_rcond=1e-12)
        f=np.array([1.,2.,1e-8j]);d=np.array([3-1j])
        a,_=new.solve(f,d);b,_=old.solve(f,d)
        np.testing.assert_allclose(a,b,atol=1e-10)
        rotated,_=new.solve(1j*f,1j*d)
        np.testing.assert_allclose(rotated,1j*a,atol=1e-10)

    def test_dependent_homogeneous_and_inconsistent_constraints(self):
        r=self.complex(7,30);base=self.complex(2,30);c=np.r_[base,base[:1]*(2+3j)]
        new=self.make(r,c,np.empty((30,0)))
        x,_=new.solve(np.ones(7),np.zeros(3))
        np.testing.assert_allclose(c@x,0.,atol=1e-12)
        with self.assertRaisesRegex(ValueError,'inconsistent'):
            new.solve(np.ones(7),[0,0,1])

    def test_zero_actions_and_no_constraints(self):
        r=np.zeros((2,9));c=np.empty((0,9));x=np.empty((9,0))
        new=self.make(r,c,x)
        value,_=new.solve(np.ones(2),np.empty(0))
        np.testing.assert_array_equal(value,np.zeros(9))
        self.assertEqual(new.report['current_basis_rank'],0)
        r=self.complex(4,9)
        new=self.make(r,c,x,radiation_rcond=1e-12)
        value,_=new.solve(np.ones(4),np.empty(0))
        np.testing.assert_allclose(value,np.linalg.pinv(r)@np.ones(4),atol=1e-12)

    def test_full_current_span_and_block_boundaries(self):
        r=self.complex(8,12);c=self.complex(6,12);x=self.complex(12,3)
        truth=self.complex(12,2)
        for block in (3,8,20):
            new=self.make(r,c,x,block_size=block)
            old=ConstrainedCurrentInverse(r,c,x)
            a,_=new.solve(r@truth,c@truth);b,_=old.solve(r@truth,c@truth)
            np.testing.assert_allclose(a,b,atol=2e-11)
            self.assertEqual(new.report['current_basis_rank'],12)

    def test_wrong_adjoint_and_memory_limit_reject_before_preparation(self):
        r=self.complex(4,16);c=self.complex(2,16);x=self.complex(16,2)
        with self.assertRaisesRegex(ValueError,'adjoint consistency'):
            ActionConstrainedCurrentInverse(16,4,2,lambda v:r@v,lambda v:r.T@v,
                lambda v:c@v,lambda v:c.conj().T@v,x)
        with self.assertRaises(MemoryError): self.make(r,c,x,memory_limit_bytes=1)
        with self.assertRaises(ValueError): self.make(r,c,x,block_size=0)


if __name__=='__main__': unittest.main()
