"""Independent direct-sum oracles for experimental scalar multilevel GPU FMM."""
import importlib.util,os,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
R=Path(__file__).resolve().parents[1];B=R/'benchmarks/dirac_gpu_fmm_20261116'

@unittest.skipUnless(os.environ.get('BEM_TEST_GPU_FMM')=='1','opt-in CUDA benchmark library')
class ScalarGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location('scalar_gpu_test_native',B/'native.py')
        cls.native=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.native)

    def test_direct_oracle_complex_channels_and_offset(self):
        rng=np.random.default_rng(711)
        for k,offset in [(40.,0.),(40.+.5j,1000.),(52.,0.),(60.,0.),(160.+.25j,0.)]:
            s=np.asfortranarray(rng.random((3,113))*.2+offset)
            t=np.asfortranarray(rng.random((3,71))*.2+offset);t[:,0]=s[:,0]
            q=np.asfortranarray(rng.normal(size=(8,113))+1j*rng.normal(size=(8,113)))
            dx=t[:,:,None]-s[:,None,:];r=np.linalg.norm(dx,axis=0)
            active=r>=1e-12;r=np.where(active,r,1.)
            green=np.where(active,np.exp(1j*k*r)/(4*np.pi*r),0.)
            ep=np.einsum('ds,ts->dt',q,green)
            eg=np.einsum('ds,ats,ts->dat',q,dx,green*(1j*k*r-1)/r**2)
            with patch.dict(os.environ,{'BEM_FMM_MULTILEVEL':'1','BEM_FMM_MAX_DEPTH':'7'}):
                plan=self.native.GPUFMM(s,t,k,digits=6,depth=3)
                try:
                    self.assertEqual(plan.info['actual_depth'],3 if abs(k)>55 else 2)
                    for near in (0,1,2):
                        with patch.dict(os.environ,{'BEM_FMM_SCALAR3_NEAR':str(near)}):
                            value=plan.apply(q)
                            for actual,expected in [(value.pot,ep),(value.grad,eg)]:
                                error=np.linalg.norm(actual-expected)/np.linalg.norm(expected)
                                self.assertLess(error,2e-9,(k,near,error))
                    for bad in [1.,np.ones(113),np.ones((3,112)),np.ones((17,113))]:
                        with self.assertRaises(ValueError):plan.apply(bad)
                finally:plan.close()
                plan.close()
                with self.assertRaisesRegex(RuntimeError,'closed'):plan.apply(q)

if __name__=='__main__':unittest.main()
