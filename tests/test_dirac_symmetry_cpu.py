"""Opt-in checks against the selected experimental Dirac CPU library."""
import os,sys,unittest
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]

@unittest.skipUnless(os.environ.get('BEM_DIRAC_LIBRARY'),'requires explicitly selected Dirac CPU library')
class DiracSymmetryCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path[:0]=[str(R/'benchmarks/clifford_dielectric_20261115'),str(R/'scripts')]
        from solve import Dirac
        from dirac_symmetry import CyclicGeometry,SurfaceTransform
        cls.d=Dirac('prism',0,3,index=1.3116+.12j,degree=3,order=24,target_order=13,near_subdiv=1,dense_kernel='simd',block_panels=1)
        cls.g=CyclicGeometry(cls.d);cls.mirror=SurfaceTransform(cls.d,np.diag([1.,-1.,1.]))
    @classmethod
    def tearDownClass(cls):cls.d.close()
    def test_rotation_mirror_operator_and_inverse(self):
        d=self.d;rng=np.random.default_rng(114);x=rng.normal(size=d.size)+1j*rng.normal(size=d.size)
        a=d.apply_reference(x)
        for t in [self.g.transforms[1],self.mirror]:
            self.assertLess(np.linalg.norm(d.apply_reference(t.apply(x))-t.apply(a))/np.linalg.norm(a),2e-11)
        self.assertLess(np.linalg.norm(self.mirror.apply(self.mirror.apply(x))-x)/np.linalg.norm(x),2e-12)
    def test_complex_index_modal_sectors_and_mirror(self):
        from dirac_symmetry import solve_modal_batch
        from tmatrix_symmetry import vertical_mirror_plan
        d=self.d;modes=[(0,2,-1),(1,3,2),(1,4,0)]
        f=np.array([d.modal_incident(m) for m in modes]);h,stats=solve_modal_batch(d,self.g,f,modes)
        for i,(p,l,m) in enumerate(modes):
            exact,s=d.solve_incident(f[i]);self.assertLess(np.linalg.norm(h[i]-exact)/np.linalg.norm(exact),1e-7)
            self.assertLess(stats[i]['residual'],1e-8)
            sign=(-1.)**(p+m+1);pf=d.modal_incident((p,l,-m))
            self.assertLess(np.linalg.norm(sign*self.mirror.apply(f[i])-pf)/np.linalg.norm(pf),2e-12)
            rhs=(pf.reshape(-1,8)*(2*d.scaling[2])).ravel()
            self.assertLess(np.linalg.norm(d.apply_reference(sign*self.mirror.apply(h[i]))-rhs)/np.linalg.norm(rhs),1e-8)
    def test_selected_rows_and_rejection(self):
        import ctypes as ct
        from solve import lib,address,check
        lib.dirac_apply_rows.argtypes=[ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_void_p,ct.c_int]
        d=self.d;rng=np.random.default_rng(711);x=np.ascontiguousarray(rng.normal(size=(3,d.size))+1j*rng.normal(size=(3,d.size)))
        rows=np.array([d.size//8-1,0,5,10],np.int32);out=np.empty((3,len(rows)*8),complex)
        check(lib.dirac_apply_rows(d.p,3,address(x),len(rows),address(rows),address(out),64))
        expected=d.apply_batch(x).reshape(3,-1,8)[:,rows].reshape(out.shape)
        self.assertLess(np.linalg.norm(out-expected)/np.linalg.norm(expected),2e-12)
        rows[0]=d.size//8
        self.assertNotEqual(lib.dirac_apply_rows(d.p,3,address(x),len(rows),address(rows),address(out),64),0)
    def test_geometry_rejects_false_rotation(self):
        from dirac_symmetry import SurfaceTransform
        a=.17;c,s=np.cos(a),np.sin(a)
        with self.assertRaises(ValueError):SurfaceTransform(self.d,[[c,-s,0],[s,c,0],[0,0,1]])

if __name__=='__main__':unittest.main()
