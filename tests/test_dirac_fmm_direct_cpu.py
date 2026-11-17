"""Independent direct-sum oracle for the optional CPU-FMM gradient kernel."""
import concurrent.futures
import ctypes as ct
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
LIBRARY=Path(os.environ.get('BEM_FMM3D_LIBRARY',ROOT/'benchmarks/clifford_fmm_20261115/FMM3D/lib/libfmm3d.so'))


@unittest.skipUnless(LIBRARY.exists(),'optional FMM3D library is not installed')
class DirectGradient(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library=ct.CDLL(str(LIBRARY))
        cls.fn=cls.library.h3ddirectcg_cpp_
        cls.fn.argtypes=[ct.c_void_p]*10;cls.fn.restype=None

    def check_case(self,nd,ns,nt,k,threshold):
        rng=np.random.default_rng(731+nd+ns+nt)
        sources=np.asfortranarray(rng.normal(size=(3,ns))*.1)
        targets=np.asfortranarray(rng.normal(size=(3,nt))*.1)
        if ns and nt:targets[:,0]=sources[:,0]
        if ns>1 and nt>1:targets[:,1]=sources[:,1]+threshold*.1
        q=np.asfortranarray(rng.normal(size=(nd,ns))+1j*rng.normal(size=(nd,ns)))
        p=np.asfortranarray(rng.normal(size=(nd,nt))+1j*rng.normal(size=(nd,nt)))
        g=np.asfortranarray(rng.normal(size=(nd,3,nt))+1j*rng.normal(size=(nd,3,nt)))
        p0,g0=p.copy(),g.copy()
        dims=list(map(ct.c_int64,(nd,ns,nt)));tol=ct.c_double(threshold);wave=np.array([k],complex)
        ptr=lambda a:a.ctypes.data_as(ct.c_void_p)
        self.fn(ct.byref(dims[0]),ptr(wave),ptr(sources),ptr(q),ct.byref(dims[1]),
                ptr(targets),ct.byref(dims[2]),ptr(p),ptr(g),ct.byref(tol))
        delta=targets[:,:,None]-sources[:,None,:]
        r=np.linalg.norm(delta,axis=0);active=r>threshold;r=np.where(active,r,1.)
        kernel=np.where(active,np.exp(1j*k*r)/(4*np.pi*r),0.)
        exact_p=p0+np.einsum('ds,ts->dt',q,kernel)
        exact_g=g0+np.einsum('ds,ats,ts->dat',q,delta,kernel*(1j*k*r-1)/r**2)
        for actual,expected in ((p,exact_p),(g,exact_g)):
            error=np.linalg.norm(actual-expected)/max(np.linalg.norm(expected),1e-300)
            self.assertLess(error,2e-12,(nd,ns,nt,k,threshold,error))

    def test_channels_tails_complex_and_additive_outputs(self):
        for nd in (1,2,3,4,8,16,17):
            for k in (0.,80.,80+10j,-3+.5j):
                with self.subTest(nd=nd,k=k):self.check_case(nd,67,19,k,1e-10)

    def test_empty_and_single(self):
        for ns,nt in ((0,0),(0,7),(9,0),(1,1),(33,1),(1,9)):
            with self.subTest(ns=ns,nt=nt):self.check_case(4,ns,nt,12.,1e-8)

    def test_parallel_independent_target_boxes(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            jobs=[pool.submit(self.check_case,nd,133,29,80+2j,1e-8) for nd in (1,2,3,4,8,16,17)]
            for job in jobs:job.result()

    def test_green_function_over_many_phase_quadrants(self):
        # A single source avoids cancellation hiding local sin/cos errors.
        # Include zero distance, quadrant boundaries, and vector tails.
        phase=np.r_[0.,np.linspace(.001,512,4097),np.arange(1,100)*np.pi/2]
        sources=np.zeros((3,1),order='F');targets=np.zeros((3,len(phase)),order='F')
        targets[0]=phase/80
        q=np.asfortranarray(np.array([[1+.3j],[-.2+.7j]],complex))
        dims=list(map(ct.c_int64,(2,1,len(phase))));threshold=ct.c_double(1e-14)
        ptr=lambda a:a.ctypes.data_as(ct.c_void_p)
        for k in (80.,-80.,80.+.5j):
            p=np.zeros((2,len(phase)),complex,order='F')
            g=np.zeros((2,3,len(phase)),complex,order='F');wave=np.array([k],complex)
            self.fn(ct.byref(dims[0]),ptr(wave),ptr(sources),ptr(q),ct.byref(dims[1]),
                    ptr(targets),ct.byref(dims[2]),ptr(p),ptr(g),ct.byref(threshold))
            r=targets[0,1:];green=np.exp(1j*k*r)/(4*np.pi*r)
            exact_p=q*green;exact_g=q*(green*(1j*k*r-1)/r)
            self.assertLess(float(np.max(abs(p[:,1:]-exact_p)/abs(exact_p))),2e-12)
            self.assertLess(float(np.max(abs(g[:,0,1:]-exact_g)/abs(exact_g))),2e-12)
            self.assertTrue(np.all(p[:,0]==0));self.assertTrue(np.all(g[:,:,0]==0))
            self.assertTrue(np.all(g[:,1:,:]==0))


class DirectBuildPatch(unittest.TestCase):
    def test_idempotence_header_dependency_and_unknown_edits(self):
        backend=ROOT/'benchmarks/clifford_fmm_20261115'
        source=backend/'FMM3D/vec-kernels/src/libkernels.cpp'
        if not source.exists():self.skipTest('optional pinned FMM3D checkout is not installed')
        spec=importlib.util.spec_from_file_location('bem_direct_patch_test',backend/'direct_kernel.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        original=source.read_text().replace(module.PATCHED_INCLUDE,module.INCLUDE).replace(module.NEW,module.OLD)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/'vec-kernels/src/libkernels.cpp'
            path.parent.mkdir(parents=True);(root/'vec-kernels/include').mkdir()
            path.write_text(original);obj=path.with_suffix('.o');obj.write_bytes(b'old')
            first=module.patch_source(root);self.assertFalse(obj.exists())
            obj.write_bytes(b'fresh')
            self.assertEqual(module.patch_source(root),first)
            self.assertEqual(obj.read_bytes(),b'fresh')
            (root/'vec-kernels/include/bem_direct.hpp').write_text('outdated header')
            module.patch_source(root);self.assertFalse(obj.exists())
            unknown=path.read_text()+'\n// unrelated local change\n';path.write_text(unknown)
            with self.assertRaisesRegex(RuntimeError,'Unrecognized'):module.patch_source(root)
            self.assertEqual(path.read_text(),unknown)


if __name__=='__main__':unittest.main()
