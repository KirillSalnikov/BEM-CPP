"""Independent algebraic and group checks for the experimental inverse."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from compact_fft_inverse import CompactInverse,D6hMaps,model_for_rank
from compact_fft_sector_storage import SectorCompactInverse

def group_maps(copies=16):
    n=24*copies;maps=np.empty((3,2,n),int)
    for copy in range(copies):
        for h in range(2):
            for v in range(2):
                for r in range(6):
                    i=24*copy+12*h+6*v+r
                    maps[0,0,i]=24*copy+12*h+6*v+(r+1)%6
                    maps[1,0,i]=24*copy+12*(1-h)+6*v+r
                    maps[2,0,i]=24*copy+12*h+6*(1-v)+(-r)%6
    maps[:,1]=1
    return D6hMaps(maps)

class CompactInverseChecks(unittest.TestCase):
    def test_woodbury_against_independent_full_solve(self):
        rng=np.random.default_rng(63);Q=rng.normal(size=(63,9))+1j*rng.normal(size=(63,9));C=(rng.normal(size=(9,63))+1j*rng.normal(size=(9,63)))/200
        x=rng.normal(size=(63,4))+1j*rng.normal(size=(63,4));model=CompactInverse(Q,C)
        exact=np.linalg.solve(np.eye(63)+Q@C,x)
        self.assertLess(np.linalg.norm(model.apply(x)-exact)/np.linalg.norm(exact),2e-13)

    def test_symmetry_complete_model_commutes_with_all_generators(self):
        group=group_maps();rng=np.random.default_rng(17);blocks=[]
        for h in [1,-1]:
            for m in [0,1,2,3]:
                for v in ([1,-1] if m in [0,3] else [0]):
                    q=group.project(rng.normal(size=(group.n,8))+1j*rng.normal(size=(group.n,8)),h,m,v)
                    q=np.linalg.qr(q)[0]
                    c=group.project(rng.normal(size=(group.n,8))+1j*rng.normal(size=(group.n,8)),h,m,v).conj().T/100
                    blocks.append({'h':h,'m':m,'v':v,'Q':q,'C':c})
        model=model_for_rank(blocks,group,128);x=rng.normal(size=(group.n,3))+1j*rng.normal(size=(group.n,3));y=model.apply(x)
        self.assertEqual(model.Q.shape,(group.n,128))
        for g in range(3):
            self.assertLess(np.linalg.norm(model.apply(group.transform(x,g))-group.transform(y,g))/np.linalg.norm(y),2e-13)
        sector_model=SectorCompactInverse(blocks,group,128)
        self.assertLess(np.linalg.norm(sector_model.apply(x)-y)/np.linalg.norm(y),2e-13)
        self.assertLess(sector_model.bytes,model.bytes)

    def test_projectors_are_idempotent_and_exhaustive(self):
        group=group_maps();rng=np.random.default_rng(7);x=rng.normal(size=(group.n,2))+1j*rng.normal(size=(group.n,2));total=np.zeros_like(x)
        for h in [1,-1]:
            for m in range(6):
                y=group.project(x,h,m);total+=y
                self.assertLess(np.linalg.norm(group.project(y,h,m)-y)/np.linalg.norm(y),2e-13)
        self.assertLess(np.linalg.norm(total-x)/np.linalg.norm(x),2e-13)

if __name__=='__main__':unittest.main()
