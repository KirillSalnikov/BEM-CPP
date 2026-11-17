"""Independent numerical oracles for the ref7 memory reductions."""
import os,sys,unittest
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]

@unittest.skipUnless(os.environ.get('BEM_DIRAC_LIBRARY'),'requires Dirac and FMM3D')
class CompactMemory(unittest.TestCase):
    def test_large_T_plan_maps_storage_and_sizes_FMM_packet(self):
        from argparse import Namespace
        from unittest.mock import patch
        sys.path[:0]=[str(R/'benchmarks/clifford_dielectric_20261115'),str(R/'scripts')]
        from memory_plan import sector_fmm_memory_plan
        args=Namespace(shape='prism',ref=7,aspect=1.6495721976846451,ka=80,index=1.3116,
            order=24,target_order=13,degree=3,near_subdiv=1,coarse_factorization='dense',
            sector_coarse_basis=0,symmetry='c6hv',coarse_cache_policy='sector',coarse_budget_gib=8,
            kernel_budget_gib=2,L=137,batch_size=1,fmm_packet=1,fmm_components=None,
            so3='mstm',no_average=False,so3_threads=4,ram_budget_gib=48)
        with patch('memory_plan.psutil.virtual_memory') as mem:
            mem.return_value.available=55*2**30
            large=sector_fmm_memory_plan(args)
            self.assertEqual(large['FMM_components'],4)
            self.assertLess(large['estimated_peak_bytes'],48*2**30)
            self.assertGreater(large['T_mapped_disk_bytes'],43*2**30)
            self.assertLess(large['T_arrays_bytes'],2**30)
            args.ram_budget_gib=40
            self.assertEqual(sector_fmm_memory_plan(args)['FMM_components'],2)
            args.ram_budget_gib=32
            with self.assertRaises(MemoryError):sector_fmm_memory_plan(args)
            args.ram_budget_gib=64;mem.return_value.available=80*2**30
            large=sector_fmm_memory_plan(args)
            self.assertEqual(large['FMM_components'],8)
            self.assertEqual(large['FMM_reference_components'],4)

    def test_translation_cache_and_streamed_channels(self):
        sys.path[:0]=[str(R/'benchmarks/clifford_dielectric_20261115'),str(R/'scripts')]
        from sector_fmm import SectorFMM
        kw=dict(ref=1,ka=10,index=1.3116+.1j,degree=3,order=24,near_subdiv=1,audit=False)
        shared=SectorFMM(**kw);plain=SectorFMM(**kw,share_near=False)
        rng=np.random.default_rng(839);x=rng.normal(size=(1,shared.size))+1j*rng.normal(size=(1,shared.size))
        try:
            self.assertLess(shared.cache_info['near_pattern_count'],shared.cache_info['near_pair_count'])
            for medium in (0,1):
                oracle=plain._near_rows(x,medium)
                self.assertLess(np.linalg.norm(shared._near_rows(x,medium)-oracle)/np.linalg.norm(oracle),2e-11)
            # Two simultaneous RHS take the original vectorized contraction;
            # a single RHS, including all eight components, streams its output.
            plain.fmm_packet=2
            expected=plain.apply_reference_batch(np.repeat(x,2,axis=0))[0];g=shared._compact_geometry
            for packet in (1,2,4,8):
                shared.fmm_components=packet
                self.assertLess(np.linalg.norm(shared.apply_reference(x[0])-expected)/np.linalg.norm(expected),2e-10)
                self.assertLess(np.linalg.norm(shared.apply_sector_batch(x)[0]-g.restrict(expected))/np.linalg.norm(g.restrict(expected)),2e-9)
            from unittest.mock import patch
            shared.fmm_reference_components=2
            with patch.object(shared.engine,'hfmm3d',wraps=shared.engine.hfmm3d) as full:
                checked=shared.apply_reference(x[0])
                self.assertEqual({call.kwargs['nd'] for call in full.call_args_list},{2})
                self.assertLess(np.linalg.norm(checked-expected)/np.linalg.norm(expected),2e-10)
            with patch.object(shared.engine,'targets',wraps=shared.engine.targets) as selected:
                checked=shared.apply_sector_batch(x)[0]
                self.assertEqual({call.kwargs['charges'].shape[0] for call in selected.call_args_list},{8})
                self.assertLess(np.linalg.norm(checked-g.restrict(expected))/np.linalg.norm(g.restrict(expected)),2e-9)
            self.assertEqual(shared._dense_factors[0].shape[0],len(g.representatives))
            self.assertIs(shared.panel_factors(g.representatives)[0],shared._dense_factors[0])
            self.assertLess(sum(t.basis.nbytes+t.components.nbytes for t in g.transforms),shared.count*12*164*8/5)
        finally:shared.close();plain.close()

if __name__=='__main__':unittest.main()
