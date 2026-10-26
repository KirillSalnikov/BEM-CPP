"""CPU checks of C6 algebra, streaming checkpoints and fallback; no GPU solves."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import tmatrix_c6 as c6
from tmatrix_reference import modes
from tmatrix_symmetry import c2x_plan


class AlgebraTest(unittest.TestCase):
    def test_reflection_audit_requires_both_symmetries_and_independent_pairs(self):
        modal=modes(3); permutation,signs,_,_=c2x_plan(3)
        eta=np.array([(-1.)**(p+l+m+1) for p,l,m in modal])
        rng=np.random.default_rng(93)
        operator=rng.normal(size=(len(modal),len(modal)))+1j*rng.normal(size=(len(modal),len(modal)))
        operator[eta[:,None]!=eta[None,:]]=0
        operator=(operator+signs[:,None]*signs[None,:]*operator[np.ix_(permutation,permutation)])/2
        columns={i:operator[:,i].copy() for i in range(len(modal))}
        self.assertTrue(c6.reflection_audit(modal,columns,1e-12)['passed'])
        positive=modal.index((0,2,1)); opposite=int(permutation[positive])
        bad={i:v.copy() for i,v in columns.items()}
        bad[positive][np.flatnonzero(eta!=eta[positive])[0]]+=1
        self.assertFalse(c6.reflection_audit(modal,bad,1e-12)['passed'])
        bad={i:v.copy() for i,v in columns.items()}
        bad[opposite]*=1.1
        report=c6.reflection_audit(modal,bad,1e-12)
        self.assertEqual(report['maximum_horizontal_error'],0)
        self.assertFalse(report['passed'])
        self.assertFalse(c6.reflection_audit(modal,{positive:columns[positive]},1e-12)['passed'])

    def test_rotation_and_reflection_plan_has_twelve_sectors(self):
        modal=modes(44); _,_,reps,_=c2x_plan(44)
        groups=c6.packets(modal,reps,True)
        self.assertEqual(len(groups),191)
        self.assertEqual(max(map(len,groups)),12)
        self.assertEqual(sorted(i for g in groups for i in g),sorted(reps))
        for group in groups:
            self.assertEqual(len({c6.sector(modal[i],True) for i in group}),len(group))

    def test_plan_has_unique_sectors_and_preserves_all_inputs(self):
        modal = modes(44)
        _, _, representatives, _ = c2x_plan(44)
        packets = c6.packets(modal, representatives)
        self.assertEqual(len(packets),382)
        self.assertEqual(sorted(i for packet in packets for i in packet),sorted(representatives))
        for packet in packets:
            self.assertEqual(len({modal[i][:2] for i in packet}),1)
            self.assertEqual(len({modal[i][2]%6 for i in packet}),len(packet))
        subset = representatives[::3]
        self.assertEqual(sorted(i for packet in c6.packets(modal,subset) for i in packet),sorted(subset))

    def test_both_phase_codes_recover_columns_and_reject_cross_talk(self):
        modal = modes(6); indices = [modal.index((1,6,m)) for m in range(6)]
        rng = np.random.default_rng(17)
        columns = rng.normal(size=(len(modal),6))+1j*rng.normal(size=(len(modal),6))
        for j,index in enumerate(indices):
            columns[[m%6 != modal[index][2]%6 for p,l,m in modal],j] = 0
        scales = np.exp(np.arange(6)*.2); combined = np.array([2.,3.])
        normalization = np.array([4*np.pi*1j**(modal[i][1]-modal[i][0]) for i in indices])*scales
        projected = (columns/normalization)@c6.phase_codes(modal,indices).T
        projected = projected.T/combined[:,None]
        actual,check = c6.decode_columns(projected,modal,indices,scales,combined,1e-10)
        self.assertTrue(check['passed'])
        for j,index in enumerate(indices):
            np.testing.assert_allclose(actual[index],columns[:,j],atol=2e-14)
        projected[1,modal.index((0,1,0))] += .1
        self.assertFalse(c6.decode_columns(projected,modal,indices,scales,combined,1e-10)[1]['passed'])
        with self.assertRaises(ValueError):
            c6.phase_codes(modal,[indices[0],modal.index((1,6,6))])

    def test_chunked_fields_equal_direct_superposition(self):
        from tmatrix_incident import regular_fields
        rng = np.random.default_rng(42); positions = rng.normal(size=(37,3))
        modal = modes(4); indices = [modal.index((p,4,m)) for p,m in [(0,0),(0,1),(0,2)]]
        fields = np.empty((2,len(positions),6),complex)
        scales,combined = c6.encode_fields(positions,modal,indices,2.,fields,chunk_size=7)
        direct = np.stack([regular_fields(positions,modal[i],2.)/scale for i,scale in zip(indices,scales)])
        expected = np.einsum('sj,jpc->spc',c6.phase_codes(modal,indices),direct)/combined[:,None,None]
        np.testing.assert_allclose(fields,expected,atol=5e-15)


class StreamingCPU(unittest.TestCase):
    def test_mirror_failure_retries_rotational_packets_across_resume(self):
        self.base[self.base.index('--order')+1]='7'
        self.base[self.base.index('--ka')+1]='3'
        self.env.update(BEM_TMATRIX_D6H_MESH='1',BEM_TEST_FAKE_CROSS_TALK='mirror')
        self.run_builder('direct')
        modal=modes(7)
        controls={i for group in c6.audit_packets(modal,7,True) for i in group}
        limit=(len(controls)+1)//2+len(c6.audit_packets(modal,7,True))+1
        flags=['--c6-pack','--mirror-pack']
        self.run_builder('packed',flags+['--max-pairs-this-run',str(limit)])
        p=self.directory/'packed'
        failed={f.name:f.read_bytes() for f in (p/'c6_rejections').glob('*.json')}
        self.assertTrue(failed)
        self.run_builder('packed',flags)
        for name,data in failed.items():
            self.assertEqual((p/'c6_rejections'/name).read_bytes(),data)
        report=json.loads((p/'summary.json').read_text())
        self.assertGreater(report['c6_packing']['rotation_only_fallback_pairs_this_run'],0)
        self.assertTrue(any(not json.loads(f.read_text())['reflection_sectors']
                            for f in (p/'c6_batches').glob('*.json')))
        np.testing.assert_allclose(np.load(p/'tmatrix.npy'),np.load(self.directory/'direct/tmatrix.npy'),rtol=1e-8,atol=1e-10)

    def test_reflection_packets_match_independent_full_T(self):
        self.base[self.base.index('--order')+1]='7'
        self.base[self.base.index('--ka')+1]='3'
        self.env['BEM_TMATRIX_D6H_MESH']='1'
        self.run_builder('direct')
        self.run_builder('mirror',['--c2-reuse','--c6-pack','--mirror-pack'])
        np.testing.assert_allclose(np.load(self.directory/'mirror/tmatrix.npy'),
                                   np.load(self.directory/'direct/tmatrix.npy'),rtol=1e-8,atol=1e-10)
        report=json.loads((self.directory/'mirror/summary.json').read_text())
        self.assertTrue(report['c6_packing']['horizontal_reflection_parity'])
        self.assertTrue(report['c6_packing']['vertical_reflection_from_C2_and_horizontal'])
        self.assertEqual(max(len(json.loads(p.read_text())['columns']) for p in
                            (self.directory/'mirror/c6_batches').glob('*.json')),12)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='bem-c6-cpu-')
        self.directory = Path(self.temporary.name)
        self.env = {k:v for k,v in os.environ.items() if not k.startswith('BEM_')}
        self.env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        self.base = [sys.executable,str(ROOT/'scripts/build_fmm_tmatrix.py'),'--shape','prism',
                     '--ka','1','--ri','1.3','--ref','1','--order','4','--tol','1e-6',
                     '--no-average','--worker-binary',str(ROOT/'tests/c6_fake_worker.py')]

    def tearDown(self):
        self.temporary.cleanup()

    def run_builder(self,name,extra=(),success=True):
        result = subprocess.run(self.base+['--out',str(self.directory/name)]+list(extra),
            cwd=ROOT,env=self.env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=45)
        if success:
            self.assertEqual(result.returncode,0,result.stdout)
        else:
            self.assertNotEqual(result.returncode,0,result.stdout)
        return result

    def test_complete_c2_c6_restart_and_missing_receipt(self):
        self.base[self.base.index('--order')+1]='7'
        self.base[self.base.index('--ka')+1]='3'
        self.run_builder('direct')
        self.run_builder('packed',['--c2-reuse','--c6-pack','--max-pairs-this-run','2'])
        self.assertFalse(json.loads((self.directory/'packed/summary.json').read_text())['complete'])
        self.run_builder('packed',['--c2-reuse','--c6-pack'])
        p=self.directory/'packed'
        reference=np.load(self.directory/'direct/tmatrix.npy')
        np.testing.assert_allclose(np.load(p/'tmatrix.npy'),reference,rtol=1e-8,atol=1e-10)
        report=json.loads((p/'summary.json').read_text())
        self.assertTrue(report['c6_packing']['independent_reference_audits_complete'])
        self.assertGreater(report['c6_packing']['packed_columns'],0)
        self.assertLess(report['new_direction_field_error'],1e-8)
        receipts=list((p/'c6_batches').glob('*.json')); self.assertTrue(receipts)
        self.assertEqual(max(len(json.loads(path.read_text())['columns']) for path in receipts),6)
        receipt=json.loads(receipts[0].read_text()); receipts[0].unlink()
        # Columns and C2 descendants without a committed receipt must be redone.
        self.run_builder('packed',['--c2-reuse','--c6-pack'])
        report=json.loads((p/'summary.json').read_text())
        self.assertGreater(len(report['pairs_this_run']),0)
        np.testing.assert_allclose(np.load(p/'tmatrix.npy'),reference,rtol=1e-8,atol=1e-10)
        receipt=json.loads(next((p/'c6_batches').glob('*.json')).read_text())
        index=receipt['columns'][0]
        path=p/'columns'/f'{index:06d}.npy'
        damaged=np.load(path); damaged[0]+=.25; np.save(path,damaged)
        self.run_builder('packed',['--c2-reuse','--c6-pack'])
        self.assertGreater(len(json.loads((p/'summary.json').read_text())['pairs_this_run']),0)
        np.testing.assert_allclose(np.load(p/'tmatrix.npy'),reference,rtol=1e-8,atol=1e-10)
        # Finite corruption of a reconstructed child must also be repaired.
        derived=next(path for path in (p/'columns').glob('*.json')
                     if json.loads(path.read_text()).get('derived_by')=='C2x_half_turn')
        path=derived.with_suffix('.npy')
        damaged=np.load(path); damaged[0]+=.25; np.save(path,damaged)
        self.run_builder('packed',['--c2-reuse','--c6-pack'])
        np.testing.assert_allclose(np.load(p/'tmatrix.npy'),reference,rtol=1e-8,atol=1e-10)
        self.run_builder('packed',['--c2-reuse','--c6-pack'])
        self.assertEqual(json.loads((p/'summary.json').read_text())['pairs_this_run'],[])

    def test_audit_rejection_never_writes_packed_columns(self):
        self.env['BEM_TEST_FAKE_CROSS_TALK']='audit'
        result=self.run_builder('reject',['--c6-pack'],success=False)
        self.assertIn('C6 packed RHS audit failed',result.stdout)
        self.assertFalse(list((self.directory/'reject/c6_batches').glob('*.json')))
        for path in (self.directory/'reject/columns').glob('*.json'):
            self.assertNotEqual(json.loads(path.read_text()).get('derived_by'),'C6_packed_RHS')

    def test_failed_packet_falls_back_and_remembers_rejection_on_restart(self):
        self.env['BEM_TEST_FAKE_CROSS_TALK']='production'
        self.run_builder('direct')
        modal=modes(4); permutation,_,_,audits=c2x_plan(4)
        controls=set(i for j in audits for i in (int(permutation[j]),j))
        groups=c6.audit_packets(modal,4); controls.update(i for group in groups for i in group)
        limit=(len(controls)+1)//2+len(groups)+1
        self.run_builder('fallback',['--c2-reuse','--c6-pack','--max-pairs-this-run',str(limit)])
        rejections=list((self.directory/'fallback/c6_rejections').glob('*.json'))
        self.assertTrue(rejections)
        before={p.name:p.read_bytes() for p in rejections}
        self.run_builder('fallback',['--c2-reuse','--c6-pack'])
        for name,data in before.items():
            self.assertEqual((self.directory/'fallback/c6_rejections'/name).read_bytes(),data)
        np.testing.assert_allclose(np.load(self.directory/'fallback/tmatrix.npy'),
                                   np.load(self.directory/'direct/tmatrix.npy'),rtol=1e-8,atol=1e-10)

    def test_seed_preserves_columns_and_rejects_different_operator(self):
        self.run_builder('seed',['--c2-reuse','--max-pairs-this-run','2'])
        original={p.name:p.read_bytes() for p in (self.directory/'seed/columns').glob('*.npy')}
        self.run_builder('packed',['--c2-reuse','--c6-pack','--seed-from',str(self.directory/'seed')])
        report=json.loads((self.directory/'packed/seed_report.json').read_text())
        self.assertEqual(report['copied_columns'],len(original))
        for name,data in original.items():
            self.assertEqual((self.directory/'packed/columns'/name).read_bytes(),data)
            self.assertEqual((self.directory/'seed/columns'/name).read_bytes(),data)
        result=self.run_builder('bad',['--c2-reuse','--c6-pack','--seed-from',str(self.directory/'seed'),'--ka','2'],success=False)
        self.assertIn('different operator/order',result.stdout)


if __name__=='__main__':
    unittest.main()
