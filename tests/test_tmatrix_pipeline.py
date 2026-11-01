import json,os,subprocess,sys,tempfile,threading,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from tmatrix_pipeline import IncidentPreparation
from tmatrix_incident import regular_fields,RegularFieldCache
from tmatrix_c6 import encode_fields
from tmatrix_reference import modes

class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.points=np.vstack((np.random.default_rng(5).normal(size=(73,3)),[[0,0,0],[0,0,1],[0,0,-1]]))
        self.modal=modes(12)
        self.group=[self.modal.index((p,8,m)) for m in range(6) for p in (0,1)]

    def test_prefetch_preserves_both_codes_scales_and_fallback(self):
        for cache in (False,True):
            fields=np.empty((2,len(self.points),6),complex)
            prep=IncidentPreparation(self.points,self.modal,7.3,fields,cache=cache,prefetch=True)
            try:
                reference=np.empty_like(fields)
                scales,combined=encode_fields(self.points,self.modal,self.group,7.3,reference,
                    reflection=True,field_cache=RegularFieldCache(self.points,7.3) if cache else None)
                prep.prefetch('packed',self.group,True)
                actual=prep.get('packed',self.group,True)
                np.testing.assert_array_equal(fields,reference)
                np.testing.assert_array_equal(actual[0],scales);np.testing.assert_array_equal(actual[1],combined)
                self.assertTrue(actual[-1]);sent=fields.copy();old_scales=actual[0].copy()
                prep.prefetch('packed',[len(self.modal)+1],True)  # unused speculative error
                index=self.modal.index((1,9,3))
                actual=prep.get('single',[index],False)
                expected=regular_fields(self.points,self.modal[index],7.3)
                expected/=np.max(abs(expected))
                np.testing.assert_allclose(fields[0],expected,atol=2e-15,rtol=2e-13)
                np.testing.assert_array_equal(fields[0],fields[1])
                np.testing.assert_array_equal(old_scales,scales)
                np.testing.assert_array_equal(sent,reference)
                self.assertFalse(actual[-1])
                prep.prefetch('packed',[len(self.modal)+1],True)
                with self.assertRaises(IndexError):prep.get('packed',[len(self.modal)+1],True)
            finally:prep.close()

    def test_producer_runs_while_main_thread_is_available(self):
        fields=np.empty((2,len(self.points),6),complex)
        prep=IncidentPreparation(self.points,self.modal,7.3,fields,prefetch=True)
        entered=threading.Event();release=threading.Event();original=prep.prepare
        def blocked(key):
            entered.set()
            if not release.wait(5):raise RuntimeError('main thread could not continue')
            return original(key)
        prep.prepare=blocked
        try:
            prep.prefetch('packed',self.group,True)
            self.assertTrue(entered.wait(5));self.assertFalse(prep.pending[1].done())
            release.set();self.assertTrue(prep.get('packed',self.group,True)[-1])
        finally:release.set();prep.close()

    def test_complete_builder_matches_serial_and_resumes(self):
        with tempfile.TemporaryDirectory() as temp:
            outputs=[]
            env={k:v for k,v in os.environ.items() if not k.startswith('BEM_')}
            env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            for prefetch,fallback,hybrid in ((False,False,False),(True,False,False),(True,True,False),
                                             (True,False,True),(True,True,True)):
                out=Path(temp)/f'{prefetch}_{fallback}_{hybrid}'
                env.pop('BEM_TEST_FAKE_CROSS_TALK',None)
                env['BEM_TEST_FAKE_HYBRID']='1' if hybrid else '0'
                if fallback:env['BEM_TEST_FAKE_CROSS_TALK']='production'
                cmd=[sys.executable,str(ROOT/'scripts/build_fmm_tmatrix.py'),'--shape','prism','--ka','1',
                     '--ri','1.3','--ref','1','--order','7','--tol','1e-6','--no-average',
                     '--c2-reuse','--c6-pack','--mirror-pack','--worker-binary',str(ROOT/'tests/c6_fake_worker.py'),'--out',str(out)]
                if prefetch:cmd+=['--incident-prefetch']
                result=subprocess.run(cmd,cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
                self.assertEqual(result.returncode,0,result.stdout)
                outputs.append(np.load(out/'tmatrix.npy'))
                summary=json.loads((out/'summary.json').read_text())
                if hybrid:self.assertTrue(any(p['hybrid_packet'] for p in summary['pairs_this_run']))
                self.assertTrue(summary['c6_packing']['independent_reference_audits_complete'])
                if fallback:self.assertTrue(list((out/'c6_rejections').glob('*.json')))
                if prefetch:
                    self.assertTrue(any(p['incident_prefetched'] for p in summary['pairs_this_run']))
                    result=subprocess.run(cmd,cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
                    self.assertEqual(result.returncode,0,result.stdout)
                    self.assertEqual(json.loads((out/'summary.json').read_text())['pairs_this_run'],[])
            np.testing.assert_array_equal(outputs[0],outputs[1])
            for other in outputs[2:]:np.testing.assert_allclose(outputs[0],other,rtol=1e-8,atol=1e-10)

    def test_hybrid_bad_residual_never_commits_either_packet(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'reject'
            env={k:v for k,v in os.environ.items() if not k.startswith('BEM_')}
            env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',BEM_TEST_FAKE_HYBRID='1',
                       BEM_TEST_FAKE_BAD_RESIDUAL='1')
            cmd=[sys.executable,str(ROOT/'scripts/build_fmm_tmatrix.py'),'--shape','prism','--ka','1',
                 '--ri','1.3','--ref','1','--order','3','--tol','1e-6','--no-average','--columns','0','4',
                 '--worker-binary',str(ROOT/'tests/c6_fake_worker.py'),'--out',str(out)]
            process=subprocess.run(cmd,cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
            self.assertNotEqual(process.returncode,0,process.stdout)
            self.assertTrue((out/'rejected_solve.json').exists())
            self.assertEqual(list((out/'columns').glob('*.npy')),[])

if __name__=='__main__':unittest.main()
