import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_tmatrix_multisize as r


class MultisizeTest(unittest.TestCase):
    def test_audited_upgrade_preserves_other_family_guards(self):
        plan={'runtime_upgrade':{'canonical_binary_digest':'original','optimized_binary_digest':'optimized'}}
        a={'parameters':{'ka':10,'ri':1.3116,'binary_digest':'original','BEM_environment':{'OTHER':'same'}}}
        b={'parameters':{'ka':11,'ri':1.3116,'binary_digest':'optimized',
                         'BEM_environment':{'OTHER':'same','BEM_FMM_MULTILEVEL_PARITY':'1'}}}
        saved=json.dumps(b,sort_keys=True)
        r.validate_family([r.family_parameters(a,plan),r.family_parameters(b,plan)])
        self.assertEqual(json.dumps(b,sort_keys=True),saved)
        b['parameters']['ri']=1.4
        with self.assertRaises(ValueError):r.validate_family([r.family_parameters(a,plan),r.family_parameters(b,plan)])
        b['parameters']['ri']=1.3116;b['parameters']['BEM_environment']['OTHER']='changed'
        with self.assertRaises(ValueError):r.validate_family([r.family_parameters(a,plan),r.family_parameters(b,plan)])
        b['parameters']['binary_digest']='unknown'
        with self.assertRaises(ValueError):r.family_parameters(b,plan)
        b['parameters']['binary_digest']='optimized';b['parameters']['BEM_environment']['BEM_FMM_MULTILEVEL_PARITY']='0'
        with self.assertRaises(ValueError):r.family_parameters(b,plan)

    def test_refinement_uses_fresh_holdouts(self):
        a,b=r.stages(10,12)
        self.assertEqual(set(b['training']),set(a['training']+a['validation']))
        self.assertFalse(set(b['validation']) & set(a['training']+a['validation']))
        for stage in (a,b):
            self.assertFalse(set(stage['training']) & set(stage['validation']))
            self.assertTrue(all(10<x<12 for x in stage['validation']))

    def test_expired_deadline_starts_no_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            sweep=r.Sweep(Path(directory),{'deadline_unix':time.time()-1})
            try:
                with patch.object(r,'run') as run:
                    with self.assertRaises(r.EndSeries):sweep.bounded([],Path(directory),{})
                    run.assert_not_called()
            finally:sweep.done.set();sweep.thread.join()

    def test_material_control_cannot_be_reused_for_another_index(self):
        plan={'ka_min':10.,'ref':4,'ri':1.3116,'wavelength_um':.532}
        lower={'converged':True,'ka':10.,'ref':5,'ri':1.3,'wavelength_um':.532}
        with self.assertRaises(ValueError):r.check_lower_control(lower,plan)
        lower['ri']=1.3116;r.check_lower_control(lower,plan)

    def test_failed_controls_cannot_publish_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);(out/'model_0').mkdir()
            r.atomic_json(out/'model_0/controls.json',{'passes_1pct_controls':False})
            with patch.object(r,'load_model') as load:
                with self.assertRaises(RuntimeError):r.produce_grid(out,{},0)
                load.assert_not_called()
            self.assertFalse((out/'size_grid.npz').exists())

    def test_frozen_fit_roundtrip_and_training_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            plan={'ka_min':1.,'ka_max':2.,'ref':4,'order':2,'ri':1.3116,
                  'stages':r.stages(1,2),'interpolation_budget':.004}
            for ka in plan['stages'][0]['training']:
                dest=r.sample_path(out,ka,4,2);dest.mkdir()
                t=np.eye(16,dtype=complex)/(ka+2-0.3j)
                np.save(dest/'tmatrix.npy',t)
                r.atomic_json(dest/'summary.json',{'complete':True,'operator':{'operator_backend':'fmm'},
                    'parameters':{'ka':ka,'ref':4,'order':2,'ri':1.3116,'ri_imag':0,'geometry':'prism'}})
            r.fit_stage(out,plan,0)
            model,_=r.load_model(out,plan,0)
            np.testing.assert_allclose(model.evaluate(1.37),np.eye(16)/(1.37+2-.3j),rtol=1e-10,atol=1e-12)
            dest=r.sample_path(out,1.,4,2)
            np.save(dest/'tmatrix.npy',np.eye(16,dtype=complex))
            with self.assertRaisesRegex(RuntimeError,'changed after the fit'):r.load_model(out,plan,0)


if __name__=='__main__':unittest.main()
