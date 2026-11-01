import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import tempfile
import json
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from select_tmatrix_order import guarded_selection,physical_key,validate_target
import run_large_tmatrix as runner

class OrderSelectionCheck(unittest.TestCase):
    def rows(self,errors):return [dict(order=i+1,acceptance_error=e) for i,e in enumerate(errors)]

    def test_nonmonotone_tail_and_guard(self):
        result=guarded_selection(self.rows([.5,.005,.02,.004,.002,.001,.0001,0]),8,.01,2)
        self.assertEqual(result['minimum_L'],4)
        self.assertEqual(result['selected_L'],6)
        self.assertEqual(result['selected_error'],.001)

    def test_guard_at_reference_requires_extension(self):
        result=guarded_selection(self.rows([.2,.02,.005,.001,0]),5,.01,2)
        self.assertIsNone(result['selected_L'])
        self.assertEqual(result['status'],'extend_reference_order')

    def test_no_gaps_duplicates_or_nan(self):
        for records in ([dict(order=1,acceptance_error=0),dict(order=3,acceptance_error=0)],
                        [dict(order=1,acceptance_error=0)]*3,self.rows([.1,float('nan'),0])):
            with self.assertRaises(ValueError):guarded_selection(records,3)

    def test_reject_reuse_with_different_physics_or_mesh(self):
        parameters=dict(shape='prism',ka=40,ri=1.3116,ri_imag=0.,prism_aspect=10/7,wavelength=.532,ref=6)
        result=dict(status='selected',selected_L=58,physical_key=physical_key(parameters))
        self.assertEqual(validate_target(result,parameters),58)
        for name,value in [('ka',41),('ri',1.3),('ri_imag',10),('prism_aspect',1),('ref',5),('wavelength',.6)]:
            with self.assertRaisesRegex(ValueError,name):validate_target(result,dict(parameters,**{name:value}))

    def test_runner_preserves_selection_and_resume_evidence(self):
        parameters=dict(shape='prism',ka=40.,ri=1.3116,ri_imag=0.,prism_aspect=10/7,wavelength=.532,ref=6)
        evidence=dict(status='selected',selected_L=58,physical_key=physical_key(parameters))
        with tempfile.TemporaryDirectory() as folder:
            args=['runner','--out',folder,'--ka','40','--ri','1.3116','--prism-aspect',str(10/7),
                  '--wavelength','.532','--ref','6','--order-study',folder+'/study','--prepare-only']
            with patch.object(sys,'argv',args),patch.object(runner,'frozen_runtime'),\
                 patch.object(runner.shutil,'disk_usage',return_value=SimpleNamespace(free=1024*2**30)),\
                 patch.object(runner.psutil,'virtual_memory',return_value=SimpleNamespace(available=60*2**30)),\
                 patch('select_tmatrix_order.audit_and_select',return_value=evidence):
                self.assertEqual(runner.main(),0)
            plan=json.loads((Path(folder)/'plan.json').read_text())
            self.assertEqual(plan['parameters']['order'],58)
            self.assertEqual(plan['order_evidence'],evidence)
            self.assertIn('--order-study',plan['resume_command'])
            self.assertIn('--order-guard',plan['resume_command'])
            self.assertEqual(plan['command'][plan['command'].index('--order')+1],'58')

if __name__=='__main__':unittest.main()
