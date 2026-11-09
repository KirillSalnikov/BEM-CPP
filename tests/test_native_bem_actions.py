"""Real-process wire controls: complex column ordering and rejected receipts.

The tiny worker is an independent dense transport fixture, not a GPU oracle.
Actual FFT/FMM identities are covered by the queued native controls.
"""
import os
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from native_bem_actions import NativeBEMActions

WORKER = r'''
import json, os, sys
from pathlib import Path
import numpy as np
p=Path(sys.argv[1]); n=4; rows=np.array([3,1])
a=np.array([[complex(i+2*j+1,2*i-j+.3) for j in range(n)] for i in range(n)])
(p/'setup.json').write_text(json.dumps(dict(dofs=n,constraint_rows=2)))
print('READY',flush=True)
for line in sys.stdin:
 v=line.split()
 if v==['QUIT']:break
 name,ident,batch,src,dst=v;batch=int(batch)
 x=np.fromfile(src,complex).reshape(batch,-1).T
 if name=='A':y=a@x
 elif name=='AT':y=a.T@x
 elif name=='AH':y=a.conj().T@x
 elif name=='C':y=a[rows]@x
 elif name=='CH':y=a[rows].conj().T@x
 else:raise RuntimeError('bad fixture action')
 if os.environ.get('BAD_RESULT'):y=y[:-1]
 np.ascontiguousarray(y.T).tofile(dst)
 if os.environ.get('BAD_RECEIPT'):ident=str(int(ident)+1)
 print('DONE',ident,.001,0,batch,0,flush=True)
'''


class NativeBEMActionsTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.script=self.root/'worker.py';self.script.write_text(WORKER)
        self.env=dict(os.environ,OPENBLAS_NUM_THREADS='1')
        self.a=np.array([[complex(i+2*j+1,2*i-j+.3) for j in range(4)] for i in range(4)])
    def tearDown(self):self.temp.cleanup()
    def start(self):
        folder=self.root/'native'
        return NativeBEMActions([sys.executable,self.script,folder],folder,env=self.env)
    def test_block_layout_and_selected_adjoint(self):
        with self.start() as native:
            x=np.arange(32).reshape(4,8)[:,::2]+1j*np.arange(16).reshape(4,4)/7
            for name,a in [('A',self.a),('AT',self.a.T),('AH',self.a.conj().T),('C',self.a[[3,1]])]:
                np.testing.assert_allclose(native.action(name,x),a@x,rtol=2e-15)
            y=x[:2];np.testing.assert_allclose(native.action('CH',y),self.a[[3,1]].conj().T@y,rtol=2e-15)
            self.assertEqual(len(native.calls),5)
            with self.assertRaises(ValueError):native.action('C',x[:2])
            with self.assertRaises(ValueError):native.action('C',np.full((4,1),np.nan))
        self.assertEqual(native.worker.returncode,0)
    def test_wrong_reply_is_not_accepted(self):
        self.env['BAD_RECEIPT']='1'
        with self.start() as native:
            with self.assertRaisesRegex(RuntimeError,'reply'):native.action('A',np.ones((4,2)))
            self.assertEqual(native.calls,[])
        self.assertEqual(native.worker.returncode,0)
    def test_incomplete_file_is_not_accepted(self):
        self.env['BAD_RESULT']='1'
        with self.start() as native:
            with self.assertRaisesRegex(RuntimeError,'incomplete'):native.action('C',np.ones((4,2)))
            self.assertEqual(native.calls,[])
        self.assertEqual(native.worker.returncode,0)


if __name__=='__main__':unittest.main()
