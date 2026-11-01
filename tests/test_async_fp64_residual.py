import sys
import unittest
from pathlib import Path
from threading import Event
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from async_fp64_residual import AsyncFP64Residual,ResidualTicket


class AsyncResidualTests(unittest.TestCase):
    def test_snapshot_is_immutable_and_cannot_certify_newer_state(self):
        started=Event();release=Event()
        def action(x):
            started.set()
            if not release.wait(5):raise RuntimeError('test worker timeout')
            return 2*x
        x=np.ones((4,2),complex);b=2*x
        with AsyncFP64Residual(action,capacity=1) as checker:
            ticket=checker.submit(x,b,label='column0',version=3)
            self.assertTrue(started.wait(5))
            x[:]=9;b[:]=100
            try:
                with self.assertRaises(RuntimeError):checker.submit(x,b,label='overflow')
                with self.assertRaises(ValueError):checker.take(ResidualTicket(ticket.token,ticket.label,ticket.version))
            finally:release.set()
            verdict=checker.take(ticket)
            self.assertTrue(verdict.accepts(1e-12))
            np.testing.assert_array_equal(verdict.candidate,np.ones((4,2)))
            self.assertFalse(verdict.candidate.flags.writeable)
            self.assertEqual(verdict.ticket.version,3)
            with self.assertRaises(ValueError):checker.take(ticket)

    def test_columnwise_gate_catches_small_bad_rhs_and_zero_rhs(self):
        b=np.array([[1e8,1e-6,0],[1e8,0,0]],complex);x=b.copy();x[0,1]+=1e-7;x[0,2]=1
        with AsyncFP64Residual(lambda x:x.copy()) as checker:
            r=checker.take(checker.submit(x,b,label='mixed norms'))
            self.assertAlmostEqual(r.relative_by_column[1],.1)
            self.assertTrue(np.isinf(r.relative_by_column[2]))
            self.assertFalse(r.accepts(1e-5))

    def test_operator_errors_propagate(self):
        with AsyncFP64Residual(lambda x:x.astype(np.complex64)) as checker:
            ticket=checker.submit(np.ones((2,1)),np.ones((2,1)),label='wrong precision')
            with self.assertRaises(RuntimeError):checker.take(ticket)


if __name__=='__main__':unittest.main()
