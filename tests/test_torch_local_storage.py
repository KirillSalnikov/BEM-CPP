"""CSR storage must preserve the complex near/mass action, including repeats."""
import sys
import unittest
import warnings
from pathlib import Path
import numpy as np
import scipy.sparse as sp
try:
    import torch
except ImportError:
    torch=None
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from symmetric_galerkin_fft import torch_local_matrix


@unittest.skipUnless(torch is not None,'Optional PyTorch dependency is not installed')
class LocalStorageTests(unittest.TestCase):
    def test_complex_duplicates_empty_rows_and_storage_bytes(self):
        a = sp.coo_matrix((np.array([1+2j, 3-1j, -1j, -2+4j]),
                           ([0, 0, 0, 3], [2, 2, 1, 3])), shape=(5, 5))
        x = np.array([1j, 2., -3j, 4-2j, 8.])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            for storage in ('coo', 'csr'):
                operator, size = torch_local_matrix(a, 'cpu', torch.complex128, storage)
                actual = operator @ torch.as_tensor(x)
                np.testing.assert_allclose(actual.numpy(), a@x, atol=1e-14)
                arrays = ((operator.crow_indices(), operator.col_indices(), operator.values())
                          if storage == 'csr' else (operator.indices(), operator.values()))
                self.assertEqual(size, sum(v.numel()*v.element_size() for v in arrays))
                if storage == 'csr':
                    self.assertEqual(operator.crow_indices().dtype, torch.int32)
                    self.assertEqual(operator.col_indices().dtype, torch.int32)

    def test_unknown_format_is_rejected(self):
        with self.assertRaises(ValueError):
            torch_local_matrix(sp.eye(2), 'cpu', torch.complex64, 'unknown')


if __name__ == '__main__':
    if torch is not None:torch.set_num_threads(1)
    unittest.main()
