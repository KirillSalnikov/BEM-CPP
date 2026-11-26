import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from tmatrix_final_assembly import assemble


def test_streamed_t_and_three_holdout_directions(tmp_path):
    rng=np.random.default_rng(2591)
    def z(shape):
        return rng.normal(size=shape)+1j*rng.normal(size=shape)
    n=29
    kernel=z((n,n))
    basis=z((20,n))
    phase=(1j)**np.arange(n)
    observed=[basis@kernel@basis[2*i:2*i+2].conj().T+0.01*z((20,2)) for i in range(3)]
    columns=tmp_path/'columns'
    columns.mkdir()
    for j in range(n):np.save(columns/f'{j:06d}.npy',kernel[:,j])
    err,power,ext=assemble(tmp_path,columns,phase,80.,basis,observed,block_columns=7)
    expected=-1j*80/(4*np.pi)*phase[:,None]*kernel*phase.conj()[None,:]
    for name,ref in [('tmatrix',expected),('kernel',kernel)]:
        mapped=np.load(tmp_path/f'{name}.npy',mmap_mode='r')
        assert mapped.flags.f_contiguous
        np.testing.assert_array_equal(mapped,ref)
    exact=np.column_stack([basis@kernel@basis[2*i:2*i+2].conj().T for i in range(3)])
    fields=np.column_stack(observed)
    np.testing.assert_allclose(err,np.linalg.norm(exact-fields)/np.linalg.norm(fields),rtol=1e-11)
    np.testing.assert_allclose(power,np.vdot(expected,expected).real)
    np.testing.assert_allclose(ext,-np.trace(expected).real)
