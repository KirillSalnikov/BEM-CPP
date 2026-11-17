"""Dispatch, physical conventions and memory guards for the unified interface."""
import argparse
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import numpy as np
import pytest
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'scripts'))
from bem_cli import register,resolve,command
from scattering_frame import frame,mueller


def args(*extra):
    p=argparse.ArgumentParser();register(p.add_subparsers())
    return p.parse_args(['solve','--ka','3',*extra])


def test_complex_index_radius_and_scale():
    a=args('--index','1.3116+10i');a.ka=None;a.radius=.7
    p=resolve(a)
    assert p['index']==[1.3116,10.]
    assert p['ka']==pytest.approx(2*np.pi*.7/.532)
    assert p['radius_um']==pytest.approx(.7)
    assert not p['accuracy_certified']


@pytest.mark.parametrize('extra',[['--method','pmchwt','--task','average'],
    ['--method','dirac','--shape','tetra'],['--shape','sphere','--symmetry','d6h'],
    ['--task','average','--orientation','0','20','0'],['--index','nan'],
    ['--index','1-1i'],['--ram-gib','-1'],['--threads','0'],['--L','4'],
    ['--method','muller','--backend','dense']])
def test_reject_invalid_and_unsupported(extra):
    with pytest.raises(ValueError):resolve(args(*extra))


def test_bullet_does_not_invent_horizontal_mirror():
    p=resolve(args('--particle','2','2','1','--task','average'))
    assert p['symmetry_selected']=='c6v'
    cmd,_=command(p)
    assert '--mirror-pack' not in cmd
    assert '--c6-pack' in cmd
    with pytest.raises(ValueError):resolve(args('--particle','2','2','1','--symmetry','d6h'))


def test_explicit_none_and_universal_fixed_orientation():
    p=resolve(args('--task','fixed','--orientation','13','47','29','--symmetry','none'))
    c,e=command(p)
    assert '--plane-euler' in c and '--plane-only' in c and '--mirror-pack' not in c
    assert e['BEM_TMATRIX_SECTOR_GMRES']=='0'
    assert c[c.index('--particle-symmetry')+1]=='off'


def test_dirac_backend_cap_and_T_only():
    p=resolve(args('--method','dirac','--ka','40','--task','tmatrix','--ram-gib','48'))
    assert p['backend']=='fmm-sector' and p['ref']>=4
    c,e=command(p)
    assert '--no-average' in c and '--coarse-factorization' in c
    assert c[c.index('--coarse-aggregate-groups')+1]=='512'
    assert float(c[c.index('--kernel-budget-gib')+1])<=8
    assert e['OPENBLAS_NUM_THREADS']=='1'


def test_full_P1_profile_enables_complete_compressed_CPU_path():
    p=resolve(args('--method','dirac','--ka','40','--task','average',
                   '--dirac-preconditioner','full-p1','--ram-gib','48'))
    assert p['backend']=='fmm-sector' and p['global_patches']==4096
    c,e=command(p)
    assert '--coarse-eliminate-full-p1' in c and '--coarse-batch-correction' in c
    assert c[c.index('--coarse-global-storage')+1]=='fp32'
    assert c[c.index('--coarse-lu-threads')+1]=='16'
    assert c[c.index('--fmm-packet')+1]=='2'
    assert e['OPENBLAS_NUM_THREADS']=='1'
    with pytest.raises(ValueError):
        resolve(args('--method','dirac','--task','fixed','--dirac-preconditioner','full-p1'))


def test_disk_rejected_before_launch():
    with patch('bem_cli.shutil.disk_usage') as d:
        d.return_value.free=1
        with pytest.raises(ValueError,match='диска'):resolve(args())


def test_lab_body_frame_preserves_rayleigh_pattern_at_arbitrary_orientation():
    theta=np.linspace(0,180,37)
    r,n,et,ep=frame([17,58,32],theta)
    np.testing.assert_allclose(r@r.T,np.eye(3),atol=3e-16)
    # Isotropic electric dipoles: project both incident polarizations transverse
    # to the observation direction, using only body coordinates.
    fields=np.array([e-n*np.einsum('nc,c->n',n,e)[:,None] for e in r[:2]])
    got=mueller(fields,et,ep,1.)
    c=np.cos(np.radians(theta))
    np.testing.assert_allclose(got[:,0,0],.5*(1+c*c),atol=5e-16)
    np.testing.assert_allclose(got[:,0,1],.5*(c*c-1),atol=5e-16)
    np.testing.assert_allclose(got[:,2,2],c,atol=5e-16)


def test_mapped_T_conventions_and_bounded_conversion():
    sys.path.insert(0,str(R/'benchmarks/clifford_dielectric_20261115'))
    from matrix_storage import convert_projected,norm_squared
    rng=np.random.default_rng(14);q=77;phase=1j**np.arange(q)
    raw=rng.normal(size=(q,q))+1j*rng.normal(size=(q,q))
    with tempfile.TemporaryDirectory() as d:
        k=np.lib.format.open_memmap(Path(d)/'kernel.npy',mode='w+',dtype=complex,shape=raw.shape,fortran_order=True)
        k[:]=raw
        t=convert_projected(k,phase,3.,Path(d)/'t.npy')
        assert isinstance(t,np.memmap)
        np.testing.assert_allclose(t,3j*phase[:,None]*raw,rtol=1e-14,atol=1e-14)
        np.testing.assert_allclose(k,4*np.pi*raw*phase[None,:],rtol=1e-14,atol=1e-14)
        assert norm_squared(t)==pytest.approx(np.linalg.norm(3j*phase[:,None]*raw)**2,rel=1e-14)


def test_completed_continue_does_not_repeat_solves(tmp_path):
    import json
    from bem_cli import continue_run
    p=resolve(args('--method','dirac','--task','tmatrix'))
    (tmp_path/'calculation.json').write_text(json.dumps(p))
    (tmp_path/'result_summary.json').write_text(json.dumps({'state':'complete'}))
    (tmp_path/'result').mkdir()
    np.save(tmp_path/'result/tmatrix.npy',np.eye(6))
    with patch('bem_cli.run_plan',side_effect=AssertionError('must not rerun')):
        assert continue_run(argparse.Namespace(directory=tmp_path))==0


def test_CPU_supervisor_without_nvidia_and_resource_stop(tmp_path):
    from bounded_bem_run import run
    with patch('bounded_bem_run.shutil.which',return_value=None):
        completed=run([sys.executable,'-c','print("ok")'],tmp_path/'ok',
                      rss_limit_gib=1,available_gib=0,disk_reserve_gib=0)
        assert completed['state']=='complete'
        stopped=run([sys.executable,'-c','import time; time.sleep(20)'],tmp_path/'stop',
                    rss_limit_gib=.000001,available_gib=0,disk_reserve_gib=0)
        assert stopped['state']=='failed'
        assert stopped['reason']=='resource reserve reached (RAM/RSS/disk)'


def test_large_dirac_plan_uses_bounded_preconditioner_and_mapped_T():
    with patch('psutil.virtual_memory') as mem:
        mem.return_value.available=54*2**30
        p=resolve(args('--method','dirac','--task','average','--ka','80','--ref','7',
                       '--aspect','1.6495721976846451','--ram-gib','48'))
    assert p['L']==137 and p['batch_size']==1
    assert p['dirac_preconditioner']=='panel'
    c,_=command(p)
    assert c[c.index('--sector-coarse-basis')+1]=='0'
    assert float(c[c.index('--matrix-budget-gib')+1])>43
    assert 'iteration count' in p['preconditioner_note']


def test_bounded_npz_archive_roundtrip_and_page_release(tmp_path):
    sys.path.insert(0,str(R/'benchmarks/clifford_dielectric_20261115'))
    import matrix_storage as ms
    a=np.lib.format.open_memmap(tmp_path/'large.npy',mode='w+',dtype=complex,shape=(1024,5000),fortran_order=True)
    a[:]=np.arange(1024)[:,None]+1j*np.arange(5000)[None,:]
    calls=[];original=ms.release_pages
    def release(value):
        calls.append(value.shape);original(value)
    with patch.object(ms,'release_pages',side_effect=release):
        with (tmp_path/'solution.npz').open('wb') as stream:ms.savez_bounded(stream,T=a,modes=[(0,1,0)])
    assert calls.count(a.shape)>=2
    with np.load(tmp_path/'solution.npz') as saved:
        np.testing.assert_array_equal(saved['T'],a)
        np.testing.assert_array_equal(saved['modes'],[(0,1,0)])
