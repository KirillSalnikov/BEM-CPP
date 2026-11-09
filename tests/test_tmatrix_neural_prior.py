import hashlib
import json
from pathlib import Path
import struct
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from tmatrix_neural_prior import configure, validate_worker


def prior(tmp_path):
    path=tmp_path/'neural.bin'
    raw=struct.pack('<16s6Q4d',b'BEM_PHYS_RITZ1',24,2,0,2,7,12,3.,1.3116,0.,10/7)+bytes(24*2*16)
    path.write_bytes(raw)
    op=dict(dofs=24,ref=2,quad=7,duffy=12,ka=3.,ri_real=1.3116,ri_imag=0.,aspect=10/7,
            geometry='regular_prism_D6h',basis='HDivBdm1')
    path.with_suffix('.bin.json').write_text(json.dumps(dict(kind='tmatrix_neural_prior_v1',
        prior_sha256=hashlib.sha256(raw).hexdigest(),operator=op,rank=2,provenance={})))
    return path


def settings(**kw):
    return dict(dict(shape='prism',obj=None,ref=2,ka=3.,ri=1.3116,ri_imag=0.,aspect=10/7),**kw)


def test_neural_prior_enables_original_checks_and_full_symmetry(tmp_path):
    env={};info=configure(prior(tmp_path),environment=env,**settings())
    assert info['rank']==2
    assert env['BEM_TMATRIX_VERIFY_RESIDUAL']=='1'
    assert env['BEM_TMATRIX_RITZ_IMAGE_MODEL']=='strict'
    assert env['BEM_TMATRIX_VERTICAL_PRECONDITIONER']=='1'
    assert env['BEM_TMATRIX_D6H_MESH']=='1'


@pytest.mark.parametrize('change',[{'ka':4.},{'ri_imag':10.},{'ref':3},{'aspect':.7},
                                  {'shape':'sphere'},{'obj':Path('external.obj')}])
def test_wrong_operator_rejected_before_launch(tmp_path,change):
    options=settings();options.update(change)
    with pytest.raises(ValueError):configure(prior(tmp_path),environment={},**options)


def test_mutation_and_conflicting_settings_rejected(tmp_path):
    path=prior(tmp_path)
    with pytest.raises(ValueError):configure(path,environment={'BEM_TMATRIX_GPU_BATCH2':'1'},**settings())
    with path.open('ab') as f:f.write(b'x')
    with pytest.raises(ValueError):configure(path,environment={},**settings())


def test_older_worker_cannot_silently_ignore_neural_sector_floor(tmp_path):
    info=configure(prior(tmp_path),environment={},**settings())
    hello=dict(system_dofs=24,physical_ritz_columns=2,physical_ritz_image_model='strict',
               native_full_fft_symmetry=True,prism_dihedral_mesh=True,vertical_preconditioner=True,
               physical_ritz_D6h_defect=1e-13)
    with pytest.raises(RuntimeError):validate_worker(info,hello)
    hello['physical_ritz_sector_floor']=0.
    validate_worker(info,hello)
    hello['physical_ritz_D6h_defect']=1e-3
    with pytest.raises(RuntimeError):validate_worker(info,hello)
