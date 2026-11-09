"""Validate a fixed neural current prior before starting a T-matrix worker."""
import hashlib
import json
import math
import os
from pathlib import Path
import struct


def configure(path, *, shape, obj, ref, ka, ri, ri_imag, aspect, environment=None):
    env = os.environ if environment is None else environment
    path = Path(path).resolve()
    manifest_path = path.with_suffix(path.suffix+'.json')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('kind') != 'tmatrix_neural_prior_v1':
        raise ValueError('Expected a trained T-matrix physical-prior manifest')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        stream.seek(0); header = stream.read(96)
    if digest != manifest['prior_sha256'] or len(header) != 96:
        raise ValueError('Neural prior changed or is incomplete')
    magic, n, rank, checksum, file_ref, quad, duffy, file_ka, re, im, file_aspect = struct.unpack('<16s6Q4d', header)
    if magic != b'BEM_PHYS_RITZ1'.ljust(16,b'\0') or n < 1 or not 1 <= rank <= 64 or path.stat().st_size != 96+16*n*rank:
        raise ValueError('Invalid physical-current prior header')
    if shape != 'prism' or obj is not None or not 2 <= ref <= 8 or quad != 7 or not 2 <= duffy <= 12:
        raise ValueError('Neural T prior requires the canonical D6h BDM prism')
    expected = dict(dofs=n, ref=ref, quad=quad, duffy=duffy, ka=ka, ri_real=ri, ri_imag=ri_imag,
                    aspect=aspect, geometry='regular_prism_D6h', basis='HDivBdm1')
    operator = manifest['operator']
    if any(operator.get(k) != v for k, v in expected.items()) or \
            (file_ref, file_ka, re, im, file_aspect, rank) != (ref, ka, ri, ri_imag, aspect, manifest['rank']):
        raise ValueError('Neural prior belongs to a different operator')
    floor=manifest.get('required_sector_floor',0.)
    if not isinstance(floor,(int,float)) or not math.isfinite(floor) or not 0<=floor<=.01:
        raise ValueError('Invalid neural sector floor')
    image_model=manifest.get('required_image_model','strict')
    if image_model not in ('strict','d6h_strict'):
        raise ValueError('Neural prior requires original FP64 image construction')
    controls = dict(BEM_TMATRIX_RITZ_PRIOR=str(path), BEM_TMATRIX_RITZ_IMAGE_MODEL=image_model,
                    BEM_TMATRIX_RITZ_SECTOR_FLOOR=str(floor),
                    BEM_TMATRIX_D6H_MESH='1', BEM_TMATRIX_SECTOR_GMRES='1',
                    BEM_TMATRIX_VERTICAL_PRECONDITIONER='1', BEM_TMATRIX_FACE_FFT='1',
                    BEM_TMATRIX_DUFFY_ORDER=str(duffy), BEM_TMATRIX_GPU_BATCH2='0',
                    BEM_TMATRIX_CIRCULANT_BLEND='.5', BEM_TMATRIX_VERIFY_RESIDUAL='1')
    for key, value in controls.items():
        if key in env and env[key] != value:
            if key in ('BEM_TMATRIX_CIRCULANT_BLEND','BEM_TMATRIX_RITZ_SECTOR_FLOOR'):
                try:
                    if float(env[key]) == float(value):continue
                except ValueError:pass
            raise ValueError('Conflicting neural T-worker setting: '+key)
    env.update(controls)
    return dict(prior_sha256=digest, manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                rank=rank, sector_floor=floor, image_model=image_model, operator=expected, provenance=manifest['provenance'])


def validate_worker(info, hello):
    if hello.get('system_dofs') != info['operator']['dofs'] or hello.get('physical_ritz_columns') != info['rank'] or \
            hello.get('physical_ritz_sector_floor') != info['sector_floor'] or \
            hello.get('physical_ritz_image_model') != info['image_model']:
        raise RuntimeError('Worker did not load the requested neural prior/sector guard')
    if not hello.get('native_full_fft_symmetry') or not hello.get('prism_dihedral_mesh') or \
            not hello.get('vertical_preconditioner'):
        raise RuntimeError('Neural T prior needs full rotation and mirror symmetry')
    error=hello.get('physical_ritz_D6h_defect',float('inf'))
    if not math.isfinite(error) or not 0<=error<=1e-8:
        raise RuntimeError('Neural inverse symmetry audit failed')
