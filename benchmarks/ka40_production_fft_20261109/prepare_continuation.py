"""Record measured pilot equivalence and freeze the ka40 continuation."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parents[2]
base=Path(__file__).resolve().parent
target=root/'runs/scice_ka40_compressed_20261109'
source=root/'runs/scice_column_ka20_40_n13116_20261104/direct/node004_ka40.00000000_ref6_L65'
sys.path.insert(0,str(root/'scripts'))
from tmatrix_checkpoint_migration import physical_parameters
from tmatrix_reference import transverse_harmonics
from run_large_tmatrix import frozen_runtime
from compressed_fft_service import validate_manifest

pilot=json.loads((base/'pilot_comparison.json').read_text())
assert pilot['status']=='complete'
runtime=target/'runtime_v2'
frozen_runtime(runtime,worker_binary=base/'bin/muller_tmatrix_fmm_server')
validate_manifest(target/'assets')
old=json.loads((source/'manifest.json').read_text())
before=json.loads((base/'physical_sources_before.json').read_text())
after={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in before}
assert before==after,'Physical source changed during integration'
rng=np.random.default_rng(20261109)
directions=rng.normal(size=(256,3));directions/=np.linalg.norm(directions,axis=1)[:,None]
theta=np.linspace(0,np.pi,181)
directions=np.vstack((directions,np.c_[np.sin(theta),np.zeros_like(theta),np.cos(theta)]))
basis=transverse_harmonics(directions,65)
fields=[]
for row in pilot['columns']:
    name=f'{row["column"]:06d}.npy'
    reference=basis@np.load(source/'columns'/name)
    candidate=basis@np.load(base/'pilot/columns'/name)
    difference=candidate-reference
    fields.append(dict(column=row['column'],
        relative_l2=float(np.linalg.norm(difference)/np.linalg.norm(reference)),
        maximum_point_difference_over_peak=float(np.max(np.linalg.norm(difference.reshape(-1,2),axis=1))/
                                                  np.max(np.linalg.norm(reference.reshape(-1,2),axis=1)))))
maximum=max(row['relative_l2'] for row in fields)
strict=max(pilot['solution']['strict_residual_x'],pilot['solution']['strict_residual_y'])
assert maximum<1e-4 and strict<2e-6
proof=dict(source_binary_digest=old['parameters']['binary_digest'],
    target_binary_digest=hashlib.sha256((runtime/'bin/muller_tmatrix_fmm_server').read_bytes()).hexdigest(),
    physical_parameters_sha256=hashlib.sha256(json.dumps(physical_parameters(old['parameters']),sort_keys=True).encode()+
                                             (source/'directions.bin').read_bytes()).hexdigest(),
    validated=True,physical_source_hashes_before=before,physical_source_hashes_after=after,
    maximum_field_relative_difference=maximum,maximum_strict_residual=strict,
    equivalence_kind='measured_native_sector_T_columns_and_reconstructed_fields',
    sample_directions=len(directions),fields=fields,
    sampling_is_not_an_operator_norm_bound=True,
    physical_hash_scope='Before and after native FFT integration; DS implementation was independently audited earlier.',
    independent_DS_audit=str(root/'benchmarks/ka40_symmetric_fft_20261108/integrated_ds_checks.json'),
    pilot_comparison_sha256=hashlib.sha256((base/'pilot_comparison.json').read_bytes()).hexdigest(),
    reuse_identical_solver_audits=False)
(base/'migration_proof.json').write_text(json.dumps(proof,indent=2)+'\n')
(target/'preparation.json').write_text(json.dumps(dict(state='validated_ready',runtime=str(runtime),
    source=str(source),case=str(target/'assets'),proof=str(base/'migration_proof.json'),
    retained_source_columns=len(list((source/'columns').glob('*.npy')))),indent=2)+'\n')
print(json.dumps(dict(maximum_field_relative_difference=maximum,maximum_strict_residual=strict,
                     runtime=str(runtime),fields=fields),indent=2))
