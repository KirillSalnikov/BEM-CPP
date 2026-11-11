"""Prepare a C6v restart; retain the original worker, mesh, basis and columns.

The campaign must be stopped. This script never starts a GPU calculation.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import numpy as np

B=Path(__file__).resolve().parent;R=B.parents[1]
sys.path.insert(0,str(R/'scripts'))
from tmatrix_checkpoint_migration import physical_parameters

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)

def main():
    base=R/'runs/scice_regular_shapes_ka20_20261111'
    source=base/'Bullet/bem';target=base/'Bullet/bem_c6v'
    old=json.loads((source/'manifest.json').read_text());params=old['parameters']
    pid=json.loads((source/'progress.json').read_text())['pid']
    try:os.kill(pid,0)
    except ProcessLookupError:pass
    else:raise RuntimeError('Stop the source builder before preparing this migration')
    runtime=base/'runtime_c6v'
    assert not runtime.exists(), 'Preserve the existing frozen runtime'
    shutil.copytree(base/'runtime',runtime,ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('build_fmm_tmatrix.py','tmatrix_symmetry.py','tmatrix_order_shell.py',
                 'tmatrix_checkpoint_migration.py','complete_tmatrix_order_shell.py'):
        shutil.copy2(R/'scripts'/name,runtime/'scripts'/name)
    hashes={str(p.relative_to(runtime)):sha(p) for p in runtime.rglob('*') if p.is_file() and p.name!='sha256.json'}
    save(runtime/'sha256.json',hashes)
    before={name:sha(base/'runtime'/name) for name in ('bin/muller_tmatrix_fmm_server',
        'scripts/tmatrix_incident.py','scripts/tmatrix_reference.py','scripts/tmatrix_projection.py','scripts/tmatrix_c6.py')}
    after={name:sha(runtime/name) for name in before};assert before==after
    assert before['bin/muller_tmatrix_fmm_server']==params['binary_digest']
    residual=max(max(d['status']['strict_residual_x'],d['status']['strict_residual_y'])
                 for p in (source/'columns').glob('*.json') for d in [json.loads(p.read_text())])
    assert np.isfinite(residual) and residual<=params['tol']
    proof=dict(source_binary_digest=params['binary_digest'],target_binary_digest=params['binary_digest'],
        physical_parameters_sha256=hashlib.sha256(json.dumps(physical_parameters(params),sort_keys=True).encode()
            +(source/'directions.bin').read_bytes()).hexdigest(),validated=True,
        physical_source_hashes_before=before,physical_source_hashes_after=after,
        maximum_field_relative_difference=0.,maximum_strict_residual=residual,
        equivalence_kind='identical_worker_and_incident_inputs',identical_incident_fields=True,
        reuse_identical_solver_audits=True,
        scope='Only the Python negative-m schedule changes. New vertical covariance audits remain mandatory.')
    proof_path=B/'migration_proof.json';save(proof_path,proof)
    config=json.loads((base/'manifest.json').read_text());save(B/'campaign_before.json',config)
    case=next(c for c in config['cases'] if c['name']=='Bullet')
    command=[s.replace(str(base/'runtime'),str(runtime)) for s in case['bem_command']]
    command[command.index('--out')+1]=str(target)
    command+=['--solver-restart-from',str(source),'--solver-restart-proof',str(proof_path),
              '--operator-cache-prefix',str(source/'operator')]
    target.mkdir(exist_ok=True)
    save(B/'restart_plan.json',dict(source=str(source),target=str(target),runtime=str(runtime),
        command=command,environment=case['bem_environment'],retained_columns=len(list((source/'columns').glob('*.json')))))
    print('Prepared',target)

if __name__=='__main__':main()
