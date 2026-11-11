"""Audited solver-only migration; preserve packed receipts and rerun coded audits."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import numpy as np
from tmatrix_symmetry import c2x_plan, c2x_partner_column, vertical_mirror_plan

SOLVER_ENV = {'BEM_TMATRIX_WAVE_RANK','BEM_TMATRIX_MBJ_OVERLAP','BEM_TMATRIX_MBJ_NODES','BEM_TMATRIX_FFT_INNER',
              'BEM_TMATRIX_CIRCULANT_BLEND','BEM_CIRCULANT_SHIFT','BEM_TMATRIX_FFT_RADIUS','BEM_PFFT_CORR_MEMORY',
              'BEM_HOST_GMRES_STRICT_INTERVAL','BEM_TMATRIX_SECTOR_GMRES','BEM_TMATRIX_ORBIT_NODES',
              'BEM_TMATRIX_SECTOR_RECYCLE_RANK','BEM_TMATRIX_SECTOR_RECYCLE_MASK',
              'BEM_TMATRIX_VERTICAL_PRECONDITIONER','BEM_SECTOR_PROJECTED_FACTOR',
              'BEM_SECTOR_FIRST_PASS_FACTOR',
              'BEM_CIRCULANT_OPPOSITE_WEIGHT','BEM_CIRCULANT_OPPOSITE_MASK',
              # Only the FP32 near kernel reads this; the strict action is unchanged.
              'BEM_FMM_P2P_LOCAL_COORDINATES',
              # The compressed inner operator retains the accurate native
              # residual gate; DS accuracy must be covered by the migration
              # proof. Neither option permits reusing identical-solver audits.
              'BEM_TMATRIX_COMPRESSED_FFT_CASE','BEM_FMM_P2P_DOUBLE_SINGLE',
              'BEM_FMM_P2P_DS_THREADS','BEM_FMM_P2P_DS_LEAF_SPLIT','BEM_FMM_P2P_DS_FACTORED',
              'BEM_TMATRIX_CHECKER_COMPARE','BEM_TMATRIX_COMPRESSED_FFT_WIRE_FP32'}
LOCATION_ENV = {'BEM_SO3_TMPDIR','BEM_SO3_BINARY_FP32','BEM_SO3_BINARY_FP64'}

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def atomic_json(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)

def physical_parameters(parameters):
    # Restart changes the Krylov basis length, not the equation or its
    # acceptance tolerance. A validated solver migration still repeats audits.
    result={k:v for k,v in parameters.items() if k not in ('binary_digest','c6_driver_digest','incident_schedule_digest','restart',
                                                        'compressed_fft_assets_digest','compressed_fft_code_digest')}
    result['BEM_environment']={k:v for k,v in result.get('BEM_environment',{}).items()
                               if k not in SOLVER_ENV | LOCATION_ENV}
    return without_vertical_schedule(result)

def without_vertical_schedule(parameters):
    """C6v column reconstruction changes scheduling, not any solved equation.

    The builder independently validates geometry and audits the new relation.
    Keep the existing C2 fingerprint; never allow a C2/C6v substitution.
    """
    result=dict(parameters)
    result.pop('vertical_reuse',None);result.pop('vertical_audit_tol',None)
    if not result.get('c2_reuse'):
        result.pop('symmetry_digest',None)
    return result

def identical_solver_parameters(parameters):
    result={k:v for k,v in parameters.items() if k not in ('c6_driver_digest','incident_schedule_digest')}
    result['BEM_environment']={k:v for k,v in result.get('BEM_environment',{}).items() if k not in LOCATION_ENV}
    return without_vertical_schedule(result)

def migrate(source,destination,parameters,directions,modal,tolerance,signature,proof_path):
    source,destination=Path(source).resolve(),Path(destination).resolve()
    if source==destination:raise ValueError('solver restart requires a new destination')
    old=json.loads((source/'manifest.json').read_text())
    direction_bytes=(source/'directions.bin').read_bytes()
    old_signature=hashlib.sha256(json.dumps(old['parameters'],sort_keys=True).encode()+direction_bytes).hexdigest()
    if old_signature!=old['signature']:raise ValueError('source manifest checksum mismatch')
    if (physical_parameters(old['parameters'])!=physical_parameters(parameters)
            or direction_bytes!=directions.tobytes() or old['modes']!=[list(m) for m in modal]):
        raise ValueError('solver migration changes the physical operator, projection, or numerical gates')
    proof=json.loads(Path(proof_path).read_text());before=proof.get('physical_source_hashes_before',{})
    physical_signature=hashlib.sha256(json.dumps(physical_parameters(parameters),sort_keys=True).encode()+direction_bytes).hexdigest()
    if (proof.get('source_binary_digest')!=old['parameters']['binary_digest']
            or proof.get('target_binary_digest')!=parameters['binary_digest']
            or proof.get('physical_parameters_sha256')!=physical_signature
            or proof.get('validated') is not True or not before
            or before!=proof.get('physical_source_hashes_after')
            or not 0<=proof.get('maximum_field_relative_difference',float('inf'))<=1e-4
            or not 0<=proof.get('maximum_strict_residual',float('inf'))<=tolerance):
        raise ValueError('missing or invalid solver-change validation proof')
    reuse_audits=proof.get('reuse_identical_solver_audits',False)
    if reuse_audits and (proof.get('equivalence_kind')!='identical_worker_and_incident_inputs'
            or proof.get('identical_incident_fields') is not True
            or proof.get('maximum_field_relative_difference')!=0
            or identical_solver_parameters(old['parameters'])!=identical_solver_parameters(parameters)):
        raise ValueError('audit reuse requires identical worker, all solver settings and incident inputs')
    report_path=destination/'solver_migration.json'
    if report_path.exists():
        report=json.loads(report_path.read_text())
        if report['source_signature']!=old_signature or report['target_signature']!=signature or report['proof_sha256']!=digest(proof_path):
            raise ValueError('solver migration source/proof changed')
        for name,expected in report['copied_data_sha256'].items():
            if digest(destination/'columns'/name)!=expected:raise ValueError('migrated column changed')
        return report
    progress=source/'progress.json'
    if progress.exists():
        pid=json.loads(progress.read_text()).get('pid')
        if pid:
            try:os.kill(pid,0)
            except ProcessLookupError:pass
            else:raise ValueError('stop source builder at a committed checkpoint before migration')
    files={int(p.stem):p for p in (source/'columns').glob('*.json')}
    count=len(modal)
    old_vertical=old['parameters'].get('vertical_reuse',False)
    permutation,signs,_,_=(vertical_mirror_plan if old_vertical else c2x_plan)(max(m[1] for m in modal))
    accepted=set();receipts={};visiting=set()
    def validate(index):
        if index in accepted:return True
        if index in visiting:raise ValueError('cyclic checkpoint provenance')
        if index not in files:return False
        visiting.add(index)
        try:
            meta=json.loads(files[index].read_text());path=files[index].with_suffix('.npy')
            if not path.exists():return False
            data=np.load(path,allow_pickle=False);s=meta['status']
            residuals=[s['strict_residual_x'],s['strict_residual_y']]
            if (not 0<=index<count or meta['mode']!=list(modal[index]) or data.shape!=(count,)
                    or not np.isfinite(data).all() or not np.isfinite(residuals).all() or max(residuals)>tolerance):
                raise ValueError('invalid source column '+str(index))
            if 'c6_batch' in meta:
                batch=meta['c6_batch']
                if not isinstance(batch,str) or len(batch)!=24 or any(c not in '0123456789abcdef' for c in batch):
                    raise ValueError('invalid packed receipt identifier')
                path_receipt=source/'c6_batches'/f'{batch}.json'
                if not path_receipt.exists():return False
                receipt=json.loads(path_receipt.read_text());origin=meta.get('source_column',index)
                if not receipt.get('passed') or receipt['signature']!=old_signature or origin not in receipt['columns']:
                    raise ValueError('invalid source packet receipt')
                if origin==index and digest(path)!=receipt['column_sha256'].get(str(index)):
                    raise ValueError('source packed column checksum mismatch')
                receipts[batch]=receipt
            derived=meta.get('derived_by')
            if derived in ('C2x_half_turn','sigma_v_xz'):
                flag='vertical_reuse' if derived=='sigma_v_xz' else 'c2_reuse'
                if not old['parameters'].get(flag) or not parameters.get(flag):
                    raise ValueError('source reconstruction requires the same target symmetry')
                origin=meta['source_column']
                if (not isinstance(origin,int) or not 0<=origin<count or modal[origin][2]<=0
                        or int(permutation[origin])!=index or not validate(origin)):return False
                parent=json.loads(files[origin].read_text())
                if parent.get('c6_batch')!=meta.get('c6_batch'):raise ValueError('C2 packed provenance mismatch')
                original=np.load(files[origin].with_suffix('.npy'),allow_pickle=False)
                if not np.array_equal(data,c2x_partner_column(original,origin,permutation,signs)):
                    raise ValueError('source C2 reconstruction mismatch')
            elif derived not in (None,'C6_packed_RHS'):raise ValueError('unsupported checkpoint provenance')
            if derived=='C6_packed_RHS' and 'c6_batch' not in meta:raise ValueError('packed column without receipt')
            accepted.add(index);return True
        finally:visiting.remove(index)
    for index in sorted(files):validate(index)
    for receipt in list(receipts.values()):
        if not all(validate(i) for i in receipt['columns']):raise ValueError('incomplete committed source packet')
    if receipts:
        audits=list((source/'c6_audits').glob('*.json'))
        if not audits or not all(json.loads(p.read_text()).get('passed') for p in audits):
            raise ValueError('source coded symmetry audits missing or failed')
    for method,name in [('C2x_half_turn','c2_audit.json'),('sigma_v_xz','vertical_audit.json')]:
        if any(json.loads(files[i].read_text()).get('derived_by')==method for i in accepted):
            if not json.loads((source/name).read_text()).get('passed'):
                raise ValueError('source symmetry audit missing or failed: '+method)
    columns=destination/'columns';columns.mkdir(exist_ok=True)
    batches=destination/'c6_batches';batches.mkdir(exist_ok=True)
    hashes={};provenance={'directory':str(source),'signature':old_signature,'proof_sha256':digest(proof_path)}
    for index in sorted(accepted):
        old_meta=files[index];old_data=old_meta.with_suffix('.npy');target=columns/old_data.name
        if target.exists() and digest(target)!=digest(old_data):raise ValueError('destination has conflicting column')
        tmp=target.with_suffix('.npy.tmp');shutil.copyfile(old_data,tmp);tmp.replace(target)
        hashes[target.name]=digest(target)
        meta=json.loads(old_meta.read_text());meta['solver_migration_source']=provenance
        atomic_json(columns/old_meta.name,meta)
    for batch,receipt in receipts.items():
        atomic_json(batches/f'{batch}.json',dict(receipt,signature=signature,solver_migration_source=provenance))
    # A changed solver repeats coded audits. Pure scheduling changes may retain
    # them only with identical executable/settings/incident formulas and an
    # explicit input-equivalence proof. Never treat this as a fresh measurement.
    archive=destination/'solver_migration_source_audits';archive.mkdir(exist_ok=True)
    for path in list((source/'c6_audits').glob('*.json'))+list(source.glob('c2_audit.json'))+list(source.glob('vertical_audit.json')):
        shutil.copyfile(path,archive/path.name)
        if reuse_audits and path.parent.name=='c6_audits':
            audit=json.loads(path.read_text())
            if audit.get('signature')!=old_signature or not all(i in accepted for i in audit['columns']):
                raise ValueError('invalid source coded audit provenance')
            target=destination/'c6_audits';target.mkdir(exist_ok=True)
            atomic_json(target/path.name,dict(audit,signature=signature,solver_migration_source=provenance,
                reused_for_identical_solver_and_inputs=True,remeasured=False))
    # Carry only a conservative tolerance policy, never a passed audit. The
    # unchanged tight/loose tolerances are part of the physical fingerprint.
    # This avoids relearning a known failed loose tolerance after a restart.
    inherited_degrees=[]
    if parameters.get('adaptive_tight_tol') is not None:
        policy_sources=[]
        for path in (source/'adaptive_tolerance').glob('*.json'):
            prior=json.loads(path.read_text())
            degree=prior.get('tight_from_degree')
            if (prior.get('signature')==old_signature and type(degree) is int
                    and 1<=degree<=max(m[1] for m in modal)):
                inherited_degrees.append(degree)
                policy_sources.append({'path':str(path),'sha256':digest(path)})
        if inherited_degrees:
            target=destination/'adaptive_tolerance';target.mkdir(exist_ok=True)
            atomic_json(target/'solver_migration_policy.json',dict(
                signature=signature,tight_from_degree=min(inherited_degrees),
                solver_migration_source=provenance,source_policies=policy_sources,
                policy_only=True,remeasured=False))
    report={'source':str(source),'source_signature':old_signature,'target_signature':signature,
            'proof_sha256':digest(proof_path),'copied_columns':len(accepted),'copied_packets':len(receipts),
            'skipped_uncommitted_columns':len(files)-len(accepted),'copied_data_sha256':hashes,
            'new_coded_audits_required':not reuse_audits,'old_residuals_are_retained_not_remeasured':True,
            'inherited_tight_from_degree':min(inherited_degrees) if inherited_degrees else None,
            'identical_solver_coded_audits_reused':bool(reuse_audits)}
    atomic_json(report_path,report);return report
