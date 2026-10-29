"""Validate the solver-only handover and continue ka=40 in a new frozen runtime."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
from tmatrix_checkpoint_migration import physical_parameters
bench=ROOT/'benchmarks/symmetry_solver_20261029'
source=ROOT/'runs/ka40_n1p3116_l0p532_aspect10over7_pipeline_20261029/main_ref6_L64'
out=ROOT/'runs/ka40_n1p3116_l0p532_aspect10over7_sectors_20261030'
binary=ROOT/'bin/muller_tmatrix_fmm_server'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for process in psutil.process_iter(['cmdline']):
 if any(Path(x).name in ('muller_tmatrix_fmm_server','muller_solver_fft_probe')for x in (process.info['cmdline']or [])):
  raise RuntimeError('GPU worker still running: '+str(process.pid))
stop=json.loads((bench/'checkpoint_stop.json').read_text());assert sha(source/'manifest.json')==stop['source_manifest_sha256']
assert len(list((source/'columns').glob('*.npy')))==stop['column_files']
control=json.loads((bench/'physical_source_control.json').read_text())['hashes']
after={name:sha(ROOT/name)for name in control};assert after==control
checks=[json.loads((ROOT/'runs/symmetry_solver_20261029'/name/'result.json').read_text())for name in ['worker_ref3_new','worker_ref6_new']]
assert all(c['passed'] and c['binary_sha256']==sha(binary)for c in checks)
old=json.loads((source/'manifest.json').read_text())
proof={'validated':True,'source_binary_digest':old['parameters']['binary_digest'],'target_binary_digest':sha(binary),
 'physical_parameters_sha256':hashlib.sha256(json.dumps(physical_parameters(old['parameters']),sort_keys=True).encode()+(source/'directions.bin').read_bytes()).hexdigest(),
 'physical_source_hashes_before':control,'physical_source_hashes_after':after,
 'maximum_field_relative_difference':max(max(c['farfield_relative_l2'])for c in checks),
 'maximum_strict_residual':max(c['maximum_strict_residual']for c in checks),
 'validation_runs':['runs/symmetry_solver_20261029/worker_ref3_old','runs/symmetry_solver_20261029/worker_ref3_new','runs/symmetry_solver_20261029/worker_ref6_new','runs/symmetry_solver_20261029/ref6_packet_l36_fast_symcirc'],
 'validation_scope':'Frozen original worker comparison at ref3; original-operator/control-solver packet comparison at ref6; full phase-code checks rerun by the migrated builder',
 'reuse_identical_solver_audits':False}
assert proof['maximum_field_relative_difference']<=1e-4 and proof['maximum_strict_residual']<=2e-6
proof_path=bench/'migration_proof.json';proof_path.write_text(json.dumps(proof,indent=2)+'\n')
cmd=[sys.executable,str(ROOT/'scripts/run_large_tmatrix.py'),'--out',str(out),'--ka','40','--ref','6','--order','64','--ri','1.3116','--tol','2e-6',
 '--prism-aspect',str(10/7),'--restart','32','--max-iters','1000','--wavelength','.532','--circulant-blend','.25',
 '--cutoff-orders','60','62','64','--mbj-nodes','50','--strict-interval','4','--incident-prefetch','--sector-gmres','--orbit-nodes','50',
 '--worker-binary',str(binary),'--solver-restart-from',str(source),'--solver-restart-proof',str(proof_path),'--prepare-only']
with (bench/'prepare_target.log').open('w')as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
base=out/'main_ref6_L64';base.mkdir(exist_ok=True)
cache=base/'operator.near'
if not cache.exists():os.link(source/'operator.near',cache)
resume=json.loads((out/'plan.json').read_text())['resume_command']
with (out/'supervisor.log').open('a')as log:
 process=subprocess.Popen(resume,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record={'supervisor_pid':process.pid,'launched_unix':time.time(),'source':str(source),'target':str(out),
 'worker_sha256':sha(binary),'source_columns':stop['column_files'],'source_packets':stop['committed_packets'],
 'proof_sha256':sha(proof_path),'new_coded_audits_required':True,'time_limit':None,'analytic_full_SO3':True}
(bench/'production_launched.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2),flush=True)
