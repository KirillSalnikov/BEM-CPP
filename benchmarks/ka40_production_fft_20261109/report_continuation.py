"""Read-only comparison of completed ka40 native packets and progress."""
import json
import statistics
import time
from pathlib import Path

root=Path(__file__).resolve().parents[2]
base=Path(__file__).resolve().parent
old=root/'runs/scice_column_ka20_40_n13116_20261104/direct/node004_ka40.00000000_ref6_L65'
new=root/'runs/scice_ka40_compressed_20261109/solution'
def read(path):
    return json.loads(path.read_text()) if path.exists() else {}
audits=[]
for path in sorted((new/'c6_audits').glob('*.json')):
    actual=read(path);reference=read(old/'c6_audits'/path.name)
    a,b=reference['status'],actual['status']
    assert actual['columns']==reference['columns']
    audits.append(dict(packet=path.stem,passed=actual['passed'],
        same_columns=True,same_requested_tolerance=a['requested_tolerance']==b['requested_tolerance'],
        old_worker_s=a['total_worker_s'],new_worker_s=b['total_worker_s'],
        worker_speedup=a['total_worker_s']/b['total_worker_s'],
        old_iterations=[a['host_iterations_x'],a['host_iterations_y']],
        new_iterations=[b['host_iterations_x'],b['host_iterations_y']],
        krylov_action_speedup=a['host_krylov_action_s']/b['host_krylov_action_s'],
        strict_check_speedup=a['host_strict_s']/b['host_strict_s'],
        maximum_column_difference=max(row['relative_column_error'] for row in
            actual['independent_reference_check']['columns'])))
packets=[]
for path in (new/'c6_batches').glob('*.json'):
    row=read(path)
    if 'solver_migration_source' in row:continue
    status=row['status']
    packets.append(dict(packet=path.stem,mtime=path.stat().st_mtime,
        columns=row['columns'],passed=row['passed'],worker_s=status['total_worker_s'],
        iterations=status['iterations'],strict_residual=max(status['strict_residual_x'],status['strict_residual_y']),
        host_krylov_action_s=status['host_krylov_action_s'],host_strict_s=status['host_strict_s']))
packets.sort(key=lambda row:row['mtime'])
progress=read(new/'progress.json')
report=dict(time=time.time(),state=progress.get('state'),
    saved_columns=len(list((new/'columns').glob('*.npy'))),
    remaining_worker_pairs=progress.get('remaining_worker_pairs',progress.get('planned_worker_pairs')),
    audits=audits,new_packets=packets)
if audits:
    report['matched_audit_worker_speedup']=sum(row['old_worker_s'] for row in audits)/sum(row['new_worker_s'] for row in audits)
if packets:
    recent=packets[-8:]
    report['recent_packet_median_s']=statistics.median(row['worker_s'] for row in recent)
    report['remaining_s_at_current_packet_rate']=report['remaining_worker_pairs']*report['recent_packet_median_s']
    report['estimate_caveat']='Extrapolation from current degrees only; excludes final T assembly/SO(3), higher degrees may change convergence.'
(base/'continuation_status.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
