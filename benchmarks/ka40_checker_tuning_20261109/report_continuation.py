"""Matched production audits and checkpoint-preserving continuation report."""
import json,statistics,time
from pathlib import Path
root=Path(__file__).resolve().parents[2];base=Path(__file__).resolve().parent
old=root/'runs/scice_ka40_compressed_20261109/solution'
new=root/'runs/scice_ka40_checker_tuned_20261109/solution'
def read(path):return json.loads(path.read_text()) if path.exists() else {}
audits=[]
for path in sorted((new/'c6_audits').glob('*.json')):
    found=read(path);before=read(old/'c6_audits'/path.name)
    a,b=before['status'],found['status']
    assert found['columns']==before['columns']
    audits.append(dict(packet=path.stem,passed=found['passed'],
        old_worker_s=a['total_worker_s'],new_worker_s=b['total_worker_s'],
        old_strict_s=a['host_strict_s'],new_strict_s=b['host_strict_s'],
        old_action_s=a['host_krylov_action_s'],new_action_s=b['host_krylov_action_s'],
        old_iterations=[a['host_iterations_x'],a['host_iterations_y']],
        new_iterations=[b['host_iterations_x'],b['host_iterations_y']],
        maximum_column_difference=max(r['relative_column_error'] for r in found['independent_reference_check']['columns'])))
packets=[]
for path in (new/'c6_batches').glob('*.json'):
    row=read(path)
    if 'solver_migration_source' in row:continue
    packets.append(dict(packet=path.stem,mtime=path.stat().st_mtime,columns=row['columns'],
                        passed=row['passed'],**row['status']))
packets.sort(key=lambda row:row['mtime'])
progress=read(new/'progress.json')
result=dict(time=time.time(),state=progress.get('state'),saved_columns=len(list((new/'columns').glob('*.npy'))),
    remaining_worker_pairs=progress.get('remaining_worker_pairs',progress.get('planned_worker_pairs')),
    audits=audits,new_packets=packets,
    migrated_columns=read(new/'solver_migration.json').get('copied_columns'),
    operator=read(new/'operator_metadata.json'))
if audits:
    result['matched_audit_speedup']=sum(r['old_worker_s'] for r in audits)/sum(r['new_worker_s'] for r in audits)
if packets:
    result['recent_packet_median_s']=statistics.median(r['total_worker_s'] for r in packets[-8:])
    result['remaining_modal_hours_estimate']=result['remaining_worker_pairs']*result['recent_packet_median_s']/3600
    result['estimate_caveat']='Current packet rate only, excludes final T assembly/SO(3); higher degrees may converge differently.'
(base/'continuation_status.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('new_packets','operator')},indent=2))
