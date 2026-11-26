"""Far-field audit on one coherent packet containing both polarizations, before orientation averaging."""
from pathlib import Path
import sys,json
import numpy as np

out=Path(__file__).resolve().parent
case,work=map(Path,sys.argv[1:])
pts=np.fromfile(case/'radiation_points.bin',float).reshape(-1,3)
basis=np.fromfile(case/'radiation_basis.bin',float).reshape(-1,13,6,3)
dofs=np.fromfile(case/'radiation_dofs.bin',np.int32).reshape(-1,6)
theta,phi=np.meshgrid(np.arange(181)*np.pi/180,np.arange(12)*np.pi/6)
dirs=np.stack((np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)),axis=-1).reshape(-1,3)
def sample(current):
    return np.einsum('tqja,tj->tqa',basis,current[dofs]).reshape(-1,3)
def radiate(x):
    j,m=np.split(x,2);sources=np.column_stack((sample(j),sample(m)))
    all_fields=[]
    for k in range(0,len(dirs),32):
        d=dirs[k:k+32]
        phase=np.exp(-20j*(d@pts.T))
        e,h=np.split(phase@sources,2,axis=1)
        all_fields.append(20j/(4*np.pi)*(-e+d*np.sum(d*e,axis=1)[:,None]+np.cross(d,h)))
    return np.vstack(all_fields)
x0=np.fromfile(work/'reference.bin',complex);x=np.fromfile(work/'compact.bin',complex)
f0=radiate(x0);f=radiate(x)
n0=np.linalg.norm(f0,axis=1)
diff=np.linalg.norm(f-f0,axis=1)
peak_delta=(abs(n0-np.linalg.norm(f,axis=1))).max()/n0.max()
report=dict(ka=20,ref=3,degree=20,modes_per_packet=12,theta_samples=181,phi_samples=12,
            reference_tol=1e-9,candidate_tol=2e-6,
            current_rel=np.linalg.norm(x-x0)/np.linalg.norm(x0),
            field_rel=np.linalg.norm(diff)/np.linalg.norm(n0),
            max_vector_difference_over_peak=diff.max()/n0.max(),max_amplitude_difference_over_peak=peak_delta)
if not report['field_rel']<1e-2:raise RuntimeError(report)
(out/'field_check.json').write_text(json.dumps(report,indent=2)+'\n')
