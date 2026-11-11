import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parent;R=B.parents[1]
sys.path.insert(0,str(R/'scripts'))
from mbs_particle_geometry import prepare,write_obj
from tmatrix_incident import regular_fields

p=argparse.ArgumentParser();p.add_argument('--shape',choices=['bullet','prism'],default='bullet')
p.add_argument('--ka',type=float,default=10);p.add_argument('--ref',type=int,default=3)
p.add_argument('--rhs',type=int,choices=[2,4,6],default=4)
p.add_argument('--packed-only',action='store_true')
a=p.parse_args();out=B/(f'{a.shape}_ka{a.ka:g}_ref{a.ref}'+('_packed' if a.packed_only else ''));out.mkdir(exist_ok=True)
if a.shape=='bullet':
 v,t,g=prepare([2,1,1]);write_obj(out/'geometry.obj',v,t);shape=str(out/'geometry.obj')
else:shape='prism'
command=[str(B/'probe'),'geometry',shape,str(a.ref),str(a.ka),'3',str(out)]
subprocess.run(command,cwd=R,check=True)
x=np.fromfile(out/'points.bin',np.float64).reshape(-1,3);names=[];metadata=[]
theta,phi=np.deg2rad([37,19]);d=np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
polar=np.array([np.cos(theta)*np.cos(phi),np.cos(theta)*np.sin(phi),-np.sin(theta)])
for pol in range(2):
 e=polar if pol==0 else np.cross(d,polar);h=np.cross(d,e)
 fields=np.exp(1j*a.ka*(x@d))[:,None]*np.r_[e,h][None,:];name='plane_'+str(pol)
 fields.astype(np.complex128).tofile(out/(name+'.fields.bin'));names.append(name);metadata.append(dict(name=name,kind='plane',direction=d.tolist(),polarization=e.tolist()))
for l in ([int(a.ka*1.4)] if a.rhs==4 else [int(a.ka*.7),int(a.ka*1.4)] if a.rhs==6 else []):
 for pol in range(2):
  m=min(5,l);mode=(pol,l,m);name=f'mode_{pol}_{l}_{m}'
  regular_fields(x,mode,a.ka).tofile(out/(name+'.fields.bin'));names.append(name);metadata.append(dict(name=name,kind='regular_spherical_wave',mode=mode))
if a.packed_only:
 from tmatrix_c6 import encode_fields
 from tmatrix_reference import modes
 L=52;l=int(a.ka*1.4);modal=modes(L);lookup={m:i for i,m in enumerate(modal)}
 indices=[lookup[(1,l,m)] for m in range(6)];fields=np.empty((2,len(x),6),complex)
 scales,combined=encode_fields(x,modal,indices,a.ka,fields)
 names=[];metadata=[]
 for phase in range(2):
  name='packed_'+str(phase);fields[phase].tofile(out/(name+'.fields.bin'));names.append(name)
  metadata.append(dict(name=name,kind='C6_packed_regular_spherical_waves',phase_code=phase,
      modes=[list(modal[i]) for i in indices],scales=scales.tolist(),combined=combined.tolist()))
(out/'tasks.txt').write_text('\n'.join(names)+'\n')
(out/'case.json').write_text(json.dumps(dict(shape=shape,shape_name=a.shape,ka=a.ka,ref=a.ref,rhs=metadata),indent=2)+'\n')
print(out,flush=True)
