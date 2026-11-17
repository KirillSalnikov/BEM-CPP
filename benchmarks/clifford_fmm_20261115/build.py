"""Reproducible local-only installation of the optional native Dirac FMM backend."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
B=Path(__file__).resolve().parent;R=B.parents[1]
PIN='cad293ae9b8ae1218dd8319d459ffcc5f1412875'
URL='https://github.com/flatironinstitute/FMM3D.git'
p=argparse.ArgumentParser();p.add_argument('--jobs',type=int,default=8)
p.add_argument('--fmm-only',action='store_true',help='preserve the separately selected Dirac binding and its build flags')
a=p.parse_args()
if not 1<=a.jobs<=32:raise ValueError('jobs must be 1..32')
source=B/'FMM3D'
if not source.exists():
    subprocess.run(['git','clone','--depth','1',URL,str(source)],check=True)
    subprocess.run(['git','fetch','--depth','1','origin',PIN],cwd=source,check=True)
    subprocess.run(['git','checkout','--detach',PIN],cwd=source,check=True)
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
if head!=PIN:raise RuntimeError('FMM3D checkout differs from the validated pin; preserve it and use the pinned checkout')
from scheduling import patch_source
scheduling=patch_source(source/'src/Helmholtz/hfmm3d.f')
from direct_kernel import patch_source as patch_direct
direct_kernel=patch_direct(source)
command=['make',f'-j{a.jobs}','lib','FAST_KER=ON','OMP=ON']
with (B/'fmm3d_build.log').open('w') as log:subprocess.run(command,cwd=source,stdout=log,stderr=subprocess.STDOUT,check=True)
if not a.fmm_only:
    subprocess.run([sys.executable,str(B.parent/'clifford_dielectric_20261115/build.py')],cwd=R,check=True)
library=source/'lib/libfmm3d.so'
(B/'native_build_manifest.json').write_text(json.dumps(dict(source_url=URL,commit=head,command=command,
    scheduling=scheduling,direct_kernel=direct_kernel,
    library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
    flags='-O3 -march=native; FAST_KER=ON; OpenMP; see build log',
    compiler=subprocess.check_output(['gfortran','--version'],text=True).splitlines()[0]),indent=2)+'\n')
print('Native FMM built; separately selected Dirac bindings preserved.' if a.fmm_only else
      'Native FMM and benchmark Dirac binding built; production Dirac binary preserved.')
