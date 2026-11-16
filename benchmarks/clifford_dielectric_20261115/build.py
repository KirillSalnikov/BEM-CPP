"""Build the bounded CPU Dirac reference without touching production binaries."""
import argparse,hashlib,json,subprocess,os
from pathlib import Path
B=Path(__file__).resolve().parent
R=B.parents[1]
p=argparse.ArgumentParser();p.add_argument('--fast-complex',action='store_true');p.add_argument('--output',type=Path,default=B/'libdirac.so');args=p.parse_args()
output=args.output.resolve();output.parent.mkdir(parents=True,exist_ok=True);temporary=output.with_suffix('.building.so')
command=['g++','-O3','-march=native','-fopenmp','-std=c++11','-fPIC','-shared','-Isrc']
if args.fast_complex:command+=['-fcx-limited-range']
command+=['src/dirac_transmission.cpp','src/mesh.cpp',str(B/'binding.cpp'),'-o',str(temporary)]
subprocess.run(command,cwd=R,check=True)
os.replace(temporary,output)
paths=[B/'matrix_storage.py',B/'sector_coarse_sparse.py',B/'sector_coarse_hierarchy.py',B/'sector_fmm.py',B/'sector_coarse_fmm.py',B.parent/'clifford_fmm_20261115/native.py',R/'src/dirac_transmission.cpp',R/'src/dirac_transmission.h',R/'src/mesh.cpp',R/'src/quadrature.h',
       B/'binding.cpp',output,B/'solve.py',B/'tmatrix.py',B/'batch_gmres.py',B/'batch_checks.py',B/'dirac_symmetry.py',B/'sector_coarse.py',B/'memory_plan.py',B/'so3_factored.py',R/'scripts/tmatrix_reference.py']
(output.parent/(output.stem+'_build_manifest.json' if output!=B/'libdirac.so' else 'build_manifest.json')).write_text(json.dumps(dict(command=command,
    compiler=subprocess.check_output(['g++','--version'],text=True).splitlines()[0],
    sha256={str(p.relative_to(R) if p.is_relative_to(R) else p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}),indent=2)+'\n')
