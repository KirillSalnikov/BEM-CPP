"""Isolated GPU potential/gradient bridge; does not replace CPU strict checks."""
import hashlib,json,subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
B=Path(__file__).resolve().parent;R=B.parents[1]
flags=['/usr/bin/nvcc','-ccbin','g++-12','-O3','-arch=sm_86','-std=c++14','-Isrc','-Xcompiler=-fPIC,-fopenmp']
sources=[R/'src/fmm.cu',R/'src/p2p.cu',B/'bridge.cpp']
def compile_one(source):
    out=B/(source.stem+'.o');cmd=flags+['-c',str(source),'-o',str(out)]
    with (B/(source.stem+'_build.log')).open('w') as log:subprocess.run(cmd,cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
    return out
with ThreadPoolExecutor(max_workers=2) as pool:objects=list(pool.map(compile_one,sources))
command=flags+['-shared',*map(str,objects),'-lcufft','-lcublas','-o',str(B/'libgpu_fmm.so')]
subprocess.run(command,cwd=R,check=True)
dependencies=sorted(set(sources+list((R/'src').glob('*.h'))+list((R/'src').glob('*.cuh'))+[B/'libgpu_fmm.so']))
(B/'build.json').write_text(json.dumps(dict(command=command,
    sha256={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}),indent=2)+'\n')
