"""Build an isolated operator; production objects, headers and worker are untouched."""
import hashlib,json,shutil,subprocess
from pathlib import Path
B=Path(__file__).resolve().parent;R=B.parents[1];S=B/'source';S.mkdir(exist_ok=True)
text=(R/'src/quadrature.h').read_text()
text=text.replace('    if (order == 4) {','''    if (order == 1) {
        q.npts=1; q.pts[0][0]=q.pts[0][1]=1./3; q.wts[0]=1.;
    } else if (order == 3) {
        q.npts=3;
        q.pts[0][0]=q.pts[0][1]=1./6;
        q.pts[1][0]=2./3; q.pts[1][1]=1./6;
        q.pts[2][0]=1./6; q.pts[2][1]=2./3;
        for(int i=0;i<3;++i)q.wts[i]=1./3;
    } else if (order == 4) {''')
(S/'quadrature.h').write_text(text)
for f in ('muller_fmm.cpp','muller_dense.cpp'):shutil.copy2(R/'src'/f,S/f)
commands=[]
flags=['-O3','-march=native','-fopenmp','-std=c++11','-I'+str(S),'-Isrc','-I/usr/include',
    '-DBEM_DEFAULT_FMM_NEAR_FP32','-DBEM_PFFT_FP32','-DBEM_FMM_CONCURRENT_MEDIA_DEFAULT','-DBEM_MULLER_GPU_ASSEMBLY_DEFAULT']
with (B/'build.log').open('w') as log:
    for src in (S/'muller_fmm.cpp',S/'muller_dense.cpp',B/'probe.cpp'):
        command=['g++',*flags,'-c',str(src),'-o',str(B/(src.stem+'.o'))];commands.append(command)
        subprocess.run(command,cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
    names=['prism_face_fft','regular_face_fft','muller_mbj','muller_mbj_fmm','muller_nodal','muller_duffy',
        'mesh','orient','muller_fmm_gpu','muller_mbj_gpu','muller_paired_gmres','fmm','p2p','pfft']
    objects=[str(R/'build/muller-fp32'/(n+'.o')) for n in names]
    command=['g++','-fopenmp','-o',str(B/'probe'),str(B/'probe.o'),str(B/'muller_fmm.o'),str(B/'muller_dense.o'),
        *objects,'-L/usr/lib/x86_64-linux-gnu','-lcudart','-lcufft','-lcusparse','-lcublas','-lm','-ldl']
    commands.append(command);subprocess.run(command,cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
(B/'build.json').write_text(json.dumps(dict(commands=commands,sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [B/'probe.cpp',S/'quadrature.h',S/'muller_fmm.cpp',S/'muller_dense.cpp',B/'probe',*map(Path,objects)]}),indent=2)+'\n')
print('Isolated reduced quadrature probe built',flush=True)
