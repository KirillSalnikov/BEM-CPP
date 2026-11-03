#!/usr/bin/env python3
"""Reproducible Linux/WSL RTX 50 builds; no display-driver installation."""
from pathlib import Path
import argparse,hashlib,json,os,platform,re,shlex,shutil,subprocess,sys,tempfile,time
from cuda_redist import install,latest_release,sha

PROBE=r'''#include <cuda_runtime.h>
#include <cstdio>
__global__ void check(int* p){*p=120;}
int main(){int device=0; if(cudaSetDevice(DEVICE)!=cudaSuccess)return 1;
cudaDeviceProp p{};if(cudaGetDeviceProperties(&p,DEVICE)!=cudaSuccess)return 2;
int* value=nullptr;int result=0;if(cudaMalloc(&value,sizeof(int))!=cudaSuccess)return 3;
check<<<1,1>>>(value);if(cudaDeviceSynchronize()!=cudaSuccess)return 4;
if(cudaMemcpy(&result,value,sizeof(int),cudaMemcpyDeviceToHost)!=cudaSuccess)return 5;
cudaFree(value);int driver=0,runtime=0;cudaDriverGetVersion(&driver);cudaRuntimeGetVersion(&runtime);
std::printf("{\"gpu\":\"%s\",\"compute\":%d,\"driver_api\":%d,\"runtime_api\":%d,\"kernel_result\":%d}\n",p.name,p.major*10+p.minor,driver,runtime,result);
return result==120 && p.major==12 && p.minor==0 ? 0:6;}
'''

def run(cmd,**kw):
    return subprocess.run([str(v) for v in cmd],check=True,**kw)

def output(cmd,**kw):return run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,**kw).stdout.strip()

def project(root):return 'mbs-fast' if (root/'cpu/Makefile').exists() else 'bem-cpp'

def gpu_arch(device=0):
    smi=shutil.which('nvidia-smi')
    if not smi and Path('/usr/lib/wsl/lib/nvidia-smi').exists():smi='/usr/lib/wsl/lib/nvidia-smi'
    if not smi:raise RuntimeError('nvidia-smi unavailable. On WSL install/update the Windows NVIDIA driver; do not install a Linux display driver.')
    row=output([smi,'--query-gpu=compute_cap','--format=csv,noheader','-i',str(device)]).splitlines()[0]
    match=re.fullmatch(r'(\d+)\.(\d+)',row.strip())
    if not match:raise RuntimeError('Cannot read GPU compute capability: '+row)
    return int(match[1])*10+int(match[2])

def compiler(cuda,requested,kind):
    text=(cuda/'include/crt/host_config.h').read_text()
    bound=re.search(r'#if\s+__GNUC__\s*>\s*(\d+)',text)
    maximum=int(bound[1]) if bound else None
    candidates=[requested] if requested else [os.environ.get('CXX'),'g++-14','g++-13','g++-12','g++']
    reasons=[]
    for item in candidates:
        if not item:continue
        path=shutil.which(item)
        if not path:continue
        major=int(output([path,'-dumpfullversion','-dumpversion']).split('.')[0])
        if maximum and major>maximum:reasons.append(f'{path}: GCC {major} > CUDA limit {maximum}');continue
        # The legacy MBS global complex type collides with GCC 15's new forward
        # declaration in std::pair. Select an unaffected supported compiler.
        if kind=='mbs-fast' and major>=15:reasons.append(f'{path}: MBS requires GCC 12–14 for its legacy complex type');continue
        return str(Path(path).resolve())
    raise RuntimeError('No compatible GNU C++ compiler. Install g++-14 or pass --cxx. '+ '; '.join(reasons))

def commands(kind,cuda,cxx,jobs,arch=120,mpicxx='mpicxx'):
    if arch!=120:raise ValueError('This installer targets RTX 50 / Blackwell compute capability 12.0 (sm_120)')
    rpath=f'-Wl,-rpath,{cuda}/lib64'
    if kind=='mbs-fast':
        gpu=['-j'+str(jobs),'GPU_ARCH=120','CUDA_PATH='+str(cuda),'CXX='+cxx,'CUDA_HOST_CXX='+cxx,'ARCH_FLAGS=-march=native','GPU_LTO=1','GPU_FAST_MATH=0','GPU_PHASE_TRIG=precise',f'LDFLAGS=-lm -lgomp -L{cuda}/lib64 {rpath} -lcufft -lcudart']
        return [('cpu',['make','-C','cpu','-j'+str(jobs),'CXX='+mpicxx,'ARCH_FLAGS=-march=native']),('gpu-fp64',['make','-C','gpu','fp64']+gpu),('gpu-fp32',['make','-C','gpu','fp32']+gpu)]
    flags=['CXX='+cxx,'CUDA_HOME='+str(cuda),'ARCH=-arch=sm_120','CUDA_CPP_STD=c++17']
    return [('cpu-reference',['make','bin/muller_nodal_demo','-j'+str(jobs)]+flags),('so3-fp64',['make','bin/tmatrix_so3_mstm','-j1']+flags),('so3-fp32',['make','bin/tmatrix_so3_mstm_fp32','-j1']+flags),('gpu',['make','muller-fp32','-j'+str(jobs)]+flags),('tmatrix-server',['make','bin/muller_tmatrix_fmm_server','-j'+str(jobs)]+flags)]

def source_files(root):
    excluded={'.git','build','bin','.venv','venv','runs','__pycache__','.pytest_cache','.cuda-local'}
    try:
        data=run(['git','-C',root,'ls-files','--cached','--others','--exclude-standard','-z'],stdout=subprocess.PIPE,stderr=subprocess.PIPE).stdout
        paths=[Path(os.fsdecode(v)) for v in data.split(b'\0') if v]
    except subprocess.CalledProcessError:
        paths=[p.relative_to(root) for p in root.rglob('*') if p.is_file() and not any(k in p.relative_to(root).parts for k in ['.git','build','bin','.venv','runs','__pycache__'])]
    return sorted(p for p in paths if not any(part in excluded for part in p.parts) and ((root/p).is_file() or (root/p).is_symlink()))

def source_digest(root,paths):
    h=hashlib.sha256()
    for path in paths:
        p=root/path;h.update(str(path).encode()+b'\0');h.update(os.readlink(p).encode() if p.is_symlink() else p.read_bytes());h.update(b'\0')
    return h.hexdigest()

def wrapper(path,argv,env):
    path.parent.mkdir(parents=True,exist_ok=True)
    lines=['#!/bin/sh','set -eu']
    for name,value in env.items():lines.append('export '+name+'='+shlex.quote(value))
    lines.append('exec '+' '.join(shlex.quote(str(v)) for v in argv)+' "$@"')
    path.write_text('\n'.join(lines)+'\n');path.chmod(0o755)

def system_deps(kind):
    if not shutil.which('apt-get'):raise RuntimeError('--install-deps currently supports apt-based Linux/WSL distributions')
    packages=['build-essential','git','python3','python3-venv','python3-pip']
    packages+=['openmpi-bin','libopenmpi-dev'] if kind=='mbs-fast' else ['gfortran','libblas-dev','liblapack-dev','libfftw3-double3']
    if kind=='mbs-fast' and shutil.which('apt-cache'):
        if 'Candidate: (none)' not in output(['apt-cache','policy','g++-14']) and output(['apt-cache','policy','g++-14']):packages.append('g++-14')
    sudo=[] if os.geteuid()==0 else ['sudo']
    run(sudo+['apt-get','update']);run(sudo+['apt-get','install','-y','--no-install-recommends']+packages)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefix',type=Path);p.add_argument('--cuda-home',type=Path,help='Use an existing toolkit >=12.8 instead of downloading a toolkit')
    p.add_argument('--cuda-version',default='latest',help='Latest stable NVIDIA release, or X.Y.Z; ignored with --cuda-home')
    p.add_argument('--cuda-root',type=Path,default=Path.home()/'.local/share/cuda');p.add_argument('--cache',type=Path,default=Path.home()/'.cache/cuda-redist')
    p.add_argument('--cxx');p.add_argument('--jobs',type=int,default=min(4,os.cpu_count() or 1));p.add_argument('--device',type=int,default=0)
    p.add_argument('--install-deps',action='store_true',help='Explicitly install host packages using apt; never installs NVIDIA display drivers')
    p.add_argument('--python',default=sys.executable,help='Python >=3.10 for the BEM launcher and analysis requirements')
    p.add_argument('--plan',action='store_true',help='Print make commands without installing or building; requires --cuda-home and --cxx')
    args=p.parse_args();root=Path(__file__).resolve().parents[1];kind=project(root)
    if args.jobs<1 or args.device<0:p.error('--jobs must be positive and --device nonnegative')
    if platform.system()!='Linux' or platform.machine() not in ['x86_64','amd64']:p.error('Supported platform: x86-64 Linux or WSL2')
    if args.plan:
        if not args.cuda_home or not args.cxx:p.error('--plan requires --cuda-home and --cxx')
        print(json.dumps(dict(project=kind,architecture='sm_120',driver_install=False,commands=commands(kind,args.cuda_home,args.cxx,args.jobs)),indent=2));return
    if args.install_deps:system_deps(kind)
    arch=gpu_arch(args.device)
    if arch!=120:raise RuntimeError(f'Expected RTX 50 / Blackwell sm_120; detected sm_{arch}')
    for tool in ['make']+(['mpicxx'] if kind=='mbs-fast' else ['gfortran']):
        if not shutil.which(tool):raise RuntimeError('Missing '+tool+'; use --install-deps or install the documented host dependencies')
    cuda=args.cuda_home.expanduser().resolve() if args.cuda_home else install(args.cuda_version,args.cuda_root,args.cache)
    version_text=output([cuda/'bin/nvcc','--version']);match=re.search(r'release (\d+)\.(\d+)',version_text)
    if not match or tuple(map(int,match.groups()))<(12,8):raise RuntimeError('Native sm_120 requires CUDA >=12.8')
    cxx=compiler(cuda,args.cxx,kind);prefix=(args.prefix or Path.home()/'.local/opt'/kind).expanduser().resolve();prefix.mkdir(parents=True,exist_ok=True)
    if prefix==root or root in prefix.parents:raise RuntimeError('--prefix must be outside the source checkout')
    env=dict(os.environ);env['PATH']=str(cuda/'bin')+':/usr/lib/wsl/lib:'+env['PATH'];env['LD_LIBRARY_PATH']=str(cuda/'lib64')+':/usr/lib/wsl/lib:'+env.get('LD_LIBRARY_PATH','');env['OMPI_CXX']=cxx
    paths=source_files(root);digest=source_digest(root,paths)
    try:revision=output(['git','-C',root,'describe','--always','--dirty'])
    except subprocess.CalledProcessError:revision='source-'+digest[:12]
    identity=hashlib.sha256(json.dumps([digest,revision,version_text,cxx,output([cxx,'--version']),arch],sort_keys=True).encode()).hexdigest()[:16]
    work=prefix/'build'/identity;work.mkdir(parents=True,exist_ok=True);source=work/'src'
    if not source.exists():
        source.mkdir()
        for relative in paths:
            target=source/relative;target.parent.mkdir(parents=True,exist_ok=True)
            if (root/relative).is_symlink():target.symlink_to(os.readlink(root/relative))
            else:shutil.copy2(root/relative,target)
    probe=work/'gpu_probe.cu';probe.write_text(PROBE.replace('DEVICE',str(args.device)))
    compat=source/('src/cuda/CudaHostCompat.h' if kind=='mbs-fast' else 'src/cuda_host_compat.h')
    with (work/'gpu_probe_build.log').open('w') as log:run([cuda/'bin/nvcc','-std=c++17','-arch=sm_120','-ccbin',cxx,'-U_GNU_SOURCE','-D_DEFAULT_SOURCE','-D_XOPEN_SOURCE=700','-include',compat,probe,'-o',work/'gpu_probe'],env=env,stdout=log,stderr=subprocess.STDOUT)
    hardware=json.loads(output([work/'gpu_probe'],env=env));assert hardware['kernel_result']==120 and hardware['compute']==120
    stages=commands(kind,cuda,cxx,args.jobs,mpicxx=shutil.which('mpicxx') or 'mpicxx');records=[]
    for name,cmd in stages:
        if kind=='mbs-fast':cmd+=['GIT_DESCRIBE='+revision]
        print('Building',kind,name,flush=True);start=time.time()
        with (work/(name+'.log')).open('w') as log:run(cmd,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
        records.append(dict(stage=name,command=cmd,seconds=time.time()-start))
    if kind=='mbs-fast':
        bins={'mbs-po-cpu':'cpu/bin/mbs_po_mpi','mbs-po-gpu-fp64':'gpu/bin/mbs_po_gpu_double','mbs-po-gpu-fp32':'gpu/bin/mbs_po_gpu_float'}
    else:
        bins={'bem-gpu':'bin/muller_nodal_fmm_demo_fp32','bem-tmatrix-server':'bin/muller_tmatrix_fmm_server','bem-so3-fp64':'bin/tmatrix_so3_mstm','bem-so3-fp32':'bin/tmatrix_so3_mstm_fp32','bem-cpu-reference':'bin/muller_nodal_demo'}
        python=Path(shutil.which(args.python) or args.python).resolve();version=output([python,'-c','import sys;print(sys.version_info.major*100+sys.version_info.minor)'])
        if int(version)<310:raise RuntimeError('BEM Python launcher needs Python >=3.10')
        check=[python,'-c','import numpy, scipy, scipy.special; assert hasattr(scipy.special,"sph_harm_y")']
        if subprocess.run([str(v) for v in check],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:
            venv=prefix/'.venv'
            if not (venv/'bin/python').exists():run([python,'-m','venv',venv])
            python=venv/'bin/python'
            run([python,'-m','pip','install','-r',source/'requirements-analysis.txt'])
            run([python]+check[1:])
        wrapper(prefix/'bin/bem',[python,source/'bem'],{'PATH':env['PATH'],'LD_LIBRARY_PATH':env['LD_LIBRARY_PATH']})
    for name,relative in bins.items():wrapper(prefix/'bin'/name,[source/relative],{'LD_LIBRARY_PATH':env['LD_LIBRARY_PATH']})
    versions={}
    for name in (['mbs-po-cpu','mbs-po-gpu-fp64','mbs-po-gpu-fp32'] if kind=='mbs-fast' else ['bem-gpu']):versions[name]=output([prefix/'bin'/name,'--version'],env=env)
    cubins={}
    if (cuda/'bin/cuobjdump').exists():
        for name in (['mbs-po-gpu-fp64','mbs-po-gpu-fp32'] if kind=='mbs-fast' else ['bem-gpu','bem-tmatrix-server']):
            listing=output([cuda/'bin/cuobjdump','--list-elf',source/bins[name]],env=env)
            if 'sm_120' not in listing:raise RuntimeError('Native sm_120 cubin missing from '+name)
            cubins[name]=True
    report=dict(project=kind,revision=revision,source_sha256=digest,architecture='sm_120',hardware=hardware,cuda_home=str(cuda),nvcc=version_text,compiler=cxx,commands=records,binaries={name:str(source/path) for name,path in bins.items()},binary_sha256={name:sha(source/path) for name,path in bins.items()},native_cubins_verified=cubins,versions=versions,driver_installed=False,fast_math_enabled=False,scope='Native sm_120 build and GPU execution probe; this does not certify full numerical convergence or all possible kernel optimizations')
    (prefix/'installation.json').write_text(json.dumps(report,indent=2)+'\n');print('Installed:',prefix/'bin');print('Manifest:',prefix/'installation.json')

if __name__=='__main__':
    try:main()
    except (RuntimeError,subprocess.CalledProcessError) as e:print('Installation failed:',e,file=sys.stderr);sys.exit(1)
