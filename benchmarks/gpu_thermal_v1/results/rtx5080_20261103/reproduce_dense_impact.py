from pathlib import Path
import subprocess,json,hashlib,os,importlib.util,argparse
parser=argparse.ArgumentParser(description='Reproduce fixed-mesh dense LU impact of the Green Hessian correction')
parser.add_argument('--output',required=True,type=Path)
args=parser.parse_args()
root=Path(__file__).resolve().parents[4];out=args.output.expanduser().absolute();out.mkdir(exist_ok=False,parents=True)
spec=importlib.util.spec_from_file_location('b',root/'scripts/benchmark_hardware.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
old=out/'old_muller_nodal.cpp';old.write_bytes(subprocess.check_output(['git','-C',str(root),'show','7d5a7be:src/muller_nodal.cpp']))
result=dict(passed=False,note='CPU dense LU controls, fixed coarse spherical discretization; not mesh-converged physical accuracy or a GPU timing score',old_commit='7d5a7be',old_kernel_source_sha256=b.sha(old),new_kernel_source_sha256=b.sha(root/'src/muller_nodal.cpp'),cases=[])
for label,nodal in [('before',old),('after',root/'src/muller_nodal.cpp')]:
 cmd=['g++','-O3','-std=c++11','-I'+str(root/'src'),'-I'+str(root/'tests/host_cuda_stubs'),str(root/'tools/muller_nodal_demo.cpp'),str(root/'src/muller_dense.cpp'),str(root/'src/muller_mbj.cpp'),str(nodal),str(root/'src/muller_duffy.cpp'),str(root/'src/mesh.cpp'),'-o',str(out/label)]
 subprocess.run(cmd,check=True)
for ka in [.01,1.,10.]:
 matrices=[];details=[]
 for label in ['before','after']:
  dest=out/(str(ka)+'_'+label+'.json');cmd=[str(out/label),'--shape','sphere','--ref','0','--ka',str(ka),'--ri','1.3116','0','--ntheta','181','--out',str(dest)]
  p=subprocess.run(cmd,check=True,capture_output=True,text=True);d=json.loads(dest.read_text());u=d['mueller'];matrix=[[theta]+[u[i][j][t] for i in range(4) for j in range(4)] for t,theta in enumerate(d['theta'])];b.validate_matrix(matrix,181)
  assert d['relative_residual']<1e-10;matrices.append(matrix);details.append(dict(command=cmd,residual=d['relative_residual'],system_dofs=d['system_dofs'],M11_back=matrix[-1][1]))
 error=b.matrix_error(matrices[1],matrices[0]);result['cases'].append(dict(ka=ka,error=error,runs=details));b.write(out/'results.json',result);print(ka,error,flush=True)
 assert error['max_scaled_M11']<.01 and error['relative_L2']<.01
result['passed']=True;b.write(out/'results.json',result)
