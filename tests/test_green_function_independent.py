"""80-digit differentiation of outgoing exp(i*k*R)/(4*pi*R).

Requires mpmath. This oracle does not call any production derivative helper.
The C++ probe exercises the actual combined kernel and its Taylor switch.
"""
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

import mpmath as mp

ROOT=Path(__file__).resolve().parents[1]
mp.mp.dps=80
cases=[]
for ke,ki in [(3+0j,4.5+0j),(1+0j,1.3116+.2j),(2+.1j,3+.4j)]:
    switch=.01/max(abs(ke),abs(ki))
    for radius in [1e-9,1e-6,switch*.5,switch*.9999,switch*1.0001,.02,.5,5.]:
        for sign in [1,-1]:
            xyz=[sign*radius*.3,sign*radius*(-.4),sign*radius*math.sqrt(.75)]
            cases.append((ke,ki,xyz))
with tempfile.TemporaryDirectory() as directory:
    probe=Path(directory)/'green_probe'
    command=shlex.split(os.environ.get('CXX','g++'))+['-O3','-std=c++11','-ffunction-sections','-fdata-sections','-I'+str(ROOT/'src'),'-I'+str(ROOT/'tests/host_cuda_stubs'),str(ROOT/'tests/green_kernel_probe.cpp'),str(ROOT/'src/muller_nodal.cpp'),str(ROOT/'src/mesh.cpp'),'-Wl,--gc-sections','-o',str(probe)]
    subprocess.run(command,check=True)
    text=''.join(' '.join(map(repr,[ke.real,ke.imag,ki.real,ki.imag]+xyz))+'\n' for ke,ki,xyz in cases)
    output=subprocess.check_output([str(probe)],input=text,text=True)
rows=output.splitlines()
assert len(rows)==len(cases)
worst={'hessian_relative':0.,'gradient_relative':0.,'scalar_relative':0.,'helmholtz_trace_scaled':0.}
for (ke,ki,xyz),line in zip(cases,rows):
    values=list(map(float,line.split()));actual=[mp.mpc(values[i],values[i+1]) for i in range(0,len(values),2)]
    ke=mp.mpc(ke);ki=mp.mpc(ki);point=tuple(map(mp.mpf,xyz))
    def green(k,x,y,z):
        r=mp.sqrt(x*x+y*y+z*z)
        return mp.exp(mp.j*k*r)/(4*mp.pi*r)
    def difference(x,y,z):return green(ke,x,y,z)-green(ki,x,y,z)
    expected_h=[];expected_ge=[];expected_gi=[]
    for i in range(3):
        order=[0,0,0];order[i]=1
        expected_ge.append(mp.diff(lambda x,y,z:green(ke,x,y,z),point,tuple(order)))
        expected_gi.append(-mp.diff(lambda x,y,z:green(ki,x,y,z),point,tuple(order)))
        for j in range(3):
            second=list(order);second[j]+=1
            expected_h.append(mp.diff(difference,point,tuple(second)))
    scalar=ke*ke*green(ke,*point)-ki*ki*green(ki,*point)
    def relative(a,b):return float(mp.sqrt(sum(abs(x-y)**2 for x,y in zip(a,b))/sum(abs(x)**2 for x in b)))
    errors=dict(hessian_relative=relative(actual[1:10],expected_h),gradient_relative=relative(actual[10:16],expected_ge+expected_gi),scalar_relative=relative(actual[:1],[scalar]))
    errors['helmholtz_trace_scaled']=float(abs(actual[1]+actual[5]+actual[9]+actual[0])/mp.sqrt(sum(abs(x)**2 for x in expected_h)))
    for key,value in errors.items():
        if not math.isfinite(value):raise AssertionError((key,value))
        worst[key]=max(worst[key],value)
    if max(errors.values())>5e-10:raise AssertionError(dict(ke=str(ke),ki=str(ki),point=list(map(str,point)),errors=errors))
print(json.dumps(dict(passed=True,cases=len(cases),precision_decimal_digits=80,mpmath_version=mp.__version__,oracle='mpmath Cartesian differentiation of exp(+ikR)/(4*pi*R)',worst_errors=worst),indent=2))
