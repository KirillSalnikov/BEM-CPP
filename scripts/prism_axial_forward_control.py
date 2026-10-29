#!/usr/bin/env python3
"""Direct two-polarization BEM solve for axial incidence; no T construction."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import tempfile
import time
import numpy as np
from build_fmm_tmatrix import receive_into, receive_message, atomic_json
from tmatrix_reference import coherency_to_mueller

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ka',type=float,required=True)
    parser.add_argument('--ref',type=int,choices=[5,6],required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    validated_mesh='main_ref5_L44' if args.ref==5 else 'mesh_ref6_L44'
    source=ROOT/'runs/ka20_d6h_q6_20261026_170903'/validated_mesh/'summary.json'
    previous=json.loads(source.read_text())
    binary=ROOT/'runs/ka20_d6h_q6_20261026_170903/runtime/bin/muller_tmatrix_fmm_server'
    if not binary.exists(): binary=ROOT/'bin/muller_tmatrix_fmm_server'
    digest=hashlib.sha256(binary.read_bytes()).hexdigest()
    if digest!=previous['parameters']['binary_digest']:
        raise RuntimeError('Worker differs from the validated ka20 build')
    theta=np.linspace(0,180,361); rad=np.radians(theta)
    directions=np.stack((np.sin(rad),np.zeros_like(rad),np.cos(rad)),axis=1)
    et=np.stack((np.cos(rad),np.zeros_like(rad),-np.sin(rad)),axis=1)
    ep=np.tile([0.,1.,0.],(len(theta),1))
    directions.tofile(out/'directions.bin')
    env={k:v for k,v in os.environ.items() if not k.startswith(('BEM_','MBS_'))}
    env.update(previous['parameters']['BEM_environment'])
    env.update(OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='1')
    # The coarse ka=60 solve can keep a longer Krylov basis within its small
    # memory footprint; the fine mesh keeps the validated 80-vector budget.
    restart=160 if args.ka>=60 and args.ref==5 else 80
    # Cache files are separate for every size and surface refinement.
    with tempfile.TemporaryDirectory(prefix='bem-axial-') as temp, (out/'worker.log').open('a') as log:
        path=str(Path(temp)/'worker.sock')
        cmd=[str(binary),path,'prism',str(args.ref),str(args.ka),'1.3','0',str(out/'directions.bin'),
             str(out/'operator'),'1e-6',str(restart),'800']
        atomic_json(out/'configuration.json',{'command':cmd,'binary_sha256':digest,'environment':{k:v for k,v in env.items() if k.startswith('BEM_')}})
        worker=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        connection=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
        try:
            while True:
                try:connection.connect(path);break
                except (FileNotFoundError,ConnectionRefusedError):
                    if worker.poll() is not None:raise RuntimeError('BEM worker startup failed; inspect worker.log')
                    time.sleep(.05)
            hello=receive_message(connection)
            positions=np.empty((hello['rhs_points'],3),dtype=np.float64)
            receive_into(connection,positions);del positions
            atomic_json(out/'operator_metadata.json',hello)
            print(json.dumps(hello),flush=True)
            plane=np.array([1.,0.,0.,0.,1.,0.,0.,0.,1.])
            connection.sendall(struct.pack('=i',2));connection.sendall(memoryview(plane).cast('B'))
            status=receive_message(connection)
            far=np.empty((2,len(theta),3),dtype=np.complex128);receive_into(connection,far)
            connection.sendall(struct.pack('=i',0));connection.close()
            worker.wait(timeout=60)
            if worker.returncode:raise RuntimeError('BEM worker failed during shutdown')
        finally:
            connection.close()
            if worker.poll() is None:
                worker.terminate()
                try:worker.wait(timeout=10)
                except subprocess.TimeoutExpired:worker.kill();worker.wait()
    if max(status['strict_residual_x'],status['strict_residual_y'])>1e-6:
        atomic_json(out/'rejected.json',status);raise RuntimeError('BEM solve failed its residual gate')
    jones=np.stack((np.einsum('ni,pni->np',et,far),np.einsum('ni,pni->np',ep,far)),axis=1)
    mueller=args.ka**2*np.array([coherency_to_mueller(np.outer(j.ravel(),j.ravel().conj())) for j in jones])
    np.savez_compressed(out/'fields.npz',theta_deg=theta,jones=jones,mueller=mueller)
    report={'complete':True,'ka':args.ka,'ref':args.ref,'shape':'hexagonal prism H/D_flats=1, a_eq=1',
            'refractive_index':[1.3,0.],'incident_direction':[0,0,1],'operator':hello,'solve':status,
            'M11_forward':float(mueller[0,0,0]),'Cext':float(-2*np.pi/args.ka*np.trace(jones[0]).imag),
            'forward_Jones_real':jones[0].real.tolist(),'forward_Jones_imag':jones[0].imag.tolist(),
            'elapsed_s':time.monotonic()-start}
    atomic_json(out/'result.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
