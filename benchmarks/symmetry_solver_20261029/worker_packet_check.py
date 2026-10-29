"""Check the actual persistent T worker with a saved two-code modal packet."""
import argparse,hashlib,json,os,socket,struct,subprocess,sys,tempfile,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
from build_fmm_tmatrix import receive_into,receive_message
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--packet',type=Path,required=True)
p.add_argument('--binary',type=Path,required=True);p.add_argument('--sector',action='store_true');p.add_argument('--reference',type=Path)
a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
packet=a.packet.resolve();meta=json.loads((packet.parent/'packet.json').read_text());ref=meta['ref'];ka=meta['ka']
directions=[]
for j in range(2):
 theta=.63+.22*j;phi=.27+.36*j;d=np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)]);e=np.array([np.cos(theta)*np.cos(phi),np.cos(theta)*np.sin(phi),-np.sin(theta)])
 angle=np.arange(721)*np.pi/720;directions.append(np.cos(angle)[:,None]*d+np.sin(angle)[:,None]*e)
directions=np.concatenate(directions);directions.tofile(out/'directions.bin')
env=os.environ.copy();env.update(BEM_TMATRIX_PRISM_ASPECT=str(10/7),BEM_TMATRIX_CIRCULANT_BLEND='.25',
 BEM_TMATRIX_MBJ_NODES='50',BEM_TMATRIX_MAX_LEAF='64',BEM_TMATRIX_SECTOR_GMRES='1'if a.sector else'0',BEM_TMATRIX_ORBIT_NODES='50')
cache=ROOT/f'runs/circulant_20261029/ref{ref}_ka40/operator' if ref in (5,6) and ka==40 else out/'operator'
with tempfile.TemporaryDirectory(prefix='bem-sector-check-')as tmp,(out/'worker.log').open('w')as log:
 address=str(Path(tmp)/'worker.sock');cmd=[str(a.binary.resolve()),address,'prism',str(ref),str(ka),'1.3116','0',str(out/'directions.bin'),str(cache),'2e-6','32','500']
 (out/'command.json').write_text(json.dumps({'command':cmd,'binary_sha256':hashlib.sha256(a.binary.read_bytes()).hexdigest(),'environment':{k:v for k,v in env.items()if k.startswith('BEM_')}},indent=2))
 process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
 connection=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);connection.settimeout(900)
 try:
  start=time.monotonic()
  while True:
   try:connection.connect(address);break
   except (FileNotFoundError,ConnectionRefusedError):
    if process.poll()is not None:raise RuntimeError('worker exited before connection')
    if time.monotonic()-start>600:raise TimeoutError('worker startup')
    time.sleep(.05)
  hello=receive_message(connection);positions=np.empty((hello['rhs_points'],3));receive_into(connection,positions)
  expected=np.fromfile(str(packet)+'.positions.bin',dtype=np.float64).reshape(-1,3)
  position_error=float(np.max(np.abs(positions-expected)))
  if position_error>1e-12:raise RuntimeError('worker geometry differs from incident samples')
  del expected,positions
  if bool(hello.get('sector_gmres',False))!=a.sector:raise RuntimeError('wrong solver selected')
  fields=np.memmap(str(packet)+'.fields.bin',dtype=np.complex128,mode='r',shape=(2,hello['rhs_points'],6))
  connection.sendall(struct.pack('=i',3));connection.sendall(memoryview(fields).cast('B'));del fields
  status=receive_message(connection);far=np.empty((2,len(directions),3),dtype=np.complex128);receive_into(connection,far)
  far.tofile(out/'farfield.bin');connection.sendall(struct.pack('=i',0));connection.close()
  if process.wait(timeout=60):raise RuntimeError('worker failed after packet')
  maximum=max(status['strict_residual_x'],status['strict_residual_y'])
  if maximum>2e-6 or not np.isfinite(maximum):raise RuntimeError('strict residual failed')
  errors=[]
  if a.reference:
   if (a.reference/'farfield.bin').exists():
    control=np.fromfile(a.reference/'farfield.bin',dtype=np.complex128).reshape(far.shape)
    errors=[float(np.linalg.norm(far[j]-control[j])/np.linalg.norm(control[j]))for j in range(2)]
   else:
    for j in range(2):
     control=np.fromfile(a.reference/f'farfield.circ0.25.rhs{j}.bin',dtype=np.complex128).reshape(721,3)
     observed=far[j,j*721:(j+1)*721];errors.append(float(np.linalg.norm(observed-control)/np.linalg.norm(control)))
   if not all(np.isfinite(errors))or max(errors)>1e-4:raise RuntimeError('far field comparison failed: '+str(errors))
  result={'passed':True,'metadata':hello,'status':status,'position_maximum_absolute_error':position_error,'farfield_relative_l2':errors,
    'maximum_strict_residual':maximum,'binary_sha256':hashlib.sha256(a.binary.read_bytes()).hexdigest()}
  (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
 finally:
  connection.close()
  if process.poll()is None:process.terminate();process.wait(timeout=30)
