"""Private CPU-vector socket bridge to the resident compressed prism operator.

Runs only as a companion of the native T worker. The native worker performs
the independent accurate residual checks, symmetry sectors and checkpointing.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
import hashlib
import json
import pickle
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

MAGIC=0x42454D4646543031


class Companion:
    """Start/stop the service with the driver's native worker lifetime."""
    def __init__(self,case,socket_path,output):
        self.case,self.socket_path,self.output=Path(case),Path(socket_path),Path(output)
        self.process=self.log=None

    def __enter__(self):
        try:
            self.log=(self.output/'external_fft.log').open('a')
            self.process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),
                '--case',str(self.case),'--socket',str(self.socket_path),
                '--report',str(self.output/'external_fft.json')],stdout=self.log,stderr=subprocess.STDOUT)
            start=time.monotonic()
            while not self.socket_path.exists():
                if self.process.poll() is not None:raise RuntimeError('FFT service startup failed; inspect external_fft.log')
                if time.monotonic()-start>60:raise RuntimeError('FFT service startup timeout')
                time.sleep(.05)
            return self
        except BaseException:
            self.__exit__(None,None,None);raise

    def __exit__(self,*_):
        if self.process is not None and self.process.poll() is None:
            try:self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
        if self.log is not None:self.log.close()


def receive_into(connection, array):
    view=memoryview(array).cast('B')
    while view:
        count=connection.recv_into(view)
        if count==0:raise EOFError('Native FFT client disconnected')
        view=view[count:]


def check_handshake(data, meta):
    magic,*values=struct.unpack('=Q9d',data)
    expected=np.array([1,meta['system_dofs'],meta['ref'],meta['ka'],*meta['ri'],
                       meta['aspect'],meta['quadrature'],meta['duffy']],float)
    if magic!=MAGIC or not np.all(np.isfinite(values)) or not np.allclose(values,expected,rtol=1e-13,atol=0):
        raise ValueError('Compressed FFT geometry/material/quadrature mismatch')


def asset_files(case):
    files=[case/n for n in ('geometry.json','streamed_preparation.json','bound_operator.pkl',
                           'near_rows.bin','near_cols.bin','near_values.bin',
                           'mass_rows.bin','mass_cols.bin','mass_values.bin')]
    if (case/'execution.json').exists():
        files.append(case/'execution.json')
        config=json.loads((case/'execution.json').read_text())
        if config.get('local_storage','csr') in ('packed','host_packed'):
            files.extend(case/'packed_local'/name for name in ('metadata.json','rows.npy','columns.npy','coefficients.npy'))
    preparation=json.loads((case/'streamed_preparation.json').read_text())
    if preparation['status']!='complete':raise ValueError('Incomplete FFT preparation')
    for row in preparation['records']:
        files.extend([case/row['file']] if row['kind']=='2D' else
                     [case/row['directory']/name for name in ('left.npy','right.npy')])
    return files


def file_digest(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):result.update(block)
    return result.hexdigest()


def prepare_manifest(case):
    case=Path(case).resolve()
    return dict(format=1,geometry=json.loads((case/'geometry.json').read_text()),
                files={str(path.relative_to(case)):file_digest(path) for path in asset_files(case)})


def validate_manifest(case):
    case=Path(case).resolve()
    expected=json.loads((case/'service_assets.json').read_text())
    found=prepare_manifest(case)
    if found!=expected:raise ValueError('Compressed FFT asset hash mismatch')
    return expected


def load_operator(case):
    import torch
    from symmetric_galerkin_fft import TorchSymmetricAction
    torch.set_num_threads(1)
    config=json.loads((case/'execution.json').read_text()) if (case/'execution.json').exists() else {}
    if set(config)-{'local_storage','spectral_budget_MiB','frequency_batch','compact_maps','symbol_storage','shared_stream_buffers'}:
        raise ValueError('Unknown FFT execution setting')
    with (case/'bound_operator.pkl').open('rb') as stream:cpu=pickle.load(stream)
    # Runtime snapshots may live elsewhere than the construction directory.
    # Read their own hashed correction/mass files, never a mutable old path.
    cpu.geometry.directory=case
    factors={}
    for row in json.loads((case/'streamed_preparation.json').read_text())['records']:
        if row['kind']=='2D':cpu.canonical[row['index']].symbol=np.load(case/row['file'],mmap_mode='c')
        else:factors[row['index']]=tuple(np.load(case/row['directory']/name,mmap_mode='r')
                                       for name in ('left.npy','right.npy'))
    local_storage=config.get('local_storage','csr')
    if local_storage in ('packed','host_packed'):
        cpu.packed_local_directory=case/'packed_local'
        packed_meta=json.loads((cpu.packed_local_directory/'metadata.json').read_text())
        if packed_meta['source_geometry']!=cpu.geometry.meta:raise ValueError('Packed correction geometry mismatch')
    else:cpu.load_correction()
    options={}
    if 'spectral_budget_MiB' in config:
        options.update(spectral_budget_bytes=int(config['spectral_budget_MiB']*1024**2),
                       stream_cache=case/'.spectral-cache',frequency_batch=config.get('frequency_batch',16))
    result=TorchSymmetricAction(cpu,'cuda',torch.complex64,factors=factors,
                               factor_storage='half',local_storage=local_storage,
                               compact_maps=config.get('compact_maps',False),
                               shared_stream_buffers=config.get('shared_stream_buffers',False),
                               symbol_storage=config.get('symbol_storage','full'),**options)
    torch.cuda.empty_cache()
    return result


def serve(case, socket_path, report_path):
    import torch
    start=time.perf_counter()
    meta=json.loads((case/'geometry.json').read_text())
    report=dict(status='listening',calls=0,action_s=0.,case=str(case),geometry=meta)
    def save():
        temporary=report_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(report_path)
    listener=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
    # Never unlink another process's live socket.
    listener.bind(str(socket_path));listener.listen(1);save()
    try:
        with listener.accept()[0] as connection:
            header=bytearray(80);receive_into(connection,header);check_handshake(header,meta)
            report['status']='loading';save()
            validate_manifest(case)
            gpu=load_operator(case);n=meta['system_dofs']
            current=np.empty(n,np.complex128)
            compact_input=compact_output=device_input=None
            # Allocate Torch scratch now, before declaring the service ready.
            with torch.inference_mode():gpu(torch.zeros(n,device='cuda',dtype=torch.complex64))
            torch.cuda.synchronize()
            report.update(status='ready',GPU_operator_bytes=gpu.bytes,setup_s=time.perf_counter()-start)
            save();connection.sendall(struct.pack('=QQ',MAGIC,n))
            previous=0
            while True:
                header=bytearray(16);receive_into(connection,header)
                operation,sequence=struct.unpack('=QQ',header)
                if operation==0:break
                if operation not in (1,2) or sequence!=previous+1:raise ValueError('Invalid FFT request sequence')
                previous=sequence
                if operation==2 and compact_input is None:
                    compact_input=torch.empty(n,dtype=torch.complex64,pin_memory=True)
                    compact_output=torch.empty_like(compact_input,pin_memory=True)
                    device_input=torch.empty(n,dtype=torch.complex64,device='cuda')
                incoming=current if operation==1 else compact_input.numpy()
                receive_into(connection,incoming)
                if not np.all(np.isfinite(incoming)):raise ValueError('Nonfinite FFT input')
                begin=time.perf_counter()
                with torch.inference_mode():
                    if operation==1:
                        value=gpu(torch.as_tensor(current,device='cuda',dtype=torch.complex64)).to(
                            'cpu',dtype=torch.complex128).numpy()
                    else:
                        device_input.copy_(compact_input,non_blocking=True)
                        compact_output.copy_(gpu(device_input),non_blocking=True)
                        torch.cuda.current_stream().synchronize()
                        value=compact_output.numpy()
                if not np.all(np.isfinite(value)):raise ValueError('Nonfinite FFT result')
                report['action_s']+=time.perf_counter()-begin;report['calls']+=1
                report['compact_calls']=report.get('compact_calls',0)+(operation==2)
                connection.sendall(struct.pack('=QQ',sequence,0));connection.sendall(memoryview(value).cast('B'))
                if report['calls']%32==0:save()
        report.update(status='complete',elapsed_s=time.perf_counter()-start);save()
    except BaseException as error:
        report.update(status='failed',error=repr(error));save();raise
    finally:
        listener.close();socket_path.unlink(missing_ok=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--write-manifest',action='store_true')
    parser.add_argument('--socket',type=Path)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    if args.write_manifest:
        result=prepare_manifest(args.case)
        (args.case/'service_assets.json').write_text(json.dumps(result,indent=2)+'\n')
    else:
        if args.socket is None or args.report is None:parser.error('--socket and --report required')
        serve(args.case.resolve(),args.socket,args.report)
