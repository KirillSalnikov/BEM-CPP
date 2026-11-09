import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from bounded_spectral_storage import StreamedSpectralBlock,half_factors,SharedStreamBuffers,decode_half_symbol
from symmetric_galerkin_fft import TorchSymmetricAction


class HalfSymbolTests(unittest.TestCase):
    def test_packed_current_symbol_matches_full_half_encoding(self):
        rng=np.random.default_rng(972)
        array=lambda shape:np.asarray(rng.normal(size=shape)+1j*rng.normal(size=shape),np.complex64)
        a,b,c=[array((23,7,5)) for _ in range(3)]
        symbol=np.concatenate((np.concatenate((a,b),axis=2),np.concatenate((c,-a),axis=2)),axis=1)
        with tempfile.TemporaryDirectory() as tmp:
            full,fs=half_factors(symbol,tmp,'full')
            packed,ps=half_factors(symbol,tmp,'packed',packed_currents=True)
            np.testing.assert_array_equal(decode_half_symbol(torch.from_numpy(full)).numpy(),
                                          decode_half_symbol(torch.from_numpy(packed)).numpy())
            np.testing.assert_array_equal(fs,ps)
            self.assertEqual(packed.nbytes*4,full.nbytes*3)
            symbol[0,-1,-1]+=1
            with self.assertRaisesRegex(ValueError,'exact negatives'):
                half_factors(symbol,tmp,'bad',packed_currents=True)

    def test_resident_half_symbol_across_batches(self):
        rng=np.random.default_rng(77)
        symbol=np.asarray(rng.normal(size=(519,7,5))+1j*rng.normal(size=(519,7,5)),np.complex64)
        symbol[4]=0;symbol[5]*=1e10;symbol[6]*=1e-10
        x=np.asarray(rng.normal(size=(519,5,3))+1j*rng.normal(size=(519,5,3)),np.complex64)
        with tempfile.TemporaryDirectory() as tmp:
            encoded,scale=half_factors(symbol,tmp,'symbol')
            decoded=(encoded[...,0].astype(np.float32)+1j*encoded[...,1].astype(np.float32))*scale
            group=dict(symbol_half=torch.from_numpy(encoded),symbol_scale=torch.from_numpy(scale))
            actual=TorchSymmetricAction.symbol_product(group,torch.from_numpy(x),7).numpy()
            expected=decoded@x
            np.testing.assert_allclose(actual,expected,rtol=3e-6,atol=1e-5)
            self.assertLess(np.linalg.norm(decoded-symbol)/np.linalg.norm(symbol),5e-4)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required for asynchronous storage')
class BoundedStorageTests(unittest.TestCase):
    def test_first_shared_use_waits_for_recycled_cuda_scratch(self):
        pool=SharedStreamBuffers('cuda')
        symbols=np.tile(np.eye(128,dtype=np.complex64),(16,1,1))
        op=StreamedSpectralBlock((symbols,),device='cuda',batch=8,pool=pool)
        x=torch.ones((16,128,2),dtype=torch.complex64,device='cuda')
        # Allocate pinned memory before the delayed device writes: a first
        # cudaHostAlloc may itself synchronize and accidentally hide the race.
        hosts=iter([torch.empty(pool.capacity,dtype=torch.uint8,pin_memory=True) for _ in range(2)])
        scratch=[torch.empty(pool.capacity,dtype=torch.uint8,device='cuda') for _ in range(2)]
        recycled=iter(scratch)
        output=torch.empty_like(x)
        # Warm BLAS and supply reusable allocations so CUDA allocation/handle
        # initialization cannot synchronize away the intended pending writes.
        torch.bmm(torch.as_tensor(symbols[:8],device='cuda'),x[:8],out=output[:8])
        original_empty=torch.empty
        def empty(*args,**kwargs):
            if kwargs.get('pin_memory'):return next(hosts)
            if kwargs.get('dtype')==torch.uint8:return next(recycled)
            if args and args[0]==tuple(x.shape):return output
            return original_empty(*args,**kwargs)
        torch.cuda.synchronize()
        pointers={s.data_ptr() for s in scratch}
        # Simulate scratch returned by the caching allocator while its former
        # writes are pending. Immediate writes from another stream are unsafe.
        torch.cuda._sleep(1000000000)
        for value in scratch:value.fill_(255)
        with patch('torch.empty',side_effect=empty):result=op(x,128)
        reused={s['gpu_raw'].data_ptr() for s in pool.slots}
        self.assertEqual(reused,pointers)
        torch.testing.assert_close(result,x,rtol=0,atol=0)

    def test_shared_slots_across_shapes_dtypes_and_streams(self):
        rng=np.random.default_rng(491)
        array=lambda shape:np.asarray(rng.normal(size=shape)+1j*rng.normal(size=shape),np.complex64)
        pool=SharedStreamBuffers('cuda');cases=[];unshared_bytes=0
        with tempfile.TemporaryDirectory() as tmp:
            for index,(nf,nr,nc,batch,half) in enumerate(((19,23,17,4,0),(35,37,29,8,1),(9,11,7,3,0),(23,22,14,4,2))):
                symbol=array((nf,nr,nc))
                if half:
                    if half==2:symbol[:,nr//2:,nc//2:]=-symbol[:,:nr//2,:nc//2]
                    encoded,scale=half_factors(symbol,tmp,str(index),packed_currents=half==2);arrays=(encoded,scale)
                    expected=decode_half_symbol(torch.from_numpy(encoded)).numpy()*scale
                else:arrays=(symbol,);expected=symbol
                op=StreamedSpectralBlock(arrays,device='cuda',batch=batch,half_matrix=half,pool=pool)
                unshared_bytes+=2*sum(min(batch,nf)*a[0].nbytes for a in arrays)
                cases.append((op,expected))
            self.assertLess(pool.bytes,unshared_bytes)
            # Queue different blocks before reading outputs, with repeated slot
            # reuse on two nondefault streams. The slot events must follow the
            # storage, not a particular block's shape or stream.
            streams=[torch.cuda.Stream(),torch.cuda.Stream()]
            outputs=[]
            for repetition in range(5):
                for index,(op,expected) in enumerate(cases):
                    x=array((len(expected),expected.shape[2],2))
                    with torch.cuda.stream(streams[(repetition+index)%2]):
                        actual=op(torch.as_tensor(x,device='cuda'),expected.shape[1])
                    outputs.append((actual,expected@x))
            torch.cuda.synchronize()
            for actual,expected in outputs:
                self.assertLess(np.linalg.norm(actual.cpu().numpy()-expected)/np.linalg.norm(expected),4e-7)
            self.assertEqual(sum(slot['gpu_raw'].numel() for slot in pool.slots),pool.bytes)
            with self.assertRaisesRegex(RuntimeError,'Register all'):
                pool.register((np.ones((3,2,2),np.complex64),),2)

    def test_tail_batch_repeat_and_changed_inputs(self):
        rng=np.random.default_rng(710)
        complex_array=lambda shape:np.asarray(rng.normal(size=shape)+1j*rng.normal(size=shape),np.complex64)
        nf,nr,nc,rank,count=35,37,29,7,3
        left,right=complex_array((nf,nr,rank)),complex_array((nf,rank,nc))
        left[2]=0;right[6]=0
        with tempfile.TemporaryDirectory() as tmp:
            u,us=half_factors(left,tmp,'left');v,vs=half_factors(right,tmp,'right')
            op=StreamedSpectralBlock((u,v,us*vs),device='cuda',batch=8,half=True)
            decoded=lambda a:np.asarray(a[...,0],np.float32)+1j*np.asarray(a[...,1],np.float32)
            ul,vr=decoded(u),decoded(v)
            symbol=complex_array((nf,nr,nc))
            full=StreamedSpectralBlock((symbol,),device='cuda',batch=8)
            half_symbol,scale=half_factors(symbol,tmp,'symbol')
            dense_half=StreamedSpectralBlock((half_symbol,scale),device='cuda',batch=8,half_matrix=True)
            for _ in range(4):
                x=complex_array((nf,nc,count));gpu=torch.as_tensor(x,device='cuda')
                for action,expected in ((op,(ul@(vr@x))*(us*vs)),(full,symbol@x),
                                        (dense_half,(decoded(half_symbol)@x)*scale)):
                    actual=action(gpu,nr).cpu().numpy()
                    self.assertLess(np.linalg.norm(actual-expected)/np.linalg.norm(expected),3e-7)
            self.assertEqual(op.bytes,op.host_bytes)
            self.assertLess(op.bytes,sum(a.nbytes for a in (u,v,us*vs)))

if __name__=='__main__':unittest.main()
