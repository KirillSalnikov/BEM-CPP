"""Local polynomial interpolation across the transverse lines of a regular face.

Each line is retained for both currents and every basis component on that line.
The fine-to-coarse right-factor action uses the transpose of the same real map.
"""
import numpy as np


class TransverseLines:
    def __init__(self, nx, dimension, *, stride, degree, edge):
        if nx < 2 or dimension % (2*nx) or stride < 1 or degree < 1 or edge < 1 or edge > nx:
            raise ValueError('Invalid transverse face layout or interpolation parameters')
        self.nx = nx
        self.channels = dimension//(2*nx)
        self.nodes = np.unique(np.r_[np.arange(0,nx,stride),np.arange(edge),np.arange(nx-edge,nx)])
        if len(self.nodes) <= degree:
            raise ValueError('Too few transverse nodes for polynomial degree')
        matrix = np.zeros((nx,len(self.nodes)),np.float32)
        matrix[self.nodes,np.arange(len(self.nodes))] = 1
        missing = np.setdiff1d(np.arange(nx),self.nodes)
        k = degree+1
        start = np.clip(np.searchsorted(self.nodes,missing)-k//2,0,len(self.nodes)-k)
        indices = start[:,None] + np.arange(k)
        nodes = self.nodes[indices]
        for j in range(k):
            weights = np.ones(len(missing))
            for q in range(k):
                if q != j:
                    weights *= (missing-nodes[:,q])/(nodes[:,j]-nodes[:,q])
            matrix[missing,indices[:,j]] = weights
        self.matrix = matrix
        self.indices = np.arange(dimension,dtype=np.int32).reshape(2,nx,-1)[:,self.nodes].reshape(-1)

    @staticmethod
    def compact_batch(first,batch,nf,guard):
        # Keep complete panels around zero frequency on both sides of the FFT.
        return guard==0 or (first>=guard and first+batch<=nf-guard)

    @staticmethod
    def apply(spectrum, matrix, channels):
        import torch
        nf, _, rhs = spectrum.shape
        output, source = matrix.shape
        # Treat real and imaginary parts as two FP32 right-hand sides. A
        # single GEMM implements the geometric map for all frequencies here.
        raw = torch.view_as_real(spectrum.contiguous()).reshape(nf,2,source,channels,rhs,2)
        raw = raw.permute(2,0,1,3,4,5).reshape(source,-1)
        mapped = (matrix @ raw).reshape(output,nf,2,channels,rhs,2)
        mapped = mapped.permute(1,2,0,3,4,5).contiguous().reshape(nf,2*output*channels,rhs,2)
        return torch.view_as_complex(mapped)
