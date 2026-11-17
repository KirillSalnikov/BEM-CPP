"""Local physical metric for cross-face compression, without dense global maps.

If B maps six edge moments to tangent quadrature vectors, S=(B.T B)**1/2
makes B S**-1 orthonormal (up to a scalar). Thus S R S**-1 is orthogonal
for the local n-cross rotation. Compress S_t**-T A S_s**-1, then return
factors of A in the original edge-moment coordinates. No physics changes.
"""
import numpy as np
import torch


class FaceBasisMetric:
    def __init__(self, block, face, device='cpu', dtype=torch.complex64):
        roots=[]; inverses=[]; conditions=[]
        for pattern in face.patterns:
            _,basis=block.pattern(face,pattern)
            b=np.asarray(basis,dtype=np.float64).transpose(0,2,1).reshape(21,6)
            gram=b.T@b
            scale=np.trace(gram)/6
            if not np.isfinite(scale) or scale<=0:
                raise ValueError('Degenerate tangent basis metric')
            eigenvalues,vectors=np.linalg.eigh(gram/scale)
            if not np.isfinite(eigenvalues).all() or eigenvalues[0]<=1e-12*eigenvalues[-1]:
                raise ValueError('Singular tangent basis metric')
            root=(vectors*np.sqrt(eigenvalues))@vectors.T
            inverse=(vectors/np.sqrt(eigenvalues))@vectors.T
            roots.append(root); inverses.append(inverse)
            conditions.append(float(np.sqrt(eigenvalues[-1]/eigenvalues[0])))
        self.root=torch.as_tensor(np.asarray(roots),device=device,dtype=dtype)
        self.inverse=torch.as_tensor(np.asarray(inverses),device=device,dtype=dtype)
        self.condition=max(conditions)
        self.patterns=len(roots)

    def rows(self, value, *, inverse=False):
        """Left multiply each six-coefficient block; accept full faces or tiles."""
        nf,nr,nc=value.shape
        if nr%(12*self.patterns):
            raise ValueError('Current/face coefficient layout mismatch')
        matrix=self.inverse if inverse else self.root
        x=value.reshape(nf,2,nr//(12*self.patterns),self.patterns,6,nc)
        return torch.einsum('pij,fcxpjr->fcxpir',matrix,x).reshape(value.shape)

    def columns(self, value, *, inverse=False):
        # Both maps are real symmetric: plain transpose, never conjugate A.
        return self.rows(value.transpose(-1,-2),inverse=inverse).transpose(-1,-2)
