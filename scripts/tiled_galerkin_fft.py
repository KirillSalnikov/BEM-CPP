"""Bound Green-function scratch space while constructing finite cross FFTs."""
import time
import numpy as np
from scipy.fft import fft
from galerkin_cross_fft import CrossGalerkinFFT

class TiledCrossGalerkinFFT(CrossGalerkinFFT):
    def build(self,point_reference=False,retain_spatial=False):
        if point_reference:raise ValueError('Use the original reference builder for point channels')
        start=time.perf_counter();t,s=self.target,self.source
        kernel=np.zeros((self.nfft,2*self.tc,2*self.sc),complex)
        for p,pt in enumerate(t.patterns):
            for q,ps in enumerate(s.patterns):
                for first in range(0,len(self.delta),4):
                    d=self.delta[first:first+4];projected,unused=self.projected_blocks(pt,ps,d)
                    del unused
                    for i in range(t.nx):
                        for j in range(s.nx):
                            a=(i*len(t.patterns)+p)*6;b=(j*len(s.patterns)+q)*6
                            k1,ke,km=(v[i,j] for v in projected)
                            kernel[d%self.nfft,a:a+6,b:b+6]=1j/self.ka*k1
                            kernel[d%self.nfft,self.tc+a:self.tc+a+6,self.sc+b:self.sc+b+6]=-1j/self.ka*k1
                            kernel[d%self.nfft,a:a+6,self.sc+b:self.sc+b+6]=ke
                            kernel[d%self.nfft,self.tc+a:self.tc+a+6,b:b+6]=km
        self.spatial=kernel if retain_spatial else None
        self.symbol=fft(kernel,axis=0);self.preparation_s=time.perf_counter()-start
        return self
