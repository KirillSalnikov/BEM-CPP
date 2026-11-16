#pragma once
// Included after the existing P2P scalar/DS arithmetic and contraction helpers.
// Combined two-medium action for one current: H_e-H_i and c_e curl_e-c_i curl_i.
// No change to the quadrature, near lists or the original Muller equation.
template<class R>
__device__ __forceinline__ void p2p_two_media_source(
    R dx,R dy,R dz,R ke_r,R ke_i,R ki_r,R ki_i,
    R ce_r,R ce_i,R ci_r,R ci_i,const R qr[3],const R qi[3],
    R cr[3],R cim[3],R ar[3],R aim[3]) {
    const R r2=dx*dx+dy*dy+dz*dz;if(r2<R(1e-24))return;
    const R ir=p2p_inverse_sqrt(r2),radius=r2*ir,scale=R(0.07957747154594767)*ir;
    R gr[2],gi[2],aa_r[2],aa_i[2],bb_r[2],bb_i[2];
#pragma unroll
    for(int medium=0;medium<2;++medium){
        const R kr=medium?ki_r:ke_r,kim=medium?ki_i:ke_i;
        R sine=0,cosine=0;p2p_pair_sine_cosine<false>(kr*radius,&sine,&cosine);
        const R attenuation=kim==R(0)?R(1):p2p_attenuation(-kim*radius);
        const R green_r=scale*attenuation*cosine,green_i=scale*attenuation*sine;
        const R grad_r=(-kim-ir)*ir,grad_i=kr*ir;
        gr[medium]=green_r*grad_r-green_i*grad_i;gi[medium]=green_r*grad_i+green_i*grad_r;
        const R k2r=kr*kr-kim*kim,k2i=R(2)*kr*kim;
        // These identities also hold for complex k; Hessian isotropic part
        // is -gradG, while H-tr(H) adds the longitudinal k^2 term below.
        aa_r[medium]=-R(3)*gr[medium]-(k2r*green_r-k2i*green_i);
        aa_i[medium]=-R(3)*gi[medium]-(k2r*green_i+k2i*green_r);
        bb_r[medium]=-gr[medium];bb_i[medium]=-gi[medium];
    }
    const R grad_r=ce_r*gr[0]-ce_i*gi[0]-ci_r*gr[1]+ci_i*gi[1];
    const R grad_i=ce_r*gi[0]+ce_i*gr[0]-ci_r*gi[1]-ci_i*gr[1];
    const R unit[3]={dx*ir,dy*ir,dz*ir};
    accumulate_vector_values(dx,dy,dz,unit,grad_r,grad_i,
        aa_r[0]-aa_r[1],aa_i[0]-aa_i[1],bb_r[0]-bb_r[1],bb_i[0]-bb_i[1],
        qr,qi,cr,cim,ar,aim);
}

#ifdef __CUDACC__
template<class R,int Tile=128>
__global__ void p2p_two_media_shared_kernel(
    const double* tgt,const double* src,const double* qxr,const double* qxi,
    const double* qyr,const double* qyi,const double* qzr,const double* qzi,
    const int* to,const int* ti,const int* no,const int* ni,int leaves,int split,
    double ker,double kei,double kir,double kii,double cer,double cei,double cir,double cii,
    double* cr,double* ci,double* ar,double* ai) {
    const int leaf=blockIdx.x/split,part=blockIdx.x%split;
    if(leaf>=leaves||to[leaf]==to[leaf+1])return;
    const int origin=ti[to[leaf]];const double ox=tgt[3*origin],oy=tgt[3*origin+1],oz=tgt[3*origin+2];
    __shared__ R data[9][Tile];
    const int count=no[leaf+1]-no[leaf],begin=no[leaf]+int((long long)count*part/split),end=no[leaf]+int((long long)count*(part+1)/split);
    for(int tb=to[leaf];tb<to[leaf+1];tb+=blockDim.x){
        const bool active=tb+threadIdx.x<to[leaf+1];const int id=active?ti[tb+threadIdx.x]:0;
        const R tx=active?R(tgt[3*id]-ox):R(0),ty=active?R(tgt[3*id+1]-oy):R(0),tz=active?R(tgt[3*id+2]-oz):R(0);
        R c_r[3]={0,0,0},c_i[3]={0,0,0},a_r[3]={0,0,0},a_i[3]={0,0,0};
        for(int base=begin;base<end;base+=Tile){const int size=min(Tile,end-base);
            for(int j=threadIdx.x;j<size;j+=blockDim.x){int s=ni[base+j];
                data[0][j]=R(src[3*s]-ox);data[1][j]=R(src[3*s+1]-oy);data[2][j]=R(src[3*s+2]-oz);
                data[3][j]=R(qxr[s]);data[4][j]=R(qxi[s]);data[5][j]=R(qyr[s]);data[6][j]=R(qyi[s]);data[7][j]=R(qzr[s]);data[8][j]=R(qzi[s]);}
            __syncthreads();
            if(active)for(int j=0;j<size;++j){const R qr[3]={data[3][j],data[5][j],data[7][j]},qi[3]={data[4][j],data[6][j],data[8][j]};
                p2p_two_media_source(tx-data[0][j],ty-data[1][j],tz-data[2][j],R(ker),R(kei),R(kir),R(kii),R(cer),R(cei),R(cir),R(cii),qr,qi,c_r,c_i,a_r,a_i);}
            __syncthreads();
        }
        if(active)for(int d=0;d<3;++d){const int dst=3*id+d;
            if(split==1){cr[dst]+=double(c_r[d]);ci[dst]+=double(c_i[d]);ar[dst]+=double(a_r[d]);ai[dst]+=double(a_i[d]);}
            else {atomicAdd(cr+dst,double(c_r[d]));atomicAdd(ci+dst,double(c_i[d]));atomicAdd(ar+dst,double(a_r[d]));atomicAdd(ai+dst,double(a_i[d]));}}
    }
}
#endif
