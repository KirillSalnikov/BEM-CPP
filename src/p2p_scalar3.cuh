// Included after the P2P arithmetic helpers in p2p.cu. Used only by the
// opt-in three-scalar multilevel path; existing vector kernels are unchanged.
namespace {
struct Scalar3Buffers {
    const double* qr[3]; const double* qi[3];
    double* pr[3]; double* pi[3];
    double* gr[3][3]; double* gi[3][3];
};

template<typename Real, bool REAL_K>
__global__ void scalar3_pot_grad_near(
    const double* target, const double* source, Scalar3Buffers data,
    const int* toff,const int* tids,const int* soff,const int* sids,
    const int* noff,const int* neighbors,double kr,double ki)
{
    const int leaf=blockIdx.x;
    const Real wave_re(kr),wave_im(ki),inv4pi(0.079577471545947667884);
    for(int it=toff[leaf]+threadIdx.x;it<toff[leaf+1];it+=blockDim.x) {
        const int tid=tids[it];
        const double tx=target[3*tid],ty=target[3*tid+1],tz=target[3*tid+2];
        Real acc[3][4][2];
        #pragma unroll
        for(int d=0;d<3;++d)for(int a=0;a<4;++a)for(int c=0;c<2;++c)acc[d][a][c]=Real(0);
        const int nb=noff[leaf],ne=noff[leaf+1];
        for(int j=nb-1;j<ne;++j) {
            const int sl=j<nb?leaf:neighbors[j];
            for(int is=soff[sl];is<soff[sl+1];++is) {
                const int sid=sids[is];
                // Subtract original FP64 coordinates before converting to a
                // two-float expansion: tiny panel separations must survive.
                const Real dx(tx-source[3*sid]),dy(ty-source[3*sid+1]),dz(tz-source[3*sid+2]);
                const Real r2=dx*dx+dy*dy+dz*dz;
                if(r2<Real(1e-24))continue;
                const Real ri=p2p_inverse_sqrt(r2),radius=r2*ri;
                Real sn,cs;p2p_pair_sine_cosine<false>(wave_re*radius,&sn,&cs);
                const Real attenuation=REAL_K?Real(1):p2p_attenuation(-wave_im*radius);
                const Real g0=attenuation*cs*inv4pi*ri,g1=attenuation*sn*inv4pi*ri;
                const Real h0=(-wave_im-ri)*ri,h1=wave_re*ri;
                const Real dg0=g0*h0-g1*h1,dg1=g0*h1+g1*h0;
                #pragma unroll
                for(int d=0;d<3;++d) {
                    const Real qr(data.qr[d][sid]),qi(data.qi[d][sid]);
                    acc[d][0][0]+=g0*qr-g1*qi;acc[d][0][1]+=g0*qi+g1*qr;
                    const Real hr=dg0*qr-dg1*qi,hi=dg0*qi+dg1*qr;
                    acc[d][1][0]+=hr*dx;acc[d][1][1]+=hi*dx;
                    acc[d][2][0]+=hr*dy;acc[d][2][1]+=hi*dy;
                    acc[d][3][0]+=hr*dz;acc[d][3][1]+=hi*dz;
                }
            }
        }
        #pragma unroll
        for(int d=0;d<3;++d) {
            data.pr[d][tid]+=double(acc[d][0][0]);data.pi[d][tid]+=double(acc[d][0][1]);
            #pragma unroll
            for(int a=0;a<3;++a) {
                data.gr[d][a][tid]+=double(acc[d][a+1][0]);
                data.gi[d][a][tid]+=double(acc[d][a+1][1]);
            }
        }
    }
}
}

void launch_p2p_scalar3_pot_grad(HelmholtzFMM& f,int precision)
{
    Scalar3Buffers b={
        {f.d_charges_re,f.d_charges2_re,f.d_charges3_re},
        {f.d_charges_im,f.d_charges2_im,f.d_charges3_im},
        {f.d_result_re,f.d_result2_re,f.d_result3_re},
        {f.d_result_im,f.d_result2_im,f.d_result3_im},
        {{f.d_gx_re_tmp_cached,f.d_gy_re_cached,f.d_gz_re_cached},
         {f.d_gx2_re_tmp_cached,f.d_gy2_re_cached,f.d_gz2_re_cached},
         {f.d_gx3_re_tmp_cached,f.d_gy3_re_cached,f.d_gz3_re_cached}},
        {{f.d_gx_im_tmp_cached,f.d_gy_im_cached,f.d_gz_im_cached},
         {f.d_gx2_im_tmp_cached,f.d_gy2_im_cached,f.d_gz2_im_cached},
         {f.d_gx3_im_tmp_cached,f.d_gy3_im_cached,f.d_gz3_im_cached}}
    };
    const int leaves=int(f.leaf_info.size());
    if(!leaves)return;
    #define SCALAR3_LAUNCH(T,R) scalar3_pot_grad_near<T,R><<<leaves,128>>>( \
        f.d_tgt_pts,f.d_src_pts,b,f.d_tgt_id_offsets_cached,f.d_tgt_ids_cached, \
        f.d_src_id_offsets_cached,f.d_src_ids_cached,f.d_leaf_near_offsets_cached, \
        f.d_leaf_near_ids_cached,f.k.real(),f.k.imag())
    if(precision==2) {
        if(f.k.imag()==0) {SCALAR3_LAUNCH(P2PCompensated,true);}
        else {SCALAR3_LAUNCH(P2PCompensated,false);}
    } else if(precision==1) {
        if(f.k.imag()==0) {SCALAR3_LAUNCH(double,true);}
        else {SCALAR3_LAUNCH(double,false);}
    } else throw std::invalid_argument("scalar3 precision must be 1 (FP64) or 2 (compensated)");
    #undef SCALAR3_LAUNCH
    CUDA_CHECK(cudaGetLastError());
}
