// Included after fmm_multilevel.cuh. Reverse the implemented traversal;
// reciprocity of the continuous Green kernel alone is not sufficient here.
#include <thrust/complex.h>
namespace {
template<class R> using FmmTComplex=thrust::complex<R>;
template<class R> __device__ FmmTComplex<R> fmm_t_phase(R kr,R ki,R dot) {
    R s,c;fmm_point_sincos(kr*dot,&s,&c);
    const R amp=ki==R(0)?R(1):fmm_point_exp(-ki*dot);
    return FmmTComplex<R>(amp*c,amp*s);
}
template<class R> __device__ FmmTComplex<R> fmm_t_read(const double* r,const double* i,size_t j) {
    return FmmTComplex<R>(R(r[j]),R(i[j]));
}
struct FmmTransposeWorkspace {
    int *offsets=nullptr,*targets=nullptr,*origins=nullptr;
    int *row_target=nullptr,*row_start=nullptr,*row_end=nullptr,*m2l_sources=nullptr,*m2l_transfers=nullptr;
    int rows=0;
    explicit FmmTransposeWorkspace(HelmholtzFMM& f) {
        auto upload=[](int*& out,const std::vector<int>& in){
            CUDA_CHECK(cudaMalloc(&out,std::max(size_t(1),in.size())*sizeof(int)));
            if(!in.empty())CUDA_CHECK(cudaMemcpy(out,in.data(),in.size()*sizeof(int),cudaMemcpyHostToDevice));};
        int pairs=0;for(const auto& level:f.m2l_level_info)pairs=std::max(pairs,level.offset+level.count);
        std::vector<int> old_target(pairs),old_source(pairs),old_transfer(pairs);
        if(pairs) {
            CUDA_CHECK(cudaMemcpy(old_target.data(),f.d_m2l_tgt,pairs*sizeof(int),cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(old_source.data(),f.d_m2l_src,pairs*sizeof(int),cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(old_transfer.data(),f.d_m2l_tidx,pairs*sizeof(int),cudaMemcpyDeviceToHost));
        }
        std::map<int,std::vector<int>> grouped;
        for(int j=0;j<pairs;++j)grouped[old_source[j]].push_back(j);
        std::vector<int> rt,rs,re,ms,mt;
        for(const auto& row:grouped){rt.push_back(row.first);rs.push_back(int(ms.size()));
            for(int j:row.second){ms.push_back(old_target[j]);mt.push_back(old_transfer[j]);}re.push_back(int(ms.size()));}
        rows=int(rt.size());upload(row_target,rt);upload(row_start,rs);upload(row_end,re);upload(m2l_sources,ms);upload(m2l_transfers,mt);
        if(!f.p2p_enabled)return;
        const int leaves=int(f.leaf_info.size());
        std::vector<int> node_to_leaf(f.n_nodes,-1),origin(f.Nt,0);
        std::vector<std::vector<int>> reverse(leaves);
        for(int j=0;j<leaves;++j)node_to_leaf[f.h_leaf_indices[j]]=j;
        for(int t=0;t<leaves;++t) {
            const int start=f.h_tgt_id_offsets[t],end=f.h_tgt_id_offsets[t+1];
            if(start==end)continue;
            for(int j=start;j<end;++j)origin[f.h_tgt_ids_flat[j]]=f.h_tgt_ids_flat[start];
            const auto append=[&](int s){if(s>=0)reverse[s].insert(reverse[s].end(),
                f.h_tgt_ids_flat.begin()+start,f.h_tgt_ids_flat.begin()+end);};
            append(t); // The forward near action includes its own leaf separately.
            const auto& leaf=f.tree.nodes[f.h_leaf_indices[t]];
            for(int j=leaf.near_start;j<leaf.near_start+leaf.near_count;++j)
                append(node_to_leaf[f.tree.near_list[j]]);
        }
        std::vector<int> row(leaves+1),flat;
        for(int s=0;s<leaves;++s){row[s]=int(flat.size());flat.insert(flat.end(),reverse[s].begin(),reverse[s].end());}
        row[leaves]=int(flat.size());
        upload(offsets,row);upload(targets,flat);upload(origins,origin);
    }
    ~FmmTransposeWorkspace(){cudaFree(offsets);cudaFree(targets);cudaFree(origins);
        cudaFree(row_target);cudaFree(row_start);cudaFree(row_end);cudaFree(m2l_sources);cudaFree(m2l_transfers);}
};
void fmm_t_clear(MultilevelArrays a,size_t size) {
    for(int d=0;d<3;++d){CUDA_CHECK(cudaMemsetAsync(a.r[d],0,size*sizeof(double)));CUDA_CHECK(cudaMemsetAsync(a.i[d],0,size*sizeof(double)));}
}
// Reverse derivative L2P: one thread per angular direction, sum target seeds.
template<class R> __global__ void fmm_t_l2p(
    const double* xyz,const double* dirs,const double* weights,const double* centers,
    const int* nodes,const int* offsets,const int* ids,int directions,
    const double* cr,const double* ci,const double* hr,const double* hi,
    MultilevelArrays out,R kr,R ki,R csr,R csi,R hsr,R hsi) {
    const int leaf=blockIdx.x,node=nodes[leaf];
    for(int l=threadIdx.x;l<directions;l+=blockDim.x) {
        const R d[3]={R(dirs[3*l]),R(dirs[3*l+1]),R(dirs[3*l+2])};
        const R dn=d[0]*d[0]+d[1]*d[1]+d[2]*d[2];
        FmmTComplex<R> sum[3]={};
        for(int j=offsets[leaf];j<offsets[leaf+1];++j) {
            const int t=ids[j];R dot=0;
            for(int a=0;a<3;++a)dot+=d[a]*R(xyz[3*t+a]-centers[3*node+a]);
            const auto phase=R(weights[l])*fmm_t_phase(kr,ki,dot);
            FmmTComplex<R> c[3],h[3],dh=0;
            for(int a=0;a<3;++a){c[a]=FmmTComplex<R>(csr,csi)*fmm_t_read<R>(cr,ci,3*t+a);
                h[a]=FmmTComplex<R>(hsr,hsi)*fmm_t_read<R>(hr,hi,3*t+a);dh+=d[a]*h[a];}
            sum[0]+=phase*(d[1]*c[0]+d[2]*c[1]+d[0]*dh-dn*h[0]);
            sum[1]+=phase*(-d[0]*c[0]+d[2]*c[2]+d[1]*dh-dn*h[1]);
            sum[2]+=phase*(-d[0]*c[1]-d[1]*c[2]+d[2]*dh-dn*h[2]);
        }
        for(int a=0;a<3;++a){out.r[a][size_t(node)*directions+l]=sum[a].real();out.i[a][size_t(node)*directions+l]=sum[a].imag();}
    }
}
// Reverse P2M: no angular quadrature weights or Helmholtz prefactor here.
template<class R> __global__ void fmm_t_p2m(
    const double* xyz,const double* dirs,const double* centers,const int* nodes,
    const int* offsets,const int* ids,int directions,MultilevelArrays in,
    MultilevelArrays out,R kr,R ki) {
    const int leaf=blockIdx.x,node=nodes[leaf];
    for(int j=offsets[leaf]+threadIdx.x;j<offsets[leaf+1];j+=blockDim.x) {
        const int s=ids[j];FmmTComplex<R> sum[3]={};
        for(int l=0;l<directions;++l) {
            R dot=0;for(int a=0;a<3;++a)dot+=R(dirs[3*l+a])*R(xyz[3*s+a]-centers[3*node+a]);
            const auto phase=fmm_t_phase(-kr,-ki,dot);
            for(int a=0;a<3;++a)sum[a]+=phase*fmm_t_read<R>(in.r[a],in.i[a],size_t(node)*directions+l);
        }
        for(int a=0;a<3;++a){out.r[a][s]=sum[a].real();out.i[a][s]=sum[a].imag();}
    }
}
__global__ void fmm_t_weight_copy(MultilevelArrays in,MultilevelArrays out,
    const int* nodes,const double* weights,int directions,int leaves,bool divide) {
    const int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=leaves*directions)return;
    const int l=j%directions;const size_t index=size_t(nodes[j/directions])*directions+l;
    const double w=divide?1./weights[l]:weights[l];
    for(int a=0;a<3;++a){out.r[a][index]=in.r[a][index]*w;out.i[a][index]=in.i[a][index]*w;}
}
// D^T = Wc S+ I Wf^-1. The interpolation is separable; only the phase
// differs from the ordinary upward transfer. Work arrays remain bounded by chunk.
template<class R> __global__ void fmm_t_down_theta(
    MultilevelArrays out,const int* nodes,const int* parents,const double* fc,const double* cc,
    const double* dirs,const double* weights,const double* te,const double* to,
    const double2* even,const double2* odd,int ntf,int ntc,int npc,double kr,double ki,
    int first,int count,const int* active) {
    const int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=count*3*ntc*npc)return;
    const int ip=id%npc,it=(id/npc)%ntc,comp=(id/(ntc*npc))%3,leaf=id/(3*ntc*npc);
    const int index=active?active[first+leaf]:first+leaf,node=nodes[index],parent=parents[index],l=it*npc+ip;
    FmmTComplex<R> value=0;
    for(int j=0;j<ntf;++j){const size_t x=((size_t(leaf)*3+comp)*ntf+j)*npc+ip;
        value+=R(te[it*ntf+j])*FmmTComplex<R>(R(even[x].x),R(even[x].y))+R(to[it*ntf+j])*FmmTComplex<R>(R(odd[x].x),R(odd[x].y));}
    double dot=0;for(int a=0;a<3;++a)dot+=dirs[3*l+a]*(fc[3*node+a]-cc[3*parent+a]);
    const auto z=weights[l]*fmm_t_phase(kr,ki,dot)*FmmTComplex<double>(value.real(),value.imag());
    atomicAdd(out.r[comp]+size_t(parent)*ntc*npc+l,z.real());atomicAdd(out.i[comp]+size_t(parent)*ntc*npc+l,z.imag());
}
// U^T = I^T S-. Input is the coarse multipole adjoint, not a local field.
__global__ void fmm_t_up_shift(MultilevelArrays in,const int* nodes,const int* parents,
    const double* fc,const double* cc,const double* dirs,double2* shifted,
    int directions,double kr,double ki,int first,int count,const int* active) {
    const int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=count*3*directions)return;
    const int l=id%directions,comp=(id/directions)%3,leaf=id/(3*directions);
    const int index=active?active[first+leaf]:first+leaf,node=nodes[index],parent=parents[index];
    double dot=0;for(int a=0;a<3;++a)dot+=dirs[3*l+a]*(fc[3*node+a]-cc[3*parent+a]);
    const auto value=fmm_t_phase(-kr,-ki,dot)*fmm_t_read<double>(in.r[comp],in.i[comp],size_t(parent)*directions+l);
    shifted[id]=make_double2(value.real(),value.imag());
}
template<class R> void fmm_t_transfer(HelmholtzFMM& f,HelmholtzFMM& c,bool reverse_down) {
    if(!f.multilevel_link)f.multilevel_link=new MultilevelLink(f,c);
    auto& a=*static_cast<MultilevelLink*>(f.multilevel_link);
    if(a.coarse!=&c)throw std::runtime_error("transpose multilevel engine mismatch");
    const int leaves=int(f.leaf_info.size());
    if(reverse_down)fmm_t_weight_copy<<<(leaves*f.L+255)/256,256>>>(multilevel_arrays(f,true),multilevel_arrays(f,false),
        f.d_leaf_idx_cached,f.d_weights_cached,f.L,leaves,true);
    const int total=reverse_down?a.down_count:a.up_count;
    const int* active=reverse_down?a.active_down:a.active_up;
    for(int first=0;first<total;first+=a.chunk) {
        const int count=std::min(a.chunk,total-first),tmp=count*3*a.ntf*a.npc;
        if(reverse_down) {
            multilevel_up_phi<R,double2><<<(tmp+255)/256,256>>>(multilevel_arrays(f,false),f.d_leaf_idx_cached,
                a.pe,a.po,a.even,a.odd,a.ntf,a.npf,a.npc,first,count,active);
            fmm_t_down_theta<R><<<(count*3*c.L+255)/256,256>>>(multilevel_arrays(c,true),f.d_leaf_idx_cached,a.parents,
                f.d_node_centers_cached,c.d_node_centers_cached,c.d_dirs_cached,c.d_weights_cached,a.te,a.to,a.even,a.odd,
                a.ntf,a.ntc,a.npc,f.k.real(),f.k.imag(),first,count,active);
        }else {
            fmm_t_up_shift<<<(count*3*c.L+255)/256,256>>>(multilevel_arrays(c,false),f.d_leaf_idx_cached,a.parents,
                f.d_node_centers_cached,c.d_node_centers_cached,c.d_dirs_cached,a.shifted,c.L,f.k.real(),f.k.imag(),first,count,active);
            multilevel_down_theta<R,double2><<<(tmp+255)/256,256>>>(a.shifted,a.te,a.to,a.even,a.odd,a.ntf,a.ntc,a.npc,count);
            multilevel_down_phi<R,double2><<<(count*3*f.L+255)/256,256>>>(multilevel_arrays(f,false),f.d_leaf_idx_cached,
                f.d_weights_cached,a.pe,a.po,a.even,a.odd,a.ntf,a.npf,a.npc,first,count,active);
        }
    }
    if(reverse_down)fmm_t_clear(multilevel_arrays(f,false),size_t(f.n_nodes)*f.L);
    else fmm_t_weight_copy<<<(leaves*f.L+255)/256,256>>>(multilevel_arrays(f,false),multilevel_arrays(f,false),
        f.d_leaf_idx_cached,f.d_weights_cached,f.L,leaves,false);
    CUDA_CHECK(cudaGetLastError());
}
void fmm_t_reverse_tree(HelmholtzFMM& f) {
    auto l=multilevel_arrays(f,true),m=multilevel_arrays(f,false);const int block=std::min(f.L,256);
    for(int level=f.tree.max_level;level>=std::max(2,f.minimum_m2l_level+1);--level)
        if(level<int(f.l2l_level_info.size()) && f.l2l_level_info[level].count) {
            const auto a=f.l2l_level_info[level];
            m2m_kernel_batch3<<<a.count,block>>>(f.d_l2l_parent,f.d_l2l_child,f.d_l2l_shift_re,f.d_l2l_shift_im,
                l.r[0],l.i[0],l.r[1],l.i[1],l.r[2],l.i[2],f.L,a.count,a.offset);
        }
    if(!f.transpose_workspace)f.transpose_workspace=new FmmTransposeWorkspace(f);
    const auto& rows=*static_cast<FmmTransposeWorkspace*>(f.transpose_workspace);
#define TRANSPOSE_M2L_ROWS(R) m2l_kernel_batch3_target_rows<R><<<rows.rows,block>>>(rows.row_target,rows.row_start,rows.row_end, \
        rows.m2l_sources,rows.m2l_transfers,f.d_transfer_re,f.d_transfer_im,l.r[0],l.i[0],l.r[1],l.i[1],l.r[2],l.i[2], \
        m.r[0],m.i[0],m.r[1],m.i[1],m.r[2],m.i[2],f.L,rows.rows,0)
    if(rows.rows) {
        if(f.near_field_fp32 && bem_env_flag_enabled("BEM_FMM_MULTILEVEL_M2L_FP32")){TRANSPOSE_M2L_ROWS(float);}
        else {TRANSPOSE_M2L_ROWS(double);}
    }
#undef TRANSPOSE_M2L_ROWS
    for(int level=std::max(1,f.minimum_m2l_level);level<f.tree.max_level;++level)
        if(level<int(f.m2m_level_info.size()) && f.m2m_level_info[level].count) {
            const auto a=f.m2m_level_info[level];
            l2l_kernel_batch3<<<a.count,block>>>(f.d_m2m_parent,f.d_m2m_child,f.d_m2m_shift_re,f.d_m2m_shift_im,
                m.r[0],m.i[0],m.r[0],m.i[0],m.r[1],m.i[1],m.r[1],m.i[1],m.r[2],m.i[2],m.r[2],m.i[2],f.L,a.count,a.offset);
        }
    CUDA_CHECK(cudaGetLastError());
}
// Gather over the explicitly reversed original near list. Geometry need not
// have symmetric target/source IDs or a symmetric sparsity pattern.
template<class R> __global__ void fmm_t_near(const double* tgt,const double* src,
    const int* offsets,const int* targets,const int* origins,const int* src_offsets,const int* src_ids,
    const double* cr,const double* ci,const double* hr,const double* hi,MultilevelArrays out,
    R kr,R ki,int splits,bool local_coordinates,bool fast_trig) {
    const int leaf=blockIdx.x/splits,part=blockIdx.x%splits;
    const int start=offsets[leaf],count=offsets[leaf+1]-start;
    const int first=start+int((long long)count*part/splits),last=start+int((long long)count*(part+1)/splits);
    for(int j=src_offsets[leaf]+threadIdx.x;j<src_offsets[leaf+1];j+=blockDim.x) {
        const int s=src_ids[j];FmmTComplex<R> sum[3]={};
        for(int p=first;p<last;++p) {
            const int t=targets[p];R d[3],r2=0;
            for(int a=0;a<3;++a){const double origin=local_coordinates?tgt[3*origins[t]+a]:0.;
                d[a]=R(tgt[3*t+a]-origin)-R(src[3*s+a]-origin);r2+=d[a]*d[a];}
            if(r2<R(1e-24))continue;
            const R inv=R(1)/sqrt(r2),r=r2*inv,inv2=inv*inv;
            FmmTComplex<R> phase;
            if(fast_trig){float sn,cs;__sincosf(float(kr*r),&sn,&cs);phase=FmmTComplex<R>(cs,sn);}
            else phase=fmm_t_phase(kr,ki,r);
            const auto green=phase*(R(.07957747154594767)*inv);
            const auto gradient=green*FmmTComplex<R>((-ki-inv)*inv,kr*inv);
            const auto ga=green*FmmTComplex<R>(R(3)*inv2+R(3)*ki*inv-(kr*kr-ki*ki),-R(3)*kr*inv-R(2)*kr*ki);
            const auto gb=green*FmmTComplex<R>(inv2+ki*inv,-kr*inv);
            FmmTComplex<R> c[3],h[3],dot=0;
            for(int a=0;a<3;++a){c[a]=fmm_t_read<R>(cr,ci,3*t+a);h[a]=fmm_t_read<R>(hr,hi,3*t+a);dot+=(d[a]*inv)*h[a];}
            sum[0]+=gradient*(d[1]*c[0]+d[2]*c[1]);
            sum[1]+=gradient*(-d[0]*c[0]+d[2]*c[2]);
            sum[2]+=gradient*(-d[0]*c[1]-d[1]*c[2]);
            for(int a=0;a<3;++a)sum[a]+=(d[a]*inv)*ga*dot+(R(2)*gb-ga)*h[a];
        }
        for(int a=0;a<3;++a){atomicAdd(out.r[a]+s,double(sum[a].real()));atomicAdd(out.i[a]+s,double(sum[a].imag()));}
    }
}
}
void HelmholtzFMM::evaluate_vector_actions_transpose(const cdouble* curl_seed,const cdouble* hessian_seed,
    cdouble* rx,cdouble* ry,cdouble* rz,HelmholtzFMM* middle,HelmholtzFMM* coarse,bool include_near) {
    if(!initialized || !batch3_device_allocated || !d_src_pts || !d_complex_tmp1)
        throw std::runtime_error("FMM transpose requires the batch3 point workspace");
    if(middle && !coarse)throw std::invalid_argument("transpose middle band requires coarse band");
    std::vector<HelmholtzFMM*> bands={this};if(middle)bands.push_back(middle);if(coarse)bands.push_back(coarse);
    for(auto* b:bands){fmm_t_clear(multilevel_arrays(*b,true),size_t(b->n_nodes)*b->L);fmm_t_clear(multilevel_arrays(*b,false),size_t(b->n_nodes)*b->L);}
    auto upload=[&](const cdouble* x,double* r,double* i){CUDA_CHECK(cudaMemcpyAsync(d_complex_tmp1,x,size_t(Nt)*3*sizeof(double2),cudaMemcpyHostToDevice));
        split_complex_kernel<<<(Nt*3+255)/256,256>>>(d_complex_tmp1,r,i,Nt*3);};
    upload(curl_seed,d_grad_re,d_grad_im);upload(hessian_seed,d_hess_re,d_hess_im);
    const int leaves=int(leaf_info.size());const cdouble pref=cdouble(0,1)*k/(16*M_PI*M_PI),cs=pref*cdouble(0,1)*k,hs=-pref*k*k;
#define TRANSPOSE_L2P(R) fmm_t_l2p<R><<<leaves,std::min(L,256)>>>(d_tgt_pts,d_dirs_cached,d_weights_cached,d_node_centers_cached, \
    d_leaf_idx_cached,d_tgt_id_offsets_cached,d_tgt_ids_cached,L,d_grad_re,d_grad_im,d_hess_re,d_hess_im,multilevel_arrays(*this,true), \
    R(k.real()),R(k.imag()),R(cs.real()),R(cs.imag()),R(hs.real()),R(hs.imag()))
    if(near_field_fp32 && bem_env_flag_enabled("BEM_FMM_MULTILEVEL_L2P_FP32")){TRANSPOSE_L2P(float);}else{TRANSPOSE_L2P(double);}
#undef TRANSPOSE_L2P
    for(size_t i=0;i+1<bands.size();++i) {
        if(near_field_fp32 && bem_env_flag_enabled("BEM_FMM_MULTILEVEL_INTERPOLATION_FP32"))fmm_t_transfer<float>(*bands[i],*bands[i+1],true);
        else fmm_t_transfer<double>(*bands[i],*bands[i+1],true);
    }
    for(int i=int(bands.size())-1;i>=0;--i) {
        fmm_t_reverse_tree(*bands[i]);
        if(i){if(near_field_fp32 && bem_env_flag_enabled("BEM_FMM_MULTILEVEL_INTERPOLATION_FP32"))fmm_t_transfer<float>(*bands[i-1],*bands[i],false);
            else fmm_t_transfer<double>(*bands[i-1],*bands[i],false);}
    }
    MultilevelArrays result={{d_charges_re,d_charges2_re,d_charges3_re},{d_charges_im,d_charges2_im,d_charges3_im}};
#define TRANSPOSE_P2M(R) fmm_t_p2m<R><<<leaves,128>>>(d_src_pts,d_dirs_cached,d_node_centers_cached,d_leaf_idx_cached, \
    d_src_id_offsets_cached,d_src_ids_cached,L,multilevel_arrays(*this,false),result,R(k.real()),R(k.imag()))
    if(near_field_fp32 && bem_env_flag_enabled("BEM_FMM_MULTILEVEL_P2M_FP32")){TRANSPOSE_P2M(float);}else{TRANSPOSE_P2M(double);}
#undef TRANSPOSE_P2M
    if(p2p_enabled && include_near) {
        if(!transpose_workspace)transpose_workspace=new FmmTransposeWorkspace(*this);
        auto& w=*static_cast<FmmTransposeWorkspace*>(transpose_workspace);
        const int splits=32;
#define TRANSPOSE_NEAR(R,LOCAL,FAST) fmm_t_near<R><<<leaves*splits,128>>>(d_tgt_pts,d_src_pts,w.offsets,w.targets,w.origins, \
    d_src_id_offsets_cached,d_src_ids_cached,d_grad_re,d_grad_im,d_hess_re,d_hess_im,result,R(k.real()),R(k.imag()),splits,LOCAL,FAST)
        if(near_field_fp32){TRANSPOSE_NEAR(float,bem_env_flag_enabled("BEM_FMM_P2P_LOCAL_COORDINATES"),k.imag()==0 && bem_env_flag_enabled("BEM_FMM_P2P_SHARED_FAST_TRIG"));}
        else {TRANSPOSE_NEAR(double,false,false);}
#undef TRANSPOSE_NEAR
    }
    CUDA_CHECK(cudaGetLastError());cdouble* output[3]={rx,ry,rz};
    for(int a=0;a<3;++a){pack_complex_kernel<<<(Ns+255)/256,256>>>(result.r[a],result.i[a],d_complex_tmp1,Ns);
        CUDA_CHECK(cudaMemcpy(output[a],d_complex_tmp1,size_t(Ns)*sizeof(double2),cudaMemcpyDeviceToHost));}
}
