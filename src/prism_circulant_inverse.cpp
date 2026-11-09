#include "prism_circulant_inverse.h"
#include <cstdio>
#include <limits>
#include <stdexcept>

void PrismCirculantInverse::build_from_inverse_kernels(const MullerP2Mesh& mesh,int ref,const Action& inverse,
        const std::vector<Complex>& side,const std::vector<Complex>& cap) {
    if(layout.system_dofs)throw std::logic_error("analytic circulant builder requires a fresh object");
    device_mapping_.reset();cpu_enabled_=false;
    layout.build(mesh,ref);local=inverse;opposite_weight_=0;opposite_mask_=0;symmetric_caps_ready=false;
    for(int kind=0;kind<2;++kind){const auto& p=layout.patches[kind?24:0];const int grid=p.nx*p.ny;
        const auto& kernel=kind?cap:side;
        if(kernel.size()!=size_t(144)*grid)throw std::invalid_argument("analytic inverse kernel dimensions");
        banks[kind].init_inverse_kernel(p.nx,p.ny,12,kind?2:24,kernel);
        input[kind].resize(size_t(kind?2:24)*12*grid);output[kind].resize(input[kind].size());
    }
}

void PrismCirculantInverse::build(const MullerP2Mesh& mesh,int ref,const Action& action,const Action& inverse,double shift,double opposite_weight,int opposite_mask) {
    if(!std::isfinite(opposite_weight) || opposite_weight<0 || opposite_weight>1 || opposite_mask<0 || opposite_mask>3)
        throw std::invalid_argument("opposite face coupling weight/mask");
    device_mapping_.reset();cpu_enabled_=false;
    opposite_weight_=opposite_weight;opposite_mask_=opposite_mask;symmetric_caps_ready=false;
    layout.build(mesh,ref);local=inverse;
    std::vector<Complex> x(layout.system_dofs),y(layout.system_dofs);
    for(int kind=0;kind<2;++kind) {
        const auto& p=layout.patches[kind?24:0];
        const int grid=p.nx*p.ny;
        const auto& opposite=layout.patches[prism_opposite_patch(kind?24:0)];
        if(coupled(kind))for(int k=kind?24:0;k<(kind?26:24);++k) {
            const auto& a=layout.patches[k];const auto& b=layout.patches[prism_opposite_patch(k)];
            const Vec3 displacement=b.origin-a.origin;
            if(a.nx!=b.nx || a.ny!=b.ny || (a.u-b.u).norm()>1e-10 || (a.v-b.v).norm()>1e-10 ||
               (a.normal+b.normal).norm()>1e-10 || std::abs(displacement.dot(a.u))>1e-10 || std::abs(displacement.dot(a.v))>1e-10)
                throw std::runtime_error("opposite FFT patches do not align");
            for(size_t j=0;j<a.dof.size();++j)if((a.dof[j]<0)!=(b.dof[j]<0))throw std::runtime_error("opposite FFT support mismatch");
        }
        int source=-1;double best=std::numeric_limits<double>::infinity();
        for(int i=0;i<p.nx;++i)for(int j=0;j<p.ny;++j) {
            bool valid=true;for(int c=0;c<12;++c)valid=valid && p.dof[p.slot(c,i,j)]>=0;
            const double distance=std::pow(i-(p.nx-1)*.5,2)+std::pow(j-(p.ny-1)*.5,2);
            if(valid && distance<best){best=distance;source=i*p.ny+j;}
        }
        if(source<0)throw std::runtime_error("no complete circulant source cell");
        std::vector<Complex> kernel(size_t(144)*grid),cross(coupled(kind)?kernel.size():0);
        for(int col=0;col<12;++col) {
            const int slot=col*grid+source,dof=p.dof[slot];
            x[dof]=p.solution_sign[slot];action(x.data(),y.data());x[dof]=0;
            for(int row=0;row<12;++row)for(int i=0;i<p.nx;++i)for(int j=0;j<p.ny;++j) {
                const int t=p.slot(row,i,j),target=p.dof[t];if(target<0)continue;
                const int dx=(i-source/p.ny+p.nx)%p.nx,dy=(j-source%p.ny+p.ny)%p.ny;
                const size_t index=(size_t(row)*12+col)*grid+dx*p.ny+dy;
                kernel[index]=double(p.rhs_sign[t])*y[target];
                if(coupled(kind))cross[index]=double(opposite.rhs_sign[t])*y[opposite.dof[t]];
            }
            std::printf("[circulant] kind=%d sampled column=%d/12\n",kind,col+1);
        }
        if(coupled(kind)) {
            std::vector<Complex> negative(kernel);
            for(size_t j=0;j<kernel.size();++j){negative[j]-=opposite_weight_*cross[j];kernel[j]+=opposite_weight_*cross[j];}
            banks[kind].init(p.nx,p.ny,12,kind?1:12,kernel,shift);
            minus[kind].init(p.nx,p.ny,12,kind?1:12,negative,shift);
        }else banks[kind].init(p.nx,p.ny,12,kind?2:24,kernel,shift);
        input[kind].resize(size_t(kind?2:24)*12*grid);output[kind].resize(input[kind].size());
    }
    std::printf("[circulant] covered=%d/%d GPU_MiB=%.3f fallback_frequencies=%d mapping_error=%.3e\n",
        layout.covered_dofs,layout.system_dofs,storage_bytes()/1048576.,fallback_count(),layout.basis_mapping_error);
}
// Add S C R (or its adjoint). In the coupled case, an orthonormal
// sum/difference transform reduces [[D,X],[X,D]] to D+X and D-X.
void PrismCirculantInverse::add_fft(const Complex* rhs,Complex* out,int kind,double blend,bool adjoint,bool symmetric) {
    const int first=kind?24:0,count=kind?2:24,pairs=count/2;
    const bool pair=coupled(kind);const size_t slots=layout.patches[first].dof.size(),half=size_t(pairs)*slots;
    const double scale=1/std::sqrt(2.);
#pragma omp parallel for schedule(static)
    for(int k=0;k<(pair?pairs:count);++k) {
        const auto& p=layout.patches[first+k];const auto& ps=adjoint?p.solution_sign:p.rhs_sign;
        const size_t base=size_t(k)*slots;
        if(pair) {
            const auto& q=layout.patches[prism_opposite_patch(first+k)];const auto& qs=adjoint?q.solution_sign:q.rhs_sign;
            for(size_t j=0;j<slots;++j) {
                const Complex a=p.dof[j]<0?Complex(0):double(ps[j])*rhs[p.dof[j]];
                const Complex b=q.dof[j]<0?Complex(0):double(qs[j])*rhs[q.dof[j]];
                input[kind][base+j]=scale*(a+b);input[kind][half+base+j]=scale*(a-b);
            }
        }else for(size_t j=0;j<slots;++j)input[kind][base+j]=p.dof[j]<0?Complex(0):double(ps[j])*rhs[p.dof[j]];
    }
    auto& positive=kind==1 && symmetric?symmetric_caps:banks[kind];
    if(adjoint)positive.apply_adjoint(input[kind].data(),output[kind].data());else positive.apply(input[kind].data(),output[kind].data());
    if(pair) {
        auto& negative=kind==1 && symmetric?symmetric_caps_minus:minus[kind];
        if(adjoint)negative.apply_adjoint(input[kind].data()+half,output[kind].data()+half);
        else negative.apply(input[kind].data()+half,output[kind].data()+half);
    }
#pragma omp parallel for schedule(static)
    for(int k=0;k<(pair?pairs:count);++k) {
        const auto& p=layout.patches[first+k];const auto& ps=adjoint?p.rhs_sign:p.solution_sign;
        const size_t base=size_t(k)*slots;
        if(pair) {
            const auto& q=layout.patches[prism_opposite_patch(first+k)];const auto& qs=adjoint?q.rhs_sign:q.solution_sign;
            for(size_t j=0;j<slots;++j)if(p.dof[j]>=0) {
                const Complex a=output[kind][base+j],b=output[kind][half+base+j];
                out[p.dof[j]]+=blend*scale*double(ps[j])*(a+b);
                out[q.dof[j]]+=blend*scale*double(qs[j])*(a-b);
            }
        }else for(size_t j=0;j<slots;++j)if(p.dof[j]>=0)out[p.dof[j]]+=blend*double(ps[j])*output[kind][base+j];
    }
}
void PrismCirculantInverse::apply(const Complex* rhs,Complex* out,double blend) {
    if(!(blend>0 && blend<=1) || rhs==out)throw std::invalid_argument("circulant blend/alias");
    local(rhs,out);
    for(const auto& p:layout.patches)for(int d:p.dof)if(d>=0)out[d]*=1-blend;
    for(int kind=0;kind<2;++kind)add_fft(rhs,out,kind,blend);
}
void PrismCirculantInverse::prepare_symmetric_caps(const std::vector<CirculantSymmetry>& group) {
    device_mapping_.reset();
    const auto& p=layout.patches.at(24);
    for(const auto& g:group)if(g.n!=p.nx || g.channel.size()!=12)throw std::invalid_argument("cap FFT symmetry dimensions");
    symmetric_caps.init_inverse_kernel(p.nx,p.ny,12,coupled(1)?1:2,average_inverse_kernel(banks[1].inverse_kernel(),group));
    if(coupled(1))symmetric_caps_minus.init_inverse_kernel(p.nx,p.ny,12,1,average_inverse_kernel(minus[1].inverse_kernel(),group));
    symmetric_caps_ready=true;
}
void PrismCirculantInverse::apply_fft_part(const Complex* rhs,Complex* out,int selected_kind,bool use_symmetric_caps) {
    if(selected_kind<-1 || selected_kind>1 || rhs==out)throw std::invalid_argument("FFT part arguments");
    if(use_symmetric_caps && !symmetric_caps_ready)throw std::logic_error("cap FFT symmetry not prepared");
    if(device_mapping_ && device_mapping_enabled_ && !cpu_enabled_ && selected_kind==-1 && use_symmetric_caps==device_mapping_symmetric_) {
        device_mapping_->apply(rhs,out);return;
    }
    std::fill(out,out+layout.system_dofs,Complex(0));
    for(int kind=0;kind<2;++kind)if(selected_kind<0 || selected_kind==kind)add_fft(rhs,out,kind,1.,false,use_symmetric_caps);
}
void PrismCirculantInverse::apply_fft_part_adjoint(const Complex* rhs,Complex* out,int selected_kind,bool use_symmetric_caps) {
    if(selected_kind<-1 || selected_kind>1 || rhs==out)throw std::invalid_argument("adjoint FFT part arguments");
    if(use_symmetric_caps && !symmetric_caps_ready)throw std::logic_error("cap FFT symmetry not prepared");
    std::fill(out,out+layout.system_dofs,Complex(0));
    for(int kind=0;kind<2;++kind)if(selected_kind<0 || selected_kind==kind)add_fft(rhs,out,kind,1.,true,use_symmetric_caps);
}
void PrismCirculantInverse::prepare_device_mapping(bool symmetric) {
    if(cpu_enabled_)throw std::logic_error("device mapping requires GPU inverse");
    if(symmetric && !symmetric_caps_ready)throw std::logic_error("cap FFT symmetry not prepared");
    std::vector<MappedBlockCirculantFFT::Term> terms;
    for(int kind=0;kind<2;++kind){
        const int first=kind?24:0,count=kind?2:24;const bool pair=coupled(kind);
        const size_t slots=layout.patches.at(first).dof.size();
        for(int branch=0;branch<(pair?2:1);++branch){
            MappedBlockCirculantFFT::Term term;term.pair_sign=pair?(branch?-1:1):0;
            term.bank=branch?(kind && symmetric?&symmetric_caps_minus:&minus[kind]):(kind && symmetric?&symmetric_caps:&banks[kind]);
            for(int k=0;k<(pair?count/2:count);++k){
                const auto& p=layout.patches[first+k];
                const auto& q=layout.patches[pair?prism_opposite_patch(first+k):first+k];
                for(size_t j=0;j<slots;++j){
                    MappedBlockCirculantFFT::Slot s;s.first=p.dof[j];
                    if(s.first>=0){s.rhs_first=p.rhs_sign[j];s.solution_first=p.solution_sign[j];}
                    if(pair){s.second=q.dof[j];if(s.second>=0){s.rhs_second=q.rhs_sign[j];s.solution_second=q.solution_sign[j];}}
                    term.slots.push_back(s);
                }
            }
            terms.push_back(std::move(term));
        }
    }
    std::unique_ptr<MappedBlockCirculantFFT> candidate(new MappedBlockCirculantFFT);
    candidate->init(layout.system_dofs,terms);
    device_mapping_.swap(candidate);device_mapping_symmetric_=symmetric;device_mapping_enabled_=true;
}
void PrismCirculantInverse::apply_adjoint(const Complex* rhs,Complex* out,const Action& local_adjoint,double blend) {
    if(!(blend>0 && blend<=1))throw std::invalid_argument("circulant blend");
    const std::vector<Complex> seed(rhs,rhs+layout.system_dofs);std::vector<Complex> masked(seed);
    // P = D B + blend S C R. Hence P^H = B^H D + blend R^T C^H S^T.
    for(const auto& p:layout.patches)for(int dof:p.dof)if(dof>=0)masked[dof]*=1-blend;
    local_adjoint(masked.data(),out);
    for(int kind=0;kind<2;++kind)add_fft(seed.data(),out,kind,blend,true);
}
void PrismCirculantInverse::prepare_batch2(bool symmetric) {
    if(symmetric && !symmetric_caps_ready)throw std::logic_error("cap FFT symmetry not prepared");
    for(int kind=0;kind<2;++kind){
        (kind && symmetric?symmetric_caps:banks[kind]).prepare_batch2();
        if(coupled(kind))(kind && symmetric?symmetric_caps_minus:minus[kind]).prepare_batch2();
        pair_input[kind].resize(2*input[kind].size());pair_output[kind].resize(2*input[kind].size());
    }
}
void PrismCirculantInverse::apply_fft_part_batch2(const Complex* first,const Complex* second,
        Complex* first_result,Complex* second_result,bool symmetric) {
    if(first==first_result || first==second_result || second==first_result || second==second_result || first_result==second_result)
        throw std::invalid_argument("FFT pair alias");
    prepare_batch2(symmetric);
    const Complex* rhs[2]={first,second};Complex* out[2]={first_result,second_result};
    for(auto* p:out)std::fill(p,p+layout.system_dofs,Complex(0));
    const double scale=1/std::sqrt(2.);
    for(int kind=0;kind<2;++kind) {
        const int begin=kind?24:0,count=kind?2:24,pairs=count/2;const bool pair=coupled(kind);
        const size_t slots=layout.patches[begin].dof.size(),span=size_t(count)*slots,half=span/2;
        auto& in=pair_input[kind];auto& result=pair_output[kind];
#pragma omp parallel for collapse(2) schedule(static)
        for(int r=0;r<2;++r)for(int k=0;k<(pair?pairs:count);++k) {
            const auto& p=layout.patches[begin+k];const size_t base=r*span+size_t(k)*slots;
            if(pair){
                const auto& q=layout.patches[prism_opposite_patch(begin+k)];
                for(size_t j=0;j<slots;++j){
                    const Complex a=p.dof[j]<0?Complex(0):double(p.rhs_sign[j])*rhs[r][p.dof[j]];
                    const Complex b=q.dof[j]<0?Complex(0):double(q.rhs_sign[j])*rhs[r][q.dof[j]];
                    in[base+j]=scale*(a+b);in[half+base+j]=scale*(a-b);
                }
            }else for(size_t j=0;j<slots;++j)in[base+j]=p.dof[j]<0?Complex(0):double(p.rhs_sign[j])*rhs[r][p.dof[j]];
        }
        auto& positive=kind && symmetric?symmetric_caps:banks[kind];
        positive.apply_batch2(in.data(),in.data()+span,result.data(),result.data()+span);
        if(pair){auto& negative=kind && symmetric?symmetric_caps_minus:minus[kind];
            negative.apply_batch2(in.data()+half,in.data()+span+half,result.data()+half,result.data()+span+half);}
#pragma omp parallel for collapse(2) schedule(static)
        for(int r=0;r<2;++r)for(int k=0;k<(pair?pairs:count);++k) {
            const auto& p=layout.patches[begin+k];const size_t base=r*span+size_t(k)*slots;
            if(pair){const auto& q=layout.patches[prism_opposite_patch(begin+k)];
                for(size_t j=0;j<slots;++j)if(p.dof[j]>=0){
                    out[r][p.dof[j]]+=scale*double(p.solution_sign[j])*(result[base+j]+result[half+base+j]);
                    out[r][q.dof[j]]+=scale*double(q.solution_sign[j])*(result[base+j]-result[half+base+j]);
                }
            }else for(size_t j=0;j<slots;++j)if(p.dof[j]>=0)out[r][p.dof[j]]+=double(p.solution_sign[j])*result[base+j];
        }
    }
}
void PrismCirculantInverse::apply_transpose(const Complex* rhs,Complex* out,const Action& local_adjoint,double blend) {
    std::vector<Complex> seed(layout.system_dofs);for(int j=0;j<layout.system_dofs;++j)seed[j]=std::conj(rhs[j]);
    apply_adjoint(seed.data(),out,local_adjoint,blend);for(int j=0;j<layout.system_dofs;++j)out[j]=std::conj(out[j]);
}
