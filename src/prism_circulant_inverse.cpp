#include "prism_circulant_inverse.h"
#include <cstdio>
#include <limits>
#include <stdexcept>

void PrismCirculantInverse::build(const MullerP2Mesh& mesh,int ref,const Action& action,const Action& inverse,double shift,double opposite_weight,int opposite_mask) {
    if(!std::isfinite(opposite_weight) || opposite_weight<0 || opposite_weight>1 || opposite_mask<0 || opposite_mask>3)
        throw std::invalid_argument("opposite face coupling weight/mask");
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
    const auto& p=layout.patches.at(24);
    for(const auto& g:group)if(g.n!=p.nx || g.channel.size()!=12)throw std::invalid_argument("cap FFT symmetry dimensions");
    symmetric_caps.init_inverse_kernel(p.nx,p.ny,12,coupled(1)?1:2,average_inverse_kernel(banks[1].inverse_kernel(),group));
    if(coupled(1))symmetric_caps_minus.init_inverse_kernel(p.nx,p.ny,12,1,average_inverse_kernel(minus[1].inverse_kernel(),group));
    symmetric_caps_ready=true;
}
void PrismCirculantInverse::apply_fft_part(const Complex* rhs,Complex* out,int selected_kind,bool use_symmetric_caps) {
    if(selected_kind<-1 || selected_kind>1 || rhs==out)throw std::invalid_argument("FFT part arguments");
    if(use_symmetric_caps && !symmetric_caps_ready)throw std::logic_error("cap FFT symmetry not prepared");
    std::fill(out,out+layout.system_dofs,Complex(0));
    for(int kind=0;kind<2;++kind)if(selected_kind<0 || selected_kind==kind)add_fft(rhs,out,kind,1.,false,use_symmetric_caps);
}
void PrismCirculantInverse::apply_adjoint(const Complex* rhs,Complex* out,const Action& local_adjoint,double blend) {
    if(!(blend>0 && blend<=1))throw std::invalid_argument("circulant blend");
    const std::vector<Complex> seed(rhs,rhs+layout.system_dofs);std::vector<Complex> masked(seed);
    // P = D B + blend S C R. Hence P^H = B^H D + blend R^T C^H S^T.
    for(const auto& p:layout.patches)for(int dof:p.dof)if(dof>=0)masked[dof]*=1-blend;
    local_adjoint(masked.data(),out);
    for(int kind=0;kind<2;++kind)add_fft(seed.data(),out,kind,blend,true);
}
void PrismCirculantInverse::apply_transpose(const Complex* rhs,Complex* out,const Action& local_adjoint,double blend) {
    std::vector<Complex> seed(layout.system_dofs);for(int j=0;j<layout.system_dofs;++j)seed[j]=std::conj(rhs[j]);
    apply_adjoint(seed.data(),out,local_adjoint,blend);for(int j=0;j<layout.system_dofs;++j)out[j]=std::conj(out[j]);
}
