#include "prism_circulant_inverse.h"
#include <cstdio>
#include <limits>
#include <stdexcept>

void PrismCirculantInverse::build(const MullerP2Mesh& mesh,int ref,const Action& action,const Action& inverse,double shift) {
    layout.build(mesh,ref);local=inverse;
    std::vector<Complex> x(layout.system_dofs),y(layout.system_dofs);
    for(int kind=0;kind<2;++kind) {
        const auto& p=layout.patches[kind?24:0];
        const int grid=p.nx*p.ny;
        int source=-1;double best=std::numeric_limits<double>::infinity();
        for(int i=0;i<p.nx;++i)for(int j=0;j<p.ny;++j) {
            bool valid=true;for(int c=0;c<12;++c)valid=valid && p.dof[p.slot(c,i,j)]>=0;
            const double distance=std::pow(i-(p.nx-1)*.5,2)+std::pow(j-(p.ny-1)*.5,2);
            if(valid && distance<best){best=distance;source=i*p.ny+j;}
        }
        if(source<0)throw std::runtime_error("no complete circulant source cell");
        std::vector<Complex> kernel(size_t(144)*grid);
        for(int col=0;col<12;++col) {
            const int slot=col*grid+source,dof=p.dof[slot];
            x[dof]=p.solution_sign[slot];action(x.data(),y.data());x[dof]=0;
            for(int row=0;row<12;++row)for(int i=0;i<p.nx;++i)for(int j=0;j<p.ny;++j) {
                const int t=p.slot(row,i,j),target=p.dof[t];if(target<0)continue;
                const int dx=(i-source/p.ny+p.nx)%p.nx,dy=(j-source%p.ny+p.ny)%p.ny;
                kernel[(size_t(row)*12+col)*grid+dx*p.ny+dy]=double(p.rhs_sign[t])*y[target];
            }
            std::printf("[circulant] kind=%d sampled column=%d/12\n",kind,col+1);
        }
        banks[kind].init(p.nx,p.ny,12,kind?2:24,kernel,shift);
        input[kind].resize(size_t(kind?2:24)*12*grid);output[kind].resize(input[kind].size());
    }
    std::printf("[circulant] covered=%d/%d GPU_MiB=%.3f fallback_frequencies=%d mapping_error=%.3e\n",
        layout.covered_dofs,layout.system_dofs,storage_bytes()/1048576.,fallback_count(),layout.basis_mapping_error);
}
void PrismCirculantInverse::apply(const Complex* rhs,Complex* out,double blend) {
    if(!(blend>0 && blend<=1))throw std::invalid_argument("circulant blend");
    local(rhs,out);
    for(int kind=0;kind<2;++kind) {
        const int first=kind?24:0,last=kind?26:24;
#pragma omp parallel for schedule(static)
        for(int index=first;index<last;++index) {
            const auto& p=layout.patches[index];const size_t base=size_t(index-first)*p.dof.size();
            for(size_t j=0;j<p.dof.size();++j)input[kind][base+j]=p.dof[j]<0?Complex(0):double(p.rhs_sign[j])*rhs[p.dof[j]];
        }
        banks[kind].apply(input[kind].data(),output[kind].data());
#pragma omp parallel for schedule(static)
        for(int index=first;index<last;++index) {
            const auto& p=layout.patches[index];const size_t base=size_t(index-first)*p.dof.size();
            for(size_t j=0;j<p.dof.size();++j)if(p.dof[j]>=0)
                out[p.dof[j]]=(1-blend)*out[p.dof[j]]+blend*double(p.solution_sign[j])*output[kind][base+j];
        }
    }
}
void PrismCirculantInverse::apply_fft_part(const Complex* rhs,Complex* out,int selected_kind) {
    if(selected_kind<-1 || selected_kind>1 || rhs==out)throw std::invalid_argument("FFT part arguments");
    std::fill(out,out+layout.system_dofs,Complex(0));
    for(int kind=0;kind<2;++kind) {
        if(selected_kind>=0 && selected_kind!=kind)continue;
        const int first=kind?24:0,last=kind?26:24;
#pragma omp parallel for schedule(static)
        for(int index=first;index<last;++index) {
            const auto& p=layout.patches[index];const size_t base=size_t(index-first)*p.dof.size();
            for(size_t j=0;j<p.dof.size();++j)input[kind][base+j]=p.dof[j]<0?Complex(0):double(p.rhs_sign[j])*rhs[p.dof[j]];
        }
        banks[kind].apply(input[kind].data(),output[kind].data());
#pragma omp parallel for schedule(static)
        for(int index=first;index<last;++index) {
            const auto& p=layout.patches[index];const size_t base=size_t(index-first)*p.dof.size();
            for(size_t j=0;j<p.dof.size();++j)if(p.dof[j]>=0)out[p.dof[j]]=double(p.solution_sign[j])*output[kind][base+j];
        }
    }
}
void PrismCirculantInverse::apply_adjoint(const Complex* rhs,Complex* out,const Action& local_adjoint,double blend) {
    if(!(blend>0 && blend<=1))throw std::invalid_argument("circulant blend");
    const std::vector<Complex> seed(rhs,rhs+layout.system_dofs);std::vector<Complex> masked(seed);
    // P = D B + blend S C R. Hence P^H = B^H D + blend R^T C^H S^T.
    for(const auto& p:layout.patches)for(int dof:p.dof)if(dof>=0)masked[dof]*=1-blend;
    local_adjoint(masked.data(),out);
    for(int kind=0;kind<2;++kind) {
        const int first=kind?24:0,last=kind?26:24;
#pragma omp parallel for schedule(static)
        for(int index=first;index<last;++index) {
            const auto& p=layout.patches[index];const size_t base=size_t(index-first)*p.dof.size();
            for(size_t j=0;j<p.dof.size();++j)input[kind][base+j]=p.dof[j]<0?Complex(0):double(p.solution_sign[j])*seed[p.dof[j]];
        }
        banks[kind].apply_adjoint(input[kind].data(),output[kind].data());
#pragma omp parallel for schedule(static)
        for(int index=first;index<last;++index) {
            const auto& p=layout.patches[index];const size_t base=size_t(index-first)*p.dof.size();
            for(size_t j=0;j<p.dof.size();++j)if(p.dof[j]>=0)out[p.dof[j]]+=blend*double(p.rhs_sign[j])*output[kind][base+j];
        }
    }
}
void PrismCirculantInverse::apply_transpose(const Complex* rhs,Complex* out,const Action& local_adjoint,double blend) {
    std::vector<Complex> seed(layout.system_dofs);for(int j=0;j<layout.system_dofs;++j)seed[j]=std::conj(rhs[j]);
    apply_adjoint(seed.data(),out,local_adjoint,blend);for(int j=0;j<layout.system_dofs;++j)out[j]=std::conj(out[j]);
}
