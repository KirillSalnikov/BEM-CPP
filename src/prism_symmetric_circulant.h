#ifndef BEM_PRISM_SYMMETRIC_CIRCULANT_H
#define BEM_PRISM_SYMMETRIC_CIRCULANT_H
#include "prism_circulant_inverse.h"
#include "prism_orbit_inverse.h"
// Symmetrize only the cheap FFT contribution. The expensive local inverse
// is already equivariant and is applied once. The covered-row mask is also
// averaged; averaging the FFT alone would leave symmetry broken by the blend.
class PrismSymmetricCirculant {
    bem_tmatrix::PrismGroupMaps group_;
    std::vector<double> coverage_;
    std::vector<std::complex<double>> input_,output_;
    std::vector<int> cap_dofs_;
    bool fast_=false;
    PrismCirculantInverse* fft_=nullptr;
    const PrismOrbitInverse* local_=nullptr;
public:
    using Complex=std::complex<double>;
    bool fast_patch_orbits()const{return fast_;}
    void build(const bem_tmatrix::D6hOrbit& orbit,PrismCirculantInverse& fft,const PrismOrbitInverse& local) {
        if(orbit.size()<=0 || fft.layout.system_dofs!=orbit.size() || fft.layout.patches.size()!=26)
            throw std::invalid_argument("symmetric circulant geometry");
        group_.build(orbit);fft_=&fft;local_=&local;const int n=orbit.size();
        input_.resize(n);output_.resize(n);coverage_.assign(n,0.);
        for(const auto& patch:fft.layout.patches)for(int d:patch.dof)if(d>=0)
            for(int g=0;g<12;++g)coverage_[group_.image[g][d]]+=1./12;
        std::vector<int> patch_of(n,-1),slot_of(n,-1);
        for(size_t p=0;p<fft.layout.patches.size();++p)for(size_t j=0;j<fft.layout.patches[p].dof.size();++j) {
            const int d=fft.layout.patches[p].dof[j];if(d<0)continue;
            if(patch_of[d]>=0)throw std::logic_error("overlapping FFT patches");
            patch_of[d]=p;slot_of[d]=j;
        }
        // Structural proof: the generator permutes entire patches, preserves
        // local channel/grid slots and supplies one equal input/output sign.
        // Every patch in a bank uses the same frequency inverse. No random
        // dot test is used to authorize this optimization.
        const auto invariant=[&](int generator,int first,int last) {
            for(int p=first;p<last;++p) {
                int target_patch=-1,common_sign=0;const auto& patch=fft.layout.patches[p];
                for(size_t j=0;j<patch.dof.size();++j) {
                    const int d=patch.dof[j];if(d<0)continue;
                    const int target=group_.image[generator][d],q=patch_of[target];
                    if(q<first || q>=last || slot_of[target]!=int(j))return false;
                    if(target_patch<0)target_patch=q;else if(target_patch!=q)return false;
                    const auto& other=fft.layout.patches[q];
                    const int sol=group_.sign[generator][d]*patch.solution_sign[j]*other.solution_sign[j];
                    const int rhs=(generator<6?1:-1)*group_.sign[generator][d]*patch.rhs_sign[j]*other.rhs_sign[j];
                    if(sol!=rhs || (common_sign && common_sign!=sol))return false;
                    common_sign=sol;
                }
            }
            return true;
        };
        fast_=invariant(1,0,24) && invariant(6,0,24) && invariant(6,24,26);
        cap_dofs_.clear();for(int p=24;p<26;++p)for(int d:fft.layout.patches[p].dof)if(d>=0)cap_dofs_.push_back(d);
    }
    void apply(const Complex* rhs,Complex* result,double blend=.25,bool force_full_group=false) {
        if(!fft_ || !local_ || rhs==result || !std::isfinite(blend) || blend<0 || blend>1)throw std::invalid_argument("symmetric circulant arguments");
        const int n=input_.size();local_->apply(rhs,result);
#pragma omp parallel for schedule(static)
        for(int i=0;i<n;++i)result[i]*=1-blend*coverage_[i];
        if(fast_ && !force_full_group) {
            fft_->apply_fft_part(rhs,output_.data(),0);
#pragma omp parallel for schedule(static)
            for(int i=0;i<n;++i)result[i]+=blend*output_[i];
            for(int g=0;g<6;++g) {
#pragma omp parallel for schedule(static)
                for(size_t j=0;j<cap_dofs_.size();++j) {
                    const int i=cap_dofs_[j];input_[i]=double(group_.sign[g][i])*rhs[group_.image[g][i]];
                }
                fft_->apply_fft_part(input_.data(),output_.data(),1);
#pragma omp parallel for schedule(static)
                for(size_t j=0;j<cap_dofs_.size();++j) {
                    const int i=cap_dofs_[j];result[group_.image[g][i]]+=blend/6.*double(group_.sign[g][i])*output_[i];
                }
            }
            return;
        }
        for(int g=0;g<12;++g) {
#pragma omp parallel for schedule(static)
            for(int i=0;i<n;++i)input_[i]=double((g<6?1:-1)*group_.sign[g][i])*rhs[group_.image[g][i]];
            fft_->apply_fft_part(input_.data(),output_.data());
#pragma omp parallel for schedule(static)
            for(int i=0;i<n;++i)result[group_.image[g][i]]+=blend/12.*double(group_.sign[g][i])*output_[i];
        }
    }
};
#endif
