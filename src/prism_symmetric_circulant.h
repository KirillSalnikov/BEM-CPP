#ifndef BEM_PRISM_SYMMETRIC_CIRCULANT_H
#define BEM_PRISM_SYMMETRIC_CIRCULANT_H
#include "prism_circulant_inverse.h"
#include "prism_orbit_inverse.h"
#include "prism_circulant_symmetry.h"
// Symmetrize only the cheap FFT contribution. The expensive local inverse
// is already equivariant and is applied once. The covered-row mask is also
// averaged; averaging the FFT alone would leave symmetry broken by the blend.
class PrismSymmetricCirculant {
    bem_tmatrix::PrismGroupMaps group_;
    std::vector<double> coverage_;
    std::vector<std::complex<double>> input_,output_;
    std::vector<int> cap_dofs_;
    bool fast_=false;
    bool full_fast_=false;
    std::vector<int> vertical_image_,vertical_sign_;
    PrismCirculantInverse* fft_=nullptr;
    const PrismOrbitInverse* local_=nullptr;
public:
    using Complex=std::complex<double>;
    bool fast_patch_orbits()const{return fast_;}
    bool full_fft_symmetry()const{return full_fast_;}
    bool prepare_full_fft_symmetry(const MullerP2Mesh& mesh) {
        full_fast_=false;if(!fast_)return false;
        bem_tmatrix::D6hOrbit::vertical_image_map(mesh,vertical_image_,vertical_sign_);
        PrismCirculantSymmetry proof(fft_->layout);
        if(!proof.unchanged(vertical_image_,vertical_sign_,-1,0,24,fft_->coupled(0)))return false;
        CirculantSymmetry rotation[2],vertical[2];
        for(int cap=0;cap<2;++cap)
            if(!proof.affine(24+cap,group_.image[1],group_.sign[1],1,rotation[cap]) ||
               !proof.affine(24+cap,vertical_image_,vertical_sign_,-1,vertical[cap]))return false;
        // Both batches share a symbol. Equal conjugations permit a common
        // overall sign on each generator, but no channel-dependent mismatch.
        const auto same=[&](const CirculantSymmetry& a,const CirculantSymmetry& b) {
            if(a.n!=b.n || a.matrix!=b.matrix || a.channel!=b.channel || a.offset!=b.offset)return false;
            const int common=a.solution_sign[0]*b.solution_sign[0];
            if(fft_->coupled(1) && common!=1)return false;
            for(size_t c=0;c<a.channel.size();++c)
                if(a.solution_sign[c]*b.solution_sign[c]!=common || a.rhs_sign[c]*b.rhs_sign[c]!=common)return false;
            return true;
        };
        if(!same(rotation[0],rotation[1]) || !same(vertical[0],vertical[1]))return false;
        std::vector<CirculantSymmetry> maps;
        auto r=CirculantSymmetry::identity(rotation[0].n,12);
        for(int k=0;k<6;++k) {
            maps.push_back(r);maps.push_back(compose(vertical[0],r));r=compose(rotation[0],r);
        }
        fft_->prepare_symmetric_caps(maps);full_fast_=true;return true;
    }
    void apply_full_fft_symmetry(const Complex* rhs,Complex* result,double blend=.25) {
        if(!full_fast_ || rhs==result || !std::isfinite(blend) || blend<0 || blend>1)
            throw std::invalid_argument("full FFT symmetry arguments");
        const int n=input_.size();local_->apply(rhs,result);
#pragma omp parallel for schedule(static)
        for(int i=0;i<n;++i) {
            result[i]*=.5*(1-blend*coverage_[i]);
            input_[vertical_image_[i]]=-double(vertical_sign_[i])*rhs[i];
        }
        local_->apply(input_.data(),output_.data());
#pragma omp parallel for schedule(static)
        for(int i=0;i<n;++i)result[vertical_image_[i]]+=.5*double(vertical_sign_[i])*(1-blend*coverage_[i])*output_[i];
        fft_->apply_fft_part(rhs,output_.data(),-1,true);
#pragma omp parallel for schedule(static)
        for(int i=0;i<n;++i)result[i]+=blend*output_[i];
    }
    void build(const bem_tmatrix::D6hOrbit& orbit,PrismCirculantInverse& fft,const PrismOrbitInverse& local) {
        if(orbit.size()<=0 || fft.layout.system_dofs!=orbit.size() || fft.layout.patches.size()!=26)
            throw std::invalid_argument("symmetric circulant geometry");
        group_.build(orbit);fft_=&fft;local_=&local;const int n=orbit.size();
        full_fast_=false;
        input_.resize(n);output_.resize(n);coverage_.assign(n,0.);
        for(const auto& patch:fft.layout.patches)for(int d:patch.dof)if(d>=0)
            for(int g=0;g<12;++g)coverage_[group_.image[g][d]]+=1./12;
        const PrismCirculantSymmetry proof(fft.layout);
        // In a coupled bank the group must also preserve the opposite pairing
        // and give both ends the same local sign (no parity-sector mixing).
        fast_=proof.unchanged(group_.image[1],group_.sign[1],1,0,24,fft.coupled(0)) &&
              proof.unchanged(group_.image[6],group_.sign[6],-1,0,24,fft.coupled(0)) &&
              proof.unchanged(group_.image[6],group_.sign[6],-1,24,26,fft.coupled(1));
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
