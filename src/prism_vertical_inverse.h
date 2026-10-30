#ifndef BEM_PRISM_VERTICAL_INVERSE_H
#define BEM_PRISM_VERTICAL_INVERSE_H
#include "tmatrix_symmetry_warm_start.h"

// Complete an already C6/horizontal-equivariant inverse to D6h by averaging
// its two cosets. Reflection acts with opposite signs on equations/currents.
// This doubles inverse actions, so it is opt-in and must win a solve benchmark.
class PrismVerticalInverse {
    bem_streaming::Action inverse_;
    std::vector<int> image_,sign_;
    bem_tmatrix::Vector input_,output_;
public:
    using Complex=bem_streaming::Complex;
    void build(const MullerP2Mesh& mesh,const bem_streaming::Action& inverse) {
        bem_tmatrix::D6hOrbit::vertical_image_map(mesh,image_,sign_);inverse_=inverse;
        const int n=image_.size();input_.resize(n);output_.resize(n);
        for(int i=0;i<n;++i)if(image_[image_[i]]!=i || sign_[i]*sign_[image_[i]]!=1)
            throw std::logic_error("vertical symmetry is not an involution");
    }
    void transform(const Complex* in,Complex* out,bool rhs)const {
        if(in==out)throw std::invalid_argument("vertical transform alias");
#pragma omp parallel for schedule(static)
        for(size_t i=0;i<image_.size();++i)out[image_[i]]=double((rhs?-1:1)*sign_[i])*in[i];
    }
    void apply(const Complex* rhs,Complex* result) {
        if(!inverse_ || rhs==result)throw std::invalid_argument("vertical inverse arguments");
        inverse_(rhs,result);transform(rhs,input_.data(),true);inverse_(input_.data(),output_.data());
#pragma omp parallel for schedule(static)
        for(size_t i=0;i<image_.size();++i)result[image_[i]]=.5*(result[image_[i]]+double(sign_[i])*output_[i]);
    }
};
#endif
