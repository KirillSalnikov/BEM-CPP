#pragma once
#include "prism_symmetry_sectors.h"
#include "streaming_image_space.h"

// Experimental fixed coarse inverse from transported physical current
// columns. Equation and current representations have opposite mirror signs.
// Closing each supplied pair under the vertical reflection, followed by
// independent sector projections, closes the space under the complete D6h.
class CoarseImageExperiment {
    using Complex=bem_streaming::Complex;
    using Vector=bem_tmatrix::Vector;
    const bem_tmatrix::PrismSymmetrySectors& sectors_;
    const bem_streaming::Action local_;
    bem_streaming::ImageSpace space_;
    std::vector<int> vertical_,sign_;
    Vector encoded_,guess_,removed_,physical_guess_,physical_removed_,residual_;
public:
    CoarseImageExperiment(const MullerP2Mesh& mesh,const bem_tmatrix::PrismSymmetrySectors& sectors,
                          const bem_streaming::Action& local)
        :sectors_(sectors),local_(local),space_(std::vector<int>(sectors.offsets.begin(),sectors.offsets.end())),
         encoded_(sectors.size()),physical_guess_(sectors.size()),physical_removed_(sectors.size()),residual_(sectors.size()) {
        bem_tmatrix::D6hOrbit::vertical_image_map(mesh,vertical_,sign_);
        if(mesh.system_dofs()!=sectors.size())throw std::invalid_argument("coarse image dimensions");
    }
    int rank()const{return space_.rank();}
    size_t bytes()const{return space_.bytes();}
    void add_d6h(const Vector& u,const Vector& image) {
        const int n=sectors_.size();
        if(int(u.size())!=n || int(image.size())!=n)throw std::invalid_argument("coarse image column size");
        Vector current(n),rhs(n),ec(n),er(n);
        for(int mirrored=0;mirrored<2;++mirrored) {
            if(!mirrored){current=u;rhs=image;}
            else for(int i=0;i<n;++i){current[vertical_[i]]=double(sign_[i])*u[i];rhs[vertical_[i]]=-double(sign_[i])*image[i];}
            sectors_.encode(current.data(),ec.data(),false);sectors_.encode(rhs.data(),er.data(),true);
            space_.add(ec,er);
        }
    }
    void guess(const Complex* rhs,Complex* out) {
        sectors_.encode(rhs,encoded_.data(),true);space_.apply(encoded_,guess_,removed_);
        sectors_.decode(guess_.data(),out,false);
    }
    void apply(const Complex* rhs,Complex* out) {
        const int n=sectors_.size();
        sectors_.encode(rhs,encoded_.data(),true);space_.apply(encoded_,guess_,removed_);
        sectors_.decode(guess_.data(),physical_guess_.data(),false);
        sectors_.decode(removed_.data(),physical_removed_.data(),true);
        for(int i=0;i<n;++i)residual_[i]=rhs[i]-physical_removed_[i];
        local_(residual_.data(),out);
        for(int i=0;i<n;++i)out[i]+=physical_guess_[i];
    }
    // The sector Arnoldi wrapper already has these equation coordinates.
    // Avoid decode->encode and decode(rhs)-decode(removed) on every action.
    void apply_encoded(const Complex* coordinates,Complex* out) {
        const int n=sectors_.size();
        std::copy(coordinates,coordinates+n,encoded_.begin());
        space_.apply(encoded_,guess_,removed_);
        for(int i=0;i<n;++i)encoded_[i]-=removed_[i];
        sectors_.decode(encoded_.data(),residual_.data(),true);
        local_(residual_.data(),out);
        sectors_.decode(guess_.data(),physical_guess_.data(),false);
        for(int i=0;i<n;++i)out[i]+=physical_guess_[i];
    }
};
