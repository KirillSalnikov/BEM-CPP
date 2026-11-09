#pragma once
#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

namespace bem_orbit_device {
struct SymmetricLayout {
    std::vector<double> coverage;
    std::vector<int> vertical_image,vertical_sign;
    void validate()const {
        const size_t n=coverage.size();
        if(!n || vertical_image.size()!=n || vertical_sign.size()!=n)
            throw std::invalid_argument("resident symmetry dimensions");
        for(size_t i=0;i<n;++i){
            const int j=vertical_image[i],s=vertical_sign[i];
            if(j<0 || size_t(j)>=n || (s!=1 && s!=-1) || !std::isfinite(coverage[i]) ||
               coverage[i]<0 || coverage[i]>1.+1e-12)
                throw std::invalid_argument("resident symmetry map/mask");
        }
        for(size_t i=0;i<n;++i){const int j=vertical_image[i];
            if(vertical_image[j]!=int(i) || vertical_sign[i]*vertical_sign[j]!=1)
                throw std::invalid_argument("resident reflection is not an involution");
        }
    }
    template<class Inverse> static SymmetricLayout capture(const Inverse& inverse){
        SymmetricLayout result;
        inverse.visit_full_symmetry_layout([&](const std::vector<double>& c,const std::vector<int>& i,const std::vector<int>& s){
            result.coverage=c;result.vertical_image=i;result.vertical_sign=s;
        });result.validate();return result;
    }
    // Independent host control of the final blend. Output must not overlap
    // reflected, whose rows are gathered using the vertical permutation.
    void combine_host(const std::complex<double>* first,const std::complex<double>* reflected,
                      const std::complex<double>* fft,std::complex<double>* out,double blend)const {
        if(!first || !reflected || !fft || !out || !std::isfinite(blend) || blend<0 || blend>1)
            throw std::invalid_argument("resident blend arguments");
        for(size_t i=0;i<coverage.size();++i){const int j=vertical_image[i];
            out[i]=.5*(1-blend*coverage[i])*first[i]+.5*double(vertical_sign[j])*(1-blend*coverage[j])*reflected[j]+blend*fft[i];
        }
    }
};
}
