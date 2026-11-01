#pragma once
#include "prism_symmetry_sectors.h"
#include "streaming_host_gmres.h"

namespace bem_tmatrix {
// Reynolds average for preconditioner images only:
//   A_model = (1/24) sum_g R_rhs(g)^* A R_current(g).
// Reflections have opposite signs in equation and current representations.
// The original action remains separate for Krylov and strict residual checks.
// Each application costs exactly 24 original actions. Storage is O(n), and
// the existing C6 x horizontal maps are borrowed. One instance per worker.
class PrismD6hModelAction {
    const PrismGroupMaps& group_;
    bem_streaming::Action action_;
    std::vector<int> vertical_;
    std::vector<int8_t> vertical_sign_;
    Vector source_, rotated_, image_;
    size_t calls_=0, actions_=0;
public:
    PrismD6hModelAction(const MullerP2Mesh& mesh,const PrismGroupMaps& group,
                       bem_streaming::Action action)
        :group_(group),action_(std::move(action)) {
        const int n=mesh.system_dofs();
        if(n<=0 || !action_)throw std::invalid_argument("D6h model action dimensions/action");
        for(int g=0;g<12;++g){
            if(int(group.image[g].size())!=n || int(group.sign[g].size())!=n)
                throw std::invalid_argument("D6h model action map dimensions");
            std::vector<bool> seen(n,false);
            for(int i=0;i<n;++i){const int j=group.image[g][i];
                if(j<0 || j>=n || seen[j] || std::abs(int(group.sign[g][i]))!=1)
                    throw std::invalid_argument("D6h model action invalid signed permutation");
                seen[j]=true;
            }
        }
        std::vector<int> signs;
        D6hOrbit::vertical_image_map(mesh,vertical_,signs);
        vertical_sign_.assign(signs.begin(),signs.end());
        source_.resize(n);rotated_.resize(n);image_.resize(n);
    }
    size_t storage_bytes()const {
        return (source_.size()+rotated_.size()+image_.size())*sizeof(Complex)+
            vertical_.size()*sizeof(int)+vertical_sign_.size()*sizeof(int8_t);
    }
    size_t model_calls()const{return calls_;}
    size_t original_actions()const{return actions_;}
    void apply(const Complex* x,Complex* y) {
        const int n=source_.size();
        std::copy(x,x+n,source_.begin()); // also permits x == y
        std::fill(y,y+n,Complex(0));
        ++calls_;
        for(int vertical=0;vertical<2;++vertical)for(int g=0;g<12;++g){
#pragma omp parallel for schedule(static)
            for(int i=0;i<n;++i){int j=group_.image[g][i],s=group_.sign[g][i];
                if(vertical){s*=vertical_sign_[j];j=vertical_[j];}
                rotated_[j]=double(s)*source_[i];
            }
            ++actions_;action_(rotated_.data(),image_.data());
            const double determinant=(vertical?-1.:1.)*(g>=6?-1.:1.);
#pragma omp parallel for schedule(static)
            for(int i=0;i<n;++i){int j=group_.image[g][i],s=group_.sign[g][i];
                if(vertical){s*=vertical_sign_[j];j=vertical_[j];}
                y[i]+=(determinant*double(s)/24.)*image_[j];
            }
        }
    }
};
}
