#pragma once
#include "prism_circulant_layout.h"
#include "prism_symmetry_sectors.h"

// Constant lattice-channel aggregates on all FFT patches. Disjoint support
// makes V orthonormal without a dense QR. Its span is closed under FULL D6h,
// including the vertical mirror. Only one A V column per group orbit is
// sampled; stabilizer projection and signed transport define an equivariant
// coarse approximation. The original physical action is never replaced.
class PrismFaceCoarseBasis {
public:
    using Complex=bem_streaming::Complex;
    using Action=bem_streaming::Action;
private:
    int n_=0,rank_=0,samples_=0,cached_root_=-1;
    std::array<std::vector<int>,24> image_,coarse_image_;
    std::array<std::vector<int8_t>,24> sign_,coarse_sign_;
    std::array<int,24> determinant_{};
    std::vector<int> column_,roots_,transport_,stabilizer_roots_;
    std::vector<int8_t> transport_sign_;
    std::vector<double> weight_;
    std::vector<Complex> cached_,scratch_;
public:
    PrismFaceCoarseBasis(const PrismCirculantLayout& layout,const bem_tmatrix::D6hOrbit& orbit,const MullerP2Mesh& mesh) {
        n_=layout.system_dofs;
        if(orbit.size()!=n_)throw std::invalid_argument("face coarse orbit dimensions");
        bem_tmatrix::PrismGroupMaps group;group.build(orbit);
        std::vector<int> vertical,vertical_sign;
        bem_tmatrix::D6hOrbit::vertical_image_map(mesh,vertical,vertical_sign);
        for(int g=0;g<12;++g) {
            image_[g]=std::move(group.image[g]);sign_[g]=std::move(group.sign[g]);
            image_[g+12].resize(n_);sign_[g+12].resize(n_);
            determinant_[g]=g<6?1:-1;determinant_[g+12]=-determinant_[g];
            for(int i=0;i<n_;++i){const int t=image_[g][i];image_[g+12][i]=vertical[t];sign_[g+12][i]=vertical_sign[t]*sign_[g][i];}
        }
        std::vector<int> raw_column(n_,-1),counts;
        std::vector<int8_t> raw_sign(n_,0);
        for(const auto& patch:layout.patches) {
            const int grid=patch.nx*patch.ny;
            for(int c=0;c<12;++c) {
                int count=0;
                for(int j=0;j<grid;++j)if(patch.dof[c*grid+j]>=0)++count;
                if(!count)continue;
                const int column=counts.size();counts.push_back(count);
                for(int j=0;j<grid;++j) {
                    const int slot=c*grid+j,d=patch.dof[slot];if(d<0)continue;
                    if(raw_column[d]>=0)throw std::logic_error("overlapping coarse aggregates");
                    raw_column[d]=column;raw_sign[d]=patch.solution_sign[slot];
                }
            }
        }
        rank_=counts.size();
        if(!rank_)throw std::logic_error("empty face coarse basis");
        std::array<std::vector<int>,24> raw_image;
        std::array<std::vector<int8_t>,24> raw_coarse_sign;
        for(int g=0;g<24;++g) {
            raw_image[g].assign(rank_,-1);raw_coarse_sign[g].assign(rank_,0);
            for(int i=0;i<n_;++i)if(raw_column[i]>=0) {
                const int c=raw_column[i],t=image_[g][i],q=raw_column[t];
                if(q<0 || counts[c]!=counts[q])throw std::logic_error("face coarse support is not D6h invariant");
                const int s=sign_[g][i]*raw_sign[i]*raw_sign[t];
                if(raw_image[g][c]<0){raw_image[g][c]=q;raw_coarse_sign[g][c]=s;}
                else if(raw_image[g][c]!=q || raw_coarse_sign[g][c]!=s)throw std::logic_error("face coarse channel is not invariant");
            }
        }
        std::vector<int> order,inverse(rank_,-1);
        for(int seed=0;seed<rank_;++seed)if(inverse[seed]<0) {
            const int root=order.size();stabilizer_roots_.push_back(root);
            for(int g=0;g<24;++g) {
                const int target=raw_image[g][seed];
                if(target<0)throw std::logic_error("incomplete coarse symmetry map");
                if(inverse[target]>=0)continue;
                inverse[target]=order.size();order.push_back(target);
                roots_.push_back(root);transport_.push_back(g);transport_sign_.push_back(raw_coarse_sign[g][seed]);
            }
        }
        column_.assign(n_,-1);weight_.assign(n_,0.);
        for(int i=0;i<n_;++i)if(raw_column[i]>=0) {
            const int c=raw_column[i];column_[i]=inverse[c];weight_[i]=double(raw_sign[i])/std::sqrt(double(counts[c]));
        }
        for(int g=0;g<24;++g) {
            coarse_image_[g].resize(rank_);coarse_sign_[g].resize(rank_);
            for(int j=0;j<rank_;++j){coarse_image_[g][j]=inverse[raw_image[g][order[j]]];coarse_sign_[g][j]=raw_coarse_sign[g][order[j]];}
        }
    }
    int size()const{return n_;}
    int rank()const{return rank_;}
    int independent_columns()const{return stabilizer_roots_.size();}
    int samples()const{return samples_;}
    const std::vector<int>& columns()const{return column_;}
    const std::vector<double>& weights()const{return weight_;}
    int column_image(int g,int j)const{return coarse_image_.at(g).at(j);}
    int column_sign(int g,int j)const{return coarse_sign_.at(g).at(j);}
    void generate(int j,Complex* out)const {
        if(j<0 || j>=rank_)throw std::invalid_argument("coarse column");
#pragma omp parallel for schedule(static)
        for(int i=0;i<n_;++i)out[i]=column_[i]==j?weight_[i]:0.;
    }
    void transform(int g,const Complex* in,Complex* out,bool rhs)const {
        if(g<0 || g>=24 || in==out)throw std::invalid_argument("face coarse transform");
        const int parity=rhs?determinant_[g]:1;
#pragma omp parallel for schedule(static)
        for(int i=0;i<n_;++i)out[image_[g][i]]=double(parity*sign_[g][i])*in[i];
    }
    void sample(int j,const Complex* x,Complex* y,const Action& action) {
        if(j<0 || j>=rank_)throw std::invalid_argument("face coarse sample column");
        const int root=roots_[j];
        if(j==root) {
            cached_.assign(n_,Complex(0));scratch_.resize(n_);action(x,scratch_.data());++samples_;
            int stabilizers=0;
            for(int g=0;g<24;++g)if(coarse_image_[g][j]==j) {
                ++stabilizers;const int parity=coarse_sign_[g][j]*determinant_[g];
#pragma omp parallel for schedule(static)
                for(int i=0;i<n_;++i)cached_[image_[g][i]]+=double(parity*sign_[g][i])*scratch_[i];
            }
            for(auto& z:cached_)z/=double(stabilizers);
            cached_root_=root;
            std::printf("[face coarse] full sample %d/%d, rank=%d\n",samples_,independent_columns(),rank_);
        }
        if(cached_root_!=root)throw std::logic_error("face coarse samples must follow orbit order");
        transform(transport_[j],cached_.data(),y,true);
        if(transport_sign_[j]<0)for(int i=0;i<n_;++i)y[i]=-y[i];
    }
};
