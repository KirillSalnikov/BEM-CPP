#ifndef BEM_STREAMING_TWO_LEVEL_H
#define BEM_STREAMING_TWO_LEVEL_H
#include "streaming_recycle_space.h"

namespace bem_streaming {
// Fixed right inverse P = M^-1 (I-C C*) + U C*, with A_inner U ~= C.
// Keep the same coarse space throughout an Arnoldi cycle AND its final
// solution update. Do not learn into this space through the inverse callback.
// One instance owns scratch for one serial solve; different solves need their
// own instances. No additional FMM actions or explicit dense inverse are used.
class TwoLevelInverse {
    const RecycleSpace& space_;
    const Action local_;
    const std::uint64_t generation_;
    const int rank_;
    std::vector<Complex> residual_,local_work_;
public:
    TwoLevelInverse(const RecycleSpace& space,const Action& local)
        : space_(space),local_(local),generation_(space.generation()),rank_(space.rank()),
          residual_(space.size()),local_work_(space.size()) {}
    void operator()(const Complex* rhs,Complex* solution) {
        if(space_.generation()!=generation_ || space_.rank()!=rank_)
            throw std::runtime_error("two-level coarse space changed during a fixed solve");
        if(!rank_) {local_(rhs,solution);return;}
        const int n=space_.size();
        // Copy first, so rhs and solution may alias.
        std::copy(rhs,rhs+n,residual_.begin());
        std::fill(solution,solution+n,Complex(0));
        space_.correct(solution,residual_.data());
        local_(residual_.data(),local_work_.data());
#pragma omp parallel for schedule(static)
        for(int i=0;i<n;++i)solution[i]+=local_work_[i];
    }
};
}
#endif
