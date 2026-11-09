#pragma once
#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace bem_taylor {
using MomentVector=std::vector<std::complex<double>>;
using MomentHistory=std::vector<MomentVector>;
using MomentConvolution=std::function<void(const MomentHistory&,MomentHistory&)>;
using MomentColumnAction=std::function<void(const MomentVector&,size_t,MomentHistory&)>;
using CentralMomentSolve=std::function<void(const MomentVector&,MomentVector&)>;
using CentralMomentAction=std::function<void(const std::complex<double>*,std::complex<double>*)>;
struct MomentResult{MomentHistory current;std::vector<double> original_rhs_residuals;};

// Every solve uses the SAME A0 callback; its preconditioner and recycle space
// remain owned by the caller across all moments. Derivative operators enter
// only through convolution, so no global dense Aq is needed here.
inline MomentResult solve_moments(const MomentHistory& b,const MomentConvolution& convolution,
        const CentralMomentSolve& solve,const CentralMomentAction& strict,double tolerance){
    if(b.empty()||b.size()>17||b.front().empty()||!convolution||!solve||!strict||!std::isfinite(tolerance)||tolerance<=0)
        throw std::invalid_argument("matrix-free Taylor moment arguments");
    const size_t n=b.front().size();for(const auto& row:b){if(row.size()!=n)throw std::invalid_argument("Taylor RHS dimensions");
        for(auto z:row)if(!std::isfinite(z.real())||!std::isfinite(z.imag()))throw std::invalid_argument("nonfinite Taylor RHS");}
    MomentResult out;out.current.assign(b.size(),MomentVector(n));MomentVector action(n);
    for(size_t j=0;j<b.size();++j){MomentVector rhs=b[j];
        if(j){MomentHistory previous;convolution(out.current,previous);
            if(previous.size()!=b.size()||previous[j].size()!=n)throw std::runtime_error("Taylor convolution shape");
            for(size_t i=0;i<n;++i)rhs[i]-=previous[j][i];}
        solve(rhs,out.current[j]);if(out.current[j].size()!=n)throw std::runtime_error("Taylor solve shape");
        strict(out.current[j].data(),action.data());double delta=0,norm=0;
        for(size_t i=0;i<n;++i){delta+=std::norm(action[i]-rhs[i]);norm+=std::norm(rhs[i]);}
        const double residual=std::sqrt(delta/std::max(norm,1e-280));out.original_rhs_residuals.push_back(residual);
        if(!std::isfinite(residual)||residual>tolerance)throw std::runtime_error("Taylor original moment RHS residual failed");
    }
    return out;
}

// After solving x_j, compute A_q x_j once, only for q<=N-j, and subtract
// these contributions from future right-hand sides. The unused current
// slots hold those RHS updates until their central solve; no extra full
// history accumulator is retained. A0, its inverse/recycling, and the true
// residual callback are identical to the ordinary history recurrence.
inline MomentResult solve_moments_incremental(const MomentHistory& b,const MomentColumnAction& column,
        const CentralMomentSolve& solve,const CentralMomentAction& strict,double tolerance){
    if(b.empty()||b.size()>17||b.front().empty()||!column||!solve||!strict||!std::isfinite(tolerance)||tolerance<=0)
        throw std::invalid_argument("incremental Taylor moment arguments");
    const size_t n=b.front().size();for(const auto& row:b){
        if(row.size()!=n)throw std::invalid_argument("incremental Taylor RHS dimensions");
        for(auto z:row)if(!std::isfinite(z.real())||!std::isfinite(z.imag()))throw std::invalid_argument("nonfinite incremental Taylor RHS");
    }
    MomentResult out;out.current=b;MomentVector action(n);
    for(size_t j=0;j<b.size();++j){
        MomentVector rhs=std::move(out.current[j]);out.current[j].assign(n,std::complex<double>(0));
        solve(rhs,out.current[j]);if(out.current[j].size()!=n)throw std::runtime_error("incremental Taylor solve shape");
        strict(out.current[j].data(),action.data());double delta=0,norm=0;
        for(size_t i=0;i<n;++i){delta+=std::norm(action[i]-rhs[i]);norm+=std::norm(rhs[i]);}
        const double residual=std::sqrt(delta/std::max(norm,1e-280));out.original_rhs_residuals.push_back(residual);
        if(!std::isfinite(residual)||residual>tolerance)throw std::runtime_error("incremental Taylor original moment RHS residual failed");
        const size_t remaining=b.size()-1-j;if(!remaining)continue;
        MomentHistory images;column(out.current[j],remaining,images);
        if(images.size()!=remaining+1)throw std::runtime_error("incremental Taylor column count");
        for(const auto& row:images){if(row.size()!=n)throw std::runtime_error("incremental Taylor column shape");
            for(auto z:row)if(!std::isfinite(z.real())||!std::isfinite(z.imag()))throw std::runtime_error("nonfinite incremental Taylor column");}
        for(size_t q=1;q<=remaining;++q)for(size_t i=0;i<n;++i)out.current[j+q][i]-=images[q][i];
    }
    return out;
}
}
