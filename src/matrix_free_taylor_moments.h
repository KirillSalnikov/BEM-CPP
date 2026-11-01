#pragma once
#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <stdexcept>
#include <vector>

namespace bem_taylor {
using MomentVector=std::vector<std::complex<double>>;
using MomentHistory=std::vector<MomentVector>;
using MomentConvolution=std::function<void(const MomentHistory&,MomentHistory&)>;
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
}
