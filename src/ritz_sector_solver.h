#pragma once
#include "prism_sector_solver.h"

namespace bem_tmatrix {
// The P2 screen applies its coarse inverse directly to sector coordinates
// for zero starts. Preserve that path in the full-T worker, without changing
// the established original-RHS warm-start correction and acceptance logic.
inline bem_streaming::HostResult solve_prism_sectors_with_ritz(
        const bem_streaming::Action& action,const bem_streaming::Action& strict,
        const bem_streaming::Action& inverse,const bem_streaming::Action& encoded_inverse,
        const PrismSymmetrySectors& sectors,const Complex* b,Complex* x,
        int restart,double tolerance,int limit,bool use_initial_guess=false,bool verbose=true,
        bem_streaming::SectorRecycleSpace* recycle=nullptr,std::array<int,12>* sector_iterations=nullptr,
        const bem_streaming::Action& encoded_action=bem_streaming::Action()) {
    if(encoded_inverse && encoded_action)throw std::invalid_argument("Ritz and fused sector action cannot be combined without a matching inverse");
    if(!encoded_inverse || use_initial_guess)
        return solve_prism_sectors(action,strict,inverse,sectors,b,x,restart,tolerance,limit,
                                   use_initial_guess,verbose,recycle,sector_iterations,encoded_action);
    const auto begin=std::chrono::steady_clock::now();const int n=sectors.size();
    Vector encoded(n),coordinates(n),physical_x(n),physical_y(n);
    sectors.encode(b,encoded.data(),true);
    const auto composite=[&](const Complex* a,Complex* out,bool full){
        encoded_inverse(a,physical_x.data());(full?strict:action)(physical_x.data(),physical_y.data());
        sectors.encode(physical_y.data(),out,true);
    };
    int interval=4;
    if(const char* setting=std::getenv("BEM_HOST_GMRES_STRICT_INTERVAL")){
        char* end=nullptr;const long value=std::strtol(setting,&end,10);
        if(end==setting || *end || value<1 || value>16)throw std::invalid_argument("sector strict interval");
        interval=int(value);
    }
    double factor=.5,first=0;
    if(const char* setting=std::getenv("BEM_SECTOR_PROJECTED_FACTOR")){
        char* end=nullptr;factor=std::strtod(setting,&end);
        if(end==setting || *end)throw std::invalid_argument("sector projected factor");
    }
    if(const char* setting=std::getenv("BEM_SECTOR_FIRST_PASS_FACTOR")){
        char* end=nullptr;first=std::strtod(setting,&end);
        if(end==setting || *end)throw std::invalid_argument("sector first pass factor");
    }
    auto result=bem_streaming::sector_gmres([&](const Complex* a,Complex* out){composite(a,out,false);},
        [&](const Complex* a,Complex* out){composite(a,out,true);},encoded.data(),coordinates.data(),sectors.offsets,
        restart,tolerance,limit,verbose,interval,recycle,factor,first);
    if(sector_iterations)*sector_iterations=result.sector_iterations;
    encoded_inverse(coordinates.data(),x);
    result.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    return result;
}
}
