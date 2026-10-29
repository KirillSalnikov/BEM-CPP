#ifndef BEM_PRISM_SECTOR_SOLVER_H
#define BEM_PRISM_SECTOR_SOLVER_H
#include "prism_symmetry_sectors.h"
#include "streaming_sector_gmres.h"
namespace bem_tmatrix {
// Physical-coordinate wrapper. A supplied symmetry warm start is checked
// by ordinary GMRES and corrected if necessary; zero starts use independent
// sector recurrences with the original full-operator residual acceptance.
inline bem_streaming::HostResult solve_prism_sectors(
        const bem_streaming::Action& action,const bem_streaming::Action& strict,
        const bem_streaming::Action& inverse,const PrismSymmetrySectors& sectors,
        const Complex* b,Complex* x,int restart,double tolerance,int limit,
        bool use_initial_guess=false,bool verbose=true) {
    const int n=sectors.size();
    if(use_initial_guess)return bem_streaming::gmres(action,strict,inverse,b,x,n,restart,tolerance,limit,verbose,true);
    const auto start=std::chrono::steady_clock::now();
    Vector encoded(n),coordinates(n),physical_rhs(n),physical_x(n),physical_y(n);
    sectors.encode(b,encoded.data(),true);
    const auto composite=[&](const Complex* a,Complex* out,bool full) {
        sectors.decode(a,physical_rhs.data(),true);inverse(physical_rhs.data(),physical_x.data());
        (full?strict:action)(physical_x.data(),physical_y.data());sectors.encode(physical_y.data(),out,true);
    };
    int interval=4;
    if(const char* value=std::getenv("BEM_HOST_GMRES_STRICT_INTERVAL")) {
        char* end=nullptr;const long parsed=std::strtol(value,&end,10);
        if(end==value || *end || parsed<1 || parsed>16)throw std::invalid_argument("sector strict interval");
        interval=parsed;
    }
    auto result=bem_streaming::sector_gmres([&](const Complex* a,Complex* out){composite(a,out,false);},
        [&](const Complex* a,Complex* out){composite(a,out,true);},encoded.data(),coordinates.data(),sectors.offsets,
        restart,tolerance,limit,verbose,interval);
    sectors.decode(coordinates.data(),physical_rhs.data(),true);inverse(physical_rhs.data(),x);
    result.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();return result;
}
}
#endif
