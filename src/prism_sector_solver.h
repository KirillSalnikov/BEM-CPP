#ifndef BEM_PRISM_SECTOR_SOLVER_H
#define BEM_PRISM_SECTOR_SOLVER_H
#include "prism_symmetry_sectors.h"
#include "streaming_sector_gmres.h"
namespace bem_tmatrix {
// Physical-coordinate wrapper. Warm starts solve a residual equation with
// the same sector recurrences and recycle spaces as zero starts. The final
// acceptance uses the original physical RHS norm, not the correction norm.
inline bem_streaming::HostResult solve_prism_sectors(
        const bem_streaming::Action& action,const bem_streaming::Action& strict,
        const bem_streaming::Action& inverse,const PrismSymmetrySectors& sectors,
        const Complex* b,Complex* x,int restart,double tolerance,int limit,
        bool use_initial_guess=false,bool verbose=true,bem_streaming::SectorRecycleSpace* recycle=nullptr,
        std::array<int,12>* sector_iterations=nullptr,
        const bem_streaming::Action& encoded_action=bem_streaming::Action()) {
    const int n=sectors.size();
    if(sector_iterations)sector_iterations->fill(0);
    const auto start=std::chrono::steady_clock::now();
    if(use_initial_guess) {
        if(n<1 || tolerance<=0 || !std::isfinite(tolerance) || restart<1 || limit<1)
            throw std::invalid_argument("sector warm-start dimensions");
        bem_streaming::HostResult result;
        const double bn=bem_streaming::norm(b,n);
        if(bn==0){std::fill(x,x+n,Complex(0));result.residual=0;return result;}
        Vector residual(n),correction(n),image(n);
        const auto measure=[&]() {
            for(int i=0;i<n;++i)residual[i]=b[i]-image[i];
            result.residual=bem_streaming::norm(residual.data(),n)/bn;
            if(!std::isfinite(result.residual))throw std::runtime_error("nonfinite sector warm-start residual");
            if(verbose)std::printf("  [sector warm start] original RHS FP64 residual %.6e\n",result.residual);
        };
        const auto verify=[&]() {
            const auto before=std::chrono::steady_clock::now();strict(x,image.data());
            result.strict_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-before).count();
            ++result.strict_checks;measure();
        };
        verify();
        // Defer these buffers until a correction is needed. Exact symmetry
        // guesses still cost just the initial full FP64 action.
        Vector checked_correction,checked_image,candidate;
        while(result.residual>tolerance && result.iterations<limit) {
            checked_correction.resize(n);checked_image.resize(n);candidate.resize(n);
            bool checked=false;
            // The correction gate evaluates the FULL physical current.
            // Subtract A*x0 only for the inner residual equation, and retain
            // the unmodified FP64 image for acceptance against the original b.
            const bem_streaming::Action correction_strict=[&](const Complex* a,Complex* out) {
                for(int i=0;i<n;++i){checked_correction[i]=a[i];candidate[i]=x[i]+a[i];}
                strict(candidate.data(),checked_image.data());
                for(int i=0;i<n;++i)out[i]=checked_image[i]-image[i];
                checked=true;
            };
            const double relative_tolerance=tolerance/result.residual;
            std::array<int,12> part_iterations{};
            const auto part=solve_prism_sectors(action,correction_strict,inverse,sectors,residual.data(),correction.data(),
                restart,relative_tolerance,limit-result.iterations,false,verbose,recycle,
                sector_iterations?&part_iterations:nullptr,encoded_action);
            if(sector_iterations)for(int s=0;s<12;++s)(*sector_iterations)[s]+=part_iterations[s];
            for(int i=0;i<n;++i)x[i]+=correction[i];
            result.iterations+=part.iterations;result.strict_checks+=part.strict_checks;
            result.deferred_strict_checks+=part.deferred_strict_checks;
            result.recycle_seconds+=part.recycle_seconds;result.recycle_fallbacks+=part.recycle_fallbacks;
            result.recycle_rank=part.recycle_rank;result.action_seconds+=part.action_seconds;
            result.inverse_seconds+=part.inverse_seconds;result.orthogonalization_seconds+=part.orthogonalization_seconds;
            result.strict_seconds+=part.strict_seconds;
            // Reapplying the preconditioner at the wrapper's exit normally
            // reproduces the checked correction exactly. A stateful or
            // nondeterministic inverse must instead trigger a fresh gate.
            if(checked && std::equal(correction.begin(),correction.end(),checked_correction.begin())) {
                image.swap(checked_image);measure();
            }else verify();
            if(part.iterations==0 && result.residual>tolerance)break;
        }
        result.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        return result;
    }
    Vector encoded(n),coordinates(n),physical_rhs(n),physical_x(n),physical_y(n);
    sectors.encode(b,encoded.data(),true);
    const auto composite=[&](const Complex* a,Complex* out,bool full) {
        // Optional fused E_rhs A B D_rhs action. Strict acceptance still
        // constructs and checks the actual physical current below, including
        // the original-current correction gate for every warm start.
        if(!full && encoded_action){encoded_action(a,out);return;}
        sectors.decode(a,physical_rhs.data(),true);inverse(physical_rhs.data(),physical_x.data());
        (full?strict:action)(physical_x.data(),physical_y.data());sectors.encode(physical_y.data(),out,true);
    };
    int interval=4;
    if(const char* value=std::getenv("BEM_HOST_GMRES_STRICT_INTERVAL")) {
        char* end=nullptr;const long parsed=std::strtol(value,&end,10);
        if(end==value || *end || parsed<1 || parsed>16)throw std::invalid_argument("sector strict interval");
        interval=parsed;
    }
    double factor=.5;
    if(const char* value=std::getenv("BEM_SECTOR_PROJECTED_FACTOR")) {
        char* end=nullptr;factor=std::strtod(value,&end);
        if(end==value || *end)throw std::invalid_argument("sector projected factor");
    }
    double first_pass_factor=0;
    if(const char* value=std::getenv("BEM_SECTOR_FIRST_PASS_FACTOR")) {
        char* end=nullptr;first_pass_factor=std::strtod(value,&end);
        if(end==value || *end)throw std::invalid_argument("sector first pass factor");
    }
    auto result=bem_streaming::sector_gmres([&](const Complex* a,Complex* out){composite(a,out,false);},
        [&](const Complex* a,Complex* out){composite(a,out,true);},encoded.data(),coordinates.data(),sectors.offsets,
        restart,tolerance,limit,verbose,interval,recycle,factor,first_pass_factor);
    if(sector_iterations)*sector_iterations=result.sector_iterations;
    sectors.decode(coordinates.data(),physical_rhs.data(),true);inverse(physical_rhs.data(),x);
    result.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();return result;
}
}
#endif
