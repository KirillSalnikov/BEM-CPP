#pragma once
#include "planar_bdm_symbol.h"
#include "host_spectral_inverse.h"
#include "krylov_deflation.h"
#include <exception>

namespace bem_osrc {
struct InverseKernel {
    std::vector<C> values;
    double maximum_frobenius_condition=0,maximum_inverse_defect=0;
};
inline InverseKernel make_inverse_kernel(const PlanarBdmSymbol& stencil,int nx,int ny,double k0,C index,
                                        double damping,int aliases=1){
    if(nx<1||ny<1)throw std::invalid_argument("analytic inverse grid");
    auto& api=bem_cpu::FftwRuntime::get();if(!api.available())throw std::runtime_error("FP64 FFTW unavailable");
    const int grid=nx*ny;const double pi=std::acos(-1.);InverseKernel result;result.values.resize(size_t(144)*grid);
    std::vector<double> conditions(grid),defects(grid);std::exception_ptr failure;
#pragma omp parallel for schedule(dynamic,1)
    for(int f=0;f<grid;++f)try{
        const double a=2*pi*(f/ny<=nx/2?f/ny:f/ny-nx)/nx,b=2*pi*(f%ny<=ny/2?f%ny:f%ny-ny)/ny;
        const auto matrix=stencil.symbol(k0,index,damping,a,b,aliases);std::vector<C> lu(matrix.begin(),matrix.end());std::vector<int> pivots;
        if(!bem_krylov::dense_lu_factor(lu,12,pivots))throw std::runtime_error("singular analytic BDM symbol");
        double norm=0,inverse_norm=0;for(auto value:matrix)norm+=std::norm(value);
        for(int j=0;j<12;++j){std::vector<C> column(12);column[j]=1.;bem_krylov::dense_lu_solve(lu,pivots,12,column.data());
            for(int i=0;i<12;++i){if(!finite(column[i]))throw std::runtime_error("nonfinite analytic BDM inverse");
                result.values[size_t(i*12+j)*grid+f]=column[i];inverse_norm+=std::norm(column[i]);
                C check=0;for(int k=0;k<12;++k)check+=matrix[i*12+k]*column[k];defects[f]=std::max(defects[f],std::abs(check-C(i==j)));}}
        conditions[f]=std::sqrt(norm*inverse_norm);
    }catch(...){
#pragma omp critical(planar_inverse_kernel_failure)
        {if(!failure)failure=std::current_exception();}
    }
    if(failure)std::rethrow_exception(failure);
    result.maximum_frobenius_condition=*std::max_element(conditions.begin(),conditions.end());
    result.maximum_inverse_defect=*std::max_element(defects.begin(),defects.end());
    if(result.maximum_inverse_defect>1e-9)throw std::runtime_error("analytic BDM inverse verification failed");
    auto raw=[](C* p){return reinterpret_cast<bem_cpu::FftwRuntime::Pair*>(p);};
    const auto plan=api.plan(nx,ny,raw(result.values.data()),raw(result.values.data()),1,64|2); // ESTIMATE, UNALIGNED
    if(!plan)throw std::runtime_error("analytic kernel inverse FFT plan");
#pragma omp parallel for schedule(static)
    for(int c=0;c<144;++c){auto* data=raw(result.values.data()+size_t(c)*grid);api.execute(plan,data,data);}
    api.destroy(plan);for(auto& value:result.values)value/=grid;
    return result;
}
}
