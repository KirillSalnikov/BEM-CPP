#pragma once
#include "krylov_deflation.h"
#include <cstdlib>
#include <dlfcn.h>
#include <numeric>
#include <stdexcept>

namespace bem_sector_experiment {
using Complex=bem_krylov::Complex;
struct LapackEigen {
    using Function=int (*)(int,char,char,int,Complex*,int,Complex*,Complex*,int,Complex*,int,Complex*,int,double*);
    void* library=nullptr;
    Function eigen=nullptr;
    LapackEigen() {
        const char* path=std::getenv("BEM_SECTOR_RITZ_LAPACK_LIBRARY");
        if(!path || !*path)throw std::runtime_error("exact harmonic Ritz experiment needs BEM_SECTOR_RITZ_LAPACK_LIBRARY");
        library=dlopen(path,RTLD_NOW|RTLD_LOCAL);
        if(!library)throw std::runtime_error(dlerror());
        eigen=reinterpret_cast<Function>(dlsym(library,"scipy_LAPACKE_zgeev_work"));
        if(!eigen)eigen=reinterpret_cast<Function>(dlsym(library,"LAPACKE_zgeev_work"));
        if(!eigen)throw std::runtime_error("LAPACKE_zgeev_work symbol unavailable");
        using Threads=void (*)(int);
        auto threads=reinterpret_cast<Threads>(dlsym(library,"scipy_openblas_set_num_threads"));
        if(threads)threads(1);
    }
    ~LapackEigen(){if(library)dlclose(library);}
};

inline int harmonic_ritz_coefficients(const std::vector<Complex>& h,int stride,int used,int requested,
                                     std::vector<Complex>& coefficients) {
    const char* setting=std::getenv("BEM_SECTOR_RITZ_EXPERIMENT");
    const int method=setting?std::atoi(setting):0;
    if(method==0)return bem_krylov::harmonic_ritz_coefficients(h,stride,used,requested,coefficients);
    if(method<0 || method>2)throw std::invalid_argument("harmonic Ritz experiment method");
    coefficients.clear();const int rank=std::min(requested,used);
    if(rank<=0 || stride<used)return 0;
    std::vector<Complex> t(size_t(used)*used),adjoint(size_t(used)*used);
    for(int i=0;i<used;++i)for(int j=0;j<used;++j){t[i*used+j]=h[i*stride+j];adjoint[j*used+i]=std::conj(h[i*stride+j]);}
    std::vector<int> pivots;
    if(!bem_krylov::dense_lu_factor(adjoint,used,pivots))return 0;
    std::vector<Complex> correction(used);correction.back()=1.;
    bem_krylov::dense_lu_solve(adjoint,pivots,used,correction.data());
    const double trailing=std::norm(h[size_t(used)*stride+used-1]);
    for(int i=0;i<used;++i)t[i*used+used-1]+=trailing*correction[i];
    if(method==2) {
        static LapackEigen api;
        std::vector<Complex> matrix(t.size()),values(used),vectors(t.size());
        for(int i=0;i<used;++i)for(int j=0;j<used;++j)matrix[j*used+i]=t[i*used+j];
        std::vector<double> real_work(2*used);Complex dummy=0,query=0;
        int info=api.eigen(102,'N','V',used,matrix.data(),used,values.data(),&dummy,1,vectors.data(),used,&query,-1,real_work.data());
        if(info || !std::isfinite(query.real()))return 0;
        const int workspace=std::max(2*used,int(query.real()));std::vector<Complex> work(workspace);
        info=api.eigen(102,'N','V',used,matrix.data(),used,values.data(),&dummy,1,vectors.data(),used,work.data(),workspace,real_work.data());
        if(info)return 0;
        std::vector<int> order(used);std::iota(order.begin(),order.end(),0);
        for(const auto& v:values)if(!std::isfinite(std::abs(v)))return 0;
        std::stable_sort(order.begin(),order.end(),[&](int a,int b){return std::abs(values[a])<std::abs(values[b]);});
        coefficients.resize(size_t(rank)*used);
        for(int k=0;k<rank;++k)std::copy(vectors.begin()+size_t(order[k])*used,vectors.begin()+size_t(order[k]+1)*used,coefficients.begin()+size_t(k)*used);
        return rank;
    }
    // Same block-inverse extraction, but converge the subspace for longer.
    auto factor=t;std::vector<int> inverse_pivots;
    if(!bem_krylov::dense_lu_factor(factor,used,inverse_pivots))return 0;
    coefficients.resize(size_t(rank)*used);
    const double normalization=1/std::sqrt(double(used));
    for(int col=0;col<rank;++col)for(int row=0;row<used;++row)
        coefficients[size_t(col)*used+row]=std::polar(normalization,2*std::acos(-1.)*(row+1)*(col+1)/(used+1));
    int accepted=bem_krylov::orthonormalize_columns(coefficients,used,rank);
    for(int iteration=0;iteration<60 && accepted;++iteration) {
        for(int col=0;col<accepted;++col)bem_krylov::dense_lu_solve(factor,inverse_pivots,used,coefficients.data()+size_t(col)*used);
        accepted=bem_krylov::orthonormalize_columns(coefficients,used,accepted);
    }
    return accepted;
}
}
