#ifndef BEM_HOST_SPECTRAL_INVERSE_H
#define BEM_HOST_SPECTRAL_INVERSE_H
#include <algorithm>
#include <complex>
#include <cstdlib>
#include <dlfcn.h>
#include <stdexcept>
#include <vector>

namespace bem_cpu {
// Optional FFTW runtime: no mandatory FFTW build dependency. Construct/destroy
// plans on the GPU owner's startup/shutdown thread. Only execute_dft runs in
// parallel, with disjoint, equally aligned arrays (FFTW new-array contract).
class FftwRuntime {
    void* library_=nullptr;
public:
    using Pair=double[2];using Plan=void*;
    Plan (*plan)(int,int,Pair*,Pair*,int,unsigned)=nullptr;
    void (*execute)(Plan,Pair*,Pair*)=nullptr;
    void (*destroy)(Plan)=nullptr;
    void* (*allocate)(size_t)=nullptr;
    void (*release)(void*)=nullptr;
    int (*alignment)(double*)=nullptr;
    FftwRuntime() {
        const char* path=std::getenv("BEM_CPU_FFTW_LIBRARY");
        library_=dlopen(path?path:"libfftw3.so.3",RTLD_NOW|RTLD_LOCAL);
        if(!library_)return;
        plan=reinterpret_cast<decltype(plan)>(dlsym(library_,"fftw_plan_dft_2d"));
        execute=reinterpret_cast<decltype(execute)>(dlsym(library_,"fftw_execute_dft"));
        destroy=reinterpret_cast<decltype(destroy)>(dlsym(library_,"fftw_destroy_plan"));
        allocate=reinterpret_cast<decltype(allocate)>(dlsym(library_,"fftw_malloc"));
        release=reinterpret_cast<decltype(release)>(dlsym(library_,"fftw_free"));
        alignment=reinterpret_cast<decltype(alignment)>(dlsym(library_,"fftw_alignment_of"));
        if(!plan||!execute||!destroy||!allocate||!release||!alignment){dlclose(library_);library_=nullptr;}
    }
    ~FftwRuntime(){if(library_)dlclose(library_);}
    bool available()const{return library_!=nullptr;}
    static FftwRuntime& get(){static FftwRuntime runtime;return runtime;}
};

class SpectralInverse {
    using C=std::complex<double>;
    FftwRuntime& api_;
    int grid_,channels_,batches_;
    std::vector<C> symbol_;
    C *input_=nullptr,*output_=nullptr;
    FftwRuntime::Plan forward_=nullptr,backward_=nullptr;
    static FftwRuntime::Pair* raw(C* p){return reinterpret_cast<FftwRuntime::Pair*>(p);}
    void cleanup() {
        if(forward_)api_.destroy(forward_);
        if(backward_)api_.destroy(backward_);
        if(input_)api_.release(input_);
        if(output_)api_.release(output_);
    }
public:
    SpectralInverse(int nx,int ny,int channels,int batches,const std::vector<C>& inverse_symbol)
        :api_(FftwRuntime::get()),grid_(nx*ny),channels_(channels),batches_(batches),symbol_(inverse_symbol) {
        if(!api_.available()||nx<1||ny<1||channels<1||batches<1||symbol_.size()!=size_t(channels)*channels*grid_)
            throw std::invalid_argument("CPU spectral inverse dimensions/library");
        const size_t bytes=size_t(grid_)*channels*batches*sizeof(C);
        try {
            input_=static_cast<C*>(api_.allocate(bytes));output_=static_cast<C*>(api_.allocate(bytes));
            if(!input_||!output_)throw std::bad_alloc();
            unsigned flags=64; // FFTW_ESTIMATE, without touching the symbol.
            for(int c=1;c<channels*batches;++c)
                if(api_.alignment(reinterpret_cast<double*>(input_+size_t(c)*grid_))!=api_.alignment(reinterpret_cast<double*>(input_)) ||
                   api_.alignment(reinterpret_cast<double*>(output_+size_t(c)*grid_))!=api_.alignment(reinterpret_cast<double*>(output_)))flags|=2; // FFTW_UNALIGNED
            forward_=api_.plan(nx,ny,raw(input_),raw(input_),-1,flags);
            backward_=api_.plan(nx,ny,raw(output_),raw(output_),1,flags);
            if(!forward_||!backward_)throw std::runtime_error("CPU FFTW plan creation failed");
        }catch(...){cleanup();throw;}
    }
    ~SpectralInverse(){cleanup();}
    SpectralInverse(const SpectralInverse&)=delete;
    SpectralInverse& operator=(const SpectralInverse&)=delete;
    size_t storage_bytes()const{return (symbol_.size()+2*size_t(grid_)*channels_*batches_)*sizeof(C);}
    // One caller at a time per instance, just like its GPU counterpart.
    void apply(const C* input,C* output,bool adjoint=false) {
        const int transforms=channels_*batches_;const double scale=1./grid_;
        std::copy(input,input+size_t(transforms)*grid_,input_);
#pragma omp parallel for schedule(static)
        for(int j=0;j<transforms;++j) {
            auto* data=raw(input_+size_t(j)*grid_);api_.execute(forward_,data,data);
        }
#pragma omp parallel for schedule(static)
        for(int task=0;task<transforms;++task) {
            const int batch=task/channels_,row=task%channels_;
            C* out=output_+size_t(task)*grid_;std::fill(out,out+grid_,C(0));
            for(int col=0;col<channels_;++col) {
                const C* a=symbol_.data()+size_t(adjoint?col*channels_+row:row*channels_+col)*grid_;
                const C* x=input_+size_t(batch*channels_+col)*grid_;
                for(int f=0;f<grid_;++f) {
                    const double ar=a[f].real(),ai=adjoint?-a[f].imag():a[f].imag();
                    const double xr=x[f].real(),xi=x[f].imag();
                    out[f]+=C(ar*xr-ai*xi,ar*xi+ai*xr);
                }
            }
            for(int f=0;f<grid_;++f)out[f]*=scale;
        }
#pragma omp parallel for schedule(static)
        for(int j=0;j<transforms;++j) {
            auto* data=raw(output_+size_t(j)*grid_);api_.execute(backward_,data,data);
        }
        std::copy(output_,output_+size_t(transforms)*grid_,output);
    }
};
}
#endif
