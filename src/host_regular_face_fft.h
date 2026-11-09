#ifndef BEM_HOST_REGULAR_FACE_FFT_H
#define BEM_HOST_REGULAR_FACE_FFT_H
#include "host_spectral_inverse.h"
#include "host_memory_estimate.h"
#include <array>
#include <cmath>
#include <limits>

namespace bem_cpu {
// FP64 finite-lattice Helmholtz derivatives. Both physical currents share the
// same six planar symbols; zero padding prevents periodic wrap-around.
class RegularFaceAction {
    using C=std::complex<double>;
    FftwRuntime& api_;
    int nx_,ny_,fx_,fy_,grid_,channels_,degree_;
    C *symbols_=nullptr,*qhat_=nullptr,*ohat_=nullptr;
    FftwRuntime::Plan forward_=nullptr,backward_=nullptr;
    size_t symbol_count_=0;
    size_t symbol_stride_=0;
    static FftwRuntime::Pair* raw(C* p){return reinterpret_cast<FftwRuntime::Pair*>(p);}
    static int padded(int n){
        if(n<1||n>4096)throw std::invalid_argument("CPU face FFT size");
        int upper=1;while(upper<2*n-1)upper*=2;
        for(int m=2*n-1;m<upper;++m){int r=m;for(int f:{2,3,5})while(r%f==0)r/=f;if(r==1)return m;}
        return upper;
    }
    void cleanup(){
        if(forward_)api_.destroy(forward_);
        if(backward_)api_.destroy(backward_);
        if(symbols_)api_.release(symbols_);
        if(qhat_)api_.release(qhat_);
        if(ohat_)api_.release(ohat_);
    }
public:
    // Explicit FFT arrays only. Plan/library allocations are reported outside
    // this estimate; this query does not load FFTW or allocate an FFT array.
    static size_t estimated_workspace_bytes(int nx,int ny,int channels,int degree=0){
        if(channels<1||channels>64||degree<0||degree>16)throw std::invalid_argument("CPU FFT estimate dimensions");
        const size_t grid=memory::mul(padded(nx),padded(ny));
        const size_t symbols=memory::mul(size_t(degree+1)*channels*(channels+1)/2*6,grid);
        const size_t vectors=memory::mul(size_t(channels)*(6*(degree+1)+12),grid);
        return memory::mul(memory::add(symbols,vectors),sizeof(C));
    }
    // All vectors are in a local planar frame. Offsets are channel-major xy.
    RegularFaceAction(int nx,int ny,std::array<double,2> u,std::array<double,2> v,
                      const std::vector<std::array<double,2>>& offsets,C k,
                      int taylor_degree=0,C k_slope=C(1))
        :api_(FftwRuntime::get()),nx_(nx),ny_(ny),fx_(padded(nx)),fy_(padded(ny)),
         grid_(fx_*fy_),channels_(int(offsets.size())),degree_(taylor_degree) {
        if(!api_.available()||channels_<1||channels_>64||!std::isfinite(k.real())||!std::isfinite(k.imag())||
           degree_<0||degree_>16||!std::isfinite(k_slope.real())||!std::isfinite(k_slope.imag())||
           !std::isfinite(u[0]+u[1]+v[0]+v[1])||std::abs(u[0]*v[1]-u[1]*v[0])<1e-30)
            throw std::invalid_argument("CPU face FFT geometry/library");
        for(auto p:offsets)if(!std::isfinite(p[0])||!std::isfinite(p[1]))throw std::invalid_argument("CPU face FFT offset");
        symbol_stride_=size_t(channels_)*(channels_+1)/2*6*grid_;
        symbol_count_=size_t(degree_+1)*symbol_stride_;
        try {
            symbols_=static_cast<C*>(api_.allocate(symbol_count_*sizeof(C)));
            qhat_=static_cast<C*>(api_.allocate(size_t(degree_+1)*channels_*6*grid_*sizeof(C)));
            ohat_=static_cast<C*>(api_.allocate(size_t(channels_)*12*grid_*sizeof(C)));
            if(!symbols_||!qhat_||!ohat_)throw std::bad_alloc();
            unsigned flags=64;
            for(C* p:{symbols_,qhat_,ohat_})for(int shift:{0,grid_})
                if(api_.alignment(reinterpret_cast<double*>(p+shift))!=api_.alignment(reinterpret_cast<double*>(qhat_)))flags|=2;
            forward_=api_.plan(fx_,fy_,raw(qhat_),raw(qhat_),-1,flags);
            backward_=api_.plan(fx_,fy_,raw(ohat_),raw(ohat_),1,flags);
            if(!forward_||!backward_)throw std::runtime_error("CPU face FFT plans");
            const int pairs=channels_*(channels_+1)/2;
            std::vector<std::array<int,2>> index;
            for(int t=0;t<channels_;++t)for(int s=0;s<=t;++s)index.push_back({{t,s}});
#pragma omp parallel for schedule(dynamic,1)
            for(int pair=0;pair<pairs;++pair){
                const int t=index[pair][0],s=index[pair][1];
                for(int f=0;f<grid_;++f){
                    C g[17][6]{};const int ix=f/fy_,iy=f%fy_;
                    const int dx=ix<nx_?ix:ix-fx_,dy=iy<ny_?iy:iy-fy_;
                    if(std::abs(dx)<nx_&&std::abs(dy)<ny_){
                        const double x=dx*u[0]+dy*v[0]+offsets[t][0]-offsets[s][0];
                        const double y=dx*u[1]+dy*v[1]+offsets[t][1]-offsets[s][1];
                        const double r2=x*x+y*y;
                        if(r2>1e-28){
                            const double r=std::sqrt(r2),ir=1/r;const C ik=C(0,1)*k;
                            C green=std::exp(ik*r)*(ir/(4*std::acos(-1.))),previous=0,previous2=0;
                            const C slope_i=C(0,1)*k_slope;
                            // Coefficients of k(delta)=k+k_slope*delta. No
                            // differencing, no factorial multiplication in AWE.
                            for(int j=0;j<=degree_;++j){
                                const C alpha=(green*(ik-ir)+slope_i*previous)*ir;
                                const C beta=(green*(3*ir*ir-3.*ik*ir-k*k)+
                                    previous*(-3.*slope_i*ir-2.*k*k_slope)-previous2*k_slope*k_slope)/r2;
                                g[j][0]=alpha*x;g[j][1]=alpha*y;
                                g[j][2]=alpha+beta*x*x;g[j][3]=beta*x*y;
                                g[j][4]=alpha+beta*y*y;g[j][5]=alpha;
                                previous2=previous;previous=green;green*=slope_i*r/double(j+1);
                            }
                        }
                    }
                    for(int j=0;j<=degree_;++j)for(int a=0;a<6;++a)
                        symbols_[size_t(j)*symbol_stride_+(size_t(pair)*6+a)*grid_+f]=g[j][a];
                }
                for(int j=0;j<=degree_;++j)for(int a=0;a<6;++a){
                    auto* p=raw(symbols_+size_t(j)*symbol_stride_+(size_t(pair)*6+a)*grid_);api_.execute(forward_,p,p);
                }
            }
        }catch(...){cleanup();throw;}
    }
    ~RegularFaceAction(){cleanup();}
    RegularFaceAction(const RegularFaceAction&)=delete;
    RegularFaceAction& operator=(const RegularFaceAction&)=delete;
    size_t workspace_bytes()const{return (symbol_count_+size_t(channels_)*(6*(degree_+1)+12)*grid_)*sizeof(C);}
    // input[(channel*6+Jxyz/Mxyz)*cells+cell]; output has twelve components,
    // curl-convention xyz followed by (H-tr(H)I) xyz for each current.
    void evaluate(const C* input,C* output){
        evaluate_coefficient(0,input,output);
    }
    // Reverse the complete finite-lattice action, including the zero-padding
    // and crop. Input has twelve target components, output six source
    // components. The same workspace is reused; in-place use is supported.
    void evaluate_transpose(const C* input,C* output){evaluate_reverse(0,input,output,false);}
    void evaluate_adjoint(const C* input,C* output){evaluate_reverse(0,input,output,true);}
    void evaluate_coefficient_transpose(int order,const C* input,C* output){evaluate_reverse(order,input,output,false);}
    void evaluate_coefficient_adjoint(int order,const C* input,C* output){evaluate_reverse(order,input,output,true);}
    // A_order input, where A_order=(d/delta)^order A / order!.
    void evaluate_coefficient(int order,const C* input,C* output){
        if(!input)throw std::invalid_argument("CPU face FFT input");
        evaluate_sum(order,1,&input,output);
    }
    // Sum_{j=0}^order A_{order-j} input[j], the coefficient required by
    // AWE. All products accumulate in frequency space: only one inverse FFT
    // per output channel. Null input[j] denotes an exact zero coefficient.
    void evaluate_convolution(int order,const C* const* input,C* output){
        evaluate_sum(order,order+1,input,output);
    }
    // All coefficients together: transform each input history once, retain
    // its spectrum, and accumulate every AWE convolution before inverse FFT.
    // All inputs are read before any output is written (cross-order alias OK).
    void evaluate_history(const C* const* input,C* const* output){
        evaluate_history_prefix(input,output,degree_,degree_);
    }
    // Only the supplied input prefix contributes. Higher derivatives of the
    // kernel are retained through output_degree, without transforming zeros
    // or computing output coefficients that the recurrence will not use.
    void evaluate_history_prefix(const C* const* input,C* const* output,int input_degree,int output_degree){
        if(!input||!output)throw std::invalid_argument("CPU face FFT history buffers");
        if(input_degree<0||input_degree>output_degree||output_degree>degree_)
            throw std::invalid_argument("CPU face FFT history prefix");
        for(int j=0;j<=output_degree;++j)if(!output[j])throw std::invalid_argument("CPU face FFT history output");
        transform_inputs(input_degree+1,input);
        for(int j=0;j<=output_degree;++j)evaluate_transformed(j,std::min(j,input_degree)+1,input,output[j]);
    }
private:
    void evaluate_reverse(int order,const C* input,C* output,bool adjoint){
        if(order<0||order>degree_||!input||!output)
            throw std::invalid_argument("CPU face FFT reverse order/buffers");
        const int cells=nx_*ny_;
        std::fill(ohat_,ohat_+size_t(channels_)*12*grid_,C(0));
#pragma omp parallel for schedule(static)
        for(int lane=0;lane<channels_*12;++lane){
            C* p=ohat_+size_t(lane)*grid_;
            for(int i=0;i<nx_;++i)for(int j=0;j<ny_;++j){
                const C z=input[size_t(lane)*cells+i*ny_+j];
                p[i*fy_+j]=adjoint?std::conj(z):z;
            }
            api_.execute(forward_,raw(p),raw(p));
        }
#pragma omp parallel for schedule(static)
        for(int task=0;task<channels_*grid_;++task){
            const int s=task/grid_,f=task%grid_;
            const int negative=((fx_-f/fy_)%fx_)*fy_+(fy_-f%fy_)%fy_;
            C sum[6]{};
            for(int t=0;t<channels_;++t){
                const bool reverse=s>t;const int hi=std::max(t,s),lo=std::min(t,s);
                const size_t pair=size_t(hi)*(hi+1)/2+lo;
                C g[6];for(int a=0;a<6;++a)
                    g[a]=symbols_[size_t(order)*symbol_stride_+(pair*6+a)*grid_+(reverse?f:negative)];
                if(reverse){g[0]=-g[0];g[1]=-g[1];}
                for(int current=0;current<2;++current){
                    C v[6];for(int a=0;a<6;++a)v[a]=ohat_[size_t(t*12+6*current+a)*grid_+f];
                    C* r=sum+3*current;
                    r[0]+=g[1]*v[0]-(g[4]+g[5])*v[3]+g[3]*v[4];
                    r[1]+=-g[0]*v[0]+g[3]*v[3]-(g[2]+g[5])*v[4];
                    r[2]+=-g[0]*v[1]-g[1]*v[2]-(g[2]+g[4])*v[5];
                }
            }
            for(int a=0;a<6;++a)qhat_[size_t(s*6+a)*grid_+f]=sum[a];
        }
#pragma omp parallel for schedule(static)
        for(int lane=0;lane<channels_*6;++lane){
            C* p=qhat_+size_t(lane)*grid_;api_.execute(backward_,raw(p),raw(p));
            for(int i=0;i<nx_;++i)for(int j=0;j<ny_;++j){
                const C z=p[i*fy_+j]/double(grid_);
                output[size_t(lane)*cells+i*ny_+j]=adjoint?std::conj(z):z;
            }
        }
    }
    void evaluate_sum(int order,int count,const C* const* input,C* output){
        if(order<0||order>degree_||count<1||count>order+1||!input||!output)
            throw std::invalid_argument("CPU face FFT Taylor order/buffers");
        transform_inputs(count,input);evaluate_transformed(order,count,input,output);
    }
    void transform_inputs(int count,const C* const* input){
        const int cells=nx_*ny_;
        const size_t qstride=size_t(channels_)*6*grid_;
        std::fill(qhat_,qhat_+size_t(count)*qstride,C(0));
#pragma omp parallel for schedule(static)
        for(int task=0;task<count*channels_*6;++task){
            const int term=task/(channels_*6),lane=task%(channels_*6);
            if(!input[term])continue;
            C* q=qhat_+size_t(task)*grid_;
            for(int i=0;i<nx_;++i)for(int j=0;j<ny_;++j)q[i*fy_+j]=input[term][size_t(lane)*cells+i*ny_+j];
            api_.execute(forward_,raw(q),raw(q));
        }
    }
    void evaluate_transformed(int order,int count,const C* const* input,C* output){
        const int cells=nx_*ny_;const size_t qstride=size_t(channels_)*6*grid_;
#pragma omp parallel for schedule(static)
        for(int task=0;task<channels_*grid_;++task){
            const int t=task/grid_,f=task%grid_;C sum[12]{};
            for(int s=0;s<channels_;++s){
                const bool reverse=s>t;const int hi=std::max(t,s),lo=std::min(t,s);
                const size_t pair=size_t(hi)*(hi+1)/2+lo;
                const int frequency=reverse?((fx_-f/fy_)%fx_)*fy_+(fy_-f%fy_)%fy_:f;
                for(int term=0;term<count;++term){
                if(!input[term])continue;
                const C* q=qhat_+size_t(term)*qstride;
                C g[6];for(int a=0;a<6;++a)g[a]=symbols_[size_t(order-term)*symbol_stride_+(pair*6+a)*grid_+frequency];
                if(reverse){g[0]=-g[0];g[1]=-g[1];}
                for(int current=0;current<2;++current){
                    const C x=q[size_t(s*6+3*current)*grid_+f],y=q[size_t(s*6+3*current+1)*grid_+f],
                            z=q[size_t(s*6+3*current+2)*grid_+f];
                    C* r=sum+6*current;
                    r[0]+=g[1]*x-g[0]*y;r[1]-=g[0]*z;r[2]-=g[1]*z;
                    r[3]+=g[3]*y-(g[4]+g[5])*x;r[4]+=g[3]*x-(g[2]+g[5])*y;r[5]-=(g[2]+g[4])*z;
                }
                }
            }
            for(int a=0;a<12;++a)ohat_[size_t(t*12+a)*grid_+f]=sum[a];
        }
#pragma omp parallel for schedule(static)
        for(int lane=0;lane<channels_*12;++lane){
            C* p=ohat_+size_t(lane)*grid_;api_.execute(backward_,raw(p),raw(p));
            for(int i=0;i<nx_;++i)for(int j=0;j<ny_;++j)output[size_t(lane)*cells+i*ny_+j]=p[i*fy_+j]/double(grid_);
        }
    }
};
}
#endif
