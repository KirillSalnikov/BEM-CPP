#include "host_regular_face_fft.h"
#include <cstdio>
#include <random>

using C=std::complex<double>;
using V=std::vector<C>;
using LC=std::complex<long double>;
double relative(const V& a,const V& b){
    double n=0,d=0;for(size_t i=0;i<a.size();++i){n+=std::norm(a[i]-b[i]);d+=std::norm(b[i]);}
    return std::sqrt(n/std::max(d,1e-280));
}
int main()try{
    const int degree=6,nx=3,ny=4,channels=3,cells=nx*ny,points=64;
    const std::array<double,2> u{{.3,0}},v{{.11,.23}};
    const std::vector<std::array<double,2>> offsets{{{0,0}},{{.07,.05}},{{.13,.11}}};
    const size_t ni=size_t(channels)*6*cells,no=size_t(channels)*12*cells;
    std::mt19937 rng(817);std::normal_distribution<double> normal;
    std::vector<V> x(degree+1,V(ni));
    for(auto& row:x)for(auto& z:row)z=C(normal(rng),normal(rng));
    // Missing input orders are exact zeros, including in the contour oracle.
    std::fill(x[2].begin(),x[2].end(),C(0));
    const C slopes[]={C(1),C(1.3116),C(1.3116,10),C(1)};
    const C centers[]={C(10),C(20*1.3116),C(3*1.3116,30),C(0)};
    double worst=0,convolution_worst=0,polynomial_worst=0;
    for(int c=0;c<4;++c){
        bem_cpu::RegularFaceAction action(nx,ny,u,v,offsets,centers[c],degree,slopes[c]);
        std::vector<V> coefficient(degree+1,V(no)),convolution(degree+1,V(no));
        const C* ptr[degree+1];for(int j=0;j<=degree;++j)ptr[j]=j==2?nullptr:x[j].data();
        for(int j=0;j<=degree;++j){
            action.evaluate_coefficient(j,x[0].data(),coefficient[j].data());
            action.evaluate_convolution(j,ptr,convolution[j].data());
        }
        std::vector<V> history(degree+1,V(no));C* outputs[degree+1];
        for(int j=0;j<=degree;++j)outputs[j]=history[j].data();
        action.evaluate_history(ptr,outputs);
        for(int j=0;j<=degree;++j)if(relative(history[j],convolution[j])>1e-13)
            throw std::runtime_error("single-transform spectral history differs from individual convolution");
        // Writing coefficient zero can overwrite a later input coefficient;
        // all forward transforms must already have completed before then.
        std::vector<V> aliased(degree+1,V(no));const C* alias_inputs[degree+1];C* alias_outputs[degree+1];
        for(int j=0;j<=degree;++j){std::copy(x[j].begin(),x[j].end(),aliased[j].begin());
            alias_inputs[j]=j==2?nullptr:aliased[j].data();alias_outputs[j]=aliased[degree-j].data();}
        action.evaluate_history(alias_inputs,alias_outputs);
        for(int j=0;j<=degree;++j)if(relative(aliased[degree-j],convolution[j])>1e-13)
            throw std::runtime_error("cross-order FFT history alias");
        // Cauchy's coefficient formula is independent of all analytic
        // derivative/jet expressions. The contour is safe here: the finite
        // Green function with excluded self-interaction is entire in k.
        std::vector<std::vector<LC>> oracle(degree+1,std::vector<LC>(no)),product(oracle);
        const double radius=.8;const long double pi=std::acos(-1.L);
        for(int p=0;p<points;++p){
            const C z=std::polar(radius,double(2*pi*p/points));
            bem_cpu::RegularFaceAction sample(nx,ny,u,v,offsets,centers[c]+slopes[c]*z);
            V fixed(no),mixed(no),input(ni);sample.evaluate(x[0].data(),fixed.data());
            for(size_t i=0;i<ni;++i)for(int j=degree;j>=0;--j)input[i]=z*input[i]+x[j][i];
            sample.evaluate(input.data(),mixed.data());
            for(int j=0;j<=degree;++j){
                const LC factor=std::polar(std::pow(static_cast<long double>(radius),-j)/points,-2*pi*p*j/points);
                for(size_t i=0;i<no;++i){oracle[j][i]+=LC(fixed[i])*factor;product[j][i]+=LC(mixed[i])*factor;}
            }
        }
        for(int j=0;j<=degree;++j){
            V reference(no),product_reference(no);
            for(size_t i=0;i<no;++i){reference[i]=C(oracle[j][i]);product_reference[i]=C(product[j][i]);}
            // At k=0 the first derivative of grad G and Hess G is exactly
            // zero (the linear term of G is spatially constant). A relative
            // error against contour roundoff has no meaning in this case.
            double e=relative(coefficient[j],reference);
            if(c==3&&j==1){
                double n=0,d=0;for(size_t i=0;i<no;++i){
                    n+=std::norm(coefficient[j][i])+std::norm(reference[i]);d+=std::norm(coefficient[0][i]);
                }
                e=std::sqrt(n/d);
                if(e>1e-13)throw std::runtime_error("nonzero derivative of static spatial kernel");
            }
            const double f=relative(convolution[j],product_reference);
            worst=std::max(worst,e);convolution_worst=std::max(convolution_worst,f);
            if(!std::isfinite(e+f)||e>2e-8||f>2e-8){
                std::fprintf(stderr,"case=%d degree=%d coefficient=%.6e convolution=%.6e\n",c,j,e,f);
                throw std::runtime_error("Taylor coefficients versus independent complex contour");
            }
        }
        // Evaluate at real off-centre sizes; this checks factorial and slope.
        for(double delta:{-.02,.02}){
            bem_cpu::RegularFaceAction shifted(nx,ny,u,v,offsets,centers[c]+slopes[c]*delta);
            V exact(no),predicted(no);shifted.evaluate(x[0].data(),exact.data());
            for(size_t i=0;i<no;++i)for(int j=degree;j>=0;--j)predicted[i]=delta*predicted[i]+coefficient[j][i];
            double e=relative(predicted,exact);polynomial_worst=std::max(polynomial_worst,e);
            if(!std::isfinite(e)||e>2e-8)throw std::runtime_error("off-centre FFT Taylor expansion");
        }
        // Shared input/output storage and reuse after nonzero/zero calls.
        V storage(no);std::copy(x[0].begin(),x[0].end(),storage.begin());
        action.evaluate_coefficient(3,storage.data(),storage.data());
        if(relative(storage,coefficient[3])>1e-13)throw std::runtime_error("Taylor FFT alias");
        const C* zero[degree+1]{};action.evaluate_convolution(degree,zero,storage.data());
        for(auto q:storage)if(std::abs(q)>1e-14)throw std::runtime_error("Taylor FFT zero/stale buffer");
        action.evaluate_history(zero,outputs);
        for(const auto& row:history)for(C q:row)if(q!=C(0))throw std::runtime_error("FFT history stale buffer");
        bool rejected=false;try{action.evaluate_coefficient(degree+1,x[0].data(),storage.data());}
        catch(const std::invalid_argument&){rejected=true;}
        if(!rejected)throw std::runtime_error("Taylor FFT degree guard");
    }
    std::printf("PASS FFT Taylor degree 0..6: independent Cauchy %.3e, spectral convolution %.3e, shifted operator %.3e; complex material, two currents, zero and alias\n",worst,convolution_worst,polynomial_worst);
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
