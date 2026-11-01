#include "helmholtz_taylor.h"
#include <cstdio>
using C=bem_frequency::C;
// Independent finite-sum formula, not the three-term recurrence under test.
C hankel(int l,C z){
    const C I(0,1);C sum=1,term=1;
    for(int a=1;a<=l;++a){term*=static_cast<long double>((l+a)*(l-a+1))/static_cast<long double>(a)*I/(2.L*z);sum+=term;}
    return std::pow(-I,l+1)*std::exp(I*z)/z*sum;
}
C transfer(C k,long double distance,int p,long double cosine){
    C sum=0,power=1;long double previous=1,current=cosine;
    for(int l=0;l<=p;++l){long double legendre=1;
        if(l==1)legendre=current;
        if(l>1){legendre=((2.L*l-1)*cosine*current-(l-1)*previous)/l;previous=current;current=legendre;}
        sum+=(2.L*l+1)*power*hankel(l,k*distance)*legendre;power*=C(0,1);
    }return sum;
}
int main()try{
    constexpr size_t degree=6;const int samples=128;const long double pi=std::acos(-1.L);
    long double worst=0,central=0,phase_error=0;
    for(C k:{C(2),C(20),C(20,5)})for(int p:{3,12,28})for(long double distance:{.3L,1.7L}){
        const C slope(1.3116,.2);const auto radial=bem_frequency::m2l_radial<degree>(k,slope,distance,p);
        const long double radius=.08L*std::abs(k/slope);
        for(long double cosine:{-.9L,.17L,.8L}){
            const auto t=bem_frequency::m2l_direction<degree>(radial,cosine);
            // A leaf phase and the k^3 Hessian L2P normalization must also be
            // differentiated. Testing the product detects omission of these.
            const auto w=bem_frequency::variable<degree>(k,slope);
            const auto factor=bem_frequency::phase<degree>(C(0,.13)*k,C(0,.13)*slope);
            const auto total=bem_frequency::product<degree>(t,bem_frequency::product<degree>(factor,
                bem_frequency::product<degree>(w,bem_frequency::product<degree>(w,w))));
            C oracle[degree+1]{},complete[degree+1]{};long double scale=0,fullscale=0;
            const C initial=transfer(k,distance,p,cosine);
            central=std::max(central,std::abs(t[0]-initial)/std::max(1e-280L,std::abs(initial)));
            for(int n=0;n<samples;++n){
                const C z=std::polar(radius,2*pi*n/samples),q=k+slope*z;
                const C value=transfer(q,distance,p,cosine),full=value*std::exp(C(0,.13)*q)*q*q*q;
                scale=std::max(scale,std::abs(value));fullscale=std::max(fullscale,std::abs(full));
                for(size_t j=0;j<=degree;++j){const C weight=std::polar(1.L/samples,-2*pi*n*j/samples);
                    oracle[j]+=weight*value;complete[j]+=weight*full;}
            }
            for(size_t j=0;j<=degree;++j){
                worst=std::max(worst,std::abs(t[j]*std::pow(radius,j)-oracle[j])/scale);
                phase_error=std::max(phase_error,std::abs(total[j]*std::pow(radius,j)-complete[j])/fullscale);
            }
        }
    }
    if(!std::isfinite(worst+central+phase_error)||std::max(worst,phase_error)>2e-10L||central>2e-9L)
        throw std::runtime_error("FMM jet versus independent Hankel contour");
    bool rejected=false;try{bem_frequency::m2l_radial<degree>(C(0),C(1),1,3);}
    catch(const std::invalid_argument&){rejected=true;}if(!rejected)throw std::runtime_error("FMM zero frequency guard");
    std::printf("PASS fixed-order FMM moments 0..6: central %.3Le, independent transfer contour %.3Le, phase and k^3 product %.3Le\n",central,worst,phase_error);
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
