#include "muller_taylor_kernel.h"
#include "muller_nodal.h"
#include <cstdio>

using C=std::complex<double>;using LC=std::complex<long double>;
int main()try{
    const int order=6,samples=64;const long double pi=std::acos(-1.L);
    const Vec3 normal(0,0,1),test(.8,.3,0),source(.2,-.7,.4),unit=Vec3(.2,.4,.7)*(1/std::sqrt(.69));
    const Vec3 rotated=test.cross(normal);
    const double dot=rotated.dot(source),along=rotated.dot(unit)*source.dot(unit);
    const double grad=test.dot(source)*normal.dot(unit)-normal.dot(source)*test.dot(unit);
    double zero_error=0,coefficient_error=0;
    for(double k:{2.,20.})for(C m:{C(1),C(1.3116),C(1.3116,10)})for(double r:{1e-6,.02,.7}){
        const auto a=bem_taylor::radial<order>(k,m,r);const double rho=k*.2;
        C actual[order+1][3];
        for(int j=0;j<=order;++j){actual[j][0]=dot*a.diagonal[j]+along*a.along[j];
            actual[j][1]=grad*a.epsilon[j];actual[j][2]=grad*a.mu[j];}
        const Vec3 separation=unit*r;
        auto exact=[&](C z,std::array<C,3>& v){
            v[0]=C(0,1)/z*muller_k1_kernel(z,m*z,separation,rotated,source);
            v[1]=muller_k2_kernel(1.,m*m,z,m*z,separation,test,normal,source);
            v[2]=muller_k2_kernel(1.,1.,z,m*z,separation,test,normal,source);
        };
        std::array<C,3> central;exact(k,central);
        LC oracle[order+1][3]{};long double scale[3]{};
        for(int p=0;p<samples;++p){
            const C z=std::polar(rho,double(2*pi*p/samples));std::array<C,3> value;exact(k+z,value);
            for(int c=0;c<3;++c)scale[c]=std::max(scale[c],static_cast<long double>(std::abs(value[c])));
            for(int j=0;j<=order;++j){
                const LC factor=std::polar(1.L/samples,-2*pi*p*j/samples);
                for(int c=0;c<3;++c)oracle[j][c]+=LC(value[c])*factor;
            }
        }
        for(int c=0;c<3;++c){
            const long double denom=std::max(1e-280L,scale[c]);
            zero_error=std::max(zero_error,double(std::abs(actual[0][c]-central[c])/denom));
            for(int j=0;j<=order;++j){
                const double e=double(std::abs(LC(actual[j][c])*std::pow(static_cast<long double>(rho),j)-oracle[j][c])/denom);
                coefficient_error=std::max(coefficient_error,e);
                if(!std::isfinite(e)||e>2e-10){std::fprintf(stderr,"k=%g m=(%g,%g) r=%g j=%d c=%d e=%g\n",k,m.real(),m.imag(),r,j,c,e);
                    throw std::runtime_error("Muller Taylor versus independent production kernel contour");}
            }
        }
        if(m==C(1))for(const auto& row:actual)for(C z:row)if(z!=C(0))throw std::runtime_error("equal media derivative");
    }
    bool rejected=false;try{bem_taylor::radial<6>(0,C(1.3),1.);}catch(const std::invalid_argument&){rejected=true;}
    if(!rejected)throw std::runtime_error("zero central frequency guard");
    std::printf("PASS Muller kernel moments 0..6: central %.3e, independent contour scaled defect %.3e; both media, small-distance branch, i/k normalization, equal media\n",zero_error,coefficient_error);
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
