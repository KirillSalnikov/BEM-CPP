#include "planar_muller_symbol.h"
#include "muller_nodal.h"
#include <cstdio>
using bem_osrc::C;
static void require(bool b,const char* message){if(!b)throw std::runtime_error(message);}
int main()try{
    double worst_inverse=0,worst_integral=0;
    for(double k0:{2.,20.,40.})for(C m:{C(1.),C(1.3116),C(1.3116,10.)})
        for(double rho:{0.,.5,1.,1.3116,2.,10.})for(double angle:{0.,.31,1.2}){
            const C ke(k0,.03*k0),ki=ke*m;const double x=k0*rho*std::cos(angle),y=k0*rho*std::sin(angle);
            const auto a=bem_osrc::muller(k0,ke,ki,1.+m*m,2.,x,y),b=bem_osrc::inverse(k0,ke,ki,1.+m*m,2.,x,y);
            for(int i=0;i<4;++i)for(int j=0;j<4;++j){C v=0;for(int k=0;k<4;++k)v+=a[i*4+k]*b[k*4+j];worst_inverse=std::max(worst_inverse,std::abs(v-C(i==j)));}
            const C root=bem_osrc::outgoing_root(ki,x*x+y*y);require(root.imag()>=0&&root.real()>=0,"outgoing branch");
            const auto d=bem_osrc::dyadic(ki,x,y),map=bem_osrc::mte(ki,1./m,x,y);
            // R Lambda R^-1 swaps longitudinal and transverse polarizations.
            const C scale=ki/m;const bem_osrc::Matrix2 swapped{{map[3],-map[2],-map[1],map[0]}};
            for(int j=0;j<4;++j)require(std::abs(d[j]-scale*swapped[j])<=2e-12*std::max(1.,std::abs(d[j])),"MtE/Green dyadic relation");
            if(m==C(1.)){require(a[0]==C(0)&&a[1]==C(0)&&a[4]==C(0)&&a[5]==C(0),"equal-media cancellation");}
        }
    require(worst_inverse<2e-11,"planar inverse identity");
    require(std::abs(bem_osrc::outgoing_root(2.,5.)-C(0,1))<1e-15,"evanescent decay");
    bool rejected=false;try{bem_osrc::dyadic(2.,2.,0.);}catch(const std::domain_error&){rejected=true;}require(rejected,"unregularized glancing mode not rejected");
    // Independent radial Fourier integral of the existing spatial Green/Hessian
    // difference. Damping makes a finite truncation exponentially accurate.
    // Angular integration is exact via J0/J2; Simpson integrates radius.
    const C ke(3.,.7),ki(4.1,1.2);const int samples=16000;const double stop=50,dr=stop/samples;
    for(double rho:{0.,1.1,3.,4.1,7.}){
        C sum[2]{};
        for(int s=0;s<=samples;++s){double r=s*dr;if(s==0)r=1e-8;
            const auto h=muller_hessian_difference(ke,ki,Vec3(r,0,0));
            const C scalar=(ke*ke*std::exp(C(0,1)*ke*r)-ki*ki*std::exp(C(0,1)*ki*r))/(4*std::acos(-1.)*r);
            const C common=(2.*scalar+h[0]+h[4])*::j0(rho*r),difference=(h[0]-h[4])*::jn(2,rho*r);
            const double weight=(s==0||s==samples?1:s%2?4:2)*std::acos(-1.)*r;
            sum[0]+=weight*(common-difference);sum[1]+=weight*(common+difference);
        }
        const auto d=bem_osrc::dyadic(ke,rho,0),e=bem_osrc::dyadic(ki,rho,0);
        for(int i=0;i<2;++i){const C integral=sum[i]*dr/3.,exact=C(0,.5)*(d[3*i]-e[3*i]);
            worst_integral=std::max(worst_integral,std::abs(integral-exact)/std::max(1.,std::abs(exact)));}
    }
    require(worst_integral<2e-7,"spatial Green Fourier integral/sign mismatch");
    std::printf("PASS planar two-media symbol: inverse %.3e, independent radial integral %.3e; MtE duality, passive root, both polarizations, glancing rejection, equal media\n",worst_inverse,worst_integral);
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
