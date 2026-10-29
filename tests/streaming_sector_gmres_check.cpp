#include "streaming_sector_gmres.h"
#include <cstdio>
using namespace bem_streaming;
int main()try {
 const int n=72;std::array<int,13> offsets;for(int s=0;s<=12;++s)offsets[s]=s*6;
 std::vector<Complex> matrix(n*n),expected(n),b(n),x(n),w(n);
 for(int s=0;s<12;++s)for(int i=0;i<6;++i)for(int j=0;j<6;++j)
    matrix[(6*s+i)*n+6*s+j]=i==j?Complex(1.+.4*s+.2*i,.1*s):Complex(.03*std::sin(i+3*j+s),.02*std::cos(2*i-j));
 for(int i=0;i<n;++i)expected[i]=i>=18 && i<24?Complex(0):Complex(std::cos(.37*i),std::sin(.27*i));
 const Action strict=[&](const Complex* a,Complex* out){for(int i=0;i<n;++i){out[i]=0;for(int j=0;j<n;++j)out[i]+=matrix[i*n+j]*a[j];}};
 strict(expected.data(),b.data());
 for(bool mixed:{false,true})for(int restart:{3,8}) {
    const Action inner=[&](const Complex* a,Complex* out){strict(a,out);if(mixed)for(int i=0;i<n;++i)out[i]+=Complex(2e-6,-1e-6)*a[(i+9)%n];};
    auto r=sector_gmres(inner,strict,b.data(),x.data(),offsets,restart,1e-10,300,false,3);
    strict(x.data(),w.data());for(int i=0;i<n;++i)w[i]-=b[i];const double e=norm(w.data(),n)/norm(b.data(),n);
    std::printf("sector mixed=%d restart=%d actions=%d residual=%.3e\n",mixed,restart,r.iterations,e);
    if(e>1e-10 || std::abs(e-r.residual)>1e-13 || !r.strict_checks)throw std::runtime_error("sector solve residual");
    if(!mixed && restart==8 && r.iterations>6)throw std::runtime_error("sectors not solved concurrently");
 }
 auto r=sector_gmres(strict,strict,b.data(),x.data(),offsets,3,1e-14,1,false);
 if(r.iterations!=1 || r.strict_checks!=1 || r.residual<=1e-14)throw std::runtime_error("sector action limit");
 std::fill(b.begin(),b.end(),Complex(0));r=sector_gmres(strict,strict,b.data(),x.data(),offsets,3,1e-10,30,false);
 if(r.iterations || r.residual || norm(x.data(),n))throw std::runtime_error("zero RHS");
 std::puts("Independent sector Krylov, strict replacement with cross-sector perturbations, zero sectors/RHS and limits passed");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
