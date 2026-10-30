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
 for(bool mixed:{false,true})for(int restart:{3,8})for(double factor:{.25,.5})for(double first:{0.,8.}) {
    const Action inner=[&](const Complex* a,Complex* out){strict(a,out);if(mixed)for(int i=0;i<n;++i)out[i]+=Complex(2e-6,-1e-6)*a[(i+9)%n];};
    auto r=sector_gmres(inner,strict,b.data(),x.data(),offsets,restart,1e-10,300,false,3,nullptr,factor,first);
    strict(x.data(),w.data());for(int i=0;i<n;++i)w[i]-=b[i];const double e=norm(w.data(),n)/norm(b.data(),n);
    std::printf("sector mixed=%d restart=%d actions=%d residual=%.3e\n",mixed,restart,r.iterations,e);
    if(e>1e-10 || std::abs(e-r.residual)>1e-13 || !r.strict_checks)throw std::runtime_error("sector solve residual");
    if(!first && !mixed && restart==8 && r.iterations>6)throw std::runtime_error("sectors not solved concurrently");
 }
 auto r=sector_gmres(strict,strict,b.data(),x.data(),offsets,3,1e-14,1,false);
 if(r.iterations!=1 || r.strict_checks!=1 || r.residual<=1e-14)throw std::runtime_error("sector action limit");
 std::fill(b.begin(),b.end(),Complex(0));r=sector_gmres(strict,strict,b.data(),x.data(),offsets,3,1e-10,30,false);
 if(r.iterations || r.residual || norm(x.data(),n))throw std::runtime_error("zero RHS");
 // The loose first-pass target is not an acceptance tolerance. This
 // diagonal system stops its first inner pass after one inaccurate step.
 {
  std::array<int,13> off;off.fill(2);off[0]=0;
  const Action diagonal=[](const Complex* a,Complex* out){out[0]=a[0];out[1]=1.01*a[1];};
  Complex rhs[2]={1.,1.},solution[2];
  const auto coarse=sector_gmres(diagonal,diagonal,rhs,solution,off,4,1e-3,20,false,3,nullptr,.5,32.);
  if(coarse.strict_checks<2 || coarse.residual>1e-3)throw std::runtime_error("coarse first pass bypassed strict acceptance");
 }
 // Slow modes differ between sectors and are reused across unrelated RHS.
 // Compare against known solutions, including a non-block-diagonal mixed
 // perturbation. Recycled inner residuals must never replace the true gate.
 {
  const int width=40,total=12*width;
  std::array<int,13> off;for(int s=0;s<=12;++s)off[s]=s*width;
  SectorRecycleSpace space(off,8);
  std::vector<Complex> answer(total),rhs(total),solution(total),check(total);
  const Action exact=[&](const Complex* a,Complex* out){
   for(int i=0;i<total;++i){int j=i%width,s=i/width;
    Complex d=j<4?Complex(.003*(j+1)*(1+.03*s),.0003*s):Complex(.5+1.5*j/width,.02*s);
    out[i]=d*a[i]+(j+1<width?Complex(.003,.001)*a[i+1]:Complex(0));}
  };
  const Action approximate=[&](const Complex* a,Complex* out){exact(a,out);for(int i=0;i<total;++i)out[i]+=Complex(1e-8,2e-8)*a[(i+width)%total];};
  int plain_steps=0,recycled_steps=0;
  for(int rhs_id=0;rhs_id<3;++rhs_id) {
   for(int i=0;i<total;++i)answer[i]=Complex(std::sin((.12+.03*rhs_id)*i),std::cos((.17+.07*rhs_id)*i));
   exact(answer.data(),rhs.data());
   const auto plain=sector_gmres(approximate,exact,rhs.data(),solution.data(),off,16,1e-9,1000,false,3);
   const auto recycled=sector_gmres(approximate,exact,rhs.data(),solution.data(),off,16,1e-9,1000,false,3,&space,.5,8.);
   exact(solution.data(),check.data());for(int i=0;i<total;++i)check[i]-=rhs[i];
   const double error=norm(check.data(),total)/norm(rhs.data(),total);
   if(error>1e-9 || std::abs(error-recycled.residual)>1e-14 || plain.residual>1e-9)throw std::runtime_error("sector recycling true residual");
   for(int i=0;i<total;++i)check[i]=solution[i]-answer[i];
   if(norm(check.data(),total)/norm(answer.data(),total)>1e-6)throw std::runtime_error("sector recycled solution");
   for(int s=0;s<12;++s)if(!space.sector(s).validate())throw std::runtime_error("sector recycle orthogonality");
   if(rhs_id){plain_steps+=plain.iterations;recycled_steps+=recycled.iterations;}
   std::printf("sector recycle RHS=%d plain=%d recycle=%d strict=%.3e rank=%d\n",rhs_id,plain.iterations,recycled.iterations,error,space.rank());
  }
  if(recycled_steps>=plain_steps)throw std::runtime_error("sector slow modes were not retained");
 }
 // Empty sectors and a deliberately inaccurate coarse model exercise the
 // strict fallback without taking even one inner step before correction.
 {
  std::array<int,13> off;off.fill(2);off[0]=0;
  SectorRecycleSpace space(off,2,1);
  std::vector<Complex> v={1.,0.,0.,1.,0.,0.},h={1.,0.,0.,1.,0.,0.},empty;
  const Action identity=[](const Complex* a,Complex* out){std::copy(a,a+2,out);};
  const Action twice=[](const Complex* a,Complex* out){out[0]=2.*a[0];out[1]=2.*a[1];};
  space.sector(0).learn(v,h,2,2,identity,empty,0);
  Complex rhs[2]={Complex(1,.3),Complex(-.4,.2)},solution[2];
  const auto fallback=sector_gmres(twice,twice,rhs,solution,off,2,1e-10,10,false,3,&space);
  if(fallback.residual>1e-10 || fallback.recycle_fallbacks!=1 || fallback.strict_checks<2 || std::abs(solution[0]-.5*rhs[0])>1e-10)
   throw std::runtime_error("sector coarse model bypassed strict check");
  for(int s=1;s<12;++s)if(space.sector(s).rank())throw std::runtime_error("empty sector learned vectors");
 }
 std::puts("Independent sector Krylov, strict replacement with cross-sector perturbations, zero sectors/RHS and limits passed");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
