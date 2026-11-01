#include "sector_orthogonalization_experiment.h"
#include <cstdio>
using namespace bem_streaming;
int main()try {
    setenv("BEM_SECTOR_RECYCLE_MIN_LEARN_EXPERIMENT","16",1);
    setenv("BEM_SECTOR_RITZ_EXPERIMENT","0",1);
    const int n=6;std::array<int,13> offsets;offsets.fill(n);offsets[0]=0;
    const double diagonal[n]={.01,.02,1.,2.,3.,4.};
    const Action exact=[&](const Complex* a,Complex* b){for(int i=0;i<n;++i)b[i]=diagonal[i]*a[i];};
    const Action approximate=[&](const Complex* a,Complex* b){for(int i=0;i<n;++i)b[i]=(diagonal[i]+1e-6)*a[i];};
    const Action identity=[&](const Complex* a,Complex* b){std::copy(a,a+n,b);};
    ExperimentalSectorRecycleSpace space(offsets,2,1);
    std::vector<Complex> basis(3*n),h(6),empty;
    basis[0]=1.;basis[n+1]=1.;basis[2*n+2]=1.;h[0]=.01;h[3]=.02;
    space.sector(0).learn(basis,h,2,2,identity,empty,0);
    if(space.rank()!=2)throw std::runtime_error("initial slow-space construction failed");
    const auto generation=space.sector(0).generation();
    Complex answer[n]={Complex(1,.1),Complex(.8,-.3),Complex(.6,.2),Complex(.9,.4),0.,0.};
    Complex rhs[n],solution[n],check[n];exact(answer,rhs);
    auto solved=sector_gmres_experiment(approximate,exact,rhs,solution,offsets,8,1e-10,60,false,2,&space);
    exact(solution,check);for(int i=0;i<n;++i)check[i]-=rhs[i];
    if(norm(check,n)>1e-10*norm(rhs,n) || solved.iterations>=16 || solved.strict_checks<2)
        throw std::runtime_error("short mixed/strict correction case failed");
    if(space.sector(0).generation()!=generation || space.rank()!=2 || !space.sector(0).validate())
        throw std::runtime_error("short correction changed established slow space");
    ExperimentalSectorRecycleSpace fresh(offsets,2,1);
    answer[0]=answer[1]=0.;exact(answer,rhs);
    solved=sector_gmres_experiment(approximate,exact,rhs,solution,offsets,8,1e-10,60,false,2,&fresh);
    exact(solution,check);for(int i=0;i<n;++i)check[i]-=rhs[i];
    if(norm(check,n)>1e-10*norm(rhs,n) || solved.iterations>=16 || fresh.rank()!=2 || !fresh.sector(0).validate())
        throw std::runtime_error("minimum learning steps blocked a fresh space");
    std::puts("Established slow space retained through short strict corrections; fresh short space still learned; original FP64 residual passed.");
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
