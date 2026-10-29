#include "streaming_host_gmres.h"
#include "streaming_two_level.h"
#include <cstdio>
using namespace bem_streaming;
static void require(bool ok,const char* message) {if(!ok)throw std::runtime_error(message);}
int main() try {
    const int n=80,m=4;
    std::vector<Complex> diagonal(n),basis(size_t(m+1)*n,0.),h(size_t(m+1)*m,0.),empty;
    for(int i=0;i<n;++i)diagonal[i]=i<3?Complex(.001*(i+1),.0003*i):Complex(1.+double(i)/n,.1);
    for(int j=0;j<m;++j) {basis[size_t(j)*n+j]=1.;h[size_t(j)*m+j]=diagonal[j];}
    const Action identity=[&](const Complex* a,Complex* b){std::copy(a,a+n,b);};
    const Action exact=[&](const Complex* a,Complex* b){for(int i=0;i<n;++i)b[i]=diagonal[i]*a[i];};
    RecycleSpace space(n,3);
    require(space.learn(basis,h,m,m,identity,empty,0)==3,"could not learn coarse space");
    require(space.validate(),"invalid coarse space");
    TwoLevelInverse coarse(space,identity);
    const Action inverse=[&](const Complex* a,Complex* b){coarse(a,b);};
    std::vector<Complex> b(n),x(n),control(n),work(n),alias;
    for(int i=0;i<n;++i)b[i]=Complex(std::sin(.17*i)+.3,std::cos(.13*i));
    coarse(b.data(),x.data());
    for(int i=0;i<n;++i)control[i]=i<3?b[i]/diagonal[i]:b[i];
    for(int i=0;i<n;++i)work[i]=x[i]-control[i];
    require(norm(work.data(),n)/norm(control.data(),n)<1e-12,"wrong two-level inverse formula");
    alias=b;coarse(alias.data(),alias.data());
    for(int i=0;i<n;++i)work[i]=alias[i]-x[i];
    require(norm(work.data(),n)<1e-12,"aliased inverse input changed result");
    const auto plain=gmres(exact,exact,identity,b.data(),control.data(),n,10,1e-10,1000,false);
    const auto accelerated=gmres(exact,exact,inverse,b.data(),x.data(),n,10,1e-10,1000,false);
    exact(x.data(),work.data());for(int i=0;i<n;++i)work[i]-=b[i];
    require(norm(work.data(),n)/norm(b.data(),n)<1e-10,"two-level solve failed strict check");
    require(accelerated.iterations<plain.iterations,"coarse inverse did not remove slow modes");
    space.clear();bool rejected=false;
    try {coarse(b.data(),x.data());}catch(const std::runtime_error&){rejected=true;}
    require(rejected,"changed coarse space was accepted during fixed solve");
    TwoLevelInverse no_coarse(space,identity);no_coarse(b.data(),x.data());
    require(x==b,"empty coarse space changed local inverse");

    // Noncommuting A and local inverse: the coarse block must be inverted
    // by A^-1, not multiplied by the local inverse a second time.
    const int block=3;
    const Complex upper(.4,-.2);
    std::vector<Complex> triangular(size_t(block)*block,0.),local_diagonal(n);
    for(int i=0;i<n;++i)local_diagonal[i]=Complex(.6+.03*i,.05);
    for(int i=0;i<block;++i) {
        triangular[size_t(i)*block+i]=Complex(.02*(i+1),.003*i);
        if(i+1<block)triangular[size_t(i)*block+i+1]=upper;
    }
    const Action local=[&](const Complex* a,Complex* out){
        for(int i=0;i<n;++i)out[i]=a[i]*local_diagonal[i];
    };
    std::vector<Complex> block_basis(size_t(block+1)*n,0.),block_h(size_t(block+1)*block,0.);
    for(int j=0;j<block;++j) {
        block_basis[size_t(j)*n+j]=1.;
        for(int i=0;i<block;++i)
            block_h[size_t(i)*block+j]=triangular[size_t(i)*block+j]*local_diagonal[j];
    }
    RecycleSpace block_space(n,block);
    require(block_space.learn(block_basis,block_h,block,block,local,empty,0)==block,
            "could not learn nonnormal coarse block");
    TwoLevelInverse block_inverse(block_space,local);block_inverse(b.data(),x.data());
    control=b;
    std::vector<int> pivots;
    require(bem_krylov::dense_lu_factor(triangular,block,pivots),"singular reference block");
    bem_krylov::dense_lu_solve(triangular,pivots,block,control.data());
    for(int i=block;i<n;++i)control[i]*=local_diagonal[i];
    for(int i=0;i<n;++i)work[i]=x[i]-control[i];
    require(norm(work.data(),n)/norm(control.data(),n)<1e-11,
            "noncommuting local/global correction has wrong order");
    std::fill(block_basis.begin(),block_basis.end(),Complex(0));
    for(int j=0;j<block;++j)block_basis[size_t(j)*n+j+block]=1.;
    require(block_space.learn(block_basis,block_h,block,block,local,empty,0)>0 &&
            block_space.rank()==block,"fixed-rank replacement did not run");
    rejected=false;
    try {block_inverse(b.data(),x.data());}catch(const std::runtime_error&){rejected=true;}
    require(rejected,"same-rank space replacement bypassed the frozen-space guard");
    std::printf("two-level inverse: %d -> %d iterations; exact action, aliasing, strict residual and frozen-space guard passed\n",
                plain.iterations,accelerated.iterations);
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
