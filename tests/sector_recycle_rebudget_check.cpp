#include "sector_recycle_rebudget.h"
#include <cstdio>
using namespace bem_streaming;
static void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
static void seed(RecycleSpace& space){
    const int n=space.size();std::vector<Complex> basis(size_t(5)*n),h(20),empty;
    for(int i=0;i<4;++i){basis[size_t(i)*n+i]=1.;h[size_t(i)*4+i]=.01*(i+1);}
    const Action identity=[&](const Complex* a,Complex* b){std::copy(a,a+n,b);};
    require(space.learn(basis,h,4,4,identity,empty,0)==4,"seed rank");
}
int main()try{
    std::array<int,13> offsets{};for(int s=0;s<=12;++s)offsets[s]=8*s;
    SectorRecycleSpace shared(offsets,4);for(int s=0;s<12;++s)seed(shared.sector(s));
    auto independent_lane=shared;
    std::array<double,12> work;work.fill(10);work[1]=work[5]=work[7]=work[11]=200;
    std::array<unsigned,12> seen;seen.fill(2);
    const auto plan=sector_budget_plan(offsets,4,4095,work,seen);
    require(plan.capacities[0]<4 && plan.capacities[1]>4,"test did not shrink and grow");
    rebudget_sector_recycle(shared,offsets,plan);
    std::vector<Complex> rhs(8,1.),x(8),r(8);
    for(int s=0;s<12;++s){
        auto& q=shared.sector(s);
        require(q.capacity()==plan.capacities[s] && q.rank()<=q.capacity() && q.validate(),"rebudget rank/orthogonality");
        require(independent_lane.sector(s).rank()==4 && independent_lane.sector(s).capacity()==4,"lane alias");
        std::fill(x.begin(),x.end(),0.);r=rhs;q.correct(x.data(),r.data());
        for(int i=0;i<8;++i)require(std::abs(.01*(i+1)*x[i]+r[i]-rhs[i])<2e-12,"rebudget lost A U=C");
    }
    require(shared.storage_gib()<=2.*sizeof(Complex)*plan.entry_budget/(1024.*1024.*1024.),"final storage budget");
    auto bad=plan;bad.capacities[1]--;bool rejected=false;
    try{rebudget_sector_recycle(shared,offsets,bad);}catch(const std::invalid_argument&){rejected=true;}
    require(rejected && shared.sector(1).capacity()==plan.capacities[1],"invalid plan changed live space");
    // An approximate image retained during shrinking cannot accept a solve.
    const Action strict=[](const Complex* a,Complex* b){for(int i=0;i<8;++i)b[i]=.02*(i+1)*a[i];};
    const Action identity=[](const Complex* a,Complex* b){std::copy(a,a+8,b);};
    std::fill(rhs.begin()+shared.sector(0).rank(),rhs.end(),0.);
    const auto solved=gmres(strict,strict,identity,rhs.data(),x.data(),8,8,1e-10,100,false,false,&shared.sector(0));
    strict(x.data(),r.data());for(int i=0;i<8;++i)r[i]-=rhs[i];
    require(norm(r.data(),8)/norm(rhs.data(),8)<1e-10 && solved.recycle_fallbacks>=1,"rebudget bypassed strict gate");
    std::puts("PASS rebudget A U=C, pruning, orthogonality, memory, independent lanes, original FP64 gate");
}catch(const std::exception& e){std::fprintf(stderr,"rebudget: %s\n",e.what());return 1;}
