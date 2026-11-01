#include "streaming_sector_gmres.h"
#include <cstdio>
using namespace bem_streaming;
static void require(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
// Two exact Arnoldi directions of a diagonal operator, with known A U = C.
static void seed(RecycleSpace& space,int first,double a,double b){
    const int n=space.size();std::vector<Complex> basis(3*n),h={a,0.,0.,b,0.,0.},empty;
    basis[first]=1.;basis[n+first+1]=1.;
    const Action identity=[&](const Complex* x,Complex* y){std::copy(x,x+n,y);};
    require(space.learn(basis,h,2,2,identity,empty,0)==2,"seed rank");
}
int main()try{
    const int n=8;const double diagonal[n]={.01,.02,2.,3.,4.,5.,6.,7.};
    RecycleSpace first(n,4),second(n,4);seed(first,0,diagonal[0],diagonal[1]);seed(second,2,diagonal[2],diagonal[3]);
    const auto generation=first.generation();
    require(first.merge_from(second)==2 && first.rank()==4,"union did not keep complementary directions");
    require(first.validate() && second.validate() && second.rank()==2,"merge damaged orthogonality/source");
    require(first.generation()==generation+1 && first.dirty(),"merge checkpoint generation");
    require(first.merge_from(second)==0 && first.merge_from(first)==0,"duplicate directions accepted");
    std::vector<Complex> rhs(n),x(n),residual(n);for(int i=0;i<n;++i)rhs[i]=Complex(1.+i,.1*i);
    residual=rhs;first.correct(x.data(),residual.data());
    for(int i=0;i<n;++i){
        require(std::abs(diagonal[i]*x[i]+residual[i]-rhs[i])<1e-12,"merged A U = C relation lost");
        require(std::abs(residual[i]-(i<4?Complex(0):rhs[i]))<1e-12,"merged span correction");
    }
    RecycleSpace bounded(n,2);seed(bounded,0,diagonal[0],diagonal[1]);bounded.merge_from(second);
    require(bounded.rank()==2 && bounded.validate(),"merge capacity");
    std::fill(x.begin(),x.end(),0.);residual=rhs;bounded.correct(x.data(),residual.data());
    require(std::abs(residual[0])+std::abs(residual[1])<1e-12 && std::abs(residual[2]-rhs[2])<1e-12,"slow directions not preferred");
    RecycleSpace disabled(n,0);require(disabled.merge_from(first)==0 && disabled.rank()==0,"disabled space grew");
    bool rejected=false;try{RecycleSpace wrong(n+1,4);first.merge_from(wrong);}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"mismatched dimension accepted");
    const std::string path="/tmp/bem-merge-check-"+std::to_string(getpid()),signature(64,'m');
    first.save(path,signature);RecycleSpace restored(n,4);require(restored.load(path,signature),"merged checkpoint roundtrip");
    std::remove(path.c_str());
    std::vector<Complex> y(n),r=rhs;restored.correct(y.data(),r.data());
    std::fill(x.begin(),x.end(),0.);residual=rhs;first.correct(x.data(),residual.data());
    require(x==y && r==residual,"roundtrip changed correction");
    std::array<int,13> offsets;for(int s=0;s<=12;++s)offsets[s]=s*n;
    SectorRecycleSpace target(offsets,4,1),source(offsets,4,3);
    seed(target.sector(0),0,1.,2.);seed(source.sector(0),2,3.,4.);seed(source.sector(1),0,1.,2.);
    require(target.merge_from(source)==2 && target.sector(0).rank()==4 && target.sector(1).rank()==0,"sector mask not preserved");
    auto different=offsets;different[1]++;SectorRecycleSpace wrong_offsets(different,4);
    rejected=false;try{target.merge_from(wrong_offsets);}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"mismatched sector coordinates accepted");
    // A merged inner model cannot accept a solve against a different strict operator.
    const Action strict=[&](const Complex* v,Complex* out){for(int i=0;i<n;++i)out[i]=2*diagonal[i]*v[i];};
    const Action identity=[&](const Complex* v,Complex* out){std::copy(v,v+n,out);};
    std::fill(rhs.begin()+4,rhs.end(),0.);
    const auto result=gmres(strict,strict,identity,rhs.data(),x.data(),n,8,1e-10,100,false,false,&restored);
    strict(x.data(),residual.data());for(int i=0;i<n;++i)residual[i]-=rhs[i];
    require(norm(residual.data(),n)/norm(rhs.data(),n)<1e-10 && result.strict_checks>=2 && result.recycle_fallbacks>=1,"merged model bypassed strict residual gate");
    std::puts("Merge: complementary spans, A U=C, rank budget, duplicate rejection, sector masks, checkpoints, strict fallback passed");
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
