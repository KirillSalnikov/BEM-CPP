#include "ritz_sector_solver.h"
#include <iostream>
using namespace bem_tmatrix;
static double relative(Vector x,const Vector& y){
    for(size_t i=0;i<x.size();++i)x[i]-=y[i];
    return bem_streaming::norm(x.data(),x.size())/std::max(bem_streaming::norm(y.data(),y.size()),1e-300);
}
int main()try {
    int cases=0;double worst=0;setenv("BEM_HOST_GMRES_STRICT_INTERVAL","2",1);
    for(int ref:{1,2}){
        MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
        auto mesh=build_muller_p2_mesh(regular_prism(6,10./7,ref,1.,0,false,true),options);
        D6hOrbit orbit;orbit.build(mesh);PrismGroupMaps maps;maps.build(orbit);PrismSymmetrySectors sectors;sectors.build(maps);
        const int n=sectors.size();Vector diagonal(n),tmp(n),encoded(n),truth(n),rhs(n),a(n),b(n),image(n);
        for(int i=0;i<n;++i)diagonal[i]=Complex(1.+.07*(i%7),.03*(i%3));
        const bem_streaming::Action inverse=[&](const Complex* x,Complex* y){for(int i=0;i<n/2;++i){y[i]=.7*x[i+n/2]/Complex(.7,-.2);y[i+n/2]=.7*x[i]/Complex(2.,.3);}};
        const bem_streaming::Action exact=[&](const Complex* x,Complex* y){
            for(int i=0;i<n/2;++i){tmp[i]=Complex(2.,.3)*x[i+n/2];tmp[i+n/2]=Complex(.7,-.2)*x[i];}
            sectors.encode(tmp.data(),encoded.data(),true);
            for(int i=0;i<n;++i)encoded[i]*=diagonal[i];
            sectors.decode(encoded.data(),y,true);
        };
        int fused_calls=0,strict_calls=0;
        const bem_streaming::Action strict=[&](const Complex* x,Complex* y){++strict_calls;exact(x,y);};
        // Deliberately biased mixed action: a fused route must still refine
        // and accept only against the unmodified physical operator/RHS.
        const bem_streaming::Action mixed=[&](const Complex* x,Complex* y){exact(x,y);for(int i=0;i<n;++i)y[i]*=1.+2e-5;};
        const bem_streaming::Action fused=[&](const Complex* x,Complex* y){++fused_calls;for(int i=0;i<n;++i)y[i]=.7*(1.+2e-5)*diagonal[i]*x[i];};
        for(int mode:{0,1,2}){
            bem_streaming::SectorRecycleSpace ra(sectors.offsets,4),rb(sectors.offsets,4);
            for(int sample=0;sample<2;++sample){
                for(int i=0;i<n;++i)truth[i]=Complex(std::sin((.031+.003*sample)*i),std::cos(.019*i));
                exact(truth.data(),rhs.data());for(int i=0;i<n;++i)a[i]=b[i]=(mode==0?0.:mode==1?.5:1.)*truth[i];
                fused_calls=strict_calls=0;std::array<int,12> counts{};
                auto result=solve_prism_sectors_with_ritz(mixed,strict,inverse,bem_streaming::Action(),sectors,rhs.data(),a.data(),12,1e-10,160,
                    mode!=0,false,&ra,&counts,fused);
                if(!strict_calls || (mode!=2 && !fused_calls) || (mode==2 && (fused_calls || result.iterations)))throw std::runtime_error("fused cold/warm dispatch");
                if(mode!=2 && result.strict_checks<2)throw std::runtime_error("biased action bypassed physical refinement");
                auto baseline=solve_prism_sectors(mixed,strict,inverse,sectors,rhs.data(),b.data(),12,1e-10,160,mode!=0,false,&rb);
                exact(a.data(),image.data());worst=std::max({worst,relative(a,truth),relative(a,b),relative(image,rhs)});
                if(result.residual>1e-10 || baseline.residual>1e-10 || worst>1e-8 || std::abs(result.residual-relative(image,rhs))>1e-13)
                    throw std::runtime_error("fused original-current acceptance/equivalence");
                ++cases;
            }
        }
        const bem_streaming::Action wrong=[&](const Complex* x,Complex* y){std::copy(x,x+n,y);};
        auto rejected=solve_prism_sectors(exact,strict,inverse,sectors,rhs.data(),a.data(),1,1e-10,1,false,false,nullptr,nullptr,wrong);
        if(rejected.residual<=1e-10)throw std::runtime_error("wrong fused action accepted from projected residual");
        ++cases;
        bool caught=false;try{solve_prism_sectors_with_ritz(exact,strict,inverse,inverse,sectors,rhs.data(),a.data(),12,1e-10,20,false,false,nullptr,nullptr,fused);}
        catch(const std::invalid_argument&){caught=true;}if(!caught)throw std::runtime_error("incompatible Ritz/fused accepted");++cases;
    }
    std::cout<<"PASS "<<cases<<" fused cold/warm/recycle/refusal controls; maximum_error="<<worst<<"\n";
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
