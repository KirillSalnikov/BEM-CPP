#include "coarse_image_experiment.h"
#include "ritz_sector_solver.h"
#include <iostream>
using namespace bem_tmatrix;
static double relative(Vector x,const Vector& y){
    for(size_t i=0;i<x.size();++i)x[i]-=y[i];
    return bem_streaming::norm(x.data(),x.size())/std::max(bem_streaming::norm(y.data(),y.size()),1e-300);
}
int main()try {
    int checks=0;double worst=0;
    for(int ref:{1,2}){
        MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
        const auto mesh=build_muller_p2_mesh(regular_prism(6,10./7,ref,1.,0,false,true),options);
        D6hOrbit group;group.build(mesh);PrismGroupMaps maps;maps.build(group);PrismSymmetrySectors sectors;sectors.build(maps);
        const int n=mesh.system_dofs();int strict_calls=0,encoded_calls=0;
        const bem_streaming::Action exact=[&](const Complex* x,Complex* y){for(int i=0;i<n/2;++i){y[i]=Complex(2.,.3)*x[i+n/2];y[i+n/2]=Complex(.7,-.2)*x[i];}};
        const bem_streaming::Action strict=[&](const Complex* x,Complex* y){++strict_calls;exact(x,y);};
        const bem_streaming::Action local=[&](const Complex* x,Complex* y){for(int i=0;i<n/2;++i){y[i]=.7*x[i+n/2]/Complex(.7,-.2);y[i+n/2]=.7*x[i]/Complex(2.,.3);}};
        CoarseImageExperiment correction(mesh,sectors,local);Vector u(n),image(n),truth(n),rhs(n),a(n),b(n);
        for(int j=0;j<2;++j){for(int i=0;i<n;++i)u[i]=Complex(std::sin((.123+.07*j)*i),std::cos((.17+.03*j)*i));
            exact(u.data(),image.data());correction.add_d6h(u,image);}
        const bem_streaming::Action inverse=[&](const Complex* x,Complex* y){correction.apply(x,y);};
        const bem_streaming::Action encoded=[&](const Complex* x,Complex* y){++encoded_calls;correction.apply_encoded(x,y);};
        for(int mode=0;mode<3;++mode){
            bem_streaming::SectorRecycleSpace recycle_a(sectors.offsets,4),recycle_b(sectors.offsets,4);
            for(int sample=0;sample<2;++sample){
                for(int i=0;i<n;++i)truth[i]=Complex(std::sin((.031+.003*sample)*i),std::cos(.019*i));
                exact(truth.data(),rhs.data());
                for(int i=0;i<n;++i)a[i]=b[i]=(mode==0?0.:mode==1?.5:1.)*truth[i];
                strict_calls=encoded_calls=0;std::array<int,12> counts{};
                const auto actual=solve_prism_sectors_with_ritz(exact,strict,inverse,encoded,sectors,rhs.data(),a.data(),16,1e-10,100,
                    mode!=0,false,&recycle_a,&counts);
                if(!strict_calls || (mode==0 && !encoded_calls) || (mode!=0 && encoded_calls))throw std::runtime_error("wrong cold/warm inverse dispatch");
                const auto original=solve_prism_sectors(exact,strict,inverse,sectors,rhs.data(),b.data(),16,1e-10,100,mode!=0,false,&recycle_b);
                exact(a.data(),image.data());worst=std::max({worst,relative(a,truth),relative(a,b),relative(image,rhs)});
                if(actual.residual>1e-10 || original.residual>1e-10 || worst>1e-9)throw std::runtime_error("original equation/current equivalence");
                ++checks;
            }
        }
    }
    std::cout<<"PASS "<<checks<<" Ritz cold/warm/recycle solves; maximum_error="<<worst<<'\n';
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
