#include "coarse_image_experiment.h"
#include <iostream>
using namespace bem_tmatrix;
int main()try {
    MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
    const auto mesh=build_muller_p2_mesh(regular_prism(6,10./7,2,1.,0,false,true),options);
    D6hOrbit group;group.build(mesh);PrismGroupMaps maps;maps.build(group);PrismSymmetrySectors sectors;sectors.build(maps);
    const int n=mesh.system_dofs();std::vector<int> vertical,sign;D6hOrbit::vertical_image_map(mesh,vertical,sign);
    const bem_streaming::Action a=[&](const Complex* x,Complex* y){for(int i=0;i<n/2;++i){y[i]=Complex(2.,.3)*x[i+n/2];y[i+n/2]=Complex(.7,-.2)*x[i];}};
    const bem_streaming::Action local=[&](const Complex* x,Complex* y){for(int i=0;i<n/2;++i){y[i]=.7*x[i+n/2]/Complex(.7,-.2);y[i+n/2]=.7*x[i]/Complex(2.,.3);}};
    CoarseImageExperiment correction(mesh,sectors,local,1e-7);
    Vector u(n),z(n),encoded(n),rotated(n),x(n),expected(n),actual(n),value(n);
    for(int k=0;k<4;++k){
        for(int i=0;i<n;++i)u[i]=Complex(std::sin((.013+.07*k)*i),std::cos((.021+.03*k)*i));
        sectors.encode(u.data(),encoded.data(),false);
        // Deliberately pure sectors plus tiny roundoff-like contamination.
        for(int s=0;s<12;++s)if(s!=(k+1)%6)for(int i=sectors.offsets[s];i<sectors.offsets[s+1];++i)encoded[i]*=1e-13;
        sectors.decode(encoded.data(),u.data(),false);a(u.data(),z.data());correction.add_d6h(u,z);
    }
    for(int i=0;i<n;++i)x[i]=Complex(std::sin(.035*i),std::cos(.027*i));
    correction.apply(x.data(),value.data());double worst=0;
    for(int g=0;g<3;++g){
        if(g<2){group.transform(x,rotated,g==1,true);group.transform(value,expected,g==1,false);}
        else for(int i=0;i<n;++i){rotated[vertical[i]]=-double(sign[i])*x[i];expected[vertical[i]]=double(sign[i])*value[i];}
        correction.apply(rotated.data(),actual.data());for(int i=0;i<n;++i)actual[i]-=expected[i];
        worst=std::max(worst,bem_streaming::norm(actual.data(),n)/bem_streaming::norm(expected.data(),n));
    }
    if(worst>2e-12)throw std::runtime_error("sparse-sector D6h inverse failed");
    std::cout<<"PASS sparse-sector rotation and both mirrors: "<<worst<<" rank="<<correction.rank()<<'\n';
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
