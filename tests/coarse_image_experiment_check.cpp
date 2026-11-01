#include "coarse_image_experiment.h"
#include "prism_vertical_inverse.h"
#include <cstdio>
using namespace bem_tmatrix;
static double relative(const Vector& a,const Vector& b){Vector d=a;for(size_t i=0;i<d.size();++i)d[i]-=b[i];return bem_streaming::norm(d.data(),d.size())/std::max(bem_streaming::norm(b.data(),b.size()),1e-300);}
int main()try {
    MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
    for(int ref:{1,2}) {
        const auto mesh=build_muller_p2_mesh(regular_prism(6,10./7,ref,1.,0,false,true),options);
        D6hOrbit group;group.build(mesh);PrismGroupMaps maps;maps.build(group);PrismSymmetrySectors sectors;sectors.build(maps);
        const int n=mesh.system_dofs();
        const bem_streaming::Action action=[&](const Complex* x,Complex* y){for(int i=0;i<n/2;++i){y[i]=Complex(2.,.3)*x[i+n/2];y[i+n/2]=Complex(.7,-.2)*x[i];}};
        const bem_streaming::Action local=[&](const Complex* x,Complex* y){for(int i=0;i<n/2;++i){y[i]=.7*x[i+n/2]/Complex(.7,-.2);y[i+n/2]=.7*x[i]/Complex(2.,.3);}};
        CoarseImageExperiment correction(mesh,sectors,local);Vector u(n),c(n),x(n),a(n),b(n),v(n),w(n),expected(n);
        for(int j=0;j<2;++j){for(int i=0;i<n;++i)u[i]=Complex(std::sin((.123+.07*j)*i),std::cos((.17+.03*j)*i));
            action(u.data(),c.data());correction.add_d6h(u,c);
            correction.apply(c.data(),a.data());if(relative(a,u)>2e-12)throw std::runtime_error("coarse exactness failed");}
        for(int i=0;i<n;++i)x[i]=Complex(std::cos(.031*i),std::sin(.019*i));
        correction.apply(x.data(),a.data());double defect=0;
        sectors.encode(x.data(),v.data(),true);correction.apply_encoded(v.data(),b.data());
        if(relative(a,b)>2e-12)throw std::runtime_error("encoded physical inverse equivalence failed");
        for(bool reflection:{false,true}){group.transform(x,v,reflection,true);correction.apply(v.data(),b.data());group.transform(a,expected,reflection,false);defect=std::max(defect,relative(b,expected));}
        std::vector<int> image,sign;D6hOrbit::vertical_image_map(mesh,image,sign);
        for(int i=0;i<n;++i)v[image[i]]=-double(sign[i])*x[i];
        correction.apply(v.data(),b.data());for(int i=0;i<n;++i)expected[image[i]]=double(sign[i])*a[i];defect=std::max(defect,relative(b,expected));
        for(int i=0;i<n;++i)w[i]=x[i]+Complex(.3,-.7)*c[i];
        correction.apply(w.data(),b.data());correction.apply(c.data(),v.data());for(int i=0;i<n;++i)expected[i]=a[i]+Complex(.3,-.7)*v[i];
        const double linearity=relative(b,expected);
        if(defect>2e-12 || linearity>2e-12)throw std::runtime_error("D6h/linearity failed");
        std::printf("ref=%d rank=%d symmetry=%.4e linearity=%.4e coarse_exactness=passed\n",ref,correction.rank(),defect,linearity);
    }
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
