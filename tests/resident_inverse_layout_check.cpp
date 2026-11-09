#include "host_symmetric_inverse_layout.h"
#include "prism_orbit_inverse.h"
#include "prism_circulant_layout.h"
#include <iostream>
using C=std::complex<double>;using V=std::vector<C>;
int main()try {
    int cases=0,rejected=0;double worst=0;
    for(int ref:{2,3})for(double aspect:{.2,10./7,5.}){
        MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
        const auto mesh=build_muller_p2_mesh(regular_prism(6,aspect,ref,1.,0,false,true),options);
        bem_tmatrix::D6hOrbit orbit;orbit.build(mesh);bem_tmatrix::PrismGroupMaps maps;maps.build(orbit);
        PrismCirculantLayout fft;fft.build(mesh,ref);const int n=orbit.size();
        bem_orbit_device::SymmetricLayout layout;layout.coverage.assign(n,0.);
        for(const auto& p:fft.patches)for(int d:p.dof)if(d>=0)for(int g=0;g<12;++g)layout.coverage[maps.image[g][d]]+=1./12;
        bem_tmatrix::D6hOrbit::vertical_image_map(mesh,layout.vertical_image,layout.vertical_sign);layout.validate();
        auto bad=[&](bem_orbit_device::SymmetricLayout q){try{q.validate();}catch(const std::invalid_argument&){++rejected;return;}throw std::runtime_error("bad symmetry accepted");};
        auto q=layout;q.coverage.clear();bad(q);q=layout;q.vertical_image.pop_back();bad(q);
        q=layout;q.vertical_image[0]=n;bad(q);q=layout;q.vertical_sign[0]=0;bad(q);
        q=layout;q.coverage[0]=-1;bad(q);q=layout;q.coverage[0]=std::numeric_limits<double>::quiet_NaN();bad(q);
        q=layout;q.coverage[0]=2;bad(q);q=layout;q.vertical_image[0]=q.vertical_image[1];bad(q);
        PrismOrbitInverse local;local.build(mesh,maps,7,[](const std::vector<int>& ids){
            const int d=4*ids.size();V a(size_t(d)*d);
            for(int i=0;i<d;++i)for(int j=0;j<d;++j)a[i*d+j]=C(.001*std::sin(i+3*j),.002*std::cos(i-j));
            for(int i=0;i<d;++i)a[i*d+i]+=C(2.+.001*i,.2);
            return a;
        });
        V rhs(n),first(n),reflected_rhs(n),second(n),fft_value(n),actual(n),expected(n);
        for(int i=0;i<n;++i){rhs[i]=C(std::sin(.13*i),std::cos(.23*i));fft_value[i]=C(std::cos(.17*i),std::sin(.11*i));}
        local.apply(rhs.data(),first.data());
        for(int i=0;i<n;++i)reflected_rhs[layout.vertical_image[i]]=-double(layout.vertical_sign[i])*rhs[i];
        local.apply(reflected_rhs.data(),second.data());
        // A non-symmetric synthetic mask makes using coverage[i] in place of
        // coverage[inverse_image[i]] observable, in addition to physical masks.
        for(bool synthetic:{false,true}){
            if(synthetic)for(int i=0;i<n;++i)layout.coverage[i]=double(i%11)/11.;
            for(double blend:{0.,.25,.5,1.}){
                for(int i=0;i<n;++i)expected[i]=.5*(1-blend*layout.coverage[i])*first[i];
                for(int i=0;i<n;++i)expected[layout.vertical_image[i]]+=.5*double(layout.vertical_sign[i])*(1-blend*layout.coverage[i])*second[i];
                for(int i=0;i<n;++i)expected[i]+=blend*fft_value[i];
                layout.combine_host(first.data(),second.data(),fft_value.data(),actual.data(),blend);
                double e=0,v=0;for(int i=0;i<n;++i){e+=std::norm(actual[i]-expected[i]);v+=std::norm(expected[i]);}
                worst=std::max(worst,std::sqrt(e/std::max(v,1e-300)));++cases;
            }
        }
    }
    if(worst>2e-14)throw std::runtime_error("resident blend disagrees with original push formula");
    std::cout<<"PASS "<<cases<<" blends; rejected="<<rejected<<" maximum_relative="<<worst<<"\n";
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
