#include "prism_wave_basis.h"
#include <cstdio>
int main()try {
    using C=std::complex<double>;
    MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
    const auto mesh=build_muller_p2_mesh(regular_prism(6,1/.7,3,1.,0,false,true),options);
    PrismWaveBasis waves(mesh,7.3);std::vector<C> v(mesh.system_dofs());waves(0,v.data());
    const double nodes[]={-.9602898564975363,-.7966664774136267,-.5255324099163290,-.1834346424956498,.1834346424956498,.5255324099163290,.7966664774136267,.9602898564975363};
    const double weights[]={.1012285362903763,.2223810344533745,.3137066458778873,.3626837833783620,.3626837833783620,.3137066458778873,.2223810344533745,.1012285362903763};
    double worst=0;
    for(size_t t=0;t<mesh.elements.size();t+=17)for(int edge=0;edge<3;++edge) {
        const auto& e=mesh.elements[t];const int id=e.topology_edges[edge];
        const bool forward=e.topology_vertices[edge]<e.topology_vertices[(edge+1)%3];
        const Vec3 delta=mesh.nodes[e.topology_vertices[(edge+1)%3]]-mesh.nodes[e.topology_vertices[edge]];
        C moments[2]={};
        for(int q=0;q<8;++q) {
            const double s=(nodes[q]+1)*.5,u=forward?s:1-s;
            const double xi=edge==0?u:edge==1?1-u:0,eta=edge==0?0:edge==1?u:1-u;
            const auto frame=evaluate_muller_frame(mesh,t,xi,eta);
            const auto basis=evaluate_muller_basis(mesh,t,frame);
            const Vec3 conormal=delta.cross(frame.normal)*(forward?1:-1);
            C flux=0;for(int j=0;j<basis.count;++j)flux+=v[basis.dofs[j]]*basis.values[j].dot(conormal);
            moments[0]+=.5*weights[q]*flux;moments[1]+=.5*weights[q]*(2*s-1)*flux;
        }
        for(int m=0;m<2;++m)worst=std::max(worst,std::abs(moments[m]-v[2*id+m]));
    }
    std::printf("wave BDM edge-moment duality maximum absolute error=%.3e\n",worst);
    if(worst>1e-12)throw std::runtime_error("wave edge moments disagree with BDM basis");
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
