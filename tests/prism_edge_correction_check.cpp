#include "prism_edge_correction.h"
#include "prism_vertical_inverse.h"
#include <cstdio>
using namespace bem_tmatrix;
double error(Vector a,const Vector& b){for(size_t i=0;i<a.size();++i)a[i]-=b[i];return bem_streaming::norm(a.data(),a.size())/std::max(bem_streaming::norm(b.data(),b.size()),1e-300);}
void require(bool ok,const char* msg){if(!ok)throw std::runtime_error(msg);}
int main()try {
    omp_set_num_threads(2);
    for(int ref:{0,1,2,3,4})for(int rings:{0,1,2,-1,-2}) {
        if(ref>2 && (rings<0||rings>1))continue;
        MullerP2BuildOptions opt;opt.edge_mode=MullerEdgeMode::HDivBdm1;
        const auto mesh=build_muller_p2_mesh(regular_prism(6,10./7,ref,1.,0,false,true),opt);
        D6hOrbit orbit;orbit.build(mesh);PrismGroupMaps maps;maps.build(orbit);const int n=mesh.system_dofs();
        PrismEdgeCorrection edge;
        const auto assemble=[&](const std::vector<int>& ids){const int d=4*ids.size();Vector a(d*d);
            for(int i=0;i<d;++i)for(int j=0;j<d;++j)a[i*d+j]=Complex(.003*std::sin(i+2*j+ids[0]),.002*std::cos(2*i-j));
            for(int i=0;i<d;++i)a[i*d+i]+=Complex(2.+.01*i,.3);
            for(int j=0;j<d;++j)std::swap(a[j],a[d+j]);
            return a;};
        edge.build(mesh,maps,rings,assemble);
        Vector x(n),a(n),b(n),c(n),d(n);for(int i=0;i<n;++i)x[i]=Complex(std::sin(.217*i),std::cos(.391*i));
        edge.apply(x.data(),a.data());
        PrismEdgeCorrection legacy;legacy.build(mesh,maps,rings,assemble,false);legacy.apply(x.data(),b.data());
        require(error(a,b)<1e-15,"compact scatter changed the inverse");
        require(edge.scatter_workspace_bytes()<=legacy.scatter_workspace_bytes(),"compact scatter grew workspace");
        for(bool mirror:{false,true}){orbit.transform(x,b,mirror,true);edge.apply(b.data(),c.data());orbit.transform(a,d,mirror,false);
            require(error(c,d)<1e-12,"rotation/horizontal reflection covariance");}
        PrismVerticalInverse vertical;vertical.build(mesh,[&](const Complex* v,Complex* out){edge.apply(v,out);});
        vertical.transform(x.data(),b.data(),true);edge.apply(b.data(),c.data());vertical.transform(a.data(),d.data(),false);
        require(error(c,d)<1e-12,"vertical reflection covariance");
        PrismEdgeCorrection exchange;const Complex v(1.7,.2);
        exchange.build(mesh,maps,rings,[&](const std::vector<int>& ids){const int d=4*ids.size();Vector a(d*d);
            for(int i=0;i<d;++i)a[i*d+(i/4)*4+(i+2)%4]=v;
            return a;});
        exchange.apply(x.data(),a.data());
        for(int i=0;i<n;++i)b[i]=exchange.covers(i)?x[(i+n/2)%n]/v:Complex(0);
        require(error(a,b)<1e-12,"overlap weights or pivoting changed exact local inverse");
        std::printf("ref=%d rings=%d sharp=%d patches=%zu covered=%zu/%d scatter_bytes=%zu->%zu D6h/pivot/overlap passed\n",ref,rings,edge.sharp_edges,edge.blocks(),edge.covered(),n,legacy.scatter_workspace_bytes(),edge.scatter_workspace_bytes());
    }
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
