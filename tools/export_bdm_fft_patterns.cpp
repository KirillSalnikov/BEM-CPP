#include "muller_nodal.h"
#include "quadrature.h"
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <vector>

// Canonical local BDM1 basis evaluated by the production implementation.
// Global edge incidence/orientation remains the responsibility of a later
// prism integration; these are the six independent local edge moments.
int main(){
    using C=std::complex<double>;
    const int tris[4][6]={{0,0,1,0,1,1},{0,0,1,1,0,1},{0,0,1,0,0,1},{1,0,1,1,0,1}};
    const auto rule=tri_quadrature(7);
    MullerP2Mesh mesh;mesh.basis_kind=MullerBasisKind::HDivBdm1;mesh.elements.resize(1);
    mesh.elements[0].topology_edges={{0,1,2}};mesh.elements[0].edge_orientations={{1,1,1}};
    Vec3 points[4][7],weighted[4][7][6];
    std::cout<<std::setprecision(17)<<"{\"patterns\":[";
    for(int t=0;t<4;++t){
        const auto& a=tris[t];Vec3 p(a[0],a[1],0),u(a[2]-a[0],a[3]-a[1],0),v(a[4]-a[0],a[5]-a[1],0);
        std::cout<<(t?",":"")<<"{\"points\":[";
        for(int q=0;q<7;++q){
            MullerFrameSample f;f.reference_xi=rule.pts[q][0];f.reference_eta=rule.pts[q][1];
            f.derivative_xi=u;f.derivative_eta=v;f.jacobian=u.cross(v).norm();f.normal=u.cross(v)*(1/f.jacobian);
            f.position=p+u*f.reference_xi+v*f.reference_eta;points[t][q]=f.position;
            const auto basis=evaluate_muller_basis(mesh,0,f);
            for(int j=0;j<6;++j)weighted[t][q][j]=basis.values[j]*(.5*rule.wts[q]*f.jacobian);
            std::cout<<(q?",":"")<<"["<<f.position.x<<","<<f.position.y<<"]";
        }
        std::cout<<"],\"weighted_basis\":[";
        for(int q=0;q<7;++q){std::cout<<(q?",":"")<<"[";
            for(int j=0;j<6;++j)std::cout<<(j?",":"")<<"["<<weighted[t][q][j].x<<","<<weighted[t][q][j].y<<"]";
            std::cout<<"]";}
        std::cout<<"]}";
    }
    std::cout<<"],\"oracle\":[";
    const Vec3 tu(.2,0,0),su(.1,.2*std::sqrt(3.)*.5,0),common(0,0,.15),origin(-.8,0,0);
    const Vec3 normal=tu.cross(common)*(1/tu.cross(common).norm());
    auto physical=[](Vec3 x,Vec3 u,Vec3 v){return u*x.x+v*x.y;};
    int cases=0;
    for(C ri:{C(1.3116),C(1.3116,.4)})for(double ka:{3.,10.,40.}){
        const int pt=cases%4,ps=(cases+1)%4,it=cases%4,is=(cases+2)%4,delta=cases%3-1;
        std::cout<<(cases?",":"")<<"{\"ka\":"<<ka<<",\"ri\":["<<ri.real()<<","<<ri.imag()<<"],\"pt\":"<<pt<<",\"ps\":"<<ps
                 <<",\"it\":"<<it<<",\"is\":"<<is<<",\"delta\":"<<delta<<",\"blocks\":[";
        for(int kind=0;kind<3;++kind){std::cout<<(kind?",":"")<<"[";
            for(int a=0;a<6;++a){std::cout<<(a?",":"")<<"[";
                for(int b=0;b<6;++b){C value=0;
                    for(int q=0;q<7;++q)for(int r=0;r<7;++r){
                        Vec3 d=origin+tu*it-su*is+common*delta+physical(points[pt][q],tu,common)-physical(points[ps][r],su,common);
                        const Vec3 test=physical(weighted[pt][q][a],tu,common),source=physical(weighted[ps][r][b],su,common);
                        if(d.norm2()<1e-24)continue;
                        value+=kind==0?muller_k1_kernel(ka,ka*ri,d,test.cross(normal),source):
                            muller_k2_kernel(1.,kind==1?ri*ri:C(1),ka,ka*ri,d,test,normal,source);
                    }
                    std::cout<<(b?",":"")<<"["<<value.real()<<","<<value.imag()<<"]";
                }std::cout<<"]";
            }std::cout<<"]";
        }std::cout<<"]}";++cases;
    }
    std::cout<<"]}\n";
}
