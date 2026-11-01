#include "bdm_mesh_transfer.h"
#include "tmatrix_symmetry_warm_start.h"
#include <cstdio>
using V=bem_tmatrix::Vector;
using C=std::complex<double>;
static double difference(const V& a,const V& b){double n=0,d=0;for(size_t i=0;i<a.size();++i){n+=std::norm(a[i]-b[i]);d+=std::norm(b[i]);}return std::sqrt(n/std::max(d,1e-300));}
static void require(bool b,const char* message){if(!b)throw std::runtime_error(message);}
// Surface curl of a globally continuous quadratic potential: its edge
// co-normal trace is continuous even at a prism crease. BDM1 represents it
// exactly on every planar triangle, including non-nested triangulations.
static V analytic(const MullerP2Mesh& mesh) {
    const auto gradient=[](const Vec3& p){return Vec3(1+.3*p.x+.2*p.y,-.7+.2*p.x-.4*p.y+.1*p.z,.4+.1*p.y+.5*p.z);};
    V out(mesh.system_dofs());std::vector<bool> done(mesh.topology_edge_count,false);
    for(size_t t=0;t<mesh.elements.size();++t){const auto& e=mesh.elements[t];
        const auto frame=evaluate_muller_frame(mesh,t,.2,.3);
        for(int k=0;k<3;++k){const int edge=e.topology_edges[k];if(done[edge])continue;done[edge]=true;
            const Vec3 p=mesh.nodes[e.nodes[k]],q=mesh.nodes[e.nodes[(k+1)%3]],edge_vector=q-p;
            const Vec3 measure=edge_vector.cross(frame.normal);
            const double a=frame.normal.cross(gradient(p)).dot(measure),b=frame.normal.cross(gradient(q)).dot(measure);
            for(int current=0;current<2;++current){const int at=current*mesh.current_dofs()+2*edge;
                const C scale=current?C(.7,-.8):C(1,.3);
                out[at]=scale*double(e.edge_orientations[k])*(a+b)*.5;out[at+1]=scale*(b-a)/6.;}
        }
    }
    return out;
}
int main()try {
    MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
    for(double aspect:{10./7,1.}) {
        const auto coarse=build_muller_p2_mesh(regular_prism(6,aspect,2,1,0,false,true),options);
        const auto fine=build_muller_p2_mesh(regular_prism(6,aspect,3,1,0,false,true),options);
        BdmMeshTransfer transfer,identity,reverse;transfer.build(coarse,fine);identity.build(coarse,coarse);
        V x(coarse.system_dofs()),y(fine.system_dofs()),z(coarse.system_dofs());
        for(size_t i=0;i<x.size();++i)x[i]=C(std::sin(.123*i),std::cos(.731*i));
        identity.apply(x.data(),z.data());require(difference(z,x)<2e-12,"identity moments");
        const V known_coarse=analytic(coarse),known_fine=analytic(fine);
        transfer.apply(known_coarse.data(),y.data());const double known_error=difference(y,known_fine);
        require(known_error<2e-12,"quadratic-potential current transfer");
        transfer.apply(x.data(),y.data());
        // Full D6h generators, for both electric and magnetic currents.
        bem_tmatrix::D6hOrbit cg,fg;cg.build(coarse);fg.build(fine);
        for(int symmetry=0;symmetry<3;++symmetry){V sx(x.size()),sy(y.size()),actual(y.size());
            if(symmetry<2){cg.transform(x,sx,symmetry==1,false);fg.transform(y,sy,symmetry==1,false);}
            else {std::vector<int> image,sign;bem_tmatrix::D6hOrbit::vertical_image_map(coarse,image,sign);
                for(size_t i=0;i<x.size();++i)sx[image[i]]=double(sign[i])*x[i];
                bem_tmatrix::D6hOrbit::vertical_image_map(fine,image,sign);
                for(size_t i=0;i<y.size();++i)sy[image[i]]=double(sign[i])*y[i];}
            transfer.apply(sx.data(),actual.data());const double error=difference(actual,sy);
            std::printf("aspect %.8g symmetry %d transfer_defect %.3e\n",aspect,symmetry,error);
            require(error<5e-12,"transfer lost D6h symmetry");
        }
        if(aspect>1.1){reverse.build(fine,coarse);reverse.apply(y.data(),z.data());require(difference(z,x)<5e-12,"nested coarse-fine-coarse identity");}
        std::printf("aspect %.8g exact polynomial error %.3e storage %zu\n",aspect,known_error,transfer.bytes());
        // A failed rebuild must not leave an apparently usable, empty map.
        auto curved=fine;const int midpoint=curved.elements[0].nodes[3];
        curved.nodes[midpoint]=curved.nodes[midpoint]+Vec3(.001,.002,.003);
        bool rejected=false;try{transfer.build(coarse,curved);}catch(const std::invalid_argument&){rejected=true;}
        require(rejected,"curved target accepted");
        rejected=false;try{transfer.apply(x.data(),y.data());}catch(const std::invalid_argument&){rejected=true;}
        require(rejected,"failed transfer remained usable");
    }
    std::puts("BDM current transfer passed");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
