#include "bdm_mesh_transfer.h"
#include "quadrature.h"
#include <cstdio>
#include <fstream>
#include <string>

using Complex=std::complex<double>;
using Vector=std::vector<Complex>;
using Key=std::array<int,3>;
struct Patch {
    Vec3 origin,e1,e2,normal;
    double area2_squared;
    std::array<double,2> coordinates(const Vec3& p)const {
        const Vec3 d=p-origin;
        return {{d.cross(e2).dot(normal)/area2_squared,e1.cross(d).dot(normal)/area2_squared}};
    }
    bool contains(const Vec3& p)const {
        if(std::abs((p-origin).dot(normal))>2e-11*std::sqrt(area2_squared))return false;
        const auto b=coordinates(p);
        return b[0]>=-2e-10 && b[1]>=-2e-10 && b[0]+b[1]<=1+2e-10;
    }
};
// An independent 3D box locator: no edge clipping or moment integration from
// BdmMeshTransfer is used to identify the coarse triangle at a new point.
struct Parents {
    std::vector<Patch> patches;
    std::vector<int> coarse_element;
    Parents(const MullerP2Mesh& coarse,const MullerP2Mesh& fine) {
        double step=0;
        for(const auto& e:coarse.elements){Patch p;
            p.origin=coarse.nodes[e.nodes[0]];p.e1=coarse.nodes[e.nodes[1]]-p.origin;p.e2=coarse.nodes[e.nodes[2]]-p.origin;
            p.normal=p.e1.cross(p.e2);p.area2_squared=p.normal.dot(p.normal);patches.push_back(p);
            step=std::max(step,std::max(p.e1.norm(),std::max(p.e2.norm(),(p.e1-p.e2).norm())));
        }
        const auto key=[&](const Vec3& p){return Key{{int(std::floor(p.x/step)),int(std::floor(p.y/step)),int(std::floor(p.z/step))}};};
        std::map<Key,std::vector<int>> boxes;
        for(size_t t=0;t<coarse.elements.size();++t){Vec3 lo(1e100,1e100,1e100),hi(-1e100,-1e100,-1e100);
            for(int j=0;j<3;++j){const auto& p=coarse.nodes[coarse.elements[t].nodes[j]];
                lo.x=std::min(lo.x,p.x);lo.y=std::min(lo.y,p.y);lo.z=std::min(lo.z,p.z);
                hi.x=std::max(hi.x,p.x);hi.y=std::max(hi.y,p.y);hi.z=std::max(hi.z,p.z);}
            const auto a=key(lo-Vec3(1e-11,1e-11,1e-11)),b=key(hi+Vec3(1e-11,1e-11,1e-11));
            for(int x=a[0];x<=b[0];++x)for(int y=a[1];y<=b[1];++y)for(int z=a[2];z<=b[2];++z)
                boxes[Key{{x,y,z}}].push_back(t);
        }
        coarse_element.assign(fine.elements.size(),-1);std::vector<int> children(coarse.elements.size(),0);
        for(size_t t=0;t<fine.elements.size();++t){const auto& e=fine.elements[t];
            const Vec3 center=(fine.nodes[e.nodes[0]]+fine.nodes[e.nodes[1]]+fine.nodes[e.nodes[2]])*(1./3);
            const auto box=boxes.find(key(center));
            if(box!=boxes.end())for(int c:box->second){const auto& p=patches[c];
                if(p.contains(fine.nodes[e.nodes[0]]) && p.contains(fine.nodes[e.nodes[1]]) && p.contains(fine.nodes[e.nodes[2]])) {
                    coarse_element[t]=c;++children[c];break;
                }
            }
            if(coarse_element[t]<0)throw std::runtime_error("fine triangle crosses coarse triangles: meshes are not nested");
        }
        for(int count:children)if(count!=4)throw std::runtime_error("coarse triangle does not have four fine children");
    }
};
static std::array<Complex,6> field(const MullerP2Mesh& mesh,int element,const MullerFrameSample& frame,const Vector& currents) {
    const auto basis=evaluate_muller_basis(mesh,element,frame);std::array<Complex,6> out{};
    for(int j=0;j<basis.count;++j)for(int current=0;current<2;++current){const auto c=currents[current*mesh.current_dofs()+basis.dofs[j]];
        out[3*current]+=c*basis.values[j].x;out[3*current+1]+=c*basis.values[j].y;out[3*current+2]+=c*basis.values[j].z;}
    return out;
}
static void check(const char* name,const MullerP2Mesh& coarse,const MullerP2Mesh& fine,const Parents& parent,
                  const BdmMeshTransfer& transfer,const Vector& x) {
    Vector y(fine.system_dofs());transfer.apply(x.data(),y.data());
    const auto q=tri_quadrature(13);
    double error=0,energy=0,max_error=0,max_value=0,max_position_error=0;
    // Every quadrature point of every fine triangle, both electric and
    // magnetic currents. No GMRES correction can conceal a transfer error.
#pragma omp parallel for reduction(+:error,energy) reduction(max:max_error,max_value,max_position_error) schedule(static)
    for(int t=0;t<int(fine.elements.size());++t)for(int j=0;j<q.npts;++j) {
        const auto f=evaluate_muller_frame(fine,t,q.pts[j][0],q.pts[j][1]);
        const int c=parent.coarse_element[t];const auto uv=parent.patches[c].coordinates(f.position);
        const auto old=evaluate_muller_frame(coarse,c,uv[0],uv[1]);
        max_position_error=std::max(max_position_error,(old.position-f.position).norm());
        const auto a=field(coarse,c,old,x),b=field(fine,t,f,y);
        double point_error=0,point_energy=0;
        for(int k=0;k<6;++k){point_error+=std::norm(a[k]-b[k]);point_energy+=std::norm(a[k]);}
        error+=point_error;energy+=point_energy;
        max_error=std::max(max_error,point_error);max_value=std::max(max_value,point_energy);
    }
    const double relative=std::sqrt(error/energy),peak=std::sqrt(max_error/max_value);
    std::printf("{\"case\":\"%s\",\"fine_elements\":%zu,\"points\":%zu,\"relative_l2\":%.17g,\"maximum_error_over_peak\":%.17g,\"maximum_position_error\":%.17g,\"four_children_per_parent\":true}\n",
        name,fine.elements.size(),fine.elements.size()*q.npts,relative,peak,max_position_error);
    if(!(relative<2e-11 && peak<2e-11 && max_position_error<2e-12))throw std::runtime_error("pointwise transfer failed");
}
int main(int argc,char** argv)try {
    MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
    const int ref=argc>1?5:2;
    const auto coarse=build_muller_p2_mesh(regular_prism(6,10./7,ref,1.,0,false,true),options);
    const auto fine=build_muller_p2_mesh(regular_prism(6,10./7,ref+1,1.,0,false,true),options);
    Parents parents(coarse,fine);BdmMeshTransfer transfer;transfer.build(coarse,fine);
    Vector x(coarse.system_dofs());
    for(size_t j=0;j<x.size();++j)x[j]=Complex(std::sin(.127*j),std::cos(.519*j));
    check("random_currents",coarse,fine,parents,transfer,x);
    for(int j=1;j<argc;++j){std::ifstream file(argv[j],std::ios::binary|std::ios::ate);
        if(!file || file.tellg()!=std::streamoff(x.size()*sizeof(Complex)))throw std::runtime_error("coarse current file size mismatch");
        file.seekg(0);if(!file.read(reinterpret_cast<char*>(x.data()),x.size()*sizeof(Complex)))throw std::runtime_error("coarse current read failed");
        check(argv[j],coarse,fine,parents,transfer,x);
    }
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
