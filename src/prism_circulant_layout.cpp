#include "prism_circulant_layout.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <stdexcept>
#include <tuple>

namespace {
std::array<double,2> coordinates(const PrismCirculantPatch& p,const Vec3& position,bool vector=false) {
    const Vec3 r=vector?position:position-p.origin;
    const double a=p.u.dot(p.u),b=p.u.dot(p.v),c=p.v.dot(p.v),d=a*c-b*b;
    if(d<=0)throw std::runtime_error("degenerate circulant lattice");
    return {{(c*p.u.dot(r)-b*p.v.dot(r))/d,(a*p.v.dot(r)-b*p.u.dot(r))/d}};
}
std::array<int,2> integer_coordinates(const PrismCirculantPatch& p,const Vec3& position) {
    const auto q=coordinates(p,position);std::array<int,2> result;
    for(int d=0;d<2;++d){result[d]=int(std::llround(q[d]));
        if(std::abs(q[d]-result[d])>2e-7)throw std::runtime_error("vertex misses circulant lattice");}
    return result;
}
}
void PrismCirculantLayout::build(const MullerP2Mesh& mesh,int ref) {
    if(mesh.basis_kind!=MullerBasisKind::HDivBdm1 || ref<2 || ref>8)
        throw std::invalid_argument("circulant prism requires BDM1 and ref in [2,8]");
    patches.clear();basis_mapping_error=0;covered_dofs=0;system_dofs=mesh.system_dofs();
    const int seg=1<<ref,edges=mesh.topology_edge_count;
    double radius=0,height=0;
    for(const auto& e:mesh.elements)for(int j=0;j<3;++j) {
        const Vec3 a=mesh.nodes[e.nodes[j]];
        radius=std::max(radius,std::hypot(a.x,a.y));height=std::max(height,2*std::abs(a.z));
    }
    const int nz=int(mesh.elements.size())/(12*seg)-seg;
    if(nz<2 || nz%2 || int(mesh.elements.size())!=12*seg*(seg+nz))
        throw std::runtime_error("circulant mesh count disagrees with regular D6h prism");
    Vec3 vertices[6];const double pi=std::acos(-1.);
    for(int f=0;f<6;++f){const double a=(2*f+1)*pi/6;vertices[f]=Vec3(radius*std::cos(a),radius*std::sin(a),0);}
    for(int f=0;f<6;++f)for(int ix=0;ix<2;++ix)for(int iz=0;iz<2;++iz) {
        const Vec3 tangent=(vertices[(f+1)%6]-vertices[f])*(1/radius);
        PrismCirculantPatch p;p.nx=seg/2;p.ny=nz/2;p.parity=(ix?-1:1)*(iz?-1:1);
        p.origin=vertices[ix?(f+1)%6:f]+Vec3(0,0,iz?.5*height:-.5*height);
        p.u=tangent*((ix?-1:1)*radius/seg);p.v=Vec3(0,0,(iz?-1:1)*height/nz);
        p.normal=tangent.cross(Vec3(0,0,1));patches.push_back(p);
    }
    for(int bottom=0;bottom<2;++bottom) {
        PrismCirculantPatch p;p.kind=1;p.nx=p.ny=2*seg;p.parity=bottom?-1:1;
        p.origin=(vertices[0]+vertices[1])*(-1)+Vec3(0,0,(bottom?-.5:.5)*height);
        p.u=vertices[0]*(1./seg);p.v=vertices[1]*(1./seg);p.normal=Vec3(0,0,p.parity);
        patches.push_back(p);
    }
    for(auto& p:patches) {
        p.dof.assign(size_t(12)*p.nx*p.ny,-1);
        p.rhs_sign.resize(p.dof.size());p.solution_sign.resize(p.dof.size());
    }
    std::vector<int> element_patch(mesh.elements.size(),-1),edge_patch(edges,-1),owner(edges,-1);
    std::vector<std::array<int,2>> endpoints(edges);
    for(int t=0;t<int(mesh.elements.size());++t) {
        const auto& e=mesh.elements[t];const auto frame=evaluate_muller_frame(mesh,t,1./3,1./3);
        int patch=-1;
        if(std::abs(frame.position.z-.5*height)<1e-8)patch=24;
        else if(std::abs(frame.position.z+.5*height)<1e-8)patch=25;
        else for(int f=0;f<6;++f) {
            const auto& first=patches[4*f];
            if(std::abs((frame.position-first.origin).dot(first.normal))<1e-8) {
                const auto q=coordinates(first,frame.position);
                patch=4*f+2*int(q[0]>seg/2.)+int(q[1]>nz/2.);break;
            }
        }
        if(patch<0 || frame.normal.dot(patches[patch].normal)<1-1e-8)
            throw std::runtime_error("circulant prism face/orientation mismatch");
        element_patch[t]=patch;
        for(int k=0;k<3;++k) {
            const int edge=e.topology_edges[k];
            int a=e.topology_vertices[k],b=e.topology_vertices[(k+1)%3];if(a>b)std::swap(a,b);
            endpoints[edge]={{a,b}};
            if(edge_patch[edge]==-1)edge_patch[edge]=patch;
            else if(edge_patch[edge]!=patch)edge_patch[edge]=-2;
        }
    }
    std::vector<int> edge_type(edges,-1),edge_i(edges),edge_j(edges),edge_sign(edges);
    for(int edge=0;edge<edges;++edge) {
        const int id=edge_patch[edge];if(id<0)continue;
        auto& p=patches[id];
        const auto a=integer_coordinates(p,mesh.nodes[endpoints[edge][0]]);
        const auto b=integer_coordinates(p,mesh.nodes[endpoints[edge][1]]);
        const int du=b[0]-a[0],dv=b[1]-a[1];
        int type=-1;
        if(std::abs(du)==1 && dv==0)type=0;
        else if(du==0 && std::abs(dv)==1)type=1;
        else if(std::abs(du)==1 && dv==(p.kind?-du:du))type=2;
        if(type<0)throw std::runtime_error("unrecognized mirrored BDM edge on circulant lattice");
        const int i=std::min(a[0],b[0]),j=std::min(a[1],b[1]);
        if(i<0||i>=p.nx||j<0||j>=p.ny)throw std::runtime_error("interior circulant edge outside patch");
        const int sign=a<b?1:-1;edge_type[edge]=type;edge_i[edge]=i;edge_j[edge]=j;edge_sign[edge]=sign;owner[edge]=id;
        for(int current=0;current<2;++current)for(int moment=0;moment<2;++moment) {
            const int slot=p.slot((current*3+type)*2+moment,i,j),orientation=moment?1:sign;
            if(p.dof[slot]>=0)throw std::runtime_error("duplicate circulant DOF slot");
            p.dof[slot]=current*mesh.current_dofs()+2*edge+moment;
            // A S_solution = S_rhs A. Reflection parity differs between
            // tested equations and unknown currents; moment 1 does not
            // flip under reversal of the global edge direction.
            p.rhs_sign[slot]=orientation*(current?p.parity:1);
            p.solution_sign[slot]=orientation*(current?1:p.parity);
            ++covered_dofs;
        }
    }
    // Audit all covered basis restrictions against translation/rotation/
    // mirror templates, not just edge coordinates. Piola basis vectors
    // acquire parity*edge_orientation for moment zero, parity for moment one.
    using Key=std::tuple<int,int,int,int>;
    std::map<Key,std::array<double,2>> canonical;
    for(int t=0;t<int(mesh.elements.size());++t) {
        const int id=element_patch[t];const auto& p=patches[id];const auto& e=mesh.elements[t];
        const auto frame=evaluate_muller_frame(mesh,t,1./3,1./3);
        const auto basis=evaluate_muller_basis(mesh,t,frame);
        int ci=0,cj=0;
        for(int k=0;k<3;++k){const auto q=integer_coordinates(p,mesh.nodes[e.nodes[k]]);ci+=q[0];cj+=q[1];}
        for(int b=0;b<basis.count;++b) {
            const int edge=basis.dofs[b]/2,moment=basis.dofs[b]%2;if(owner[edge]!=id)continue;
            auto value=coordinates(p,basis.values[b],true);
            const double sign=p.parity*(moment?1:edge_sign[edge]);for(auto& v:value)v*=sign;
            const Key key(p.kind,2*edge_type[edge]+moment,ci-3*edge_i[edge],cj-3*edge_j[edge]);
            auto found=canonical.emplace(key,value);
            if(!found.second) {
                const auto& reference=found.first->second;
                const double error=std::hypot(value[0]-reference[0],value[1]-reference[1])/
                    std::max(std::hypot(reference[0],reference[1]),1e-30);
                basis_mapping_error=std::max(basis_mapping_error,error);
            }
        }
    }
    if(basis_mapping_error>2e-7)throw std::runtime_error("BDM translation/mirror basis audit failed");
    interface_dofs=system_dofs-covered_dofs;
}
