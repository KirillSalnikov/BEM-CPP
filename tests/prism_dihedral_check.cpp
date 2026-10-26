#include "mesh.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <map>
#include <set>
#include <stdexcept>

using Point=std::array<long long,3>;
using Triangle=std::array<Point,3>;
Point key(const Vec3& v) { return {{std::llround(v.x*1e9),std::llround(v.y*1e9),std::llround(v.z*1e9)}}; }
Triangle triangle(const Mesh& mesh,int index,int transform) {
    Triangle out;
    for(int j=0;j<3;++j) {
        Vec3 v=mesh.verts[mesh.tris[3*index+j]];
        if(transform==1) v=Vec3(.5*v.x-std::sqrt(.75)*v.y,std::sqrt(.75)*v.x+.5*v.y,v.z);
        if(transform==2) v.z=-v.z;
        if(transform==3) v.y=-v.y;
        out[j]=key(v);
    }
    std::sort(out.begin(),out.end()); return out;
}
int missing(const Mesh& mesh,int transform) {
    std::set<Triangle> original;
    for(int i=0;i<mesh.nt();++i) original.insert(triangle(mesh,i,0));
    int count=0;
    for(int i=0;i<mesh.nt();++i) count+=!original.count(triangle(mesh,i,transform));
    return count;
}
void check_closed_outward(const Mesh& mesh) {
    std::map<std::pair<int,int>,std::pair<int,int>> edges;
    for(int i=0;i<mesh.nt();++i) {
        const Vec3 a=mesh.verts[mesh.tris[3*i]],b=mesh.verts[mesh.tris[3*i+1]],c=mesh.verts[mesh.tris[3*i+2]];
        const Vec3 u=b-a,v=c-a;
        const Vec3 n(u.y*v.z-u.z*v.y,u.z*v.x-u.x*v.z,u.x*v.y-u.y*v.x);
        if(n.x*a.x+n.y*a.y+n.z*a.z<=0.)
            throw std::runtime_error("degenerate or inward triangle");
        for(int j=0;j<3;++j) {
            const int x=mesh.tris[3*i+j],y=mesh.tris[3*i+(j+1)%3];
            auto& value=edges[std::minmax(x,y)];
            ++value.first; value.second+=(x<y ? 1:-1);
        }
    }
    for(const auto& edge:edges)
        if(edge.second.first!=2 || edge.second.second!=0)
            throw std::runtime_error("nonconforming or inconsistently oriented edge");
}
int main() {
    for(int ref : {0,1,2,5,6}) {
        Mesh mesh=regular_prism(6,1.,ref,1.,0,false,true);
        check_closed_outward(mesh);
        int rotation=missing(mesh,1),horizontal=missing(mesh,2),vertical=missing(mesh,3);
        if(rotation||horizontal||vertical) throw std::runtime_error("D6h mesh symmetry failure");
        std::printf("{\"ref\":%d,\"triangles\":%d,\"system_dofs\":%d,\"C6_missing\":%d,\"horizontal_missing\":%d,\"vertical_missing\":%d,\"closed_outward\":true}\n",
            ref,mesh.nt(),6*mesh.nt(),rotation,horizontal,vertical);
    }
    Mesh original=regular_prism(6,1.,2,1.);
    if(missing(original,1)||!missing(original,2)) throw std::runtime_error("control mesh did not expose the old reflection defect");
}
