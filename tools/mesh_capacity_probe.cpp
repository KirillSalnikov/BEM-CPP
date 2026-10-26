#include "mesh.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

int main(int argc,char**argv) {
    const int low=argc>1?std::atoi(argv[1]):4;
    const int high=argc>2?std::atoi(argv[2]):7;
    const double index=argc>3?std::atof(argv[3]):1.3;
    for(int ref=low;ref<=high;++ref) {
        const Mesh mesh=regular_prism(6,1.,ref,1.);
        double edge=0,radius=0;
        for(const Vec3& v:mesh.verts)radius=std::max(radius,v.norm());
        for(int t=0;t<mesh.nt();++t) {
            Vec3 a,b,c;mesh.tri_verts(t,a,b,c);
            edge=std::max(edge,std::max((a-b).norm(),std::max((a-c).norm(),(b-c).norm())));
        }
        std::printf("{\"ref\":%d,\"vertices\":%d,\"triangles\":%d,\"system_dofs\":%lld,\"maximum_edge\":%.17g,\"bounding_radius\":%.17g,\"ka_at_8_equivalent_P2_points_per_inner_wavelength\":%.17g}\n",
            ref,mesh.nv(),mesh.nt(),6LL*mesh.nt(),edge,radius,4*std::acos(-1.)/(8*index*edge));
    }
}
