#include "prism_circulant_layout.h"
#include "muller_fmm.h"
#include <algorithm>
#include <cstdio>
#include <stdexcept>
using C=std::complex<double>;
std::vector<C> block(const MullerFmmOperator& op,const PrismCirculantPatch& p,int i,int j,int dx,int dy) {
    std::vector<int> groups;
    for(int t=0;t<3;++t)for(int source=0;source<2;++source) {
        const int d=p.dof[p.slot(2*t,i+(source?0:dx),j+(source?0:dy))];
        if(d<0)throw std::runtime_error("incomplete test cell");
        groups.push_back(d/2);
    }
    std::sort(groups.begin(),groups.end());groups.erase(std::unique(groups.begin(),groups.end()),groups.end());
    const auto a=assemble_muller_nodal_block(op,groups);const int n=groups.size()*4;
    std::vector<C> canonical(144);
    for(int r=0;r<12;++r)for(int c=0;c<12;++c) {
        const int rs=p.slot(r,i+dx,j+dy),cs=p.slot(c,i,j),rd=p.dof[rs],cd=p.dof[cs];
        const int rr=4*(std::lower_bound(groups.begin(),groups.end(),(rd%op.current_dofs)/2)-groups.begin())+2*(rd/op.current_dofs)+rd%2;
        const int cc=4*(std::lower_bound(groups.begin(),groups.end(),(cd%op.current_dofs)/2)-groups.begin())+2*(cd/op.current_dofs)+cd%2;
        canonical[r*12+c]=double(p.rhs_sign[rs]*p.solution_sign[cs])*a[rr*n+cc];
    }
    return canonical;
}
int main()try {
    for(double aspect:{1.,1./.7}) {
        MullerFmmOperator op;MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
        op.mesh=build_muller_p2_mesh(regular_prism(6,aspect,4,1.,0,false,true),options);
        op.current_dofs=op.mesh.current_dofs();op.system_dofs=op.mesh.system_dofs();
        op.k_exterior=4.;op.k_interior=4*1.3116;op.epsilon_interior=1.3116*1.3116;
        PrismCirculantLayout layout;layout.build(op.mesh,4);
        double maximum=0,far_maximum=0;
        for(int kind=0;kind<2;++kind)for(int dx:{0,1,3}) {
            const int first=kind?24:0,last=kind?26:24;
            const int i=kind?15:2,j=kind?15:3;
            const auto reference=block(op,layout.patches[first],i,j,dx,0);
            for(int p=first;p<last;++p) {
                const auto actual=block(op,layout.patches[p],i,j,dx,0);
                double a=0,b=0;for(int k=0;k<144;++k){a+=std::norm(actual[k]-reference[k]);b+=std::norm(reference[k]);}
                if(dx==3)far_maximum=std::max(far_maximum,std::sqrt(a/b));
                else maximum=std::max(maximum,std::sqrt(a/b));
            }
        }
        std::printf("aspect=%.9g near_block_error=%.3e separated_block_error=%.3e\n",aspect,maximum,far_maximum);
        // The local assembler uses Duffy order 4, which is not invariant
        // to all triangle-vertex reorderings. Use the established 1e-4
        // physical symmetry gate for touching supports, but demand roundoff
        // agreement for separated supports with symmetric triangle rules.
        if(maximum>1e-4 || far_maximum>1e-10)throw std::runtime_error("physical block covariance failed");
    }
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
