#include "prism_circulant_layout.h"
#include <cstdio>
#include <set>
int main()try {
    MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
    for(double aspect:{1.,1./.7})for(int ref:{2,3,5}) {
        const auto mesh=build_muller_p2_mesh(regular_prism(6,aspect,ref,1.,0,false,true),options);
        PrismCirculantLayout layout;layout.build(mesh,ref);
        std::set<int> covered;
        for(const auto& p:layout.patches)for(size_t i=0;i<p.dof.size();++i)if(p.dof[i]>=0) {
            if(!covered.insert(p.dof[i]).second)return 1;
            if(p.rhs_sign[i]*p.solution_sign[i]!=p.parity)return 2;
        }
        if(int(covered.size())!=layout.covered_dofs || layout.interface_dofs<=0)return 3;
        std::printf("aspect=%.9g ref=%d covered=%d/%d basis_audit=%.3e\n",aspect,ref,layout.covered_dofs,
                    layout.system_dofs,layout.basis_mapping_error);
    }
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 9;}
