#include "prism_circulant_layout.h"
#include "prism_circulant_symmetry.h"
#include "prism_symmetry_sectors.h"
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
        bem_tmatrix::D6hOrbit orbit;orbit.build(mesh);
        bem_tmatrix::PrismGroupMaps group;group.build(orbit);
        std::vector<int> vi,vs;bem_tmatrix::D6hOrbit::vertical_image_map(mesh,vi,vs);
        PrismCirculantSymmetry proof(layout);
        if(!proof.unchanged(group.image[1],group.sign[1],1,0,24,true) ||
           !proof.unchanged(group.image[6],group.sign[6],-1,0,24,true) || !proof.unchanged(group.image[6],group.sign[6],-1,24,26,true) ||
           !proof.unchanged(vi,vs,-1,0,24,true))throw std::runtime_error("side/H FFT symmetry proof failed");
        for(int p=0;p<26;++p) {
            const int q=prism_opposite_patch(p);const auto& a=layout.patches[p];const auto& b=layout.patches[q];
            if(prism_opposite_patch(q)!=p || p==q || (a.u-b.u).norm()>1e-12 || (a.v-b.v).norm()>1e-12 ||
               (a.normal+b.normal).norm()>1e-12)throw std::runtime_error("opposite lattice alignment");
            for(size_t j=0;j<a.dof.size();++j)if((a.dof[j]<0)!=(b.dof[j]<0))throw std::runtime_error("opposite zero mask");
            const int g=(2*(p/4)+5)%6;int common=0;
            for(size_t j=0;j<a.dof.size();++j)if(a.dof[j]>=0) {
                const int d=a.dof[j],v=vi[d],t=p<24?group.image[g][v]:group.image[6][d];
                const int sign=p<24?int(group.sign[g][v])*vs[d]:group.sign[6][d];
                if(t!=b.dof[j])throw std::runtime_error("opposite reflection permutation");
                const int sol=sign*a.solution_sign[j]*b.solution_sign[j],rhs=-sign*a.rhs_sign[j]*b.rhs_sign[j];
                if(sol!=rhs || (common && common!=sol))throw std::runtime_error("opposite reflection signs");
                common=sol;
            }
        }
        CirculantSymmetry cap_rotation,cap_vertical;
        for(int p=24;p<26;++p) {
            CirculantSymmetry r,v;
            if(!proof.affine(p,group.image[1],group.sign[1],1,r) || !proof.affine(p,vi,vs,-1,v))
                throw std::runtime_error("cap FFT affine symmetry proof failed");
            if(p==24){cap_rotation=r;cap_vertical=v;}
            else if(r.solution_sign!=cap_rotation.solution_sign || r.rhs_sign!=cap_rotation.rhs_sign ||
                    v.solution_sign!=cap_vertical.solution_sign || v.rhs_sign!=cap_vertical.rhs_sign)
                throw std::runtime_error("cap parity sectors mixed by rotation/vertical mirror");
            auto r6=CirculantSymmetry::identity(r.n,r.channel.size());
            for(int k=0;k<6;++k)r6=compose(r,r6);
            auto v2=compose(v,v),vrv=compose(v,compose(r,v)),r5=CirculantSymmetry::identity(r.n,r.channel.size());
            for(int k=0;k<5;++k)r5=compose(r,r5);
            for(int c=0;c<12;++c)for(int i=0;i<r.n;++i)for(int j=0;j<r.n;++j) {
                for(const auto* id:{&r6,&v2})if(id->channel[c]!=c || id->point(c,i,j)!=std::array<int,2>{{i,j}} || id->solution_sign[c]!=1 || id->rhs_sign[c]!=1)
                    throw std::runtime_error("cap group identity failed");
                if(vrv.channel[c]!=r5.channel[c] || vrv.point(c,i,j)!=r5.point(c,i,j) || vrv.solution_sign[c]!=r5.solution_sign[c] || vrv.rhs_sign[c]!=r5.rhs_sign[c])
                    throw std::runtime_error("cap group conjugation failed");
            }
        }
        std::printf("aspect=%.9g ref=%d covered=%d/%d basis_audit=%.3e\n",aspect,ref,layout.covered_dofs,
                    layout.system_dofs,layout.basis_mapping_error);
    }
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 9;}
