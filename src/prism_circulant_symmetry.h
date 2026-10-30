#pragma once
#include "block_circulant_symmetry.h"
#include "prism_circulant_layout.h"

// Structural proofs on ALL mesh DOFs, including the zero-extension mask.
// Failure is an optimization miss, never permission to change the operator.
struct PrismCirculantSymmetry {
    const PrismCirculantLayout& layout;
    std::vector<int> patch_of,slot_of;
    explicit PrismCirculantSymmetry(const PrismCirculantLayout& l):layout(l),patch_of(l.system_dofs,-1),slot_of(l.system_dofs,-1) {
        for(size_t p=0;p<l.patches.size();++p)for(size_t s=0;s<l.patches[p].dof.size();++s) {
            const int d=l.patches[p].dof[s];if(d<0)continue;
            if(patch_of[d]>=0)throw std::logic_error("overlapping FFT patches");
            patch_of[d]=p;slot_of[d]=s;
        }
    }
    template<class Sign> bool unchanged(const std::vector<int>& image,const std::vector<Sign>& sign,int determinant,int first,int last,bool paired=false)const {
        std::vector<int> targets(layout.patches.size(),-1),signs(layout.patches.size());
        for(int p=first;p<last;++p) {
            int target_patch=-1,common=0;const auto& a=layout.patches[p];
            for(size_t j=0;j<a.dof.size();++j) {
                const int d=a.dof[j];if(d<0)continue;
                const int t=image[d],q=patch_of[t];
                if(q<first || q>=last || slot_of[t]!=int(j))return false;
                if(target_patch<0)target_patch=q;else if(target_patch!=q)return false;
                const auto& b=layout.patches[q];
                const int sol=sign[d]*a.solution_sign[j]*b.solution_sign[j];
                const int rhs=determinant*sign[d]*a.rhs_sign[j]*b.rhs_sign[j];
                if(sol!=rhs || (common && common!=sol))return false;
                common=sol;
            }
            targets[p]=target_patch;signs[p]=common;
        }
        if(paired)for(int p=first;p<last;++p) {
            const int q=prism_opposite_patch(p);
            if(q<first || q>=last || targets[p]<0 || targets[q]!=prism_opposite_patch(targets[p]) || signs[p]!=signs[q])return false;
        }
        return true;
    }
    template<class Sign> bool affine(int p,const std::vector<int>& image,const std::vector<Sign>& sign,int determinant,CirculantSymmetry& map)const {
        const auto& a=layout.patches[p];if(a.nx!=a.ny)return false;
        const int n=a.nx,grid=n*n,c=a.dof.size()/grid;
        map=CirculantSymmetry::identity(n,c);
        bool found=false;int target_patch=-1;
        for(int ch=0;ch<c && !found;++ch)for(int i=0;i<n-1 && !found;++i)for(int j=0;j<n-1 && !found;++j) {
            const int d=a.dof[a.slot(ch,i,j)],dx=a.dof[a.slot(ch,i+1,j)],dy=a.dof[a.slot(ch,i,j+1)];
            if(d<0 || dx<0 || dy<0)continue;
            const int t=slot_of[image[d]],tx=slot_of[image[dx]],ty=slot_of[image[dy]];
            if(t<0 || tx<0 || ty<0)return false;
            map.matrix={{map.mod(tx%grid/n-t%grid/n),map.mod(ty%grid/n-t%grid/n),map.mod(tx%n-t%n),map.mod(ty%n-t%n)}};
            found=true;
        }
        if(!found)return false;
        for(int ch=0;ch<c;++ch) {
            found=false;
            for(int i=0;i<n;++i)for(int j=0;j<n;++j) {
                const int s=a.slot(ch,i,j),d=a.dof[s];if(d<0)continue;
                const int t=image[d],q=patch_of[t],slot=slot_of[t];
                if(q<24 || q>=26 || slot<0)return false;
                if(target_patch<0)target_patch=q;else if(q!=target_patch)return false;
                const auto& b=layout.patches[q];if(b.nx!=n || b.ny!=n)return false;
                const int sol=sign[d]*a.solution_sign[s]*b.solution_sign[slot];
                const int rhs=determinant*sign[d]*a.rhs_sign[s]*b.rhs_sign[slot];
                if(!found) {
                    map.channel[ch]=slot/grid;
                    map.offset[ch]={{map.mod(slot%grid/n-map.matrix[0]*i-map.matrix[1]*j),map.mod(slot%n-map.matrix[2]*i-map.matrix[3]*j)}};
                    map.solution_sign[ch]=sol;map.rhs_sign[ch]=rhs;found=true;
                }
                const auto xy=map.point(ch,i,j);
                if(map.channel[ch]!=slot/grid || xy[0]!=slot%grid/n || xy[1]!=slot%n || sol!=map.solution_sign[ch] || rhs!=map.rhs_sign[ch])return false;
            }
            if(!found)return false;
        }
        map.validate();const auto& b=layout.patches[target_patch];
        for(int ch=0;ch<c;++ch)for(int i=0;i<n;++i)for(int j=0;j<n;++j) {
            const auto xy=map.point(ch,i,j);
            if((a.dof[a.slot(ch,i,j)]<0)!=(b.dof[b.slot(map.channel[ch],xy[0],xy[1])]<0))return false;
        }
        return true;
    }
};
