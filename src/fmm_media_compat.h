#pragma once
#include "fmm.h"
// Both media are built from the same Muller quadrature. Validate all leaf
// permutations and adjacency before suppressing either ordinary near action.
inline bool fmm_two_media_near_compatible(const HelmholtzFMM& a,const HelmholtzFMM& b) {
    if(!a.initialized||!b.initialized||!a.p2p_enabled||!b.p2p_enabled||a.Ns!=b.Ns||a.Nt!=b.Nt||a.n_nodes!=b.n_nodes||
       !a.d_leaf_near_source_offsets_cached||!a.d_leaf_near_source_ids_cached||
       !b.d_leaf_near_source_offsets_cached||!b.d_leaf_near_source_ids_cached||
       a.h_leaf_indices!=b.h_leaf_indices||a.h_tgt_id_offsets!=b.h_tgt_id_offsets||a.h_src_id_offsets!=b.h_src_id_offsets||
       a.h_tgt_ids_flat!=b.h_tgt_ids_flat||a.h_src_ids_flat!=b.h_src_ids_flat||a.tree.near_list!=b.tree.near_list||
       a.tree.nodes.size()!=b.tree.nodes.size())return false;
    for(size_t j=0;j<a.tree.nodes.size();++j){const auto& x=a.tree.nodes[j];const auto& y=b.tree.nodes[j];
        if(x.near_start!=y.near_start||x.near_count!=y.near_count||x.half_size!=y.half_size)return false;
        for(int d=0;d<3;++d)if(x.center[d]!=y.center[d])return false;}
    return true;
}
