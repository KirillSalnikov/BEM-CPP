#ifndef BEM_SECTOR_RECYCLE_REBUDGET_H
#define BEM_SECTOR_RECYCLE_REBUDGET_H
#include "sector_recycle_budget.h"
#include "streaming_sector_gmres.h"

namespace bem_streaming {
// Join all users first. Construct/prune a new space transactionally, keeping
// U and its operator image C together. Peak temporary storage includes both
// copies; the entry budget bounds the final space, not this one-time peak.
inline int rebudget_sector_recycle(SectorRecycleSpace& space,
        const std::array<int,13>& offsets,const SectorBudgetPlan& plan) {
    if(!space.matches(offsets))throw std::invalid_argument("sector rebudget offsets");
    size_t entries=0;
    for(int s=0;s<12;++s){
        const int dim=offsets[s+1]-offsets[s],cap=plan.capacities[s];
        if(cap<0 || cap>64 || cap>dim || cap!=plan.capacities[mirror_sector(s)])
            throw std::invalid_argument("sector rebudget capacities");
        entries+=size_t(dim)*cap;
    }
    if(entries!=plan.entries || entries>plan.entry_budget)throw std::invalid_argument("sector rebudget memory");
    SectorRecycleSpace next(offsets,0);int retained=0;
    for(int s=0;s<12;++s){
        next.sector(s)=RecycleSpace(std::max(1,offsets[s+1]-offsets[s]),plan.capacities[s]);
        next.sector(s).merge_from(space.sector(s));retained+=next.sector(s).rank();
    }
    space=std::move(next);return retained;
}
}
#endif
