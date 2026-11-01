#ifndef BEM_SECTOR_RECYCLE_BUDGET_H
#define BEM_SECTOR_RECYCLE_BUDGET_H
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace bem_streaming {
// Pure allocation policy: learn only from past sector iteration counts and
// freeze the resulting plan before a held-out sequence. Equal capacities for
// m and -m preserve the vertical-mirror pairing. Budget is vector entries, not
// number of columns, because sector dimensions need not be equal.
struct SectorBudgetPlan {
    std::array<int,12> capacities{};
    size_t entry_budget=0,entries=0;
    bool trained=false;
};
inline int mirror_sector(int s){return (s/6)*6+(6-s%6)%6;}
inline SectorBudgetPlan sector_budget_plan(const std::array<int,13>& offsets,
        int uniform_capacity,unsigned mask,const std::array<double,12>& work,
        const std::array<unsigned,12>& observations,unsigned minimum_observations=2) {
    if(offsets[0]!=0 || uniform_capacity<0 || uniform_capacity>64 || mask>4095 || !minimum_observations)
        throw std::invalid_argument("sector rank budget dimensions");
    SectorBudgetPlan result;bool ready=true,any_work=false;
    std::array<int,12> dimensions{};
    for(int s=0;s<12;++s){
        dimensions[s]=offsets[s+1]-offsets[s];
        if(dimensions[s]<0 || !std::isfinite(work[s]) || work[s]<0)throw std::invalid_argument("sector rank budget observations");
        if(bool(mask&(1u<<s))!=bool(mask&(1u<<mirror_sector(s))))throw std::invalid_argument("sector rank mask breaks mirror pairing");
        if(dimensions[s] && (mask&(1u<<s))){
            result.capacities[s]=std::min(uniform_capacity,dimensions[s]);
            result.entry_budget+=size_t(dimensions[s])*result.capacities[s];
            ready=ready && observations[s]>=minimum_observations;any_work=any_work || work[s]>0;
        }
    }
    result.entries=result.entry_budget;
    if(!ready || !uniform_capacity || !result.entry_budget || !any_work)return result;
    for(int s=0;s<12;++s)if((mask&(1u<<s)) && bool(dimensions[s])!=bool(dimensions[mirror_sector(s)]))
        throw std::invalid_argument("empty mirror sector");
    struct Group{std::vector<int> sectors;size_t cost=0;double score=0;int rank=0,ceiling=64;};
    std::vector<Group> groups;
    for(int s=0;s<12;++s){
        if(s>mirror_sector(s) || !dimensions[s] || !(mask&(1u<<s)))continue;
        Group g;g.sectors.push_back(s);if(mirror_sector(s)!=s)g.sectors.push_back(mirror_sector(s));
        for(int q:g.sectors){
            if(!dimensions[q])throw std::invalid_argument("empty mirror sector");
            g.cost+=dimensions[q];g.score+=work[q]/observations[q];g.ceiling=std::min(g.ceiling,dimensions[q]);
        }
        g.ceiling=std::min(g.ceiling,2*uniform_capacity);
        g.rank=std::min(g.ceiling,uniform_capacity/2);
        groups.push_back(g);
    }
    result.capacities.fill(0);result.entries=0;
    for(const auto& g:groups)result.entries+=g.cost*g.rank;
    if(result.entries>result.entry_budget)throw std::logic_error("minimum rank exceeds budget");
    while(true){
        int best=-1;double priority=-1;
        for(size_t i=0;i<groups.size();++i){const auto& g=groups[i];
            if(g.rank==g.ceiling || g.cost>result.entry_budget-result.entries)continue;
            const double value=g.score/(double(g.cost)*(g.rank+1));
            if(value>priority){priority=value;best=int(i);}
        }
        if(best<0)break;
        ++groups[best].rank;result.entries+=groups[best].cost;
    }
    for(const auto& g:groups)for(int s:g.sectors)result.capacities[s]=g.rank;
    result.trained=true;return result;
}

// The policy sees completed production packets only. The caller invokes
// freeze_after_batch AFTER both solver lanes have joined, before any next RHS.
// No held-out packet can change the allocation, including exact warm starts.
class SectorBudgetLearner {
    std::array<int,13> offsets_;
    int capacity_;unsigned mask_,training_packets_,packets_=0;
    std::array<double,12> work_{};
    std::array<unsigned,12> observations_{};
    SectorBudgetPlan plan_;bool frozen_=false;
public:
    SectorBudgetLearner(const std::array<int,13>& offsets,int capacity,unsigned mask,unsigned packets):
        offsets_(offsets),capacity_(capacity),mask_(mask),training_packets_(packets) {
        plan_=sector_budget_plan(offsets_,capacity_,mask_,work_,observations_);
    }
    void record_packet(const std::array<int,12>& x,const std::array<int,12>& y) {
        if(!training_packets_ || frozen_)return;
        for(int s=0;s<12;++s)if(x[s]<0 || y[s]<0)throw std::invalid_argument("negative sector iterations");
        for(int s=0;s<12;++s)for(int count:{x[s],y[s]})if(count>0){work_[s]+=count;++observations_[s];}
        ++packets_;
    }
    bool freeze_after_batch() {
        if(!training_packets_ || frozen_ || packets_<training_packets_)return false;
        plan_=sector_budget_plan(offsets_,capacity_,mask_,work_,observations_);frozen_=true;return true;
    }
    const SectorBudgetPlan& plan()const{return plan_;}
    bool frozen()const{return frozen_;}
    unsigned packets()const{return packets_;}
};
}
#endif
