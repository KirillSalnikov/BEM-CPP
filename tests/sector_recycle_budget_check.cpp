#include "sector_recycle_budget.h"
#include <cstdio>
using namespace bem_streaming;
int main()try{
    std::array<int,13> offsets{};for(int s=0;s<=12;++s)offsets[s]=s*100;
    std::array<double,12> work;work.fill(40);std::array<unsigned,12> seen;seen.fill(2);
    auto p=sector_budget_plan(offsets,16,4095,work,seen);
    if(!p.trained||p.entries!=p.entry_budget)throw std::runtime_error("equal-work budget");
    for(int c:p.capacities)if(c!=16)throw std::runtime_error("equal work changed uniform ranks");
    work[1]=work[5]=work[7]=work[11]=200;p=sector_budget_plan(offsets,16,4095,work,seen);
    if(!p.trained||p.capacities[1]<=16||p.capacities[0]>=16)throw std::runtime_error("hard sectors got no additional rank");
    for(int s=0;s<12;++s)if(p.capacities[s]!=p.capacities[mirror_sector(s)])throw std::runtime_error("mirror rank mismatch");
    for(int s=0;s<=12;++s)offsets[s]=s==0?0:offsets[s-1]+47+7*(s%3);
    p=sector_budget_plan(offsets,16,4095,work,seen);size_t entries=0;
    for(int s=0;s<12;++s)entries+=size_t(offsets[s+1]-offsets[s])*p.capacities[s];
    if(entries!=p.entries||entries>p.entry_budget)throw std::runtime_error("weighted memory budget exceeded");
    seen[3]=1;p=sector_budget_plan(offsets,16,4095,work,seen);
    if(p.trained)throw std::runtime_error("allocation used unseen sector");
    for(int c:p.capacities)if(c!=16)throw std::runtime_error("insufficient training changed ranks");
    seen.fill(2);p=sector_budget_plan(offsets,0,4095,work,seen);
    if(p.entries||p.entry_budget||p.trained)throw std::runtime_error("disabled budget grew");
    auto zeros=work;zeros.fill(0);p=sector_budget_plan(offsets,16,4095,zeros,seen);
    if(p.trained || p.entries!=p.entry_budget)throw std::runtime_error("zero work changed ranks");
    p=sector_budget_plan(offsets,16,0,work,seen);
    if(p.trained || p.entries)throw std::runtime_error("empty mask grew");
    std::array<int,12> counts;counts.fill(40);
    SectorBudgetLearner learner(offsets,16,4095,3);
    learner.record_packet(counts,counts);learner.record_packet(counts,counts);
    if(learner.freeze_after_batch())throw std::runtime_error("early budget freeze");
    counts[1]=counts[5]=200;learner.record_packet(counts,counts);learner.record_packet(counts,counts);
    if(!learner.freeze_after_batch() || !learner.plan().trained || learner.packets()!=4)
        throw std::runtime_error("safe batch boundary did not freeze budget");
    const auto frozen=learner.plan().capacities;counts.fill(1000000);
    learner.record_packet(counts,counts);
    if(learner.freeze_after_batch() || learner.packets()!=4 || learner.plan().capacities!=frozen)
        throw std::runtime_error("held-out packet changed trained allocation");
    int rejected=0;
    try{sector_budget_plan(offsets,16,2,work,seen);}catch(const std::invalid_argument&){++rejected;}
    work[0]=std::numeric_limits<double>::quiet_NaN();
    try{sector_budget_plan(offsets,16,4095,work,seen);}catch(const std::invalid_argument&){++rejected;}
    if(rejected!=2)throw std::runtime_error("invalid budget accepted");
    std::puts("PASS sector memory budget, held-out training guard, mirror pairs, masks and invalid inputs");
}catch(const std::exception& e){std::fprintf(stderr,"sector budget: %s\n",e.what());return 1;}
