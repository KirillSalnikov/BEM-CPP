#include "host_hybrid_scheduler.h"
#include <thread>
#include <iostream>
using bem_streaming::HybridScheduler;
static void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
int main()try {
    const auto owner=std::this_thread::get_id();
    std::atomic<int> ready(0),paired(0),singles(0);
    HybridScheduler scheduler(20000);
    const auto work=[&](int lane){
        for(int step=0;step<12;++step){
            ++ready;while(ready.load()<2*(step+1))std::this_thread::yield();
            const double input=3*lane+step+.5;double output=-100;
            scheduler.gpu_pairable(lane,1,&input,&output,[&]{++singles;output=2*input;},
                [&](const void* a,void* b,const void* c,void* d){
                    require(std::this_thread::get_id()==owner,"batched action changed GPU owner");
                    ++paired;*static_cast<double*>(b)=2**static_cast<const double*>(a);
                    *static_cast<double*>(d)=2**static_cast<const double*>(c);
                });
            require(output==2*input,"batched inputs/outputs crossed lanes");
        }
    };
    scheduler.run([&]{work(0);},[&]{work(1);},false);
    require(paired==12 && singles==0,"ready tasks were not paired");
    require(scheduler.statistics(0).gpu_paired_actions==12 && scheduler.statistics(1).paired_kinds[1]==12,"paired statistics");
    HybridScheduler different(50);int wrong=0;std::atomic<int> count(0);
    const auto uneven=[&](int lane){for(int i=0;i<(lane?2:7);++i){
        int x=i+1,y=0;
        different.gpu_pairable(lane,1+lane,&x,&y,[&]{y=x;++count;},
            [&](const void*,void*,const void*,void*){++wrong;});
        require(x==y,"incompatible task did not use single action");
    }};
    different.run([&]{uneven(0);},[&]{uneven(1);},false);
    require(wrong==0 && count==9,"incompatible/finished lanes incorrectly paired");
    HybridScheduler failure(20000);std::atomic<int> arrived(0),caught(0);bool propagated=false;
    const auto fail=[&](int lane){
        int x=1,y=0;++arrived;while(arrived<2)std::this_thread::yield();
        try{failure.gpu_pairable(lane,1,&x,&y,[]{},
            [](const void*,void*,const void*,void*){throw std::runtime_error("injected paired failure");});}
        catch(...){++caught;throw;}
    };
    try{failure.run([&]{fail(0);},[&]{fail(1);},false);}
    catch(const std::runtime_error& e){propagated=std::string(e.what())=="injected paired failure";}
    require(propagated && caught==2,"paired exception did not drain both lanes");
    HybridScheduler gates(1000);int checks=0;std::atomic<int> operations(0);
    gates.run([&]{gates.strict(0,[&]{++checks;},[&]{++checks;});},[&]{
        int x=1,y=0;gates.gpu_pairable(1,1,&x,&y,[&]{y=x;++operations;},
            [](const void*,void*,const void*,void*){throw std::runtime_error("strict check was paired");});
        require(y==1,"unpaired action failed");
    });
    require(checks==1 && operations==1,"strict gate lost");
    std::cout<<"PASS: independent buffers, equal-kind pairing, single fallback, unequal lane lengths, strict gates and paired exception draining\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
