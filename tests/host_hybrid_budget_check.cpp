#include "host_hybrid_scheduler.h"
#include <thread>
#include <iostream>

int main()try {
    using bem_streaming::HybridScheduler;
    const auto owner=std::this_thread::get_id();
    for(int budget:{0,1,2}) {
        HybridScheduler scheduler(0,budget);
        std::atomic<bool> finished(false);
        std::atomic<int> host_calls(0),device_calls(0),host_live(0);
        double outputs[4]={0,0,0,0};
        scheduler.run([&]{
            for(int k=0;k<4;++k) {
                const double input=k+1.;
                scheduler.strict(0,[&,k,input]{
                    if(std::this_thread::get_id()!=owner)throw std::runtime_error("GPU ownership");
                    ++device_calls;outputs[k]=3*input;
                },[&,k,input]{
                    if(std::this_thread::get_id()==owner || host_live.fetch_add(1))
                        throw std::runtime_error("CPU isolation");
                    ++host_calls;std::this_thread::sleep_for(std::chrono::milliseconds(5));
                    outputs[k]=3*input;--host_live;
                });
            }
            finished.store(true);
        },[&]{while(!finished.load())scheduler.gpu(1,[]{
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        });});
        for(int k=0;k<4;++k)if(outputs[k]!=3.*(k+1))throw std::runtime_error("strict action changed");
        if(host_calls<1 || (budget && host_calls>budget) || host_calls+device_calls!=4 ||
           scheduler.statistics(0).cpu_checks!=host_calls || scheduler.statistics(0).gpu_checks!=device_calls)
            throw std::runtime_error("budget dropped a check or wrong routing");
        if(budget==1 && (host_calls!=1 || device_calls!=3))throw std::runtime_error("one-check policy");
    }
    HybridScheduler tail(0,1);std::atomic<bool> host_started(false),other_finished(false);
    tail.run([&]{tail.strict(0,[]{throw std::runtime_error("tail should run on CPU");},[&]{
        host_started.store(true);
        while(!other_finished.load())std::this_thread::yield();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    });},[&]{
        while(!host_started.load())tail.gpu(1,[]{});
        other_finished.store(true);
    });
    const auto times=tail.idle_statistics();
    if(!(times.cpu_tail_seconds>0 && times.cpu_tail_seconds<=times.cpu_check_seconds &&
         times.cpu_check_seconds<=times.empty_queue_seconds))throw std::runtime_error("tail wait accounting");
    for(int budget:{-1,65537}) {
        bool rejected=false;try{HybridScheduler invalid(0,budget);}catch(const std::invalid_argument&){rejected=true;}
        if(!rejected)throw std::runtime_error("invalid budget accepted");
    }
    std::cout<<"PASS CPU budgets0/1/2 preserve all strict actions, fallback ownership and tail wait accounting\n";
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
