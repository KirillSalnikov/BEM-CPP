#ifndef BEM_HOST_THREAD_AFFINITY_H
#define BEM_HOST_THREAD_AFFINITY_H
#include <atomic>
#include <cstdlib>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <sched.h>
#include <sstream>
#include <stdexcept>
namespace bem_cpu {
inline void configure_threads(int count,const char* affinity_name) {
    if(count<1)throw std::invalid_argument("CPU thread count");
#ifdef _OPENMP
    omp_set_dynamic(0);omp_set_num_threads(count);
#endif
    const char* value=std::getenv(affinity_name);if(!value)return;
    cpu_set_t mask;CPU_ZERO(&mask);std::stringstream input(value);std::string item;
    while(std::getline(input,item,',')) {
        const auto dash=item.find('-');const int first=std::stoi(item.substr(0,dash));
        const int last=dash==std::string::npos?first:std::stoi(item.substr(dash+1));
        if(first<0||last<first||last>=CPU_SETSIZE)throw std::invalid_argument("CPU affinity range");
        for(int cpu=first;cpu<=last;++cpu)CPU_SET(cpu,&mask);
    }
    if(!CPU_COUNT(&mask))throw std::invalid_argument("empty CPU affinity");
    std::atomic<int> failed(0);
#pragma omp parallel
    {if(sched_setaffinity(0,sizeof(mask),&mask)!=0)failed.store(1);}
    if(failed.load())throw std::runtime_error("cannot set CPU affinity");
}
}
#endif
