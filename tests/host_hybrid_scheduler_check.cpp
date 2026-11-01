#include "host_hybrid_scheduler.h"
#include "streaming_host_gmres.h"
#include <thread>
#include <iostream>
int main()try {
    using namespace bem_streaming;
    const auto owner=std::this_thread::get_id();
    const int n=48;std::atomic<int> overlap(0),host_active(0),gpu_active(0);
    const Action exact=[&](const Complex* x,Complex* y) {
        for(int i=0;i<n;++i)y[i]=Complex(.4+.04*i,.01*i)*x[i]+(i+1<n?Complex(.05,.02)*x[i+1]:Complex(0));
    };
    std::vector<Complex> known[2],rhs[2],x[2];HostResult results[2];
    for(int lane=0;lane<2;++lane) {
        known[lane].resize(n);rhs[lane].resize(n);x[lane].resize(n);
        for(int i=0;i<n;++i)known[lane][i]=Complex(std::sin(.13*i+.4*lane),std::cos(.19*i-.3*lane));
        exact(known[lane].data(),rhs[lane].data());
    }
    HybridScheduler scheduler;
    const auto solve=[&](int lane) {
        const Action gpu=[&](const Complex* a,Complex* b) {scheduler.gpu(lane,[&] {
            if(std::this_thread::get_id()!=owner || gpu_active.fetch_add(1)!=0)throw std::runtime_error("GPU owner/serialization");
            if(host_active.load())++overlap;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));exact(a,b);--gpu_active;
        });};
        const Action strict=[&](const Complex* a,Complex* b) {
            const std::vector<Complex> snapshot(a,a+n);
            scheduler.strict(lane,[&]{exact(snapshot.data(),b);},[&] {
                if(std::this_thread::get_id()==owner || host_active.fetch_add(1)!=0)throw std::runtime_error("CPU checker isolation");
                std::this_thread::sleep_for(std::chrono::milliseconds(8));exact(snapshot.data(),b);--host_active;
            });
        };
        const Action identity=[&](const Complex* a,Complex* b){std::copy(a,a+n,b);};
        results[lane]=gmres(gpu,strict,identity,rhs[lane].data(),x[lane].data(),n,12,1e-10,200,false);
    };
    scheduler.run([&]{solve(0);},[&]{solve(1);});
    for(int lane=0;lane<2;++lane) {
        std::vector<Complex> residual(n);exact(x[lane].data(),residual.data());
        for(int i=0;i<n;++i)residual[i]-=rhs[lane][i];
        const double measured=norm(residual.data(),n)/norm(rhs[lane].data(),n);
        if(measured>1e-10 || std::abs(measured-results[lane].residual)>1e-14)throw std::runtime_error("hybrid original RHS gate");
    }
    if(!overlap.load() || scheduler.statistics(0).cpu_checks+ scheduler.statistics(1).cpu_checks<1)
        throw std::runtime_error("no measured useful overlap");
    // A background failure must reach the caller after both lanes drain.
    HybridScheduler failing;bool propagated=false;
    try {failing.run([&]{failing.strict(0,[]{},[]{throw std::runtime_error("injected CPU failure");});},
                    [&]{for(int i=0;i<4;++i)failing.gpu(1,[]{std::this_thread::sleep_for(std::chrono::milliseconds(1));});});}
    catch(const std::runtime_error& e){propagated=std::string(e.what())=="injected CPU failure";}
    if(!propagated)throw std::runtime_error("background failure lost");
    HybridScheduler handoff;bool handoff_failed=false,first_finished=false;
    try {handoff.run([&]{handoff.strict(0,[]{},[]{});first_finished=true;},[]{},true,
                     []{throw std::runtime_error("injected handoff failure");});}
    catch(const std::runtime_error& e){handoff_failed=std::string(e.what())=="injected handoff failure";}
    if(!handoff_failed||!first_finished)throw std::runtime_error("handoff failure not drained");
    std::cout<<"PASS: original residuals, isolated CPU snapshot, GPU owner, useful overlap="<<overlap.load()<<", exception draining\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
