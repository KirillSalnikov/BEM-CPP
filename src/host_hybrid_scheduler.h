#ifndef BEM_HOST_HYBRID_SCHEDULER_H
#define BEM_HOST_HYBRID_SCHEDULER_H
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace bem_streaming {
// Two independent solver lanes. All GPU calls, including preconditioning,
// execute on the constructing thread. Only the independent host operator may
// run concurrently. A strict call blocks its owning solver until completion;
// the other solver supplies useful GPU work. No projected residual is accepted.
class HybridScheduler {
public:
    using Work=std::function<void()>;
    using PairWork=std::function<void(const void*,void*,const void*,void*)>;
    struct Statistics {int cpu_checks=0,gpu_checks=0,gpu_actions=0,gpu_paired_actions=0;std::array<int,3> paired_kinds{{0,0,0}};double cpu_seconds=0,pair_wait_seconds=0;};
private:
    struct Task {
        int lane;Work gpu,cpu;bool strict;
        int pair_kind=0;const void* input=nullptr;void* output=nullptr;PairWork pair;
        std::promise<void> completion;
        Task(int l,Work g,Work c,bool s):lane(l),gpu(std::move(g)),cpu(std::move(c)),strict(s){}
    };
    std::mutex mutex_;
    std::condition_variable changed_;
    std::deque<std::shared_ptr<Task>> tasks_;
    std::array<std::atomic<bool>,2> done_,checking_;
    std::array<Statistics,2> statistics_;
    std::future<void> cpu_work_;
    bool ran_=false;
    int pair_wait_us_=0;
    void submit(int lane,Work gpu,Work cpu,bool strict,int pair_kind=0,
                const void* input=nullptr,void* output=nullptr,PairWork pair=PairWork()) {
        if(lane<0||lane>1)throw std::invalid_argument("hybrid lane");
        auto task=std::make_shared<Task>(lane,std::move(gpu),std::move(cpu),strict);
        task->pair_kind=pair_kind;task->input=input;task->output=output;task->pair=std::move(pair);
        auto result=task->completion.get_future();
        if(strict)checking_[lane].store(true);
        {std::lock_guard<std::mutex> lock(mutex_);tasks_.push_back(task);}
        changed_.notify_one();
        try {result.get();}catch(...) {if(strict)checking_[lane].store(false);throw;}
        if(strict)checking_[lane].store(false);
    }
    static void execute(const std::shared_ptr<Task>& task,const Work& work) {
        try {work();task->completion.set_value();}
        catch(...) {task->completion.set_exception(std::current_exception());}
    }
public:
    explicit HybridScheduler(int pair_wait_us=0):pair_wait_us_(pair_wait_us) {
        if(pair_wait_us<0 || pair_wait_us>1000000)throw std::invalid_argument("hybrid pair wait");
        for(int j=0;j<2;++j){done_[j].store(false);checking_[j].store(false);}
    }
    void gpu(int lane,Work work) {submit(lane,std::move(work),Work(),false);}
    // Equal nonzero kinds promise the same operator/layout for both inputs.
    // Each lane keeps its buffers immutable until its completion is signalled.
    void gpu_pairable(int lane,int kind,const void* input,void* output,Work single,PairWork pair) {
        if(kind<=0 || !input || !output || !single || !pair)throw std::invalid_argument("hybrid pair task");
        submit(lane,std::move(single),Work(),false,kind,input,output,std::move(pair));
    }
    void strict(int lane,Work gpu,Work cpu) {submit(lane,std::move(gpu),std::move(cpu),true);}
    const Statistics& statistics(int lane) const {return statistics_.at(lane);}
    void run(const Work& first,const Work& second,bool stagger=true,const Work& before_second=Work()) {
        if(ran_)throw std::logic_error("hybrid scheduler is single use");
        if(before_second&&!stagger)throw std::invalid_argument("recycle handoff requires staggered lanes");
        ran_=true;
        std::array<std::future<void>,2> solvers;
        const std::array<Work,2> jobs{{first,second}};
        bool started[2]={false,false};
        const auto launch=[&](int lane) {
            started[lane]=true;
            // Lane zero is stopped at its first gate, or has finished. A
            // caller can safely copy its learned recycle space here.
            try {
                if(lane==1 && before_second)before_second();
                solvers[lane]=std::async(std::launch::async,[&,lane] {
                try {jobs[lane]();}
                catch(...) {done_[lane].store(true);changed_.notify_one();throw;}
                done_[lane].store(true);changed_.notify_one();
                });
            }catch(...) {
                std::promise<void> failed;solvers[lane]=failed.get_future();
                failed.set_exception(std::current_exception());done_[lane].store(true);
            }
        };
        launch(0);if(!stagger)launch(1);
        while(!done_[0].load()||!started[1]||!done_[1].load()) {
            if(cpu_work_.valid() && cpu_work_.wait_for(std::chrono::seconds(0))==std::future_status::ready)
                cpu_work_.get();
            if(!started[1]&&done_[0].load())launch(1);
            std::shared_ptr<Task> task;
            {std::unique_lock<std::mutex> lock(mutex_);
                if(tasks_.empty())changed_.wait_for(lock,std::chrono::milliseconds(2));
                if(!tasks_.empty()){task=tasks_.front();tasks_.pop_front();}
            }
            if(!task)continue;
            // Start a fresh independent packet when the first lane needs its
            // first strict gate. Starting both at once can synchronize their
            // gates and leave no useful GPU work to hide a slow host check.
            if(task->strict && !started[1])launch(1);
            const int other=1-task->lane;
            if(task->pair && started[other] && !done_[other].load() && !checking_[other].load()) {
                const auto before=std::chrono::steady_clock::now();
                std::shared_ptr<Task> partner;
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    // At most one outstanding task exists per lane. A different
                    // operation must run before that lane can submit a partner;
                    // do not wait for it while holding the GPU owner thread.
                    if(tasks_.empty() && pair_wait_us_)
                        changed_.wait_for(lock,std::chrono::microseconds(pair_wait_us_),[&]{
                            return !tasks_.empty() || done_[other].load() || checking_[other].load();});
                    if(!tasks_.empty() && tasks_.front()->lane==other &&
                       tasks_.front()->pair_kind==task->pair_kind && tasks_.front()->pair) {
                        partner=tasks_.front();tasks_.pop_front();
                    }
                }
                statistics_[task->lane].pair_wait_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-before).count();
                if(partner) {
                    ++statistics_[task->lane].gpu_actions;++statistics_[other].gpu_actions;
                    ++statistics_[task->lane].gpu_paired_actions;++statistics_[other].gpu_paired_actions;
                    if(task->pair_kind<3){++statistics_[task->lane].paired_kinds[task->pair_kind];++statistics_[other].paired_kinds[task->pair_kind];}
                    try {
                        task->pair(task->input,task->output,partner->input,partner->output);
                        task->completion.set_value();partner->completion.set_value();
                    }catch(...) {
                        const auto failure=std::current_exception();
                        task->completion.set_exception(failure);partner->completion.set_exception(failure);
                    }
                    continue;
                }
            }
            const bool overlap=task->strict && bool(task->cpu) && !cpu_work_.valid()
                && started[other] && !done_[other].load() && !checking_[other].load();
            if(overlap) {
                try {
                    cpu_work_=std::async(std::launch::async,[this,task] {
                        const auto before=std::chrono::steady_clock::now();execute(task,task->cpu);
                        statistics_[task->lane].cpu_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-before).count();
                        changed_.notify_one();
                    });
                    ++statistics_[task->lane].cpu_checks;
                }catch(...) {
                    // Resource exhaustion while starting a host thread must
                    // not strand a solver waiting for this task's promise.
                    ++statistics_[task->lane].gpu_checks;
                    execute(task,task->gpu);
                }
            }else {
                if(task->strict)++statistics_[task->lane].gpu_checks;
                else ++statistics_[task->lane].gpu_actions;
                execute(task,task->gpu);
            }
        }
        if(cpu_work_.valid())cpu_work_.get();
        // Both lanes have drained before propagating an exception, including
        // exceptions from an operator. No caller-owned buffers can outlive us.
        std::exception_ptr failure;
        for(auto& solver:solvers)try{solver.get();}catch(...){if(!failure)failure=std::current_exception();}
        if(failure)std::rethrow_exception(failure);
    }
};
}
#endif
