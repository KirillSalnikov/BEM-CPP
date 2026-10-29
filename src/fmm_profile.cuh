// Included by fmm.cu. Record stages without synchronizing between kernels:
// nested far-field calls append to the enclosing host action's timeline.
// Concurrent media have independent per-thread streams and timelines. Their
// elapsed times include contention and must not be summed as exclusive time.
#include <string>

class FmmCudaTimeline {
    struct Marker { cudaEvent_t event; const char* name; int depth; };
    std::vector<Marker> markers;
    FmmCudaTimeline* previous;
    bool enabled;
    double kr, ki;
    bool mixed;
    static FmmCudaTimeline*& current() {
        static thread_local FmmCudaTimeline* value = nullptr;
        return value;
    }
public:
    FmmCudaTimeline(bool on, double real_k, double imag_k, bool fp32)
        : previous(current()), enabled(on), kr(real_k), ki(imag_k), mixed(fp32) {
        if (enabled) { current() = this; mark("start"); }
    }
    static void mark(const char* name, int depth = -1) {
        FmmCudaTimeline* p = current();
        if (!p) return;
        Marker m = {nullptr, name, depth};
        CUDA_CHECK(cudaEventCreate(&m.event));
        CUDA_CHECK(cudaEventRecord(m.event));
        p->markers.push_back(m);
    }
    ~FmmCudaTimeline() {
        if (!enabled) return;
        current() = previous;
        CUDA_CHECK(cudaEventSynchronize(markers.back().event));
        // One line per medium, so parallel host threads cannot interleave it.
        std::string line = "  [FMM stage profile";
        char buffer[160];
        std::snprintf(buffer, sizeof(buffer), " k=%.6g%+.6gi mixed=%d]", kr, ki, int(mixed));
        line += buffer;
        for (size_t i = 1; i < markers.size(); ++i) {
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, markers[i-1].event, markers[i].event));
            std::snprintf(buffer, sizeof(buffer), " %s(d%d)=%.3fms",
                          markers[i].name, markers[i].depth, ms);
            line += buffer;
        }
        std::printf("%s\n", line.c_str());
        for (const Marker& m : markers) CUDA_CHECK(cudaEventDestroy(m.event));
    }
    FmmCudaTimeline(const FmmCudaTimeline&) = delete;
    FmmCudaTimeline& operator=(const FmmCudaTimeline&) = delete;
};
