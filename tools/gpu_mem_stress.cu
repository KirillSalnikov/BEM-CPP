#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

__device__ __forceinline__ unsigned long long mix64(unsigned long long x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

__global__ void write_pattern(unsigned long long* data, size_t n, unsigned long long seed) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = tid; i < n; i += stride) {
        data[i] = mix64(seed ^ static_cast<unsigned long long>(i));
    }
}

__global__ void verify_and_burn(const unsigned long long* data, size_t n,
                                unsigned long long seed, unsigned long long* errors,
                                double* sink, int fp_iters) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    unsigned long long local_errors = 0;
    double acc = 1.0 + 1.0e-9 * static_cast<double>(tid & 1023);
    for (size_t i = tid; i < n; i += stride) {
        unsigned long long want = mix64(seed ^ static_cast<unsigned long long>(i));
        unsigned long long got = data[i];
        if (got != want) {
            local_errors++;
        }
        double x = static_cast<double>((got ^ want) & 0xffffULL) * 1.0e-7 + acc;
        #pragma unroll 4
        for (int k = 0; k < fp_iters; ++k) {
            x = x * 1.00000000000013 + 0.99999999999991;
            x = x - 0.99999999999987;
        }
        acc += x * 1.0e-18;
    }
    if (local_errors) {
        atomicAdd(errors, local_errors);
    }
    if (tid < 1024) {
        sink[tid] = acc;
    }
}

static void usage(const char* argv0) {
    std::fprintf(stderr,
        "Usage: %s --device N [--mb MB] [--seconds SEC] [--fp-iters N]\n",
        argv0);
}

static bool parse_int_arg(int argc, char** argv, const char* name, long long* out) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0) {
            *out = std::atoll(argv[i + 1]);
            return true;
        }
    }
    return false;
}

static void check(cudaError_t rc, const char* what) {
    if (rc != cudaSuccess) {
        std::fprintf(stderr, "CUDA_ERROR %s: %s\n", what, cudaGetErrorString(rc));
        std::exit(2);
    }
}

int main(int argc, char** argv) {
    long long device = -1;
    long long mb = 8192;
    long long seconds = 60;
    long long fp_iters = 32;
    parse_int_arg(argc, argv, "--device", &device);
    parse_int_arg(argc, argv, "--mb", &mb);
    parse_int_arg(argc, argv, "--seconds", &seconds);
    parse_int_arg(argc, argv, "--fp-iters", &fp_iters);
    if (device < 0 || mb <= 0 || seconds <= 0 || fp_iters < 0) {
        usage(argv[0]);
        return 2;
    }

    check(cudaSetDevice(static_cast<int>(device)), "cudaSetDevice");
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, static_cast<int>(device)), "cudaGetDeviceProperties");

    size_t bytes = static_cast<size_t>(mb) * 1024ULL * 1024ULL;
    size_t n = bytes / sizeof(unsigned long long);
    unsigned long long* data = nullptr;
    unsigned long long* errors = nullptr;
    double* sink = nullptr;
    check(cudaMalloc(&data, n * sizeof(unsigned long long)), "cudaMalloc(data)");
    check(cudaMalloc(&errors, sizeof(unsigned long long)), "cudaMalloc(errors)");
    check(cudaMalloc(&sink, 1024 * sizeof(double)), "cudaMalloc(sink)");

    int blocks = prop.multiProcessorCount * 8;
    int threads = 256;
    std::printf("GPU_STRESS_START device=%lld name=\"%s\" mb=%lld seconds=%lld fp_iters=%lld blocks=%d threads=%d\n",
                device, prop.name, mb, seconds, fp_iters, blocks, threads);
    std::fflush(stdout);

    auto start = std::chrono::steady_clock::now();
    int iter = 0;
    unsigned long long total_errors = 0;
    while (true) {
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start).count();
        if (elapsed >= static_cast<double>(seconds)) {
            break;
        }
        unsigned long long seed = 0x1234567800000000ULL ^ static_cast<unsigned long long>(iter);
        check(cudaMemset(errors, 0, sizeof(unsigned long long)), "cudaMemset(errors)");
        write_pattern<<<blocks, threads>>>(data, n, seed);
        check(cudaGetLastError(), "write_pattern launch");
        verify_and_burn<<<blocks, threads>>>(data, n, seed, errors, sink, static_cast<int>(fp_iters));
        check(cudaGetLastError(), "verify_and_burn launch");
        check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        unsigned long long host_errors = 0;
        check(cudaMemcpy(&host_errors, errors, sizeof(host_errors), cudaMemcpyDeviceToHost), "cudaMemcpy(errors)");
        total_errors += host_errors;
        iter++;
        now = std::chrono::steady_clock::now();
        elapsed = std::chrono::duration<double>(now - start).count();
        static double last_report = -1.0e30;
        if (host_errors || elapsed - last_report >= 10.0) {
            last_report = elapsed;
            std::printf("GPU_STRESS_PROGRESS device=%lld iter=%d elapsed=%.1f errors=%llu total_errors=%llu\n",
                        device, iter, elapsed,
                        static_cast<unsigned long long>(host_errors),
                        static_cast<unsigned long long>(total_errors));
            std::fflush(stdout);
        }
    }

    check(cudaFree(sink), "cudaFree(sink)");
    check(cudaFree(errors), "cudaFree(errors)");
    check(cudaFree(data), "cudaFree(data)");
    check(cudaDeviceReset(), "cudaDeviceReset");
    std::printf("GPU_STRESS_DONE device=%lld iterations=%d total_errors=%llu status=%s\n",
                device, iter, static_cast<unsigned long long>(total_errors),
                total_errors == 0 ? "PASS" : "FAIL");
    return total_errors == 0 ? 0 : 1;
}
