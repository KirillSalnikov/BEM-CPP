#pragma once
#include <complex>
#include <functional>
#include <memory>
#include <vector>

// Frozen global coarse correction P = B + (V-BAV)(V*AV)^-1 V*.
// V is orthonormal. The full A is sampled only during setup. There are no
// extra full-operator actions per application. Arrays live on the GPU.
class GlobalCoarseInverse {
public:
    using Complex=std::complex<double>;
    using Action=std::function<void(const Complex*,Complex*)>;
    using Generator=std::function<void(int,Complex*)>;
    GlobalCoarseInverse();
    ~GlobalCoarseInverse();
    GlobalCoarseInverse(const GlobalCoarseInverse&)=delete;
    GlobalCoarseInverse& operator=(const GlobalCoarseInverse&)=delete;
    void build(int n,int candidates,const Generator& generate,const Action& action,const Action& local);
    void set_rank(int rank);
    void apply(const Complex* rhs,Complex* output,double weight=1.);
    int rank()const;
    int available_rank()const;
    size_t storage_bytes()const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

// The same multiplicative coarse correction with disjoint, real, orthonormal
// aggregate columns V. V is implicit: projection costs O(n), and only
// D = V - B A V is stored densely on the GPU. Sampling may use symmetry.
class DisjointCoarseInverse {
public:
    using Complex=std::complex<double>;
    using Action=std::function<void(const Complex*,Complex*)>;
    using Sample=std::function<void(int,const Complex*,Complex*)>;
    DisjointCoarseInverse();
    ~DisjointCoarseInverse();
    DisjointCoarseInverse(const DisjointCoarseInverse&)=delete;
    DisjointCoarseInverse& operator=(const DisjointCoarseInverse&)=delete;
    void build(int columns,const std::vector<int>& column_of,const std::vector<double>& weight,
               const Sample& sample,const Action& local);
    void apply(const Complex* rhs,Complex* output,double weight=1.);
    int rank()const;
    size_t storage_bytes()const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
