#ifndef BEM_MULLER_MBJ_H
#define BEM_MULLER_MBJ_H

#include "muller_dense.h"
#include <complex>
#include <string>
#include <vector>

struct MullerFmmOperator;

struct MullerMbjBlock {
    std::vector<int> dofs;
    std::vector<std::complex<double>> lu;
    std::vector<int> pivots;
    int core_dof_begin = 0;
    int core_dof_end = 0;
};

struct MullerMbjPreconditioner {
    int system_dofs = 0;
    int scalar_nodes_per_block = 50;
    int overlap_nodes = 0;
    bool stores_inverse = false;
    double ordering_seconds = 0.0;
    double assembly_seconds = 0.0;
    double factorization_seconds = 0.0;
    double setup_seconds = 0.0;
    double coarse_setup_seconds = 0.0;
    double cache_io_seconds = 0.0;
    int setup_threads = 1;
    int coarse_rank = 0;
    bool cache_hit = false;
    std::string cache_path;
    std::vector<MullerMbjBlock> blocks;
    std::vector<std::complex<double>> coarse_action;
    std::vector<std::complex<double>> coarse_update;
    std::vector<std::complex<double>> coarse_gram_lu;
    std::vector<int> coarse_gram_pivots;

    void build(
        const MullerDenseSystem& system,
        int requested_scalar_nodes_per_block = 50,
        int requested_overlap_nodes = 0);
    void build(
        const MullerFmmOperator& op,
        int requested_scalar_nodes_per_block = 50,
        int requested_overlap_nodes = 0);
    void build_cached(
        const MullerFmmOperator& op,
        int requested_scalar_nodes_per_block,
        int requested_overlap_nodes,
        const std::string& path);
    void build_coarse(
        MullerFmmOperator& op,
        int requested_rank);
    void load_neural(
        const MullerFmmOperator& op,
        const std::string& path);
    void apply(
        const std::complex<double>* rhs,
        std::complex<double>* solution) const;
    double storage_megabytes() const;
};

#endif
