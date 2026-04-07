#ifndef BLOCK_GMRES_H
#define BLOCK_GMRES_H

#include <complex>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
typedef std::complex<double> cdouble;

class BemFmmOperator;
class NearFieldPrecond;

// Persistent workspace for paired GMRES — allocated once, reused across orientations.
// Avoids cudaMalloc/cudaFree and cublasCreate/Destroy per solve call.
struct GmresPairedWorkspace {
    int n;          // system size (2*N)
    int restart;    // restart parameter

    // GPU Krylov bases: (n, restart+1) each
    cuDoubleComplex *d_V1, *d_V2;
    cuDoubleComplex *d_w1, *d_w2;
    cuDoubleComplex *d_h1, *d_h2;

    // cuBLAS handle (reused)
    cublasHandle_t handle;

    // Host workspace (reused)
    std::vector<cdouble> h_r1, h_r2, h_w1, h_w2, h_v1, h_v2, h_z1, h_z2;
    std::vector<cdouble> h_x1, h_x2;
    std::vector<cdouble> h_hvec1, h_hvec2;
    std::vector<cdouble> h_Z1, h_Z2;  // preconditioned vectors for solution update

    // Hessenberg and Givens
    std::vector<cdouble> H1, H2;
    std::vector<cdouble> cs1, sn1, s1, y1;
    std::vector<cdouble> cs2, sn2, s2, y2;

    bool allocated;

    GmresPairedWorkspace() : n(0), restart(0), d_V1(0), d_V2(0),
        d_w1(0), d_w2(0), d_h1(0), d_h2(0), allocated(false) {}

    // Allocate all memory for given system size and restart
    void init(int n_, int restart_, bool use_precond);

    // Free all memory
    void cleanup();

    ~GmresPairedWorkspace() { if (allocated) cleanup(); }
};

// Solve Z*x1=b1 and Z*x2=b2 simultaneously using paired GMRES
// Both systems share the same operator Z, using batched matvec
// Returns total number of matvec evaluations
// If ws != nullptr, uses persistent workspace (avoids GPU alloc/free per call)
int gmres_solve_paired(BemFmmOperator& op,
    const cdouble* b1, const cdouble* b2,
    cdouble* x1, cdouble* x2,
    int restart = 100, double tol = 1e-4, int maxiter = 300,
    bool verbose = true, NearFieldPrecond* precond = nullptr,
    GmresPairedWorkspace* ws = nullptr);

#endif
