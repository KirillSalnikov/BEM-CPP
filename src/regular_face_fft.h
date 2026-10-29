#ifndef BEM_REGULAR_FACE_FFT_H
#define BEM_REGULAR_FACE_FFT_H

#include <complex>
#include <cstddef>
#include <memory>
#include <vector>

// Exact linear convolution of point-quadrature Helmholtz derivatives on a
// finite planar lattice with repeated offsets (e.g. triangle quadrature).
// No grid interpolation, periodic-boundary assumption, or inter-face P2P.
// Singular coincident point terms are omitted, as in the FMM point action;
// an eventual Galerkin caller must retain its singular/adjacent correction.
// The experimental PrismFaceFFT backend combines this with cross-face FMM.
// It is opt-in; end-to-end speed must be measured before production use.
class RegularFaceFFT {
public:
    using Complex = std::complex<double>;
    RegularFaceFFT();
    ~RegularFaceFFT();
    RegularFaceFFT(const RegularFaceFFT&) = delete;
    RegularFaceFFT& operator=(const RegularFaceFFT&) = delete;

    // Offsets are channel-major xyz. Source and target cells use identical
    // lattice steps; distinct offsets allow translated parallel faces.
    void init(int nx, int ny, const double step_u[3], const double step_v[3],
              const std::vector<double>& source_offsets,
              const std::vector<double>& target_offsets, Complex k);
    // charges[(channel*3+component)*nx*ny + ix*ny+iy].
    // result[(channel*6+component)*nx*ny + ix*ny+iy]:
    // (d_y J_x-d_x J_y, d_z J_x-d_x J_z, d_z J_y-d_y J_z),
    // followed by (H - trace(H) I) J, matching the Muller/FMM convention.
    // Zero charges mask unused triangle types/cells. Output includes all
    // target cells; the caller selects its active quadrature points.
    void evaluate(const Complex* charges, Complex* result);
    // Device-resident path. The caller fills device_input(), then launches
    // evaluate_device(); device_output() remains on the same CUDA stream.
    Complex* device_input();
    const Complex* device_output() const;
    void evaluate_device();
    void cleanup();
    std::size_t storage_bytes() const;
    static std::size_t estimated_array_bytes(int nx, int ny, int channels);
private:
    struct Impl;
    std::unique_ptr<Impl> p;
};

#endif
