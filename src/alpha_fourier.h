#ifndef BEM_ALPHA_FOURIER_H
#define BEM_ALPHA_FOURIER_H

#include <array>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

namespace bem_alpha {
using Complex = std::complex<double>;
using Field = std::array<Complex, 6>;
using Jones = std::array<Complex, 4>;

// Fourier coefficients of the final, laboratory-frame Jones amplitudes.
// The input fields use [alpha][angle][xyz] for two incident polarizations.
// mapper(alpha, angle, field) must be linear in field and have Fourier
// support [-2,2] in alpha (the two polarization/frame rotations).
// Modes are stored in ascending order, starting at -sample_count/2-2.
// Keeping the extra four modes avoids aliasing the rotations at Nyquist.
template <class Mapper>
std::vector<Jones> jones_modes(
    const std::vector<Complex>& field_x,
    const std::vector<Complex>& field_y,
    int sample_count, int angle_count, const Mapper& mapper)
{
    if (sample_count < 2 || angle_count < 1 ||
        field_x.size() != static_cast<size_t>(sample_count) * angle_count * 3 ||
        field_y.size() != field_x.size())
        throw std::invalid_argument("invalid alpha Fourier field dimensions");
    const double two_pi = 2.0 * std::acos(-1.0);
    const int first_mode = -sample_count / 2;
    std::vector<Complex> weights(static_cast<size_t>(sample_count) * sample_count);
    for (int mode = 0; mode < sample_count; ++mode)
        for (int sample = 0; sample < sample_count; ++sample)
            weights[static_cast<size_t>(mode) * sample_count + sample] =
                std::polar(1.0 / sample_count,
                    -two_pi * (first_mode + mode) * sample / sample_count);
    std::vector<Jones> result(
        static_cast<size_t>(sample_count + 4) * angle_count, Jones{});
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int angle = 0; angle < angle_count; ++angle) {
        // Each of the 24 entries in the Jones projection matrix is a
        // degree-two trigonometric polynomial: five samples recover it exactly.
        std::array<std::array<Jones, 6>, 5> map_modes{};
        for (int sample = 0; sample < 5; ++sample) {
            const double alpha = two_pi * sample / 5;
            for (int channel = 0; channel < 6; ++channel) {
                Field unit{};
                unit[channel] = 1.0;
                const Jones mapped = mapper(alpha, angle, unit);
                for (int shift = -2; shift <= 2; ++shift) {
                    const Complex weight = std::polar(0.2, -shift * alpha);
                    for (int component = 0; component < 4; ++component)
                        map_modes[shift + 2][channel][component] +=
                            weight * mapped[component];
                }
            }
        }
        for (int mode = 0; mode < sample_count; ++mode) {
            Field coefficient{};
            for (int sample = 0; sample < sample_count; ++sample) {
                const Complex weight =
                    weights[static_cast<size_t>(mode) * sample_count + sample];
                const size_t offset =
                    (static_cast<size_t>(sample) * angle_count + angle) * 3;
                for (int axis = 0; axis < 3; ++axis) {
                    coefficient[axis] += weight * field_x[offset + axis];
                    coefficient[axis + 3] += weight * field_y[offset + axis];
                }
            }
            for (int shift = -2; shift <= 2; ++shift) {
                Jones& destination = result[
                    static_cast<size_t>(mode + shift + 2) * angle_count + angle];
                for (int channel = 0; channel < 6; ++channel)
                    for (int component = 0; component < 4; ++component)
                        destination[component] += coefficient[channel] *
                            map_modes[shift + 2][channel][component];
            }
        }
    }
    return result;
}
} // namespace bem_alpha
#endif
