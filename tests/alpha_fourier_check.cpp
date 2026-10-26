#include "alpha_fourier.h"
#include <algorithm>
#include <cstdio>

using bem_alpha::Complex;
using bem_alpha::Field;
using bem_alpha::Jones;

int main()
{
    const double two_pi = 2.0 * std::acos(-1.0);
    double worst_error = 0.0;
    for (int count : {7, 8, 32}) {
        const int angles = 3;
        const auto field_at = [=](double alpha, int angle) {
            Field field{};
            for (int channel = 0; channel < 6; ++channel)
                for (int mode = -count / 2; mode <= (count - 1) / 2; ++mode) {
                    const Complex coefficient(
                        std::sin(1.3 * mode + channel + 0.4 * angle),
                        std::cos(0.7 * mode - 0.3 * channel + angle));
                    field[channel] += coefficient * std::polar(1.0, mode * alpha) / static_cast<double>(count);
                }
            return field;
        };
        const auto mapper = [](double alpha, int angle, const Field& field) {
            Jones result{};
            for (int output = 0; output < 4; ++output)
                for (int input = 0; input < 6; ++input) {
                    const double phase = output + 0.3 * input + 0.2 * angle;
                    result[output] += field[input] * (
                        Complex(std::cos(alpha + phase), std::sin(2 * alpha - phase)) +
                        0.2 * std::polar(1.0, -2 * alpha + phase));
                }
            return result;
        };
        std::vector<Complex> x(static_cast<size_t>(count) * angles * 3), y(x.size());
        for (int sample = 0; sample < count; ++sample)
            for (int angle = 0; angle < angles; ++angle) {
                const Field field = field_at(two_pi * sample / count, angle);
                for (int axis = 0; axis < 3; ++axis) {
                    const size_t offset = (static_cast<size_t>(sample) * angles + angle) * 3;
                    x[offset + axis] = field[axis];
                    y[offset + axis] = field[axis + 3];
                }
            }
        const auto modes = bem_alpha::jones_modes(x, y, count, angles, mapper);
        for (int angle = 0; angle < angles; ++angle) {
            std::array<Complex, 16> analytic{}, numerical{};
            for (int mode = 0; mode < count + 4; ++mode) {
                const Jones& j = modes[static_cast<size_t>(mode) * angles + angle];
                for (int i = 0; i < 4; ++i)
                    for (int k = 0; k < 4; ++k)
                        analytic[4 * i + k] += j[i] * std::conj(j[k]);
            }
            // Independent direct integration of the known analytic field,
            // including negative Nyquist modes and degree-two frame rotations.
            for (int sample = 0; sample < 1024; ++sample) {
                const double alpha = two_pi * sample / 1024;
                const Jones j = mapper(alpha, angle, field_at(alpha, angle));
                for (int i = 0; i < 4; ++i)
                    for (int k = 0; k < 4; ++k)
                        numerical[4 * i + k] += j[i] * std::conj(j[k]) / 1024.0;
            }
            for (int i = 0; i < 16; ++i)
                worst_error = std::max(worst_error,
                    std::abs(analytic[i] - numerical[i]) / std::max(1.0, std::abs(numerical[i])));
        }
    }
    std::printf("alpha Fourier coherency maximum error: %.3e\n", worst_error);
    return worst_error < 1e-12 ? 0 : 1;
}
