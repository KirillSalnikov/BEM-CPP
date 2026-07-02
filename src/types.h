#ifndef BEM_TYPES_H
#define BEM_TYPES_H

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <complex>

// CUDA error checking
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// Complex double on host
typedef std::complex<double> cdouble;

// Constants
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
static const double INV4PI = 1.0 / (4.0 * M_PI);

// Simple 3D vector (host)
struct Vec3 {
    double x, y, z;
    Vec3() : x(0), y(0), z(0) {}
    Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
    Vec3 operator+(const Vec3& b) const { return {x+b.x, y+b.y, z+b.z}; }
    Vec3 operator-(const Vec3& b) const { return {x-b.x, y-b.y, z-b.z}; }
    Vec3 operator*(double s) const { return {x*s, y*s, z*s}; }
    double dot(const Vec3& b) const { return x*b.x + y*b.y + z*b.z; }
    Vec3 cross(const Vec3& b) const {
        return {y*b.z - z*b.y, z*b.x - x*b.z, x*b.y - y*b.x};
    }
    double norm() const { return sqrt(x*x + y*y + z*z); }
    double norm2() const { return x*x + y*y + z*z; }
    Vec3 normalized() const { double n = norm(); return {x/n, y/n, z/n}; }
};

inline Vec3 operator*(double s, const Vec3& v) { return v*s; }

// Timing helper
#include <chrono>
struct Timer {
    std::chrono::high_resolution_clock::time_point t0;
    Timer() { reset(); }
    void reset() { t0 = std::chrono::high_resolution_clock::now(); }
    double elapsed_ms() const {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    double elapsed_s() const { return elapsed_ms() / 1000.0; }
};

#endif // BEM_TYPES_H
