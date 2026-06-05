#ifndef BEM_ORIENT_H
#define BEM_ORIENT_H

#include "types.h"
#include <vector>

// ZYZ Euler rotation matrix
struct Mat3 {
    double m[3][3];
    Vec3 row(int i) const { return Vec3(m[i][0], m[i][1], m[i][2]); }
    Vec3 col(int j) const { return Vec3(m[0][j], m[1][j], m[2][j]); }
    Vec3 operator*(const Vec3& v) const {
        return Vec3(m[0][0]*v.x + m[0][1]*v.y + m[0][2]*v.z,
                    m[1][0]*v.x + m[1][1]*v.y + m[1][2]*v.z,
                    m[2][0]*v.x + m[2][1]*v.y + m[2][2]*v.z);
    }
    Mat3 T() const {
        Mat3 r;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                r.m[i][j] = m[j][i];
        return r;
    }
};

Mat3 euler_rotation(double alpha, double beta, double gamma);

// Gauss-Legendre nodes and weights on [-1, 1]
void gauss_legendre(int n, std::vector<double>& nodes, std::vector<double>& weights);

// Orientation quadrature info
struct Orientation {
    Mat3 RT;      // R^T = inverse rotation
    double weight;
};

// Generate orientation quadrature points
std::vector<Orientation> generate_orientations(int n_alpha, int n_beta, int n_gamma);

#endif
