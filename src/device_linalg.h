#ifndef BEM_DEVICE_LINALG_H
#define BEM_DEVICE_LINALG_H

#include "types.h"

void device_complex_zero(double2* x, int n);
void device_complex_copy(double2* dst, const double2* src, int n);
void device_complex_sub(double2* out, const double2* a, const double2* b, int n);
void device_complex_axpy(double2* y, double2 alpha, const double2* x, int n);
void device_complex_scale(double2* y, const double2* x, double alpha, int n);
double device_complex_norm(const double2* x, int n);
void device_complex_norm_pair(const double2* x1, const double2* x2, int n,
                              double* norm1, double* norm2);
double2 device_complex_dot(const double2* a, const double2* b, int n);
void device_complex_dot_pair(const double2* a1, const double2* b1,
                             const double2* a2, const double2* b2,
                             int n, double2* dot1, double2* dot2);

#endif
