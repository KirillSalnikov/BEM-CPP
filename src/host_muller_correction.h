#ifndef BEM_HOST_MULLER_CORRECTION_H
#define BEM_HOST_MULLER_CORRECTION_H

#include <complex>

namespace bem_host {

// The host Muller matvec only consumes K1*J, K1*M, K2eps*M and K2mu*J.
// Keep each row's original addition order, including its already assembled
// regular-field value, while reading the correction CSR once for both currents.
// Output arrays are distinct from each other and from input.
template<class Entry>
inline void correct_current_pair(
    int dofs, const int* offsets, const Entry* entries,
    const std::complex<double>* input,
    std::complex<double>* k1_j, std::complex<double>* k1_m,
    std::complex<double>* k2e_m, std::complex<double>* k2m_j)
{
#pragma omp parallel for schedule(static)
    for (int row=0;row<dofs;++row) {
        std::complex<double> a=k1_j[row],b=k1_m[row],c=k2e_m[row],d=k2m_j[row];
        for (int index=offsets[row];index<offsets[row+1];++index) {
            const Entry& entry=entries[index];
            const std::complex<double> j=input[entry.column],m=input[dofs+entry.column];
            a+=entry.k1*j;
            b+=entry.k1*m;
            c+=entry.k2_epsilon*m;
            d+=entry.k2_mu*j;
        }
        k1_j[row]=a;k1_m[row]=b;k2e_m[row]=c;k2m_j[row]=d;
    }
}

} // namespace bem_host
#endif
