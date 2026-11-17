#pragma once
#include <array>
#include <complex>
#include <cmath>
#include <stdexcept>

namespace bem_pfft {
using Complex=std::complex<double>;

// Exact regrouping of tensor-product interpolation weights by displacement.
// Cost: 3*(p+1)^2 + (2p+1)^3, instead of (p+1)^6 per near pair.
inline std::array<Complex,10> tensor_correlation(
    int nodes,const double* target,const double* source,const int* displacement,
    int range,const Complex* const* kernel,int components)
{
    if(nodes<1||nodes>8||range<0||components<1||components>10)
        throw std::invalid_argument("pFFT tensor correlation dimensions");
    double weights[3][15]={};const int width=2*nodes-1,offset=nodes-1;
    for(int axis=0;axis<3;++axis)
        for(int a=0;a<nodes;++a)for(int b=0;b<nodes;++b)
            weights[axis][a-b+offset]+=target[axis*nodes+a]*source[axis*nodes+b];
    std::array<Complex,10> out{};const size_t span=2*range+1;
    for(int a=0;a<width;++a){const int x=displacement[0]+a-offset;
        if(std::abs(x)>range)continue;
        for(int b=0;b<width;++b){const int y=displacement[1]+b-offset;
            if(std::abs(y)>range)continue;
            const double xy=weights[0][a]*weights[1][b];
            for(int c=0;c<width;++c){const int z=displacement[2]+c-offset;
                if(std::abs(z)>range)continue;
                const double w=xy*weights[2][c];
                if(w==0)continue;
                const size_t index=(size_t(x+range)*span+size_t(y+range))*span+size_t(z+range);
                for(int q=0;q<components;++q)out[q]+=w*kernel[q][index];
            }
        }
    }
    return out;
}
}
