#pragma once
#include "streaming_recycle_space.h"
#include <cstring>

namespace coarse_ritz_experiment {
using Vector=std::vector<bem_streaming::Complex>;
// Read our own fresh public RecycleSpace::save output. U is still in the
// right-preconditioned sector coordinates here; callers must decode it with
// rhs=true and apply the coarse inverse before physical mesh transfer.
inline std::vector<Vector> read_columns(const std::string& path,int n,const std::string& signature) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    if(!f || n<1 || signature.size()!=64)throw std::runtime_error("Ritz checkpoint open/dimensions");
    const auto bytes=f.tellg();f.seekg(0);
    char magic[16]={},stored[64]={};std::uint64_t dimensions[3]={};
    f.read(magic,16);f.read(stored,64);f.read(reinterpret_cast<char*>(dimensions),sizeof(dimensions));
    const char expected[16]="BEM_T_RECYCLE_1";const auto rank=dimensions[1];
    if(!f || std::memcmp(magic,expected,16) || std::string(stored,64)!=signature || dimensions[0]!=std::uint64_t(n) ||
        rank<1 || rank>64 || bytes!=std::streamoff(104+2*rank*n*sizeof(bem_streaming::Complex)))throw std::runtime_error("Ritz checkpoint header");
    std::vector<Vector> columns(rank,Vector(n));Vector image(n);std::uint64_t checksum=1469598103934665603ULL;
    for(int block=0;block<2;++block)for(size_t j=0;j<rank;++j){auto& v=block?image:columns[j];
        if(!f.read(reinterpret_cast<char*>(v.data()),v.size()*sizeof(v[0])))throw std::runtime_error("Ritz checkpoint data");
        const auto* data=reinterpret_cast<const unsigned char*>(v.data());
        for(size_t k=0;k<v.size()*sizeof(v[0]);++k){checksum^=data[k];checksum*=1099511628211ULL;}
        for(auto z:v)if(!std::isfinite(z.real()) || !std::isfinite(z.imag()))throw std::runtime_error("Ritz checkpoint nonfinite");
    }
    if(checksum!=dimensions[2])throw std::runtime_error("Ritz checkpoint checksum");
    return columns;
}
}
