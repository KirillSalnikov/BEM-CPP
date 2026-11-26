#pragma once
#include <array>
#include <complex>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>
#include <unistd.h>

// Cache only the sampled columns. FFT inversion, basis mapping and the
// independent end-to-end operator/residual checks still run on every launch.
namespace prism_samples {
using Vector = std::vector<std::complex<double>>;
inline std::uint64_t checksum(const Vector& self,const Vector& opposite) {
    std::uint64_t h=14695981039346656037ULL;
    for(const Vector* v : {&self,&opposite}) {
        const auto* bytes=reinterpret_cast<const unsigned char*>(v->data());
        for(size_t i=0;i<v->size()*sizeof(Vector::value_type);++i)h=(h^bytes[i])*1099511628211ULL;
    }
    return h;
}
inline bool read(const std::string& file,const std::string& key,Vector& self,Vector& opposite,int kind=0) {
    std::ifstream in(file,std::ios::binary);
    std::array<std::uint64_t,4> h;
    char id[64];
    if(!in.read(reinterpret_cast<char*>(h.data()),sizeof(h)) || h[0]!=(0x5052534d53414d31ULL+kind) ||
       h[1]!=self.size() || h[2]!=opposite.size() || !in.read(id,sizeof(id)) ||
       key!=std::string(id,sizeof(id)))return false;
    for(Vector* v:{&self,&opposite})if(!v->empty() && !in.read(reinterpret_cast<char*>(v->data()),v->size()*sizeof(Vector::value_type)))return false;
    return in.peek()==std::char_traits<char>::eof() && !in.bad() && checksum(self,opposite)==h[3];
}
inline bool write(const std::string& file,const std::string& key,const Vector& self,const Vector& opposite,int kind=0) {
    if(key.size()!=64)return false;
    const std::string tmp=file+"."+std::to_string(getpid())+".tmp";
    std::ofstream out(tmp,std::ios::binary|std::ios::trunc);
    const std::array<std::uint64_t,4> h={{0x5052534d53414d31ULL+kind,self.size(),opposite.size(),checksum(self,opposite)}};
    out.write(reinterpret_cast<const char*>(h.data()),sizeof(h));out.write(key.data(),key.size());
    for(const Vector* v:{&self,&opposite})if(!v->empty())out.write(reinterpret_cast<const char*>(v->data()),v->size()*sizeof(Vector::value_type));
    out.close();
    if(out && std::rename(tmp.c_str(),file.c_str())==0)return true;
    std::remove(tmp.c_str());return false;
}
}
