#pragma once
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace bem_ritz {
using Complex=std::complex<double>;
using Vector=std::vector<Complex>;
struct Parameters {
    std::uint64_t dofs=0,columns=0,ref=0,quad=7,duffy=6;
    double ka=0,ri_real=0,ri_imag=0,aspect=0;
};
// Portable between our little-endian IEEE-754 hosts. Columns are physical
// BDM currents on the TARGET mesh, never encoded right-preconditioned vectors.
// Their source/training provenance belongs to the accompanying experiment
// manifest; these parameters prevent accidental material/mesh reuse.
static constexpr size_t header_bytes=96;
inline std::uint64_t hash(std::uint64_t h,const void* p,size_t n) {
    const auto* bytes=static_cast<const unsigned char*>(p);
    for(size_t i=0;i<n;++i){h^=bytes[i];h*=1099511628211ULL;}
    return h;
}
inline void validate(const Parameters& p) {
    const std::uint16_t endian=1;
    if(*reinterpret_cast<const unsigned char*>(&endian)!=1 || sizeof(double)!=8 || !std::numeric_limits<double>::is_iec559)
        throw std::runtime_error("physical Ritz file needs little-endian IEEE-754 doubles");
    if(!p.dofs || p.dofs>std::uint64_t(std::numeric_limits<int>::max()) || !p.columns || p.columns>64 ||
       p.ref<2 || p.ref>8 || p.quad!=7 || p.duffy<2 || p.duffy>12 ||
       !std::isfinite(p.ka) || p.ka<=0 || !std::isfinite(p.ri_real) || p.ri_real<=0 ||
       !std::isfinite(p.ri_imag) || p.ri_imag<0 || !std::isfinite(p.aspect) || p.aspect<=0)
        throw std::invalid_argument("invalid physical Ritz parameters");
}
inline bool same(double a,double b) {
    return std::isfinite(a) && std::isfinite(b) && std::abs(a-b)<=8*std::numeric_limits<double>::epsilon()*std::max({1.,std::abs(a),std::abs(b)});
}
inline void header(std::ostream& out,const Parameters& p,std::uint64_t checksum) {
    const char magic[16]="BEM_PHYS_RITZ1";
    const std::uint64_t dims[6]={p.dofs,p.columns,checksum,p.ref,p.quad,p.duffy};
    const double values[4]={p.ka,p.ri_real,p.ri_imag,p.aspect};
    out.write(magic,16);out.write(reinterpret_cast<const char*>(dims),sizeof(dims));
    out.write(reinterpret_cast<const char*>(values),sizeof(values));
}
template<class Source> void write(const std::string& path,const Parameters& p,Source source) {
    validate(p);
    if(std::ifstream(path,std::ios::binary).good())throw std::runtime_error("refusing to replace physical Ritz file");
    std::ofstream out(path,std::ios::binary);header(out,p,0);
    Vector column(p.dofs);std::uint64_t checksum=1469598103934665603ULL;
    for(size_t j=0;j<p.columns;++j){
        source(j,column);
        if(column.size()!=p.dofs)throw std::runtime_error("physical Ritz source size");
        for(auto z:column)if(!std::isfinite(z.real())||!std::isfinite(z.imag()))throw std::runtime_error("nonfinite physical Ritz current");
        checksum=hash(checksum,column.data(),column.size()*sizeof(Complex));
        out.write(reinterpret_cast<const char*>(column.data()),column.size()*sizeof(Complex));
    }
    out.seekp(0);header(out,p,checksum);out.close();
    if(!out)throw std::runtime_error("physical Ritz write failed");
}
class Reader {
    std::ifstream stream_;
    Parameters parameters_;
public:
    Reader(const std::string& path,const Parameters& expected):stream_(path,std::ios::binary|std::ios::ate) {
        if(!stream_)throw std::runtime_error("cannot open physical Ritz file");
        const auto bytes=stream_.tellg();stream_.seekg(0);
        char magic[16]={};std::uint64_t dims[6]={};double values[4]={};
        stream_.read(magic,16);stream_.read(reinterpret_cast<char*>(dims),sizeof(dims));
        stream_.read(reinterpret_cast<char*>(values),sizeof(values));
        const char wanted[16]="BEM_PHYS_RITZ1";
        if(!stream_ || std::memcmp(magic,wanted,16))throw std::runtime_error("physical Ritz header");
        parameters_.dofs=dims[0];parameters_.columns=dims[1];parameters_.ref=dims[3];parameters_.quad=dims[4];parameters_.duffy=dims[5];
        parameters_.ka=values[0];parameters_.ri_real=values[1];parameters_.ri_imag=values[2];parameters_.aspect=values[3];
        const auto& p=parameters_;validate(p);
        if(p.dofs!=expected.dofs || p.ref!=expected.ref || p.quad!=expected.quad || p.duffy!=expected.duffy ||
           !same(p.ka,expected.ka) || !same(p.ri_real,expected.ri_real) || !same(p.ri_imag,expected.ri_imag) || !same(p.aspect,expected.aspect) ||
           (expected.columns && p.columns!=expected.columns) ||
           bytes!=std::streamoff(header_bytes+p.dofs*p.columns*sizeof(Complex)))
            throw std::runtime_error("physical Ritz file does not match the target operator");
        // Validate the whole file before any expensive operator image is made.
        Vector column(p.dofs);std::uint64_t checksum=1469598103934665603ULL;
        for(size_t j=0;j<p.columns;++j){
            read(j,column);checksum=hash(checksum,column.data(),column.size()*sizeof(Complex));
        }
        if(checksum!=dims[2])throw std::runtime_error("physical Ritz checksum");
    }
    const Parameters& parameters()const{return parameters_;}
    void read(size_t j,Vector& column) {
        if(j>=parameters_.columns)throw std::out_of_range("physical Ritz column");
        column.resize(parameters_.dofs);stream_.seekg(header_bytes+j*column.size()*sizeof(Complex));
        if(!stream_.read(reinterpret_cast<char*>(column.data()),column.size()*sizeof(Complex)))throw std::runtime_error("physical Ritz current read");
        for(auto z:column)if(!std::isfinite(z.real())||!std::isfinite(z.imag()))throw std::runtime_error("nonfinite physical Ritz current");
    }
};
}
