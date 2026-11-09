#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

namespace bem_sector_device {
using Complex=std::complex<double>;
// Exact dictionaries: both IEEE754 components (including signed zero) are
// keys. Repeated orbit matrices share entries; no rounding or phase fitting.
struct Layout {
    int dofs=0;
    std::array<int,13> sectors{};
    std::vector<int> orbit_offsets,physical,solution,rhs,orbit_matrix,matrix_offsets;
    std::vector<std::uint8_t> matrix_dimensions;
    std::vector<std::uint16_t> weights;
    std::vector<Complex> palette;
    size_t storage_bytes()const {
        return sizeof(int)*(orbit_offsets.size()+physical.size()+solution.size()+rhs.size()+orbit_matrix.size()+matrix_offsets.size())+
               sizeof(std::uint8_t)*matrix_dimensions.size()+sizeof(std::uint16_t)*weights.size()+sizeof(Complex)*palette.size();
    }
    void validate()const {
        const size_t blocks=orbit_matrix.size(),matrices=matrix_dimensions.size();
        if(dofs<1 || sectors[0]!=0 || sectors[12]!=dofs || !blocks || !matrices || palette.empty() || palette.size()>65536 ||
           physical.size()!=size_t(dofs) || solution.size()!=size_t(dofs) || rhs.size()!=size_t(dofs) ||
           orbit_offsets.size()!=blocks+1 || orbit_offsets.front()!=0 || orbit_offsets.back()!=dofs ||
           matrix_offsets.size()!=matrices+1 || matrix_offsets.front()!=0 || size_t(matrix_offsets.back())!=weights.size())
            throw std::invalid_argument("sector device dimensions");
        for(int s=0;s<12;++s)if(sectors[s]>sectors[s+1] || sectors[s+1]-sectors[s]!=sectors[(s+6)%12+1]-sectors[(s+6)%12])
            throw std::invalid_argument("sector device parity dimensions");
        for(auto value:palette)if(!std::isfinite(value.real()) || !std::isfinite(value.imag()))throw std::invalid_argument("nonfinite sector palette");
        for(auto index:weights)if(size_t(index)>=palette.size())throw std::invalid_argument("sector palette index");
        for(size_t m=0;m<matrices;++m){
            const int r=matrix_dimensions[m],lo=matrix_offsets[m],hi=matrix_offsets[m+1];
            if(r<1 || r>12 || lo<0 || hi<lo || hi-lo!=r*r || size_t(hi)>weights.size())throw std::invalid_argument("sector orbit matrix shape");
            for(int j=0;j<r;++j)for(int k=0;k<r;++k){
                Complex dot=0;for(int i=0;i<r;++i)dot+=std::conj(palette[weights[lo+j*r+i]])*palette[weights[lo+k*r+i]];
                if(std::abs(dot-Complex(j==k?1.:0.))>5e-13)throw std::invalid_argument("sector orbit matrix is not unitary");
            }
        }
        for(size_t b=0;b<blocks;++b){
            const int r=orbit_offsets[b+1]-orbit_offsets[b],m=orbit_matrix[b];
            if(orbit_offsets[b]<0 || r<1 || r>12 || m<0 || size_t(m)>=matrices || matrix_dimensions[m]!=r)
                throw std::invalid_argument("sector orbit mapping");
        }
        for(const auto* permutation:{&physical,&solution,&rhs}){
            std::vector<bool> seen(dofs,false);
            for(int i:*permutation){if(i<0 || i>=dofs || seen[i])throw std::invalid_argument("sector index is not a permutation");seen[i]=true;}
        }
        for(int i=0;i<dofs;++i){
            const int s=int(std::upper_bound(sectors.begin(),sectors.end(),solution[i])-sectors.begin())-1;
            const int t=int(std::upper_bound(sectors.begin(),sectors.end(),rhs[i])-sectors.begin())-1;
            if(t!=(s+6)%12)throw std::invalid_argument("sector equation/current parity mismatch");
        }
    }
    template<class Transform> static Layout capture(const Transform& transform){
        Layout result;result.dofs=transform.size();result.sectors=transform.offsets;
        result.orbit_offsets.push_back(0);result.matrix_offsets.push_back(0);
        std::map<std::array<std::uint64_t,2>,std::uint16_t> value_index;
        std::map<std::vector<std::uint16_t>,int> matrix_index;
        transform.visit_orbit_transforms([&](const std::vector<int>& physical,const std::vector<int>& solution,
                                            const std::vector<int>& rhs,const std::vector<Complex>& values){
            const size_t r=physical.size();
            if(!r || r>12 || solution.size()!=r || rhs.size()!=r || values.size()!=r*r ||
               result.physical.size()+r>size_t(std::numeric_limits<int>::max()))throw std::invalid_argument("sector export shape");
            std::vector<std::uint16_t> indices;
            for(auto value:values){
                const double parts[2]={value.real(),value.imag()};std::array<std::uint64_t,2> key;
                std::memcpy(key.data(),parts,sizeof(parts));auto found=value_index.find(key);
                if(found==value_index.end()){
                    if(result.palette.size()>=65536)throw std::invalid_argument("sector palette exceeds uint16 capacity");
                    found=value_index.emplace(key,std::uint16_t(result.palette.size())).first;result.palette.push_back(value);
                }
                indices.push_back(found->second);
            }
            auto found=matrix_index.find(indices);
            if(found==matrix_index.end()){
                if(result.weights.size()+indices.size()>size_t(std::numeric_limits<int>::max()))throw std::invalid_argument("sector matrix catalog overflow");
                const int index=result.matrix_dimensions.size();found=matrix_index.emplace(indices,index).first;
                result.matrix_dimensions.push_back(r);result.weights.insert(result.weights.end(),indices.begin(),indices.end());
                result.matrix_offsets.push_back(result.weights.size());
            }
            result.orbit_matrix.push_back(found->second);
            result.physical.insert(result.physical.end(),physical.begin(),physical.end());
            result.solution.insert(result.solution.end(),solution.begin(),solution.end());result.rhs.insert(result.rhs.end(),rhs.begin(),rhs.end());
            result.orbit_offsets.push_back(result.physical.size());
        });result.validate();return result;
    }
    void apply_host(const Complex* input,Complex* output,bool encode,bool equation)const {
        if(!input || !output)throw std::invalid_argument("sector transform null vector");
        const auto a=reinterpret_cast<std::uintptr_t>(input),b=reinterpret_cast<std::uintptr_t>(output);
        if((a>b?a-b:b-a)<size_t(dofs)*sizeof(Complex))throw std::invalid_argument("sector transform overlap");
        const auto& coordinate=equation?rhs:solution;
        for(size_t block=0;block<orbit_matrix.size();++block){const int lo=orbit_offsets[block],r=orbit_offsets[block+1]-lo,base=matrix_offsets[orbit_matrix[block]];
            for(int i=0;i<r;++i){Complex sum=0;
                for(int j=0;j<r;++j){const auto value=palette[weights[base+(encode?i*r+j:j*r+i)]];
                    sum+=(encode?std::conj(value):value)*input[encode?physical[lo+j]:coordinate[lo+j]];}
                output[encode?coordinate[lo+i]:physical[lo+i]]=sum;
            }
        }
    }
};
}
