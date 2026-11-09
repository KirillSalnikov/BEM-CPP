#pragma once
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace bem_orbit_device {
using Complex=std::complex<double>;
struct Block {
    std::vector<Complex> lu; // Existing row-major P A = L U factors.
    std::vector<int> pivots,images;
    std::vector<std::int8_t> signs;
    std::vector<double> weight;
};

// Canonical packed layout of the established 12-image orbit inverse.
// Scatter is a deterministic row sum, including mirror-fixed stabilizers;
// it needs neither atomics nor a change to the factorization/precision.
struct Layout {
    int dofs=0;
    std::vector<int> dimensions,pivot_offsets,slot_offsets,pivots,input_indices;
    std::vector<std::uint64_t> lu_offsets;
    std::vector<std::int8_t> input_signs;
    std::vector<Complex> lu;
    std::vector<int> row_offsets,scatter_slots;
    std::vector<double> scatter_weights;

    void validate()const {
        const size_t blocks=dimensions.size(),slots=input_indices.size();
        if(dofs<=0 || !blocks || slots>size_t(std::numeric_limits<int>::max()) || pivot_offsets.size()!=blocks ||
           slot_offsets.size()!=blocks || lu_offsets.size()!=blocks || input_signs.size()!=slots ||
           row_offsets.size()!=size_t(dofs)+1 || row_offsets[0]!=0 || row_offsets.back()!=int(slots) ||
           scatter_slots.size()!=slots || scatter_weights.size()!=slots)
            throw std::invalid_argument("packed orbit dimensions");
        size_t p=0,s=0,a=0;
        for(size_t b=0;b<blocks;++b){const int d=dimensions[b];
            if(d<=0 || d>2048 || d%4 || pivot_offsets[b]!=int(p) || slot_offsets[b]!=int(s) || lu_offsets[b]!=a ||
               p+d>pivots.size() || s+12*d>slots || a+size_t(d)*d>lu.size())throw std::invalid_argument("packed orbit offsets");
            for(int i=0;i<d;++i)if(pivots[p+i]<i || pivots[p+i]>=d || std::abs(lu[a+size_t(i)*d+i])<1e-24)
                throw std::invalid_argument("packed orbit pivots/diagonal");
            p+=d;s+=12*d;a+=size_t(d)*d;
        }
        if(p!=pivots.size() || s!=slots || a!=lu.size())throw std::invalid_argument("trailing packed orbit data");
        for(auto z:lu)if(!std::isfinite(z.real()) || !std::isfinite(z.imag()))throw std::invalid_argument("nonfinite packed orbit factors");
        for(size_t i=0;i<slots;++i)if(input_indices[i]<0 || input_indices[i]>=dofs || (input_signs[i]!=1 && input_signs[i]!=-1))
            throw std::invalid_argument("packed orbit input map");
        std::vector<bool> seen(slots,false);
        for(int i=0;i<dofs;++i){
            if(row_offsets[i]<0 || row_offsets[i]>=row_offsets[i+1] || row_offsets[i+1]>int(slots))throw std::invalid_argument("packed orbit row offsets");
            double coverage=0;
            for(int k=row_offsets[i];k<row_offsets[i+1];++k){const int index=scatter_slots[k];const double weight=scatter_weights[k];
                if(index<0 || index>=int(slots) || seen[index] || input_indices[index]!=i || !std::isfinite(weight) || weight==0 || std::abs(weight)>1)
                    throw std::invalid_argument("packed orbit scatter");
                seen[index]=true;coverage+=std::abs(weight);
            }
            if(std::abs(coverage-1)>64*std::numeric_limits<double>::epsilon())throw std::invalid_argument("packed orbit coverage");
        }
    }

    static Layout build(int n,const std::vector<Block>& blocks) {
        if(n<=0 || blocks.empty())throw std::invalid_argument("empty orbit device layout");
        Layout out;out.dofs=n;
        std::vector<int> owner(n,-1),counts(n,0),slot_destination;
        std::vector<double> coverage(n,0),slot_weight;
        for(size_t b=0;b<blocks.size();++b){
            const auto& block=blocks[b];const size_t d=block.pivots.size();
            if(!d || d>2048 || d%4 || block.lu.size()!=d*d || block.images.size()!=12*d ||
               block.signs.size()!=12*d || block.weight.size()!=d ||
               out.input_indices.size()+12*d>size_t(std::numeric_limits<int>::max()))
                throw std::invalid_argument("orbit device block dimensions");
            for(size_t k=0;k<d;++k){
                if(block.pivots[k]<int(k) || block.pivots[k]>=int(d) || !std::isfinite(block.weight[k]) ||
                   block.weight[k]<=0 || block.weight[k]>1 || std::abs(block.lu[k*d+k])<1e-24)
                    throw std::invalid_argument("orbit device pivots/diagonal/weight");
            }
            for(auto z:block.lu)if(!std::isfinite(z.real()) || !std::isfinite(z.imag()))throw std::invalid_argument("nonfinite orbit device factors");
            out.dimensions.push_back(int(d));out.pivot_offsets.push_back(int(out.pivots.size()));
            out.slot_offsets.push_back(int(out.input_indices.size()));out.lu_offsets.push_back(out.lu.size());
            out.pivots.insert(out.pivots.end(),block.pivots.begin(),block.pivots.end());out.lu.insert(out.lu.end(),block.lu.begin(),block.lu.end());
            for(int g=0;g<12;++g){
                std::vector<int> selected;
                for(size_t i=0;i<d;++i){
                    const int index=block.images[g*d+i],sign=block.signs[g*d+i];
                    if(index<0 || index>=n || (sign!=1 && sign!=-1) || (owner[index]>=0 && owner[index]!=int(b)))
                        throw std::invalid_argument("orbit device image/sign/ownership");
                    owner[index]=int(b);++counts[index];coverage[index]+=block.weight[i];selected.push_back(index);
                    out.input_indices.push_back(index);out.input_signs.push_back((g<6?1:-1)*sign);
                    slot_destination.push_back(index);slot_weight.push_back(block.weight[i]*sign);
                }
                std::sort(selected.begin(),selected.end());
                if(std::adjacent_find(selected.begin(),selected.end())!=selected.end())throw std::invalid_argument("noninjective orbit image");
            }
        }
        out.row_offsets.push_back(0);
        for(int i=0;i<n;++i){
            if(owner[i]<0 || std::abs(coverage[i]-1)>64*std::numeric_limits<double>::epsilon())
                throw std::invalid_argument("incomplete orbit device coverage");
            out.row_offsets.push_back(out.row_offsets.back()+counts[i]);
        }
        out.scatter_slots.resize(slot_destination.size());out.scatter_weights.resize(slot_destination.size());
        auto cursor=out.row_offsets;
        for(size_t slot=0;slot<slot_destination.size();++slot){
            const int entry=cursor[slot_destination[slot]]++;
            out.scatter_slots[entry]=int(slot);out.scatter_weights[entry]=slot_weight[slot];
        }
        return out;
    }
    template<class Inverse> static Layout capture(const Inverse& inverse) {
        int n=0;std::vector<Block> blocks;
        inverse.visit_factorized_blocks([&](int size,const std::vector<Complex>& lu,const std::vector<int>& pivots,
                const std::vector<int>& images,const std::vector<std::int8_t>& signs,const std::vector<double>& weights){
            if(n && n!=size)throw std::invalid_argument("inconsistent orbit export dimensions");
            n=size;Block block;block.lu=lu;block.pivots=pivots;block.images=images;block.signs=signs;block.weight=weights;
            blocks.push_back(std::move(block));
        });
        return build(n,blocks);
    }
    // CPU oracle for export/layout controls, not a production replacement.
    void apply_host_reference(const Complex* input,Complex* output)const {
        if(dofs<=0 || !input || !output)throw std::invalid_argument("orbit layout application");
        std::vector<Complex> values(input_indices.size());
        for(size_t i=0;i<values.size();++i)values[i]=double(input_signs[i])*input[input_indices[i]];
        for(size_t b=0;b<dimensions.size();++b){
            const int d=dimensions[b];const auto* a=lu.data()+lu_offsets[b];const auto* p=pivots.data()+pivot_offsets[b];
            for(int g=0;g<12;++g){auto* v=values.data()+slot_offsets[b]+g*d;
                for(int k=0;k<d;++k)std::swap(v[k],v[p[k]]);
                for(int k=0;k<d;++k)for(int i=k+1;i<d;++i)v[i]-=a[i*d+k]*v[k];
                for(int k=d-1;k>=0;--k){for(int j=k+1;j<d;++j)v[k]-=a[k*d+j]*v[j];v[k]/=a[k*d+k];}
            }
        }
        for(int i=0;i<dofs;++i){Complex sum=0;
            for(int k=row_offsets[i];k<row_offsets[i+1];++k)sum+=scatter_weights[k]*values[scatter_slots[k]];
            output[i]=sum;
        }
    }
};
}
