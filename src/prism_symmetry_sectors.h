#ifndef BEM_PRISM_SYMMETRY_SECTORS_H
#define BEM_PRISM_SYMMETRY_SECTORS_H
#include "tmatrix_symmetry_warm_start.h"
#include <cstdint>
#include <numeric>

namespace bem_tmatrix {
// The commuting subgroup C6 x horizontal reflection, not all of D6h.
struct PrismGroupMaps {
    std::array<std::vector<int>,12> image;
    std::array<std::vector<int8_t>,12> sign;
    void build(const D6hOrbit& orbit) {
        const int n=orbit.size();
        std::vector<int> rotation,rs,mirror,ms;
        orbit.image_map(false,false,rotation,rs);orbit.image_map(true,false,mirror,ms);
        for(int g=0;g<12;++g){image[g].resize(n);sign[g].resize(n);}
        for(int i=0;i<n;++i) {
            image[0][i]=i;sign[0][i]=1;
            image[6][i]=mirror[i];sign[6][i]=ms[i];
            for(int p:{0,6})for(int g=p+1;g<p+6;++g) {
                const int previous=image[g-1][i];
                image[g][i]=rotation[previous];sign[g][i]=rs[previous]*sign[g-1][i];
            }
        }
    }
};

class PrismSymmetrySectors {
    struct Orbit {
        std::vector<int> dofs,sector,solution_coordinate,rhs_coordinate;
        // Columns are orthonormal eigenvectors of the solution representation.
        std::vector<Complex> weights;
    };
    std::vector<Orbit> orbits_;
    int n_=0;
public:
    std::array<int,13> offsets{};
    int size()const{return n_;}
    template<class Visitor> void visit_orbit_transforms(Visitor visitor)const {
        if(n_<1)throw std::logic_error("sector transforms not built");
        for(const auto& o:orbits_)visitor(o.dofs,o.solution_coordinate,o.rhs_coordinate,o.weights);
    }
    size_t storage_bytes()const {
        size_t bytes=0;for(const auto& o:orbits_)bytes+=sizeof(Orbit)+
            sizeof(int)*(o.dofs.size()+o.sector.size()+o.solution_coordinate.size()+o.rhs_coordinate.size())+
            sizeof(Complex)*o.weights.size();return bytes;
    }
    void build(const PrismGroupMaps& group) {
        n_=group.image[0].size();orbits_.clear();offsets.fill(0);
        std::vector<bool> seen(n_,false);
        const double pi=std::acos(-1.);
        for(int seed=0;seed<n_;++seed)if(!seen[seed]) {
            Orbit o;
            for(int g=0;g<12;++g)o.dofs.push_back(group.image[g][seed]);
            std::sort(o.dofs.begin(),o.dofs.end());o.dofs.erase(std::unique(o.dofs.begin(),o.dofs.end()),o.dofs.end());
            for(int d:o.dofs){if(seen[d])throw std::logic_error("overlapping group orbits");seen[d]=true;}
            const int r=o.dofs.size();
            for(int s=0;s<12;++s) {
                Vector v(r,0.);
                for(int g=0;g<12;++g) {
                    const int j=std::lower_bound(o.dofs.begin(),o.dofs.end(),group.image[g][seed])-o.dofs.begin();
                    const Complex character=std::polar(1.,-2*pi*(s%6)*(g%6)/6.)*double(s>=6 && g>=6?-1:1);
                    v[j]+=double(group.sign[g][seed])*character/12.;
                }
                const double length=bem_streaming::norm(v.data(),r);
                if(length<1e-10)continue;
                o.sector.push_back(s);++offsets[s+1];
                for(auto z:v)o.weights.push_back(z/length);
            }
            if(int(o.sector.size())!=r)throw std::logic_error("incomplete symmetry basis");
            orbits_.push_back(std::move(o));
        }
        std::array<int,12> counts{};for(int s=0;s<12;++s)counts[s]=offsets[s+1];
        for(int s=0;s<12;++s)if(counts[s]!=counts[(s+6)%12])throw std::logic_error("unequal equation/current sector dimensions");
        for(int s=0;s<12;++s)offsets[s+1]+=offsets[s];
        auto sol=offsets,rhs=offsets;
        for(auto& o:orbits_)for(int s:o.sector) {
            o.solution_coordinate.push_back(sol[s]++);
            o.rhs_coordinate.push_back(rhs[(s+6)%12]++);
        }
    }
    void encode(const Complex* full,Complex* coordinates,bool rhs)const {
        if(full==coordinates)throw std::invalid_argument("sector transform aliases input");
#pragma omp parallel for schedule(static)
        for(size_t k=0;k<orbits_.size();++k) {
            const auto& o=orbits_[k];const int r=o.dofs.size();
            const auto& index=rhs?o.rhs_coordinate:o.solution_coordinate;
            for(int j=0;j<r;++j){Complex z=0;for(int i=0;i<r;++i)z+=std::conj(o.weights[j*r+i])*full[o.dofs[i]];coordinates[index[j]]=z;}
        }
    }
    void decode(const Complex* coordinates,Complex* full,bool rhs)const {
        if(full==coordinates)throw std::invalid_argument("sector transform aliases input");
#pragma omp parallel for schedule(static)
        for(size_t k=0;k<orbits_.size();++k) {
            const auto& o=orbits_[k];const int r=o.dofs.size();
            const auto& index=rhs?o.rhs_coordinate:o.solution_coordinate;
            for(int i=0;i<r;++i){Complex z=0;for(int j=0;j<r;++j)z+=o.weights[j*r+i]*coordinates[index[j]];full[o.dofs[i]]=z;}
        }
    }
};
}
#endif
