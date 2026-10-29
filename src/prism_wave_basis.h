#pragma once
#include "muller_nodal.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

// BDM1 edge moments of tangential plane-wave traces. The moment integrals
// are analytic, with the same globally oriented topology edges as the BEM.
// Each direction gives two polarizations in each of the two current spaces.
class PrismWaveBasis {
    std::vector<Vec3> centers_,edges_;
    double k_;
    static double radical(int n,int base) {
        double x=0,w=1.;while(n){w/=base;x+=(n%base)*w;n/=base;}return x;
    }
public:
    PrismWaveBasis(const MullerP2Mesh& mesh,double k):k_(k) {
        if(mesh.basis_kind!=MullerBasisKind::HDivBdm1 || !std::isfinite(k) || k<=0)
            throw std::invalid_argument("wave coarse basis requires BDM1 and real positive k");
        centers_.resize(mesh.topology_edge_count);edges_.resize(centers_.size());
        for(const auto& e:mesh.elements)for(int j=0;j<3;++j) {
            int a=e.topology_vertices[j],b=e.topology_vertices[(j+1)%3];if(a>b)std::swap(a,b);
            const int id=e.topology_edges[j];centers_[id]=(mesh.nodes[a]+mesh.nodes[b])*.5;edges_[id]=mesh.nodes[b]-mesh.nodes[a];
        }
    }
    void operator()(int candidate,std::complex<double>* out)const {
        const int direction=candidate/4,polarization=candidate%2,current=(candidate%4)/2;
        const double z=2*radical(direction+1,2)-1,phi=2*std::acos(-1.)*radical(direction+1,3),r=std::sqrt(1-z*z);
        const Vec3 d(r*std::cos(phi),r*std::sin(phi),z);
        const Vec3 axis=std::abs(z)<.9?Vec3(0,0,1):Vec3(1,0,0);
        const Vec3 p0=axis.cross(d)*(1./axis.cross(d).norm()),p=polarization?d.cross(p0):p0;
        const int edges=edges_.size();std::fill(out,out+4*edges,std::complex<double>(0));
#pragma omp parallel for schedule(static)
        for(int e=0;e<edges;++e) {
            const double x=.5*k_*d.dot(edges_[e]),x2=x*x;
            const double sinc=std::abs(x)<1e-3?1-x2/6+x2*x2/120:std::sin(x)/x;
            const double j1=std::abs(x)<1e-3?x*(1./3-x2/30+x2*x2/840):(std::sin(x)-x*std::cos(x))/x2;
            const auto amplitude=p.dot(edges_[e])*std::exp(std::complex<double>(0,k_*d.dot(centers_[e])));
            out[2*current*edges+2*e]=amplitude*sinc;
            out[2*current*edges+2*e+1]=amplitude*std::complex<double>(0,j1);
        }
    }
};
