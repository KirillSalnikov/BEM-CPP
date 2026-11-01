#pragma once
#include "muller_nodal.h"
#include "quadrature.h"
#include "matrix_free_taylor_moments.h"
#include <array>

namespace bem_taylor {
// Fixed mesh, nondispersive exterior, k=k0+delta. All coefficients are
// derivatives divided by factorial. No dense incident/radiation matrix.
inline void check_direction(const Vec3& d){
    if(!std::isfinite(d.x)||!std::isfinite(d.y)||!std::isfinite(d.z)||std::abs(d.dot(d)-1)>1e-10)
        throw std::invalid_argument("Taylor field requires a unit direction");
}
template<size_t N> MomentHistory plane_wave_moments(const MullerP2Mesh& mesh,double k,
        const Vec3& direction,const Vec3& electric,int quad=13){
    static_assert(N<=16,"Taylor field order exceeds supported range");
    check_direction(direction);
    if(!std::isfinite(k)||k<=0||!std::isfinite(electric.dot(electric))||
       std::abs(electric.dot(direction))>1e-10*std::max(1.,electric.norm()))
        throw std::invalid_argument("Taylor incident field parameters");
    using C=std::complex<double>;const int n=mesh.current_dofs();
    MomentHistory out(N+1,MomentVector(2*n));const auto rule=tri_quadrature(quad);
    const Vec3 magnetic=direction.cross(electric);
    for(size_t t=0;t<mesh.elements.size();++t)for(int q=0;q<rule.npts;++q){
        const auto sample=evaluate_muller_frame(mesh,t,rule.pts[q][0],rule.pts[q][1]);
        const auto basis=evaluate_muller_basis(mesh,t,sample);
        const double weight=.5*rule.wts[q]*sample.jacobian;
        const C phase(0,direction.dot(sample.position));std::array<C,N+1> exponential;
        exponential[0]=std::exp(k*phase);
        for(size_t j=1;j<=N;++j)exponential[j]=exponential[j-1]*phase/double(j);
        for(int a=0;a<basis.count;++a){const Vec3 rotated=basis.values[a].cross(sample.normal);
            const int i=basis.dofs[a];const double e=-weight*rotated.dot(electric),h=weight*rotated.dot(magnetic);
            for(size_t j=0;j<=N;++j){out[j][i]+=e*exponential[j];out[j][n+i]+=h*exponential[j];}}
    }
    return out;
}

template<size_t N> MomentHistory farfield_moments(const MullerP2Mesh& mesh,double k,
        const MomentHistory& currents,const std::vector<Vec3>& directions,int quad=13){
    static_assert(N<=16,"Taylor field order exceeds supported range");
    using C=std::complex<double>;const int n=mesh.current_dofs();
    if(!std::isfinite(k)||k<=0||currents.size()!=N+1)throw std::invalid_argument("Taylor radiation parameters");
    for(const auto& v:currents){if(v.size()!=size_t(2*n))throw std::invalid_argument("Taylor radiation current dimensions");
        for(C z:v)if(!std::isfinite(z.real())||!std::isfinite(z.imag()))throw std::invalid_argument("Taylor radiation nonfinite current");}
    for(const auto& d:directions)check_direction(d);
    using Current=std::array<std::array<C,6>,N+1>;
    struct Sample{Vec3 position;Current weighted;};
    const auto rule=tri_quadrature(quad);std::vector<Sample> samples(mesh.elements.size()*rule.npts);
    // Project each current history once, reused by all observation directions.
#pragma omp parallel for schedule(static)
    for(size_t t=0;t<mesh.elements.size();++t)for(int q=0;q<rule.npts;++q){
        const auto frame=evaluate_muller_frame(mesh,t,rule.pts[q][0],rule.pts[q][1]);
        const auto basis=evaluate_muller_basis(mesh,t,frame);const double weight=.5*rule.wts[q]*frame.jacobian;
        auto& s=samples[t*rule.npts+q];s.position=frame.position;s.weighted=Current{};
        for(int a=0;a<basis.count;++a){const auto& v=basis.values[a];const int i=basis.dofs[a];
            for(size_t j=0;j<=N;++j)for(int c=0;c<2;++c){const C z=weight*currents[j][c*n+i];
                s.weighted[j][3*c]+=v.x*z;s.weighted[j][3*c+1]+=v.y*z;s.weighted[j][3*c+2]+=v.z*z;}}
    }
    MomentHistory out(N+1,MomentVector(3*directions.size()));
#pragma omp parallel for schedule(static)
    for(size_t p=0;p<directions.size();++p){const auto& d=directions[p];Current transformed{};
        for(const auto& sample:samples){const C phase(0,-d.dot(sample.position));std::array<C,N+1> exponential;
            exponential[0]=std::exp(k*phase);for(size_t j=1;j<=N;++j)exponential[j]=exponential[j-1]*phase/double(j);
            for(size_t j=0;j<=N;++j)for(size_t q=0;q<=j;++q)for(int a=0;a<6;++a)
                transformed[j][a]+=exponential[q]*sample.weighted[j-q][a];}
        std::array<C,3> previous{};
        for(size_t j=0;j<=N;++j){const auto& v=transformed[j];const C dot=d.x*v[0]+d.y*v[1]+d.z*v[2];
            const std::array<C,3> field={{v[0]-d.x*dot-d.y*v[5]+d.z*v[4],
                v[1]-d.y*dot-d.z*v[3]+d.x*v[5],v[2]-d.z*dot-d.x*v[4]+d.y*v[3]}};
            for(int a=0;a<3;++a)out[j][3*p+a]=C(0,-INV4PI)*(k*field[a]+previous[a]);
            previous=field;
        }
    }
    return out;
}
}
