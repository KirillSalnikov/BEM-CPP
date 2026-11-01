#include "muller_taylor_fields.h"
#include "muller_dense.h"
#include <cstdio>
#include <numeric>
#include <random>
using C=std::complex<double>;using LC=std::complex<long double>;
using V=bem_taylor::MomentVector;using H=bem_taylor::MomentHistory;
constexpr int N=6,S=48;
double relative(const V& a,const V& b){double e=0,n=0;for(size_t i=0;i<a.size();++i){e+=std::norm(a[i]-b[i]);n+=std::norm(b[i]);}return std::sqrt(e/std::max(n,1e-280));}
V polynomial(const H& a,C z){V v(a.front().size());for(auto it=a.rbegin();it!=a.rend();++it)for(size_t i=0;i<v.size();++i)v[i]=z*v[i]+(*it)[i];return v;}
int main()try{
    MullerP2BuildOptions opts;opts.edge_mode=MullerEdgeMode::HDivBdm1;
    const auto mesh=build_muller_p2_mesh(regular_prism(6,10./7,0,1.,0,false,true),opts);
    const int n=mesh.current_dofs();const double k=3,radius=.4;
    const Vec3 d=Vec3(.3,.4,.5)*(1/std::sqrt(.5)),e=Vec3(.8,-.6,0),h=d.cross(e);
    std::vector<Vec3> directions={d,d*(-1.),Vec3(0,0,1),Vec3(0,1,0)};
    H currents(N+1,V(2*n));std::mt19937 rng(751);std::normal_distribution<double> normal;
    for(auto& v:currents)for(auto& z:v)z=C(normal(rng),normal(rng));
    const auto far=bem_taylor::farfield_moments<N>(mesh,k,currents,directions);
    double central=0,coefficient=0,transverse=0,nearby=0;
    for(const auto& pol:{e,h}){
        const auto rhs=bem_taylor::plane_wave_moments<N>(mesh,k,d,pol);
        const auto central_rhs=muller_nodal_planewave_rhs(mesh,k,pol,d,13);
        central=std::max(central,relative(rhs[0],central_rhs));
        std::vector<std::vector<LC>> rhs_sum(N+1,std::vector<LC>(2*n)),far_sum(N+1,std::vector<LC>(3*directions.size()));
        for(int q=0;q<S;++q){const long double theta=2*std::acos(-1.L)*q/S;const C delta=radius*std::exp(C(0,double(theta)));
            const auto sampled_rhs=muller_nodal_planewave_rhs(mesh,k+delta,pol,d,13);const auto x=polynomial(currents,delta);
            V sampled_far;muller_nodal_farfield(mesh,x.data(),x.data()+n,k+delta,directions,sampled_far,13);
            for(int j=0;j<=N;++j){const LC phase=std::exp(LC(0,-j*theta))/static_cast<long double>(S);
                for(size_t i=0;i<sampled_rhs.size();++i)rhs_sum[j][i]+=LC(sampled_rhs[i])*phase;
                for(size_t i=0;i<sampled_far.size();++i)far_sum[j][i]+=LC(sampled_far[i])*phase;}
        }
        const double rhs_norm=std::sqrt(std::real(std::inner_product(rhs[0].begin(),rhs[0].end(),rhs[0].begin(),C(0),std::plus<C>(),[](C a,C b){return std::conj(a)*b;})));
        double far_norm=0;for(C z:far[0])far_norm+=std::norm(z);far_norm=std::sqrt(far_norm);
        for(int j=0;j<=N;++j){double er=0,ef=0;const double scale=std::pow(radius,j);
            for(size_t i=0;i<rhs[0].size();++i)er+=std::norm(LC(rhs[j][i])*static_cast<long double>(scale)-rhs_sum[j][i]);
            for(size_t i=0;i<far[0].size();++i)ef+=std::norm(LC(far[j][i])*static_cast<long double>(scale)-far_sum[j][i]);
            coefficient=std::max({coefficient,std::sqrt(er)/rhs_norm,std::sqrt(ef)/far_norm});}
        for(double delta:{-.03,.03}){const auto approximate=polynomial(rhs,delta),exact=muller_nodal_planewave_rhs(mesh,k+delta,pol,d,13);
            nearby=std::max(nearby,relative(approximate,exact));}
    }
    V actual;muller_nodal_farfield(mesh,currents[0].data(),currents[0].data()+n,k,directions,actual,13);central=std::max(central,relative(far[0],actual));
    for(int j=0;j<=N;++j)for(size_t p=0;p<directions.size();++p){const auto& d=directions[p];
        const C dot=d.x*far[j][3*p]+d.y*far[j][3*p+1]+d.z*far[j][3*p+2];
        const double norm=std::sqrt(std::norm(far[j][3*p])+std::norm(far[j][3*p+1])+std::norm(far[j][3*p+2]));
        transverse=std::max(transverse,std::abs(dot)/std::max(norm,1e-280));}
    for(double delta:{-.03,.03}){const auto x=polynomial(currents,delta);muller_nodal_farfield(mesh,x.data(),x.data()+n,k+delta,directions,actual,13);
        nearby=std::max(nearby,relative(polynomial(far,delta),actual));}
    auto zero=currents;for(auto& v:zero)std::fill(v.begin(),v.end(),C(0));
    for(const auto& v:bem_taylor::farfield_moments<N>(mesh,k,zero,directions))for(C z:v)if(z!=C(0))throw std::runtime_error("zero current");
    bool invalid=false;try{bem_taylor::plane_wave_moments<N>(mesh,k,Vec3(0,0,2),e);}catch(const std::invalid_argument&){invalid=true;}
    if(!invalid||central>2e-12||coefficient>2e-12||transverse>2e-12||nearby>2e-9)throw std::runtime_error("physical Taylor fields mismatch");
    std::printf("PASS physical incident/radiation moments: central %.6e, contour scaled %.6e, transverse %.6e, nearby %.6e; both polarizations, order %d\n",central,coefficient,transverse,nearby,N);
}catch(const std::exception& error){std::fprintf(stderr,"%s\n",error.what());return 1;}
