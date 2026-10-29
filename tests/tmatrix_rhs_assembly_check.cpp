#include "tmatrix_rhs_assembly.h"
#include "muller_dense.h"
#include "quadrature.h"
#include <cstdio>
#ifdef _OPENMP
#include <omp.h>
#endif

using Complex=std::complex<double>;
struct CachedPoint {int element;MullerFrameSample sample;double weight;};
void require(bool value,const char* message) {if(!value)throw std::runtime_error(message);}
double relative_error(const std::vector<Complex>& a,const std::vector<Complex>& b) {
    double error=0,length=0;
    for(size_t i=0;i<a.size();++i) {error+=std::norm(a[i]-b[i]);length+=std::norm(b[i]);}
    return std::sqrt(error/std::max(length,1e-300));
}
int main() try {
    const auto rule=tri_quadrature(13);
    for(int kind:{0,1,2}) {
        const bool bdm=kind!=0,curved=kind!=1;
        MullerP2BuildOptions options;options.edge_mode=bdm?MullerEdgeMode::HDivBdm1:MullerEdgeMode::Smooth;
        options.project_edge_nodes_to_sphere=curved;
        const auto mesh=build_muller_p2_mesh(curved?icosphere(1.,1):regular_prism(6,1.,1,1.,0,false,true),options);
        std::vector<CachedPoint> cache;
        for(int e=0;e<int(mesh.elements.size());++e)for(int q=0;q<rule.npts;++q) {
            auto frame=evaluate_muller_frame(mesh,e,rule.pts[q][0],rule.pts[q][1]);
            cache.push_back({e,frame,.5*rule.wts[q]*frame.jacobian});
        }
        std::vector<Complex> fields(cache.size()*6),reference(mesh.system_dofs(),0.),single_thread;
        // All E/H components are independent complex inputs; no plane-wave
        // relation can hide a sign, component or current-offset error.
        for(size_t i=0;i<fields.size();++i)fields[i]=Complex(std::sin(.017*i),std::cos(.023*i));
        for(size_t point=0;point<cache.size();++point) {
            const int e=point/rule.npts,q=point%rule.npts;
            const auto frame=evaluate_muller_frame(mesh,e,rule.pts[q][0],rule.pts[q][1]);
            const auto basis=evaluate_muller_basis(mesh,e,frame);
            const Complex* f=fields.data()+point*6;
            for(int a=0;a<basis.count;++a) {
                const Vec3 t=basis.values[a].cross(frame.normal)*(.5*rule.wts[q]*frame.jacobian);
                reference[basis.dofs[a]]-=t.x*f[0]+t.y*f[1]+t.z*f[2];
                reference[mesh.current_dofs()+basis.dofs[a]]+=t.x*f[3]+t.y*f[4]+t.z*f[5];
            }
        }
        for(int threads:{1,4})for(int chunk:{3,64,512,2048}) {
#ifdef _OPENMP
            omp_set_num_threads(threads);
#endif
            bem_tmatrix::StreamingRhsAssembler assembler(rule.npts,chunk);
            std::vector<Complex> rhs(mesh.system_dofs(),0.);
            for(size_t begin=0;begin<cache.size();begin+=assembler.chunk_points()) {
                const size_t count=std::min(assembler.chunk_points(),cache.size()-begin);
                assembler.add(mesh,cache,begin,count,fields.data()+6*begin,rhs);
            }
            const double error=relative_error(rhs,reference);
            require(error<2e-14,"cached element integrals disagree with pointwise reference");
            if(single_thread.empty())single_thread=rhs;
            else require(rhs==single_thread,"thread or chunk count changed the result");
            std::printf("RHS basis=%s curved=%d threads=%d chunk=%d error=%.3e\n",bdm?"BDM1":"P2",curved,threads,chunk,error);
        }
        // Independently implemented tested plane-wave RHS, including curved
        // geometry and the electric/magnetic equation signs.
        const Vec3 propagation(0.,0.,1.),electric(1.,0.,0.),magnetic=propagation.cross(electric);
        const Complex k(3.,.1);
        for(size_t point=0;point<cache.size();++point) {
            const Complex phase=std::exp(Complex(0,1)*k*propagation.dot(cache[point].sample.position));
            for(int c=0;c<3;++c) {
                const double ec=c==0?electric.x:c==1?electric.y:electric.z;
                const double hc=c==0?magnetic.x:c==1?magnetic.y:magnetic.z;
                fields[6*point+c]=phase*ec;fields[6*point+3+c]=phase*hc;
            }
        }
        bem_tmatrix::StreamingRhsAssembler assembler(rule.npts,64);
        std::vector<Complex> rhs(mesh.system_dofs(),0.);
        for(size_t begin=0;begin<cache.size();begin+=assembler.chunk_points())
            assembler.add(mesh,cache,begin,std::min(assembler.chunk_points(),cache.size()-begin),fields.data()+6*begin,rhs);
        const auto plane=muller_nodal_planewave_rhs(mesh,k,electric,propagation,13);
        require(relative_error(rhs,plane)<2e-14,"independent plane-wave RHS disagrees");
        bool rejected=false;
        try {assembler.add(mesh,cache,1,rule.npts,fields.data(),rhs);}
        catch(const std::invalid_argument&) {rejected=true;}
        require(rejected,"partial element chunk accepted");
        rejected=false;
        try {assembler.add(mesh,cache,0,rule.npts,nullptr,rhs);}
        catch(const std::invalid_argument&) {rejected=true;}
        require(rejected,"null field buffer accepted");
    }
    return 0;
} catch(const std::exception& e) {std::fprintf(stderr,"%s\n",e.what());return 1;}
