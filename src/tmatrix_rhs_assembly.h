#ifndef BEM_TMATRIX_RHS_ASSEMBLY_H
#define BEM_TMATRIX_RHS_ASSEMBLY_H

#include "muller_nodal.h"
#include <algorithm>
#include <array>
#include <complex>
#include <stdexcept>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace bem_tmatrix {
// Integrate a bounded number of complete elements in parallel, then scatter
// in mesh order. Shared edge/node DOFs never receive concurrent writes.
// Point must provide the existing cache's element, sample and weight fields;
// the template keeps this CPU-only helper independent of CUDA/FMM headers.
class StreamingRhsAssembler {
    using Complex=std::complex<double>;
    struct ElementIntegral {
        int count=0;
        std::array<int,12> dofs;
        std::array<Complex,12> electric,magnetic;
    };
    int points_per_element_;
    int thread_limit_;
    std::vector<ElementIntegral> integrals_;
public:
    explicit StreamingRhsAssembler(int points_per_element,int chunk_elements=2048,int thread_limit=4)
        :points_per_element_(points_per_element),thread_limit_(thread_limit) {
        if(points_per_element<1 || chunk_elements<1 || thread_limit<1)
            throw std::invalid_argument("invalid RHS chunk dimensions");
        integrals_.resize(chunk_elements);
    }
    size_t chunk_points() const {return size_t(points_per_element_)*integrals_.size();}
    size_t workspace_bytes() const {return integrals_.size()*sizeof(ElementIntegral);}
    int thread_limit() const {return thread_limit_;}

    template<class Point>
    void add(const MullerP2Mesh& mesh,const std::vector<Point>& points,
             size_t begin,size_t count,const Complex* fields,std::vector<Complex>& rhs) {
        if(points.size()!=mesh.elements.size()*size_t(points_per_element_) ||
           rhs.size()!=size_t(mesh.system_dofs()) || begin>points.size() || count>points.size()-begin ||
           begin%points_per_element_ || count%points_per_element_ || count>chunk_points() ||
           (count && fields==nullptr))
            throw std::invalid_argument("RHS chunk must contain complete cached elements");
        const int first_element=int(begin/points_per_element_);
        const int elements=int(count/points_per_element_);
        if(elements && (points[begin].element!=first_element ||
                        points[begin+count-1].element!=first_element+elements-1))
            throw std::invalid_argument("RHS cache element order mismatch");
#ifdef _OPENMP
        const int threads=std::min(thread_limit_,omp_get_max_threads());
#pragma omp parallel for schedule(static) if(elements>=32) num_threads(threads)
#endif
        for(int e=0;e<elements;++e) {
            auto& integral=integrals_[e];
            integral.electric.fill(Complex(0));integral.magnetic.fill(Complex(0));
            if(mesh.basis_kind==MullerBasisKind::HDivBdm1) {
                const auto& element=mesh.elements[first_element+e];
                integral.count=6;
                for(int a=0;a<6;++a)integral.dofs[a]=2*element.topology_edges[a/2]+a%2;
                for(int q=0;q<points_per_element_;++q) {
                    const size_t local=size_t(e)*points_per_element_+q;
                    const auto& point=points[begin+local];
                    const auto& frame=point.sample;
                    const double factor=point.weight/frame.jacobian;
                    const Vec3 u=frame.derivative_xi.cross(frame.normal)*factor;
                    const Vec3 v=frame.derivative_eta.cross(frame.normal)*factor;
                    const Complex* f=fields+local*6;
                    const Complex eu=u.x*f[0]+u.y*f[1]+u.z*f[2],ev=v.x*f[0]+v.y*f[1]+v.z*f[2];
                    const Complex hu=u.x*f[3]+u.y*f[4]+u.z*f[5],hv=v.x*f[3]+v.y*f[4]+v.z*f[5];
                    const double xi=frame.reference_xi,eta=frame.reference_eta;
                    // Factor the two Piola directions out of the six BDM1
                    // tests. This remains valid on curved quadratic panels.
                    const double reference[6][2]={
                        {xi,eta-1.},{3.*xi,-6.*xi-3.*eta+3.},
                        {xi,eta},{-3.*xi,3.*eta},
                        {xi-1.,eta},{3.*xi+6.*eta-3.,-3.*eta}};
                    for(int a=0;a<6;++a) {
                        integral.electric[a]-=reference[a][0]*eu+reference[a][1]*ev;
                        integral.magnetic[a]+=reference[a][0]*hu+reference[a][1]*hv;
                    }
                }
                for(int edge=0;edge<3;++edge) {
                    integral.electric[2*edge]*=element.edge_orientations[edge];
                    integral.magnetic[2*edge]*=element.edge_orientations[edge];
                }
                continue;
            }
            for(int q=0;q<points_per_element_;++q) {
                const size_t local=size_t(e)*points_per_element_+q;
                const auto& point=points[begin+local];
                const auto basis=evaluate_muller_basis(mesh,first_element+e,point.sample);
                if(q==0) {
                    integral.count=basis.count;
                    std::copy_n(basis.dofs.begin(),basis.count,integral.dofs.begin());
                }
                const Complex* f=fields+local*6;
                for(int a=0;a<basis.count;++a) {
                    const Vec3 t=basis.values[a].cross(point.sample.normal)*point.weight;
                    integral.electric[a]-=t.x*f[0]+t.y*f[1]+t.z*f[2];
                    integral.magnetic[a]+=t.x*f[3]+t.y*f[4]+t.z*f[5];
                }
            }
        }
        const int current_dofs=mesh.current_dofs();
        for(int e=0;e<elements;++e) {
            const auto& integral=integrals_[e];
            for(int a=0;a<integral.count;++a) {
                rhs[integral.dofs[a]]+=integral.electric[a];
                rhs[current_dofs+integral.dofs[a]]+=integral.magnetic[a];
            }
        }
    }
};
}
#endif
