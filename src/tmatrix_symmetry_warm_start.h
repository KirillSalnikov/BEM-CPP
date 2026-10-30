#ifndef BEM_TMATRIX_SYMMETRY_WARM_START_H
#define BEM_TMATRIX_SYMMETRY_WARM_START_H

#include "muller_nodal.h"
#include "streaming_host_gmres.h"
#include <array>
#include <map>

namespace bem_tmatrix {
using bem_streaming::Complex;
using Vector=std::vector<Complex>;

// Signed edge-moment permutations for C6 rotation and horizontal reflection.
// BDM1 moment zero changes sign when the canonical edge direction reverses;
// moment one does not. Under reflection, electric/magnetic currents have
// opposite parity. The tested equations have the opposite reflection parity
// to the unknowns: A S_solution = S_rhs A, with S_rhs = -S_solution.
class D6hOrbit {
    struct Map { std::vector<int> source, sign; };
    int edges_=0;
    Map rotation_,reflection_;
    using Key=std::array<long long,3>;
    static Key key(const Vec3& p) {
        return {{std::llround(p.x*1e10),std::llround(p.y*1e10),std::llround(p.z*1e10)}};
    }
    static Map make_map(const MullerP2Mesh& mesh,bool reflection,bool vertical=false) {
        using Edge=std::pair<int,int>;
        const int local[3][2]={{0,1},{1,2},{2,0}};
        std::map<Edge,int> by_edge;
        std::map<int,int> vertices;
        std::vector<Edge> edge_vertices(mesh.topology_edge_count);
        for(const auto& element:mesh.elements)for(int e=0;e<3;++e) {
            int a=element.topology_vertices[local[e][0]],b=element.topology_vertices[local[e][1]];
            if(a>b)std::swap(a,b);
            vertices[a]=-1;vertices[b]=-1;
            by_edge[{a,b}]=element.topology_edges[e];
            edge_vertices[element.topology_edges[e]]={a,b};
        }
        std::map<Key,int> by_position;
        for(const auto& vertex:vertices)
            if(!by_position.emplace(key(mesh.nodes[vertex.first]),vertex.first).second)
                throw std::runtime_error("D6h warm start: duplicate vertex position");
        for(auto& vertex:vertices) {
            const Vec3 p=mesh.nodes[vertex.first];
            const double sine=std::sqrt(3.)*.5;
            const Vec3 q=vertical?Vec3(p.x,-p.y,p.z):reflection?Vec3(p.x,p.y,-p.z):Vec3(.5*p.x-sine*p.y,sine*p.x+.5*p.y,p.z);
            const Key center=key(q);
            int target=-1;
            const auto exact=by_position.find(center);
            if(exact!=by_position.end())target=exact->second;
            else {
                for(int dx=-1;dx<=1;++dx)for(int dy=-1;dy<=1;++dy)for(int dz=-1;dz<=1;++dz) {
                    const Key candidate={{center[0]+dx,center[1]+dy,center[2]+dz}};
                    const auto found=by_position.find(candidate);
                    if(found!=by_position.end() && (mesh.nodes[found->second]-q).norm()<1e-9)
                        target=found->second;
                }
            }
            if(target<0)throw std::runtime_error("D6h warm start: mesh is not invariant");
            vertex.second=target;
        }
        Map result;
        result.source.assign(mesh.topology_edge_count,-1);
        result.sign.resize(mesh.topology_edge_count);
        for(int source=0;source<mesh.topology_edge_count;++source) {
            const auto endpoints=edge_vertices[source];
            const int a=vertices.at(endpoints.first),b=vertices.at(endpoints.second);
            const auto found=by_edge.find({std::min(a,b),std::max(a,b)});
            if(found==by_edge.end() || result.source[found->second]>=0)
                throw std::runtime_error("D6h warm start: edge mapping is not bijective");
            result.source[found->second]=source;
            result.sign[found->second]=a<b?1:-1;
        }
        return result;
    }
public:
    void build(const MullerP2Mesh& mesh) {
        if(mesh.basis_kind!=MullerBasisKind::HDivBdm1)
            throw std::runtime_error("D6h warm start requires BDM1 edge moments");
        edges_=mesh.topology_edge_count;
        rotation_=make_map(mesh,false);reflection_=make_map(mesh,true);
    }
    int size() const { return 4*edges_; }
    // Extra generator of D6h outside the commuting C6 x horizontal-mirror
    // subgroup. Its equation signs are the negatives of these current signs.
    static void vertical_image_map(const MullerP2Mesh& mesh,std::vector<int>& target,std::vector<int>& sign) {
        if(mesh.basis_kind!=MullerBasisKind::HDivBdm1)throw std::invalid_argument("vertical map requires BDM1");
        D6hOrbit helper;helper.edges_=mesh.topology_edge_count;
        helper.reflection_=make_map(mesh,true,true);helper.image_map(true,false,target,sign);
    }
    // Source-to-target signed permutation, also used by orbit-local inverses.
    void image_map(bool reflection,bool rhs,std::vector<int>& target,std::vector<int>& sign) const {
        target.resize(size());sign.resize(size());
        const Map& map=reflection?reflection_:rotation_;
        for(int edge=0;edge<edges_;++edge)for(int current=0;current<2;++current) {
            const int offset=current*2*edges_;
            const int parity=reflection?((current==0?-1:1)*(rhs?-1:1)):1;
            for(int moment=0;moment<2;++moment) {
                const int source=offset+2*map.source[edge]+moment;
                target[source]=offset+2*edge+moment;
                sign[source]=parity*(moment?1:map.sign[edge]);
            }
        }
    }
    void transform(const Vector& source,Vector& target,bool reflection,bool rhs) const {
        if(int(source.size())!=size() || &source==&target)
            throw std::invalid_argument("invalid D6h transform buffers");
        target.resize(size());
        const Map& map=reflection?reflection_:rotation_;
#pragma omp parallel for schedule(static)
        for(int edge=0;edge<edges_;++edge)for(int current=0;current<2;++current) {
            const int offset=current*2*edges_;
            const double parity=reflection?((current==0?-1.:1.)*(rhs?-1.:1.)):1.;
            target[offset+2*edge]=parity*map.sign[edge]*source[offset+2*map.source[edge]];
            target[offset+2*edge+1]=parity*source[offset+2*map.source[edge]+1];
        }
    }
};

struct WarmStartResult {
    bool fitted=false;
    int rank=0;
    double relative_rhs_error=1.,amplification=0.,seconds=0.;
};

// Fit RHS2 in the twelve-element orbit of RHS1. Carry the QR coefficients
// through to the orbit of solution1, without storing twelve solution vectors.
// This is only an initial guess: GMRES must check its complete FP64 residual.
inline WarmStartResult symmetry_guess(const D6hOrbit& group,const Vector& rhs1,
    const Vector& solution1,const Vector& rhs2,Vector& guess) {
    const auto begin=std::chrono::steady_clock::now();
    const int n=group.size();
    if(int(rhs1.size())!=n || int(rhs2.size())!=n || int(solution1.size())!=n)
        throw std::invalid_argument("D6h warm start dimensions disagree");
    WarmStartResult result;
    const double norm1=bem_streaming::norm(rhs1.data(),n),norm2=bem_streaming::norm(rhs2.data(),n);
    guess.assign(n,0.);
    if(norm1==0 || norm2==0)return result;
    std::vector<Vector> basis;
    std::vector<std::array<Complex,12>> transforms;
    Vector orbit=rhs1,scratch;
    for(int g=0;g<12;++g) {
        if(g==6)group.transform(rhs1,orbit,true,true);
        else if(g%6) { group.transform(orbit,scratch,false,true);orbit.swap(scratch); }
        Vector v=orbit;
        std::array<Complex,12> coefficients={};coefficients[g]=1.;
        for(int pass=0;pass<2;++pass)for(size_t j=0;j<basis.size();++j) {
            const Complex a=bem_streaming::dot(basis[j].data(),v.data(),n);
#pragma omp parallel for schedule(static)
            for(int i=0;i<n;++i)v[i]-=a*basis[j][i];
            for(int k=0;k<12;++k)coefficients[k]-=a*transforms[j][k];
        }
        const double length=bem_streaming::norm(v.data(),n);
        if(length<1e-8*norm1)continue;
#pragma omp parallel for schedule(static)
        for(int i=0;i<n;++i)v[i]/=length;
        for(auto& value:coefficients)value/=length;
        basis.push_back(std::move(v));transforms.push_back(coefficients);
    }
    result.rank=int(basis.size());
    Vector residual=rhs2;
    std::array<Complex,12> coefficients={};
    for(size_t j=0;j<basis.size();++j) {
        const Complex a=bem_streaming::dot(basis[j].data(),rhs2.data(),n);
#pragma omp parallel for schedule(static)
        for(int i=0;i<n;++i)residual[i]-=a*basis[j][i];
        for(int k=0;k<12;++k)coefficients[k]+=a*transforms[j][k];
    }
    result.relative_rhs_error=bem_streaming::norm(residual.data(),n)/norm2;
    for(const auto& value:coefficients)result.amplification+=std::abs(value)*norm1/norm2;
    result.fitted=std::isfinite(result.relative_rhs_error) && result.relative_rhs_error<1e-7 &&
        std::isfinite(result.amplification) && result.amplification<32.;
    if(result.fitted) {
        orbit=solution1;
        for(int g=0;g<12;++g) {
            if(g==6)group.transform(solution1,orbit,true,false);
            else if(g%6) { group.transform(orbit,scratch,false,false);orbit.swap(scratch); }
#pragma omp parallel for schedule(static)
            for(int i=0;i<n;++i)guess[i]+=coefficients[g]*orbit[i];
        }
    }
    result.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    return result;
}
}
#endif
