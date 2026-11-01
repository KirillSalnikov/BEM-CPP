#ifndef BEM_PRISM_ORBIT_INVERSE_H
#define BEM_PRISM_ORBIT_INVERSE_H
#include "prism_symmetry_sectors.h"
#include <exception>
struct MullerFmmOperator;

// An additive block inverse on orbits of local patches. Each seed patch is
// factorized once. Its signed images cover the surface; stabilizer duplicates
// are averaged, including edges fixed by the horizontal mirror.
class PrismOrbitInverse {
public:
    using Complex=std::complex<double>;
    using Assemble=std::function<std::vector<Complex>(const std::vector<int>&)>;
private:
    struct Block {
        std::vector<Complex> lu;
        std::vector<int> pivots,images;
        std::vector<int8_t> signs;
        std::vector<double> weight;
    };
    std::vector<Block> blocks_;
    int n_=0;
    static void factor(Block& b,int n) {
        b.pivots.resize(n);
        for(int k=0;k<n;++k) {
            int pivot=k;for(int i=k+1;i<n;++i)if(std::abs(b.lu[i*n+k])>std::abs(b.lu[pivot*n+k]))pivot=i;
            if(std::abs(b.lu[pivot*n+k])<1e-24)throw std::runtime_error("singular orbit MBJ block");
            b.pivots[k]=pivot;if(pivot!=k)for(int j=0;j<n;++j)std::swap(b.lu[k*n+j],b.lu[pivot*n+j]);
            for(int i=k+1;i<n;++i){const Complex a=b.lu[i*n+k]/b.lu[k*n+k];b.lu[i*n+k]=a;for(int j=k+1;j<n;++j)b.lu[i*n+j]-=a*b.lu[k*n+j];}
        }
    }
    static void solve(const Block& b,Complex* v,int n) {
        for(int k=0;k<n;++k)if(b.pivots[k]!=k)std::swap(v[k],v[b.pivots[k]]);
        for(int k=0;k<n;++k)for(int i=k+1;i<n;++i)v[i]-=b.lu[i*n+k]*v[k];
        for(int k=n-1;k>=0;--k){for(int j=k+1;j<n;++j)v[k]-=b.lu[k*n+j]*v[j];v[k]/=b.lu[k*n+k];}
    }
public:
    double setup_seconds=0;
    static size_t estimated_storage_bytes(const bem_tmatrix::PrismGroupMaps& group,int system_dofs,int nodes){
        if(system_dofs<4 || system_dofs%4 || nodes<1 || group.image.size()!=12 || int(group.image[0].size())!=system_dofs)
            throw std::invalid_argument("orbit inverse storage dimensions");
        const int edges=system_dofs/4;std::vector<bool> visited(edges,false);size_t seeds=0;
        for(int e=0;e<edges;++e)if(!visited[e]){
            ++seeds;for(int g=0;g<12;++g)visited.at(group.image[g][2*e]/2)=true;
        }
        size_t bytes=0;
        for(size_t first=0;first<seeds;first+=nodes){
            const size_t d=4*std::min(size_t(nodes),seeds-first);
            bytes+=sizeof(Block)+sizeof(Complex)*d*d+sizeof(int)*13*d+12*d+sizeof(double)*d;
        }
        return bytes;
    }
    size_t block_count()const{return blocks_.size();}
    size_t storage_bytes()const {
        size_t total=0;for(const auto& b:blocks_)total+=sizeof(Block)+sizeof(Complex)*b.lu.size()+
            sizeof(int)*(b.pivots.size()+b.images.size())+b.signs.size()+sizeof(double)*b.weight.size();return total;
    }
    void build(const MullerFmmOperator& op,const bem_tmatrix::PrismGroupMaps& group,int nodes);
    void build(const MullerP2Mesh& mesh,const bem_tmatrix::PrismGroupMaps& group,int nodes,const Assemble& assemble) {
        const auto start=std::chrono::steady_clock::now();
        if(nodes<1 || mesh.basis_kind!=MullerBasisKind::HDivBdm1)throw std::invalid_argument("orbit inverse basis/block size");
        n_=4*mesh.topology_edge_count;const int edges=n_/4;
        if(int(group.image[0].size())!=n_)throw std::invalid_argument("orbit inverse group size");
        std::vector<bool> visited(edges,false);std::vector<int> seeds,cardinality(edges,0);
        const double pi=std::acos(-1.);
        for(int e=0;e<edges;++e)if(!visited[e]) {
            std::vector<int> members;for(int g=0;g<12;++g)members.push_back(group.image[g][2*e]/2);
            std::sort(members.begin(),members.end());members.erase(std::unique(members.begin(),members.end()),members.end());
            int seed=-1;double height=-1e100,angle=1e100;
            for(int member:members) {
                visited[member]=true;cardinality[member]=members.size();
                const auto& p=mesh.current_dof_points[2*member];double a=std::atan2(p.y,p.x);
                if(a<0)a+=2*pi;
                if(a>2*pi-1e-9)a=0;
                if(p.z>height+1e-9 || (std::abs(p.z-height)<1e-9 && a<angle-1e-9)) {seed=member;height=p.z;angle=a;}
            }
            seeds.push_back(seed);
        }
        // Local Morton ordering in the representative wedge, not full-surface
        // ordering followed by a symmetry projection.
        double lo[3]={1e100,1e100,1e100},hi[3]={-1e100,-1e100,-1e100};
        for(int e:seeds){const auto& p=mesh.current_dof_points[2*e];const double v[3]={p.x,p.y,p.z};for(int a=0;a<3;++a){lo[a]=std::min(lo[a],v[a]);hi[a]=std::max(hi[a],v[a]);}}
        std::vector<std::pair<uint64_t,int>> ordered;
        for(int e:seeds) {
            const auto& p=mesh.current_dof_points[2*e];const double v[3]={p.x,p.y,p.z};uint64_t code=0;
            for(int a=0;a<3;++a) {
                const uint32_t q=uint32_t(std::llround((hi[a]>lo[a]?(v[a]-lo[a])/(hi[a]-lo[a]):0)*((1u<<20)-1)));
                for(int bit=0;bit<20;++bit)code|=uint64_t((q>>bit)&1)<<(3*bit+a);
            }
            ordered.emplace_back(code,e);
        }
        std::sort(ordered.begin(),ordered.end());
        blocks_.clear();blocks_.resize((seeds.size()+nodes-1)/nodes);
        std::exception_ptr failure;
#pragma omp parallel for schedule(dynamic,1)
        for(size_t k=0;k<blocks_.size();++k)try {
            std::vector<int> selected;for(size_t j=k*nodes;j<std::min(ordered.size(),(k+1)*nodes);++j)selected.push_back(ordered[j].second);
            auto& b=blocks_[k];const int d=4*selected.size();
            b.images.resize(12*d);b.signs.resize(12*d);b.weight.resize(d);
            for(size_t j=0;j<selected.size();++j)for(int channel=0;channel<4;++channel) {
                const int dof=(channel/2)*2*edges+2*selected[j]+channel%2,local=4*j+channel;
                b.weight[local]=cardinality[selected[j]]/12.;
                for(int g=0;g<12;++g){b.images[g*d+local]=group.image[g][dof];b.signs[g*d+local]=group.sign[g][dof];}
            }
            b.lu=assemble(selected);if(int(b.lu.size())!=d*d)throw std::runtime_error("orbit block assembly size");factor(b,d);
        }catch(...) {
#pragma omp critical(prism_orbit_build_failure)
            {if(!failure)failure=std::current_exception();}
        }
        if(failure)std::rethrow_exception(failure);
        setup_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    }
    void apply(const Complex* rhs,Complex* result)const {
        if(!n_)throw std::logic_error("orbit inverse is not initialized");
        std::vector<Complex> seed;if(rhs==result){seed.assign(rhs,rhs+n_);rhs=seed.data();}
        std::fill(result,result+n_,Complex(0));
        // Different seed blocks contain disjoint edge orbits. Mirror-fixed
        // rows overlap only inside one block and are accumulated serially.
#pragma omp parallel for schedule(static)
        for(size_t k=0;k<blocks_.size();++k) {
            const auto& b=blocks_[k];const int d=b.pivots.size();std::vector<Complex> v(d);
            for(int g=0;g<12;++g) {
                for(int i=0;i<d;++i)v[i]=double((g<6?1:-1)*b.signs[g*d+i])*rhs[b.images[g*d+i]];
                solve(b,v.data(),d);
                for(int i=0;i<d;++i)result[b.images[g*d+i]]+=b.weight[i]*double(b.signs[g*d+i])*v[i];
            }
        }
    }
};
#endif
