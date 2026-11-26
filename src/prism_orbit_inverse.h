#ifndef BEM_PRISM_ORBIT_INVERSE_H
#define BEM_PRISM_ORBIT_INVERSE_H
#include "prism_symmetry_sectors.h"
#include <exception>
#include <memory>
#include "prism_sparse_inverse.h"
struct MullerFmmOperator;

// An additive block inverse on orbits of local patches. Each seed patch is
// factorized once. Its signed images cover the surface; stabilizer duplicates
// are averaged, including edges fixed by the horizontal mirror.
class PrismOrbitInverse {
public:
    using Complex=std::complex<double>;
    using Assemble=std::function<std::vector<Complex>(const std::vector<int>&)>;
private:
    using ShortPair=std::array<int16_t,2>;
    struct Block {
        std::vector<Complex> lu;
        std::vector<std::complex<float>> compact_rows;
        std::vector<ShortPair> short_rows;
        std::vector<double> row_scale;
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
    using OrbitValues=std::array<Complex,12>;
    // The twelve images use the same LU and pivots. Streaming the factors
    // once also makes each inner update a contiguous set of independent RHS.
    static void solve_images(const Block& b,OrbitValues* values,int n) {
        for(int i=0;i<n;++i)if(b.pivots[i]!=i)std::swap(values[i],values[b.pivots[i]]);
        for(int i=0;i<n;++i) {
            auto& row=values[i];
            const Complex* lu=b.lu.data()+size_t(i)*n;
            for(int j=0;j<i;++j) {
                const Complex coefficient=lu[j];
                #pragma omp simd
                for(int g=0;g<12;++g)row[g]-=coefficient*values[j][g];
            }
        }
        for(int i=n-1;i>=0;--i) {
            auto& row=values[i];
            const Complex* lu=b.lu.data()+size_t(i)*n;
            for(int j=i+1;j<n;++j) {
                const Complex coefficient=lu[j];
                #pragma omp simd
                for(int g=0;g<12;++g)row[g]-=coefficient*values[j][g];
            }
            for(auto& v:row)v/=lu[i];
        }
    }
    template<class Scalar> static Complex coefficient(const Scalar& value,double){return Complex(value);}
    static Complex coefficient(const ShortPair& value,double scale){return Complex(value[0]*scale,value[1]*scale);}
    struct SplitOrbit {double real[12],imag[12];};
    template<class Scalar>
    static void apply_rows(const Scalar* rows,int core,int width,const SplitOrbit* input,OrbitValues* out,
                           const double* scale=nullptr) {
        // BDM1 has four rows per core edge. Two rows share the same twelve
        // gathered images and fit in SIMD registers during the dot products.
        for(int i=0;i<core;i+=2) {
            SplitOrbit sum[2]{};
            for(int j=0;j<width;++j)for(int k=0;k<2;++k) {
                const Complex v=coefficient(rows[size_t(i+k)*width+j],scale?scale[i+k]:1.);
                #pragma omp simd
                for(int g=0;g<12;++g){
                    sum[k].real[g]+=v.real()*input[j].real[g]-v.imag()*input[j].imag[g];
                    sum[k].imag[g]+=v.real()*input[j].imag[g]+v.imag()*input[j].real[g];
                }
            }
            for(int k=0;k<2;++k)for(int g=0;g<12;++g)
                out[i+k][g]=Complex(sum[k].real[g],sum[k].imag[g]);
        }
    }
    static void solve_adjoint(const Block& b,Complex* v,int n) {
        // PA=LU, so A^-H=P^T L^-H U^-H. Reverse the pivot sequence last.
        for(int k=0;k<n;++k){for(int j=0;j<k;++j)v[k]-=std::conj(b.lu[j*n+k])*v[j];v[k]/=std::conj(b.lu[k*n+k]);}
        for(int k=n-1;k>=0;--k)for(int j=k+1;j<n;++j)v[k]-=std::conj(b.lu[j*n+k])*v[j];
        for(int k=n-1;k>=0;--k)if(b.pivots[k]!=k)std::swap(v[k],v[b.pivots[k]]);
    }
public:
    double setup_seconds=0;
    static size_t estimated_storage_bytes(const bem_tmatrix::PrismGroupMaps& group,int system_dofs,int nodes,int halo=0,int row_bits=64){
        if((row_bits!=64 && row_bits!=32 && row_bits!=16) || system_dofs<4 || system_dofs%4 || nodes<1 || halo<0 || group.image.size()!=12 || int(group.image[0].size())!=system_dofs)
            throw std::invalid_argument("orbit inverse storage dimensions");
        const int edges=system_dofs/4;std::vector<bool> visited(edges,false);size_t seeds=0;
        for(int e=0;e<edges;++e)if(!visited[e]){
            ++seeds;for(int g=0;g<12;++g)visited.at(group.image[g][2*e]/2)=true;
        }
        size_t bytes=0;
        for(size_t first=0;first<seeds;first+=nodes){
            const size_t core_edges=std::min(size_t(nodes),seeds-first);
            const size_t core=4*core_edges,d=core+4*std::min(size_t(halo),seeds-core_edges);
            bytes+=sizeof(Block)+(core!=d?size_t(row_bits/4):sizeof(Complex))*core*d+((core!=d && row_bits==16)?sizeof(double)*core:0)+sizeof(int)*(12*d+(core==d?d:0))+12*d+sizeof(double)*core;
        }
        return bytes;
    }
    size_t block_count()const{return blocks_.size();}
    size_t storage_bytes()const {
        size_t total=0;for(const auto& b:blocks_)total+=sizeof(Block)+sizeof(Complex)*b.lu.size()+
            sizeof(std::complex<float>)*b.compact_rows.size()+sizeof(ShortPair)*b.short_rows.size()+sizeof(double)*b.row_scale.size()+sizeof(int)*(b.pivots.size()+b.images.size())+b.signs.size()+sizeof(double)*b.weight.size();return total;
    }
    void build(const MullerFmmOperator& op,const bem_tmatrix::PrismGroupMaps& group,int nodes,int halo=0,int row_bits=64);
    void build(const MullerP2Mesh& mesh,const bem_tmatrix::PrismGroupMaps& group,int nodes,const Assemble& assemble,int halo=0,int row_bits=64) {
        const auto start=std::chrono::steady_clock::now();
        if((row_bits!=64 && row_bits!=32 && row_bits!=16) || nodes<1 || halo<0 || mesh.basis_kind!=MullerBasisKind::HDivBdm1)throw std::invalid_argument("orbit inverse basis/block size");
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
        std::vector<Vec3> locations;
        std::unique_ptr<bem_spai::NearestEdges> nearby;
        if(halo) {
            locations.reserve(ordered.size());
            for(const auto& entry:ordered)locations.push_back(mesh.current_dof_points[2*entry.second]);
            nearby.reset(new bem_spai::NearestEdges(locations));
        }
        blocks_.clear();blocks_.resize((seeds.size()+nodes-1)/nodes);
        std::exception_ptr failure;
#pragma omp parallel for schedule(dynamic,1)
        for(size_t k=0;k<blocks_.size();++k)try {
            std::vector<int> selected;for(size_t j=k*nodes;j<std::min(ordered.size(),(k+1)*nodes);++j)selected.push_back(ordered[j].second);
            const int core=4*selected.size();
            if(nearby) {
                const size_t begin=k*nodes,end=begin+selected.size();
                std::map<int,double> closest;
                for(size_t j=begin;j<end;++j)for(int candidate:nearby->nearest(j,selected.size()+halo)) {
                    if(size_t(candidate)>=begin && size_t(candidate)<end)continue;
                    const Vec3 delta=locations[candidate]-locations[j];
                    const double distance=delta.x*delta.x+delta.y*delta.y+delta.z*delta.z;
                    auto item=closest.emplace(candidate,distance);
                    if(!item.second)item.first->second=std::min(distance,item.first->second);
                }
                std::vector<std::pair<double,int>> sorted;
                for(const auto& item:closest)sorted.emplace_back(item.second,item.first);
                const size_t count=std::min(size_t(halo),sorted.size());
                std::partial_sort(sorted.begin(),sorted.begin()+count,sorted.end());
                for(size_t j=0;j<count;++j)selected.push_back(ordered[sorted[j].second].second);
            }
            auto& b=blocks_[k];const int d=4*selected.size();
            b.images.resize(12*d);b.signs.resize(12*d);b.weight.resize(d);
            for(size_t j=0;j<selected.size();++j)for(int channel=0;channel<4;++channel) {
                const int dof=(channel/2)*2*edges+2*selected[j]+channel%2,local=4*j+channel;
                b.weight[local]=cardinality[selected[j]]/12.;
                for(int g=0;g<12;++g){b.images[g*d+local]=group.image[g][dof];b.signs[g*d+local]=group.sign[g][dof];}
            }
            b.lu=assemble(selected);if(int(b.lu.size())!=d*d)throw std::runtime_error("orbit block assembly size");factor(b,d);
            if(core<d) {
                // Only the core writes output. Store exactly these inverse rows.
                std::vector<Complex> rows,v(d);
                if(row_bits==16){b.short_rows.resize(size_t(core)*d);b.row_scale.resize(core);}
                else if(row_bits==32)b.compact_rows.resize(size_t(core)*d);
                else rows.resize(size_t(core)*d);
                for(int i=0;i<core;++i) {
                    std::fill(v.begin(),v.end(),Complex(0));v[i]=1.;
                    solve_adjoint(b,v.data(),d);
                    if(row_bits==16){
                        double peak=0;
                        for(const auto& z:v){
                            if(!std::isfinite(z.real()) || !std::isfinite(z.imag()))throw std::overflow_error("nonfinite restricted inverse row");
                            peak=std::max(peak,std::max(std::abs(z.real()),std::abs(z.imag())));
                        }
                        b.row_scale[i]=std::max(peak/32767.,std::numeric_limits<double>::min());
                    }
                    for(int j=0;j<d;++j){
                        const Complex value=std::conj(v[j]);
                        const size_t index=size_t(i)*d+j;
                        if(row_bits==16){
                            const double scale=b.row_scale[i];
                            b.short_rows[index]={{static_cast<int16_t>(std::round(value.real()/scale)),
                                                  static_cast<int16_t>(std::round(value.imag()/scale))}};
                        }else if(row_bits==32){
                            const double bound=std::numeric_limits<float>::max();
                            if(!(std::abs(value.real())<=bound && std::abs(value.imag())<=bound))
                                throw std::overflow_error("restricted orbit coefficient exceeds FP32 range");
                            b.compact_rows[index]=std::complex<float>(value);
                        }else rows[index]=value;
                    }
                }
                b.lu.swap(rows);std::vector<int>().swap(b.pivots);
                b.weight.resize(core);b.weight.shrink_to_fit();
            }
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
        // Output cores contain disjoint edge orbits. Mirror-fixed
        // rows overlap only inside one block and are accumulated serially.
#pragma omp parallel for schedule(static)
        for(size_t k=0;k<blocks_.size();++k) {
            const auto& b=blocks_[k];const int d=b.images.size()/12;
            const int core=b.weight.size();
            std::vector<OrbitValues> solved(core);
            if(b.pivots.empty()){
                std::vector<SplitOrbit> input(d);
                for(int j=0;j<d;++j)for(int g=0;g<12;++g){
                    const Complex v=double((g<6?1:-1)*b.signs[g*d+j])*rhs[b.images[g*d+j]];
                    input[j].real[g]=v.real();input[j].imag[g]=v.imag();
                }
                if(!b.short_rows.empty())apply_rows(b.short_rows.data(),core,d,input.data(),solved.data(),b.row_scale.data());
                else if(!b.compact_rows.empty())apply_rows(b.compact_rows.data(),core,d,input.data(),solved.data());
                else apply_rows(b.lu.data(),core,d,input.data(),solved.data());
            }else{
                for(int j=0;j<d;++j)for(int g=0;g<12;++g)
                    solved[j][g]=double((g<6?1:-1)*b.signs[g*d+j])*rhs[b.images[g*d+j]];
                solve_images(b,solved.data(),d);
            }
            for(int g=0;g<12;++g)for(int i=0;i<core;++i)
                result[b.images[g*d+i]]+=b.weight[i]*double(b.signs[g*d+i])*solved[i][g];
        }
    }
    void apply_adjoint(const Complex* rhs,Complex* result)const {
        if(!n_)throw std::logic_error("orbit inverse is not initialized");
        std::vector<Complex> seed;if(rhs==result){seed.assign(rhs,rhs+n_);rhs=seed.data();}
        std::fill(result,result+n_,Complex(0));
        // Halo contributions can share output entries in the adjoint.
        const bool restricted=std::any_of(blocks_.begin(),blocks_.end(),[](const Block& b){return b.pivots.empty();});
#pragma omp parallel for schedule(static) if(!restricted)
        for(size_t k=0;k<blocks_.size();++k){
            const auto& b=blocks_[k];const int d=b.images.size()/12;
            if(b.pivots.empty()){
                const int core=b.weight.size();
                for(int g=0;g<12;++g)for(int i=0;i<core;++i){
                    const Complex v=b.weight[i]*double(b.signs[g*d+i])*rhs[b.images[g*d+i]];
                    for(int j=0;j<d;++j){
                        const size_t index=size_t(i)*d+j;
                        const Complex coef=!b.short_rows.empty()?coefficient(b.short_rows[index],b.row_scale[i]):
                            !b.compact_rows.empty()?Complex(b.compact_rows[index]):b.lu[index];
                        result[b.images[g*d+j]]+=double((g<6?1:-1)*b.signs[g*d+j])*std::conj(coef)*v;
                    }
                }
                continue;
            }
            std::vector<Complex> v(d);
            for(int g=0;g<12;++g){
                for(int i=0;i<d;++i)v[i]=b.weight[i]*double(b.signs[g*d+i])*rhs[b.images[g*d+i]];
                solve_adjoint(b,v.data(),d);
                for(int i=0;i<d;++i)result[b.images[g*d+i]]+=double((g<6?1:-1)*b.signs[g*d+i])*v[i];
            }
        }
    }
    // Read-only export for a separately validated device implementation.
    // Preserve seed-block order, LU pivots, image signs and stabilizer weights.
    template<class Visitor> void visit_factorized_blocks(Visitor visitor)const {
        if(!n_)throw std::logic_error("orbit inverse is not initialized");
        for(const auto& block:blocks_)if(block.pivots.empty())throw std::logic_error("resident LU export requires disjoint orbit blocks");
        for(const auto& block:blocks_)
            visitor(n_,block.lu,block.pivots,block.images,block.signs,block.weight);
    }
};
#endif
