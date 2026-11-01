#pragma once
#include "prism_symmetry_sectors.h"
#include <queue>
#include <exception>

// Experimental geometric row-SPAI. Each row minimizes ||m A - e_i^T|| on
// a larger test stencil than its permitted nonzero stencil. This is NOT a
// sum of overlapping square block inverses. Construction uses local BDM1
// entries only; application is independent of the FFT/FMM operator.
namespace bem_spai {
using C=std::complex<double>;
using V=std::vector<C>;

class NearestEdges {
    struct Node { int edge,left,right,axis; };
    std::vector<Vec3> points_;
    std::vector<Node> tree_;
    int root_=-1;
    static double coordinate(const Vec3& p,int axis){return axis==0?p.x:axis==1?p.y:p.z;}
    int split(std::vector<int>& ids,int first,int last,int depth){
        if(first==last)return -1;
        const int mid=(first+last)/2,axis=depth%3;
        std::nth_element(ids.begin()+first,ids.begin()+mid,ids.begin()+last,[&](int a,int b){
            const double x=coordinate(points_[a],axis),y=coordinate(points_[b],axis);return x<y||(x==y&&a<b);});
        const int node=tree_.size();tree_.push_back({ids[mid],-1,-1,axis});
        const int left=split(ids,first,mid,depth+1),right=split(ids,mid+1,last,depth+1);
        tree_[node].left=left;tree_[node].right=right;return node;
    }
    using Neighbor=std::pair<double,int>;
    void query(int node,const Vec3& p,int count,std::priority_queue<Neighbor>& heap)const{
        if(node<0)return;
        const auto& q=tree_[node];const auto d=points_[q.edge]-p;
        const Neighbor candidate{d.x*d.x+d.y*d.y+d.z*d.z,q.edge};
        if(int(heap.size())<count)heap.push(candidate);
        else if(candidate<heap.top()){heap.pop();heap.push(candidate);}
        const double delta=coordinate(p,q.axis)-coordinate(points_[q.edge],q.axis);
        query(delta<0?q.left:q.right,p,count,heap);
        if(int(heap.size())<count || delta*delta<=heap.top().first)query(delta<0?q.right:q.left,p,count,heap);
    }
public:
    explicit NearestEdges(std::vector<Vec3> points):points_(std::move(points)){
        std::vector<int> ids(points_.size());std::iota(ids.begin(),ids.end(),0);tree_.reserve(ids.size());root_=split(ids,0,ids.size(),0);
    }
    std::vector<int> nearest(int seed,int count)const{
        if(seed<0||seed>=int(points_.size())||count<1)throw std::invalid_argument("SPAI nearest stencil");
        count=std::min(count,int(points_.size()));std::priority_queue<Neighbor> heap;query(root_,points_[seed],count,heap);
        std::vector<Neighbor> ordered;while(!heap.empty()){ordered.push_back(heap.top());heap.pop();}
        std::sort(ordered.begin(),ordered.end());std::vector<int> result{seed};
        for(const auto& p:ordered)if(p.second!=seed && int(result.size())<count)result.push_back(p.second);
        return result;
    }
};

// Column-pivoted, twice-reorthogonalized QR. Four right-hand sides are solved
// together. No normal equations: their squared condition number is avoidable.
// A and B are column-major; returned X is column-major. Rank loss is explicit.
inline V least_squares(V a,int rows,int cols,const V& b,int nrhs,int& rank,double rcond=1e-11){
    if(rows<cols||cols<1||nrhs<1||a.size()!=size_t(rows)*cols||b.size()!=size_t(rows)*nrhs||!(rcond>0&&rcond<1))
        throw std::invalid_argument("SPAI QR dimensions");
    std::vector<int> permutation(cols);std::iota(permutation.begin(),permutation.end(),0);
    V r(size_t(cols)*cols),qtb(size_t(cols)*nrhs),x(size_t(cols)*nrhs);
    rank=0;double scale=0;
    for(int k=0;k<cols;++k){
        int pivot=k;double norm2=-1;
        for(int j=k;j<cols;++j){double n=0;for(int i=0;i<rows;++i)n+=std::norm(a[size_t(j)*rows+i]);
            if(!std::isfinite(n))throw std::runtime_error("nonfinite SPAI matrix");
            if(n>norm2){norm2=n;pivot=j;}}
        const double length=std::sqrt(norm2);if(k==0)scale=length;
        if(length<=rcond*scale || length==0)break;
        if(pivot!=k){for(int i=0;i<rows;++i)std::swap(a[size_t(k)*rows+i],a[size_t(pivot)*rows+i]);
            for(int i=0;i<k;++i)std::swap(r[i*cols+k],r[i*cols+pivot]);
            std::swap(permutation[k],permutation[pivot]);}
        C* q=a.data()+size_t(k)*rows;for(int i=0;i<rows;++i)q[i]/=length;r[k*cols+k]=length;
        for(int h=0;h<nrhs;++h)for(int i=0;i<rows;++i)qtb[size_t(h)*cols+k]+=std::conj(q[i])*b[size_t(h)*rows+i];
        for(int j=k+1;j<cols;++j)for(int pass=0;pass<2;++pass){C dot=0;C* v=a.data()+size_t(j)*rows;
            for(int i=0;i<rows;++i)dot+=std::conj(q[i])*v[i];
            r[k*cols+j]+=dot;
            for(int i=0;i<rows;++i)v[i]-=dot*q[i];}
        ++rank;
    }
    for(int h=0;h<nrhs;++h){V solution(cols);
        for(int i=rank-1;i>=0;--i){C v=qtb[size_t(h)*cols+i];for(int j=i+1;j<rank;++j)v-=r[i*cols+j]*solution[j];solution[i]=v/r[i*cols+i];}
        for(int i=0;i<cols;++i){if(!std::isfinite(std::abs(solution[i])))throw std::runtime_error("nonfinite SPAI solution");x[size_t(h)*cols+permutation[i]]=solution[i];}}
    return x;
}

class PrismSparseInverse {
    struct Patch { V coefficients;std::vector<int> inputs;std::vector<int8_t> input_signs;
        std::array<int,96> outputs;std::array<int8_t,96> output_signs;double weight=0;int rank=0;double defect=0; };
    std::vector<Patch> patches_;
    int n_=0,columns_=0;
public:
    using Assemble=std::function<V(const std::vector<int>&)>;
    double setup_seconds=0,maximum_fit_defect=0;
    int rank_deficient_patches=0;
    size_t patches()const{return patches_.size();}
    size_t storage_bytes()const{size_t bytes=0;for(const auto& p:patches_)bytes+=sizeof(Patch)+p.coefficients.size()*sizeof(C)+p.inputs.size()*sizeof(int)+p.input_signs.size();return bytes;}
    void build(const MullerP2Mesh& mesh,const bem_tmatrix::PrismGroupMaps& group,int support_edges,int test_edges,const Assemble& assemble){
        const auto start=std::chrono::steady_clock::now();
        if(mesh.basis_kind!=MullerBasisKind::HDivBdm1||support_edges<1||test_edges<support_edges)throw std::invalid_argument("SPAI BDM1/stencil parameters");
        n_=4*mesh.topology_edge_count;const int edges=n_/4;
        for(int g=0;g<12;++g)if(int(group.image[g].size())!=n_||int(group.sign[g].size())!=n_)throw std::invalid_argument("SPAI group dimensions");
        support_edges=std::min(support_edges,edges);test_edges=std::min(test_edges,edges);columns_=4*support_edges;
        std::vector<int> vertical,vs;bem_tmatrix::D6hOrbit::vertical_image_map(mesh,vertical,vs);
        const auto image=[&](int g,int i){return g<12?group.image[g][i]:vertical[group.image[g-12][i]];};
        const auto sign=[&](int g,int i){return g<12?int(group.sign[g][i]):vs[group.image[g-12][i]]*int(group.sign[g-12][i]);};
        std::vector<bool> seen(edges,false);std::vector<int> seeds;std::vector<Vec3> points;
        for(int e=0;e<edges;++e){points.push_back(mesh.current_dof_points[2*e]);if(!seen[e]){seeds.push_back(e);for(int g=0;g<24;++g)seen[image(g,2*e)/2]=true;}}
        NearestEdges nearest(std::move(points));patches_.clear();patches_.resize(seeds.size());std::exception_ptr failure;
#pragma omp parallel for schedule(dynamic,1)
        for(size_t index=0;index<seeds.size();++index)try{
            const int seed=seeds[index],rows=4*test_edges,cols=columns_;const auto selected=nearest.nearest(seed,test_edges);
            const V square=assemble(selected);if(square.size()!=size_t(rows)*rows)throw std::runtime_error("SPAI assembly dimensions");
            V matrix(size_t(rows)*cols),rhs(size_t(rows)*4);
            // Transpose, not conjugate transpose: m A=e_i^T, x=m^T.
            for(int j=0;j<cols;++j)for(int i=0;i<rows;++i)matrix[size_t(j)*rows+i]=square[size_t(j)*rows+i];
            for(int h=0;h<4;++h)rhs[size_t(h)*rows+h]=1.;
            auto& p=patches_[index];p.coefficients=least_squares(matrix,rows,cols,rhs,4,p.rank);
            for(int h=0;h<4;++h){double residual=0;for(int i=0;i<rows;++i){C z=-rhs[size_t(h)*rows+i];
                for(int j=0;j<cols;++j)z+=matrix[size_t(j)*rows+i]*p.coefficients[size_t(h)*cols+j];
                residual+=std::norm(z);}p.defect=std::max(p.defect,std::sqrt(residual));}
            std::vector<int> members;for(int g=0;g<24;++g)members.push_back(image(g,2*seed)/2);
            std::sort(members.begin(),members.end());members.erase(std::unique(members.begin(),members.end()),members.end());p.weight=members.size()/24.;
            p.inputs.resize(24*cols);p.input_signs.resize(24*cols);
            for(int g=0;g<24;++g){const int det=((g%12)<6?1:-1)*(g<12?1:-1);
                for(int h=0;h<4;++h){const int dof=(h/2)*2*edges+2*seed+h%2;p.outputs[4*g+h]=image(g,dof);p.output_signs[4*g+h]=sign(g,dof);}
                for(int j=0;j<cols;++j){const int dof=((j%4)/2)*2*edges+2*selected[j/4]+j%2;
                    p.inputs[g*cols+j]=image(g,dof);p.input_signs[g*cols+j]=det*sign(g,dof);}}
        }catch(...){
#pragma omp critical(prism_spai_build_failure)
            {if(!failure)failure=std::current_exception();}
        }
        if(failure){patches_.clear();n_=0;std::rethrow_exception(failure);}
        rank_deficient_patches=0;maximum_fit_defect=0;
        for(const auto& p:patches_){rank_deficient_patches+=p.rank<columns_;maximum_fit_defect=std::max(maximum_fit_defect,p.defect);}
        setup_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    }
    void apply(const C* rhs,C* result)const{
        if(!n_)throw std::logic_error("SPAI not initialized");
        V copy;if(rhs==result){copy.assign(rhs,rhs+n_);rhs=copy.data();}
        std::fill(result,result+n_,C(0));
        // Distinct patches own disjoint full D6h output orbits; no atomics or
        // per-thread full-surface scatter buffers are needed.
#pragma omp parallel for schedule(static)
        for(size_t index=0;index<patches_.size();++index){const auto& p=patches_[index];
            for(int g=0;g<24;++g)for(int h=0;h<4;++h){C value=0;
                for(int j=0;j<columns_;++j)value+=p.coefficients[size_t(h)*columns_+j]*double(p.input_signs[g*columns_+j])*rhs[p.inputs[g*columns_+j]];
                result[p.outputs[4*g+h]]+=p.weight*double(p.output_signs[4*g+h])*value;}}
    }
};
} // namespace bem_spai
