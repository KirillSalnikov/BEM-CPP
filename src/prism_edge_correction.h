#ifndef BEM_PRISM_EDGE_CORRECTION_H
#define BEM_PRISM_EDGE_CORRECTION_H
#include "prism_symmetry_sectors.h"
#include <exception>
#include <functional>
#include <omp.h>

// Experimental overlapping Schwarz correction on both sides of sharp edges.
// Principal blocks come from the original BEM operator. This is a numerical
// local wedge model, not an asserted closed-form dielectric wedge solution.
// Average all 24 signed D6h images, then normalize by row multiplicity.
class PrismEdgeCorrection {
public:
    using C=std::complex<double>;
    using Assemble=std::function<std::vector<C>(const std::vector<int>&)>;
private:
    struct Block {std::vector<C> lu;std::vector<int> pivot,image;std::vector<int8_t> sign;};
    int n_=0,threads_=1;
    std::vector<Block> blocks_;
    std::vector<int> counts_,covered_;
    std::vector<int> scatter_index_;
    bool compact_workspace_=false;
    std::vector<std::vector<C>> workspace_;
    static void factor(Block& b,int n) {
        b.pivot.resize(n);
        for(int k=0;k<n;++k) {
            int p=k;for(int i=k+1;i<n;++i)if(std::abs(b.lu[i*n+k])>std::abs(b.lu[p*n+k]))p=i;
            if(std::abs(b.lu[p*n+k])<1e-24)throw std::runtime_error("singular edge patch");
            b.pivot[k]=p;if(p!=k)for(int j=0;j<n;++j)std::swap(b.lu[k*n+j],b.lu[p*n+j]);
            for(int i=k+1;i<n;++i){const C v=b.lu[i*n+k]/b.lu[k*n+k];b.lu[i*n+k]=v;
                for(int j=k+1;j<n;++j)b.lu[i*n+j]-=v*b.lu[k*n+j];}
        }
    }
    static void solve(const Block& b,C* x,int n) {
        for(int k=0;k<n;++k)if(k!=b.pivot[k])std::swap(x[k],x[b.pivot[k]]);
        for(int k=0;k<n;++k)for(int i=k+1;i<n;++i)x[i]-=b.lu[i*n+k]*x[k];
        for(int k=n-1;k>=0;--k){for(int j=k+1;j<n;++j)x[k]-=b.lu[k*n+j]*x[j];x[k]/=b.lu[k*n+k];}
    }
public:
    int sharp_edges=0,max_block_dofs=0;
    double setup_s=0;
    int size()const{return n_;}
    size_t blocks()const{return blocks_.size();}
    size_t covered()const{return covered_.size();}
    bool covers(int i)const{return counts_.at(i)>0;}
    bool compact_workspace()const{return compact_workspace_;}
    size_t scatter_workspace_bytes()const {
        size_t bytes=scatter_index_.size()*sizeof(int);
        for(const auto& w:workspace_)bytes+=w.size()*sizeof(C);return bytes;
    }
    size_t bytes()const {
        size_t bytes=sizeof(int)*(counts_.size()+covered_.size()+scatter_index_.size());
        for(const auto& w:workspace_)bytes+=sizeof(C)*w.size();
        for(const auto& b:blocks_)bytes+=sizeof(C)*b.lu.size()+sizeof(int)*(b.pivot.size()+b.image.size())+b.sign.size();
        return bytes;
    }
    void build(const MullerP2Mesh& mesh,const bem_tmatrix::PrismGroupMaps& group,int rings,const Assemble& assemble,
               bool compact_workspace=true) {
        const auto start=std::chrono::steady_clock::now();
        if(mesh.basis_kind!=MullerBasisKind::HDivBdm1||rings< -3||rings>5)throw std::invalid_argument("edge patch parameters");
        // Negative depth selects entire geometric ribs before expanding the
        // support. It retains nonlocal coupling along each rib in one block.
        const bool whole_rib=rings<0;const int depth=std::abs(rings);
        n_=mesh.system_dofs();const int edges=mesh.topology_edge_count;
        std::vector<int> vertical,vertical_sign;
        bem_tmatrix::D6hOrbit::vertical_image_map(mesh,vertical,vertical_sign);
        std::vector<std::vector<int>> image(24),sign(24),support(edges),neighbors(edges);
        for(int g=0;g<24;++g){image[g].resize(n_);sign[g].resize(n_);
            for(int i=0;i<n_;++i){const int j=group.image[g%12][i];
                image[g][i]=g<12?j:vertical[j];sign[g][i]=group.sign[g%12][i]*(g<12?1:vertical_sign[j]);}}
        std::vector<Vec3> normal(mesh.elements.size()),face_normals;
        std::vector<int> face(mesh.elements.size());
        for(size_t t=0;t<mesh.elements.size();++t){normal[t]=evaluate_muller_frame(mesh,t,1./3,1./3).normal;
            int f=0;for(;f<int(face_normals.size());++f)if(normal[t].dot(face_normals[f])>1-1e-10)break;
            if(f==int(face_normals.size()))face_normals.push_back(normal[t]);
            face[t]=f;
            for(int e:mesh.elements[t].topology_edges){support[e].push_back(t);
                for(int f:mesh.elements[t].topology_edges)if(e!=f)neighbors[e].push_back(f);}}
        std::vector<bool> sharp(edges,false),visited(edges,false);sharp_edges=0;
        for(int e=0;e<edges;++e){if(support[e].size()!=2)throw std::runtime_error("edge patch requires a closed mesh");
            sharp[e]=normal[support[e][0]].dot(normal[support[e][1]])<.99;sharp_edges+=sharp[e];}
        std::vector<std::vector<int>> seeds;
        const auto face_pair=[&](int e){return std::minmax(face[support[e][0]],face[support[e][1]]);};
        for(int e=0;e<edges;++e)if(sharp[e]&&!visited[e]){
            std::vector<int> initial;
            if(whole_rib){const auto pair=face_pair(e);for(int f=0;f<edges;++f)if(sharp[f]&&face_pair(f)==pair)initial.push_back(f);}
            else initial.push_back(e);
            seeds.push_back(initial);
            for(int edge:initial)for(int g=0;g<24;++g){const int f=image[g][2*edge]/2;
                if(!sharp[f])throw std::runtime_error("sharp-edge mask breaks D6h");
                visited[f]=true;}}
        blocks_.clear();blocks_.resize(seeds.size());counts_.assign(n_,0);covered_.clear();max_block_dofs=0;
        std::exception_ptr failure;
#pragma omp parallel for schedule(dynamic,1)
        for(size_t p=0;p<seeds.size();++p)try {
            std::vector<int> selected=seeds[p],frontier=seeds[p];
            for(int step=0;step<depth;++step){std::vector<int> next;
                for(int e:frontier)for(int f:neighbors[e])if(std::find(selected.begin(),selected.end(),f)==selected.end()){
                    selected.push_back(f);next.push_back(f);}frontier.swap(next);}
            std::sort(selected.begin(),selected.end());auto& b=blocks_[p];const int d=4*selected.size();
            b.lu=assemble(selected);if(b.lu.size()!=size_t(d)*d)throw std::runtime_error("edge block dimension");factor(b,d);
            b.image.resize(24*d);b.sign.resize(24*d);
            for(size_t e=0;e<selected.size();++e)for(int c=0;c<4;++c){const int dof=(c/2)*2*edges+2*selected[e]+c%2;
                for(int g=0;g<24;++g){b.image[g*d+4*e+c]=image[g][dof];b.sign[g*d+4*e+c]=sign[g][dof];}}
        }catch(...){
#pragma omp critical(edge_build_failure)
            {if(!failure)failure=std::current_exception();}}
        if(failure)std::rethrow_exception(failure);
        for(const auto& b:blocks_){max_block_dofs=std::max(max_block_dofs,int(b.pivot.size()));for(int i:b.image)++counts_[i];}
        for(int i=0;i<n_;++i){if(counts_[i])covered_.push_back(i);
            for(int g=0;g<24;++g)if(counts_[i]!=counts_[image[g][i]])throw std::runtime_error("edge coverage breaks D6h");}
        threads_=omp_get_max_threads();
        compact_workspace_=compact_workspace && sizeof(int)*size_t(n_)+sizeof(C)*threads_*covered_.size()<sizeof(C)*threads_*size_t(n_);
        std::vector<int>().swap(scatter_index_);
        if(compact_workspace_){scatter_index_.assign(n_,-1);for(size_t j=0;j<covered_.size();++j)scatter_index_[covered_[j]]=int(j);}
        // Release a previous dense scatter allocation when rebuilding the
        // same inverse; vector assignment alone can retain its old capacity.
        workspace_.clear();
        workspace_.assign(threads_,std::vector<C>(compact_workspace_?covered_.size():size_t(n_)));
        setup_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    }
    // Not reentrant: each instance owns its per-thread scatter workspace.
    void apply(const C* rhs,C* result) {
        if(!n_||rhs==result||omp_get_max_threads()>threads_)throw std::invalid_argument("edge apply arguments/threads");
        // Clear every workspace even if the runtime chooses fewer threads.
#pragma omp parallel for schedule(static)
        for(int t=0;t<threads_;++t)std::fill(workspace_[t].begin(),workspace_[t].end(),C(0));
#pragma omp parallel
        {
            auto& out=workspace_[omp_get_thread_num()];
            std::vector<C> v(max_block_dofs);
#pragma omp for schedule(static)
            for(size_t task=0;task<blocks_.size()*24;++task){const auto& b=blocks_[task/24];const int d=b.pivot.size(),g=task%24;
                const int rhs_parity=((g%12<6)?1:-1)*(g<12?1:-1);
                for(int i=0;i<d;++i)v[i]=double(rhs_parity*b.sign[g*d+i])*rhs[b.image[g*d+i]];
                solve(b,v.data(),d);
                for(int i=0;i<d;++i){const int row=b.image[g*d+i];
                    out[compact_workspace_?scatter_index_[row]:row]+=double(b.sign[g*d+i])*v[i];}}
        }
        std::fill(result,result+n_,C(0));
#pragma omp parallel for schedule(static)
        for(size_t j=0;j<covered_.size();++j){const int i=covered_[j];C value=0;
            for(int t=0;t<threads_;++t)value+=workspace_[t][compact_workspace_?j:size_t(i)];
            result[i]=value/double(counts_[i]);}
    }
};
#endif
