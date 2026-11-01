#pragma once
#include "muller_taylor_kernel.h"
#include "muller_nodal.h"
#include "muller_duffy.h"
#include "quadrature.h"
#include <cstdint>
#include <map>
#include <set>

namespace bem_taylor {
// Sparse Duffy-minus-regular derivative corrections, normalized to the final
// Muller blocks (i/k is already included in diagonal). Only adjacency pairs
// are integrated; rigidly congruent BDM pairs share all frequency moments.
template<size_t N> class SparseCorrection {
public:
    using Series=std::array<C,N+1>;
    struct Values{Series diagonal{},epsilon{},mu{};};
    struct Entry{int column;Values value;};
    std::vector<int> offsets;
    std::vector<Entry> entries;
    size_t pairs=0,templates=0;
private:
    struct Adjacency{int count=0;int t[3]{-1,-1,-1},s[3]{-1,-1,-1};};
    static Adjacency adjacent(const MullerP2Element& t,const MullerP2Element& s){
        Adjacency a;for(int i=0;i<3;++i)for(int j=0;j<3;++j)if(t.topology_vertices[i]==s.topology_vertices[j]){
            a.t[a.count]=i;a.s[a.count++]=j;}return a;
    }
    static void remap(MullerDuffyPoint& p,const Adjacency& a){
        if(a.count==1){muller_duffy_remap_shared_vertex(p.test_xi,p.test_eta,a.t[0]);muller_duffy_remap_shared_vertex(p.trial_xi,p.trial_eta,a.s[0]);}
        if(a.count==2){muller_duffy_remap_shared_edge(p.test_xi,p.test_eta,a.t[0],a.t[1]);muller_duffy_remap_shared_edge(p.trial_xi,p.trial_eta,a.s[0],a.s[1]);}
    }
    using Key=std::vector<std::int64_t>;
    static Key key(const MullerP2Mesh& mesh,int t,int s,const Adjacency& a,double quantization){
        const auto& e=mesh.elements[t];const Vec3 origin=mesh.nodes[e.nodes[0]],d1=mesh.nodes[e.nodes[1]]-origin,d2=mesh.nodes[e.nodes[2]]-origin;
        const Vec3 u=d1*(1/d1.norm()),normal=d1.cross(d2)*(1/d1.cross(d2).norm()),v=normal.cross(u);
        Key k{a.count};for(int j=0;j<3;++j){k.push_back(a.t[j]);k.push_back(a.s[j]);}
        for(int element:{t,s})for(int node:mesh.elements[element].nodes){const Vec3 d=mesh.nodes[node]-origin;
            for(double x:{d.dot(u),d.dot(v),d.dot(normal)})k.push_back(std::llround(x/quantization));}
        return k;
    }
public:
    size_t storage_bytes()const{return offsets.size()*sizeof(int)+entries.size()*sizeof(Entry);}
    SparseCorrection(const MullerP2Mesh& mesh,double k,C m,int quad,int duffy,bool reuse=true){
        if(mesh.basis_kind!=MullerBasisKind::HDivBdm1||mesh.elements.empty())throw std::invalid_argument("Taylor correction requires BDM1 mesh");
        const auto regular=tri_quadrature(quad);
        const std::vector<MullerDuffyPoint> rules[3]={muller_duffy_rule(duffy,MullerDuffyAdjacency::Coincident),
            muller_duffy_rule(duffy,MullerDuffyAdjacency::EdgeAdjacent),muller_duffy_rule(duffy,MullerDuffyAdjacency::VertexAdjacent)};
        const int nt=mesh.elements.size(),n=mesh.current_dofs();
        int maxvertex=0;for(const auto& e:mesh.elements)for(int v:e.topology_vertices)maxvertex=std::max(maxvertex,v);
        std::vector<std::vector<int>> incident(maxvertex+1),neighbors(nt);
        for(int t=0;t<nt;++t)for(int v:mesh.elements[t].topology_vertices)incident[v].push_back(t);
        for(int t=0;t<nt;++t){for(int v:mesh.elements[t].topology_vertices)neighbors[t].insert(neighbors[t].end(),incident[v].begin(),incident[v].end());
            std::sort(neighbors[t].begin(),neighbors[t].end());neighbors[t].erase(std::unique(neighbors[t].begin(),neighbors[t].end()),neighbors[t].end());pairs+=neighbors[t].size();}
        std::vector<MullerBasisSample> center(nt);std::vector<MullerFrameSample> frames(size_t(nt)*regular.npts);
        for(int t=0;t<nt;++t){center[t]=evaluate_muller_basis(mesh,t,evaluate_muller_frame(mesh,t,1./3,1./3));
            if(center[t].count!=6)throw std::invalid_argument("Taylor BDM basis count");
            for(int q=0;q<regular.npts;++q)frames[size_t(t)*regular.npts+q]=evaluate_muller_frame(mesh,t,regular.pts[q][0],regular.pts[q][1]);}
        Vec3 lo=mesh.nodes.front(),hi=lo;for(const auto& v:mesh.nodes){lo.x=std::min(lo.x,v.x);lo.y=std::min(lo.y,v.y);lo.z=std::min(lo.z,v.z);
            hi.x=std::max(hi.x,v.x);hi.y=std::max(hi.y,v.y);hi.z=std::max(hi.z,v.z);}
        const double quantum=std::max(1.,(hi-lo).norm())*1e-11;
        std::map<Key,int> ids;std::vector<std::pair<int,int>> representatives;std::vector<std::vector<int>> mapping(nt);
        for(int t=0;t<nt;++t)for(int s:neighbors[t]){const auto a=adjacent(mesh.elements[t],mesh.elements[s]);
            Key identity=reuse?key(mesh,t,s,a,quantum):Key{t,s};auto found=ids.emplace(identity,int(representatives.size()));
            if(found.second)representatives.emplace_back(t,s);
            mapping[t].push_back(found.first->second);}
        templates=representatives.size();using Block=std::array<Values,36>;std::vector<Block> blocks(templates);
#pragma omp parallel for schedule(dynamic,1)
        for(size_t id=0;id<templates;++id){const int t=representatives[id].first,s=representatives[id].second;
            const auto a=adjacent(mesh.elements[t],mesh.elements[s]);auto& block=blocks[id];
            auto sample=[&](const MullerFrameSample& test,const MullerFrameSample& trial,double weight){
                const Vec3 d=test.position-trial.position;const double r=d.norm();if(r<1e-13)return;
                const Vec3 unit=d*(1/r);const auto kernel=radial<N>(k,m,r);
                const auto bt=evaluate_muller_basis(mesh,t,test),bs=evaluate_muller_basis(mesh,s,trial);weight*=test.jacobian*trial.jacobian;
                for(int row=0;row<6;++row)for(int col=0;col<6;++col){const Vec3 v=bt.values[row],w=bs.values[col],rot=v.cross(test.normal);
                    const double dot=rot.dot(w),along=rot.dot(unit)*w.dot(unit),grad=v.dot(w)*test.normal.dot(unit)-test.normal.dot(w)*v.dot(unit);
                    auto& value=block[row*6+col];for(size_t j=0;j<=N;++j){value.diagonal[j]+=weight*(dot*kernel.diagonal[j]+along*kernel.along[j]);
                        value.epsilon[j]+=weight*grad*kernel.epsilon[j];value.mu[j]+=weight*grad*kernel.mu[j];}}
            };
            for(auto p:rules[t==s?0:a.count==2?1:2]){remap(p,a);sample(evaluate_muller_frame(mesh,t,p.test_xi,p.test_eta),evaluate_muller_frame(mesh,s,p.trial_xi,p.trial_eta),p.weight);}
            for(int i=0;i<regular.npts;++i)for(int j=0;j<regular.npts;++j)sample(frames[size_t(t)*regular.npts+i],frames[size_t(s)*regular.npts+j],-.25*regular.wts[i]*regular.wts[j]);
        }
        // Colour target elements, making row-map assembly race-free.
        std::vector<std::vector<int>> colors;std::vector<std::set<int>> used;
        for(int t=0;t<nt;++t){size_t color=0;for(;color<colors.size();++color){bool conflict=false;for(int a=0;a<6;++a)conflict|=used[color].count(center[t].dofs[a]);if(!conflict)break;}
            if(color==colors.size()){colors.emplace_back();used.emplace_back();}colors[color].push_back(t);for(int a=0;a<6;++a)used[color].insert(center[t].dofs[a]);}
        std::vector<std::map<int,Values>> rows(n);
        for(const auto& color:colors){
#pragma omp parallel for schedule(static)
            for(size_t it=0;it<color.size();++it){const int t=color[it];
                for(size_t q=0;q<neighbors[t].size();++q){const int s=neighbors[t][q],id=mapping[t][q];const auto rep=representatives[id];
                    for(int a=0;a<6;++a)for(int b=0;b<6;++b){double sign=1;
                        if(a%2==0)sign*=mesh.elements[t].edge_orientations[a/2]*mesh.elements[rep.first].edge_orientations[a/2];
                        if(b%2==0)sign*=mesh.elements[s].edge_orientations[b/2]*mesh.elements[rep.second].edge_orientations[b/2];
                        auto& dst=rows[center[t].dofs[a]][center[s].dofs[b]];const auto& src=blocks[id][6*a+b];
                        for(size_t j=0;j<=N;++j){dst.diagonal[j]+=sign*src.diagonal[j];dst.epsilon[j]+=sign*src.epsilon[j];dst.mu[j]+=sign*src.mu[j];}
                    }
                }
            }
        }
        offsets.resize(n+1);for(int row=0;row<n;++row){offsets[row]=entries.size();for(const auto& item:rows[row])entries.push_back({item.first,item.second});}
        offsets[n]=entries.size(); // Retain derivative-only entries even when order zero cancels.
    }
};
}
