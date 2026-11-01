#pragma once
#include "muller_nodal.h"
#include <algorithm>
#include <cmath>
#include <exception>
#include <map>
#include <stdexcept>

// Canonical BDM1 edge moments of the source field on a second triangulation
// of the SAME planar faces. Split every target edge at source-triangle
// boundaries and integrate the linear trace exactly (two Gauss points).
// This is interpolation of currents, not a restriction of equation vectors.
class BdmMeshTransfer {
    struct Triangle {
        int element, face;
        Vec3 p, e1, e2;
        double g11, g12, g22, determinant;
        std::array<double,3> barycentric(const Vec3& x) const {
            const Vec3 r=x-p;
            const double a=r.dot(e1),b=r.dot(e2);
            const double xi=(a*g22-b*g12)/determinant,eta=(b*g11-a*g12)/determinant;
            return {{1-xi-eta,xi,eta}};
        }
    };
    struct Face {
        Vec3 origin, normal, u, v;
        double xmin=1e100,xmax=-1e100,ymin=1e100,ymax=-1e100,step=0,area=0;
        int nx=0,ny=0;
        std::vector<int> triangles;
        std::vector<std::vector<int>> bins;
        int ix(double x)const{return std::max(0,std::min(nx-1,int(std::floor((x-xmin)/step))));}
        int iy(double y)const{return std::max(0,std::min(ny-1,int(std::floor((y-ymin)/step))));}
        std::array<double,2> uv(const Vec3& p)const {const auto d=p-origin;return {{d.dot(u),d.dot(v)}};}
    };
    using Row=std::vector<std::pair<int,double>>;
    std::vector<Row> rows_;
    int source_current_=0,target_current_=0;
    bool ready_=false;
public:
    size_t bytes()const {size_t n=sizeof(Row)*rows_.size();for(const auto& r:rows_)n+=r.size()*sizeof(r[0]);return n;}
    int source_size()const{return 2*source_current_;}
    int target_size()const{return 2*target_current_;}
    void build(const MullerP2Mesh& source,const MullerP2Mesh& target) {
        ready_=false;
        if(source.basis_kind!=MullerBasisKind::HDivBdm1 || target.basis_kind!=MullerBasisKind::HDivBdm1)
            throw std::invalid_argument("BDM transfer requires edge moments");
        source_current_=source.current_dofs();target_current_=target.current_dofs();
        rows_.clear();rows_.resize(target_current_);
        std::vector<Triangle> triangles;
        std::vector<Face> faces;
        double scale=0;for(const auto& p:source.nodes)scale=std::max(scale,p.norm());
        if(!(scale>0))throw std::invalid_argument("empty source geometry");
        const double distance_tolerance=2e-10*scale;
        const auto face_of=[&](const Vec3& p,const Vec3& normal) {
            for(size_t j=0;j<faces.size();++j)
                if(normal.dot(faces[j].normal)>1-1e-12 &&
                   std::abs((p-faces[j].origin).dot(faces[j].normal))<distance_tolerance)return int(j);
            return -1;
        };
        for(size_t t=0;t<source.elements.size();++t) {
            const auto& e=source.elements[t];Triangle a;
            a.element=t;a.p=source.nodes[e.nodes[0]];
            a.e1=source.nodes[e.nodes[1]]-a.p;a.e2=source.nodes[e.nodes[2]]-a.p;
            const auto cross=a.e1.cross(a.e2);const double area2=cross.norm();
            if(!(area2>0))throw std::runtime_error("degenerate source triangle");
            const Vec3 normal=cross*(1/area2);
            a.face=face_of(a.p,normal);
            if(a.face<0){Face f;f.origin=a.p;f.normal=normal;f.u=a.e1*(1/a.e1.norm());f.v=normal.cross(f.u);
                a.face=faces.size();faces.push_back(std::move(f));}
            // Refuse curved P2 patches: the line clipping below is planar.
            for(int k=0;k<3;++k)if((source.nodes[e.nodes[3+k]]-
                (source.nodes[e.nodes[k]]+source.nodes[e.nodes[(k+1)%3]])*.5).norm()>distance_tolerance)
                throw std::invalid_argument("BDM transfer supports flat triangles only");
            a.g11=a.e1.dot(a.e1);a.g12=a.e1.dot(a.e2);a.g22=a.e2.dot(a.e2);
            a.determinant=a.g11*a.g22-a.g12*a.g12;
            auto& f=faces[a.face];f.area+=area2*.5;f.triangles.push_back(triangles.size());
            for(int k=0;k<3;++k){const auto uv=f.uv(source.nodes[e.nodes[k]]);
                f.xmin=std::min(f.xmin,uv[0]);f.xmax=std::max(f.xmax,uv[0]);
                f.ymin=std::min(f.ymin,uv[1]);f.ymax=std::max(f.ymax,uv[1]);}
            triangles.push_back(a);
        }
        for(auto& f:faces) {
            f.step=std::sqrt(2*f.area/f.triangles.size());
            f.nx=std::max(1,int(std::ceil((f.xmax-f.xmin)/f.step)));
            f.ny=std::max(1,int(std::ceil((f.ymax-f.ymin)/f.step)));
            f.bins.resize(size_t(f.nx)*f.ny);
            for(int t:f.triangles){const auto& e=source.elements[triangles[t].element];
                double x0=1e100,y0=1e100,x1=-1e100,y1=-1e100;
                for(int k=0;k<3;++k){const auto uv=f.uv(source.nodes[e.nodes[k]]);
                    x0=std::min(x0,uv[0]);x1=std::max(x1,uv[0]);y0=std::min(y0,uv[1]);y1=std::max(y1,uv[1]);}
                for(int y=f.iy(y0-distance_tolerance);y<=f.iy(y1+distance_tolerance);++y)
                    for(int x=f.ix(x0-distance_tolerance);x<=f.ix(x1+distance_tolerance);++x)f.bins[y*f.nx+x].push_back(t);
            }
        }
        std::vector<std::pair<int,int>> owners(target.topology_edge_count,{-1,-1});
        for(const auto& e:target.elements)for(int k=0;k<3;++k)
            if((target.nodes[e.nodes[3+k]]-(target.nodes[e.nodes[k]]+
                target.nodes[e.nodes[(k+1)%3]])*.5).norm()>distance_tolerance)
                throw std::invalid_argument("BDM transfer supports flat triangles only");
        for(size_t t=0;t<target.elements.size();++t)for(int k=0;k<3;++k)
            if(owners[target.elements[t].topology_edges[k]].first<0)owners[target.elements[t].topology_edges[k]]={int(t),k};
        std::exception_ptr failure;
#pragma omp parallel for schedule(dynamic,64)
        for(int edge=0;edge<target.topology_edge_count;++edge)try {
            const int t=owners[edge].first,k=owners[edge].second;
            if(t<0)throw std::runtime_error("unowned target edge");
            const auto& e=target.elements[t];
            const Vec3 p=target.nodes[e.nodes[k]],q=target.nodes[e.nodes[(k+1)%3]],direction=q-p;
            const double length=direction.norm();
            const auto frame=evaluate_muller_frame(target,t,1./3,1./3);
            const Vec3 conormal=direction.cross(frame.normal)*(1/length);
            const int fi=face_of(p,frame.normal);
            if(fi<0 || std::abs((q-faces[fi].origin).dot(faces[fi].normal))>distance_tolerance)
                throw std::invalid_argument("source/target faces differ");
            const auto& f=faces[fi];const auto uv0=f.uv(p),uv1=f.uv(q);
            std::vector<int> candidates;
            for(int y=f.iy(std::min(uv0[1],uv1[1])-distance_tolerance);y<=f.iy(std::max(uv0[1],uv1[1])+distance_tolerance);++y)
                for(int x=f.ix(std::min(uv0[0],uv1[0])-distance_tolerance);x<=f.ix(std::max(uv0[0],uv1[0])+distance_tolerance);++x)
                    candidates.insert(candidates.end(),f.bins[y*f.nx+x].begin(),f.bins[y*f.nx+x].end());
            std::sort(candidates.begin(),candidates.end());candidates.erase(std::unique(candidates.begin(),candidates.end()),candidates.end());
            struct Interval {double lo,hi;int triangle;};std::vector<Interval> intervals;
            std::vector<double> cuts={0,1};
            for(int index:candidates){const auto& a=triangles[index];const auto b0=a.barycentric(p),b1=a.barycentric(q);
                double lo=0,hi=1;bool possible=true;
                for(int j=0;j<3;++j){const double slope=b1[j]-b0[j];
                    if(std::abs(slope)<1e-13){if(b0[j]<-1e-10)possible=false;}
                    else if(slope>0)lo=std::max(lo,-b0[j]/slope);else hi=std::min(hi,-b0[j]/slope);}
                if(possible && hi-lo>1e-12){intervals.push_back({lo,hi,index});cuts.push_back(lo);cuts.push_back(hi);}
            }
            std::sort(cuts.begin(),cuts.end());cuts.erase(std::unique(cuts.begin(),cuts.end(),[](double a,double b){return std::abs(a-b)<1e-11;}),cuts.end());
            std::map<int,std::array<double,2>> weights;
            for(size_t j=1;j<cuts.size();++j){const double lo=cuts[j-1],hi=cuts[j],mid=(lo+hi)*.5;
                if(hi-lo<1e-12)continue;
                int chosen=-1;for(const auto& a:intervals)if(mid>=a.lo-1e-12 && mid<=a.hi+1e-12){chosen=a.triangle;break;}
                if(chosen<0)throw std::runtime_error("target edge leaves source triangulation");
                const auto& a=triangles[chosen];
                for(double sign:{-1.,1.}){const double z=mid+sign*(hi-lo)/(2*std::sqrt(3.));
                    const auto b=a.barycentric(p+direction*z);
                    const auto sample=evaluate_muller_frame(source,a.element,b[1],b[2]);
                    const auto basis=evaluate_muller_basis(source,a.element,sample);
                    for(int c=0;c<basis.count;++c){const double w=(hi-lo)*.5*length*basis.values[c].dot(conormal);
                        auto& v=weights[basis.dofs[c]];v[0]+=e.edge_orientations[k]*w;v[1]+=(2*z-1)*w;}
                }
            }
            for(const auto& entry:weights)for(int moment=0;moment<2;++moment)
                if(std::abs(entry.second[moment])>1e-14)rows_[2*edge+moment].push_back({entry.first,entry.second[moment]});
        }catch(...) {
#pragma omp critical(bdm_transfer_failure)
            {if(!failure)failure=std::current_exception();}
        }
        if(failure){rows_.clear();std::rethrow_exception(failure);}
        ready_=true;
    }
    void apply(const std::complex<double>* source,std::complex<double>* target)const {
        if(!ready_ || !source || !target || source==target || rows_.size()!=size_t(target_current_))
            throw std::invalid_argument("invalid BDM transfer buffers/state");
#pragma omp parallel for schedule(static)
        for(int i=0;i<target_current_;++i){std::complex<double> a=0,b=0;
            for(const auto& v:rows_[i]){a+=v.second*source[v.first];b+=v.second*source[source_current_+v.first];}
            target[i]=a;target[target_current_+i]=b;
        }
    }
};
