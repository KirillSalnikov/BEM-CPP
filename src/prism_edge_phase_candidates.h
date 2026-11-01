#pragma once
#include "prism_edge_phase_basis.h"
#include "mbs_surface_seed.h"
#include <string>

namespace bem_edge_phase {
struct Line {Vec3 center,tangent;double length;};
inline std::vector<Line> prism_feature_lines(const MullerP2Mesh& mesh) {
    struct Segment {Vec3 start,end,normal;int visits=0;};
    std::vector<Segment> segments(mesh.topology_edge_count);
    struct Range {Vec3 origin,tangent;double lo,hi;};std::vector<Range> ranges;
    for(size_t t=0;t<mesh.elements.size();++t){const auto& e=mesh.elements[t];
        const auto frame=evaluate_muller_frame(mesh,t,1./3,1./3);
        for(int j=0;j<3;++j){auto& edge=segments[e.topology_edges[j]];
            if(edge.visits++==0){edge.start=mesh.nodes[e.topology_vertices[j]];edge.end=mesh.nodes[e.topology_vertices[(j+1)%3]];edge.normal=frame.normal;continue;}
            if(edge.visits!=2)throw std::runtime_error("nonmanifold phase feature edge");
            if(edge.normal.dot(frame.normal)>1-1e-9)continue;
            Vec3 tangent=edge.end-edge.start;tangent=tangent*(1/tangent.norm());
            for(int axis=0;axis<3;++axis)if(std::abs(component(tangent,axis))>1e-10){if(component(tangent,axis)<0)tangent=tangent*(-1.);break;}
            const Vec3 origin=edge.start-tangent*tangent.dot(edge.start);
            double lo=tangent.dot(edge.start),hi=tangent.dot(edge.end);if(lo>hi)std::swap(lo,hi);
            bool merged=false;
            for(auto& range:ranges)if((range.tangent-tangent).norm()<1e-8 && (range.origin-origin).norm()<1e-8){
                range.lo=std::min(range.lo,lo);range.hi=std::max(range.hi,hi);merged=true;break;}
            if(!merged)ranges.push_back({origin,tangent,lo,hi});
        }
    }
    for(const auto& edge:segments)if(edge.visits!=2)throw std::runtime_error("open phase feature mesh");
    std::vector<Line> lines;for(const auto& r:ranges)lines.push_back({r.origin+r.tangent*((r.lo+r.hi)*.5),r.tangent,r.hi-r.lo});
    if(lines.size()!=18)throw std::runtime_error("expected 18 physical edges of hexagonal prism");
    std::sort(lines.begin(),lines.end(),[](const Line& a,const Line& b){
        if(std::abs(a.center.z-b.center.z)>1e-8)return a.center.z<b.center.z;
        if(std::abs(a.center.x-b.center.x)>1e-8)return a.center.x<b.center.x;
        return a.center.y<b.center.y;});
    return lines;
}
struct Candidate {Specification spec;int polarization,ray_depth;std::string source;double ray_strength;};
inline std::vector<Candidate> candidates(const MullerP2Mesh& mesh,const MbsSurfaceSeed& rays,double k) {
    if(!rays.internal_traces || std::abs(rays.ri-1.3116)>1e-12 || std::abs(rays.aspect-10./7)>1e-12)
        throw std::invalid_argument("P13 requires matching internal ray traces for n=1.3116, aspect=10/7");
    for(size_t t=0;t<mesh.elements.size();++t){const auto f=evaluate_muller_frame(mesh,t,1./3,1./3);rays.face(f.normal,f.position);}
    const auto features=prism_feature_lines(mesh);
    struct Source {Line line;Vec3 phase;double path,strength;int depth;std::string label;};std::vector<Source> sources;
    for(bool vertical:{true,false}) {
        const auto found=std::find_if(features.begin(),features.end(),[&](const Line& e){return (std::abs(e.tangent.z)>.9)==vertical && (vertical||e.center.z>0);});
        if(found==features.end())throw std::runtime_error("missing primary diffraction edge");
        sources.push_back({*found,rays.direction,rays.direction.dot(found->center),1.,-1,vertical?"incident_vertical":"incident_rim"});
    }
    // Freeze two genuine interior beam-boundary lines, including reflected
    // phases. Only their geometry and phase enter the basis, not MBS currents
    // or its far field. Fresnel strength chooses training lines before RHS tests.
    for(int depth:{1,2}) {
        Source best;double score=-1;
        for(const auto& ray:rays.rays)if(ray.depth==depth){double strength=0;for(auto z:ray.j)strength+=std::norm(z);
            for(size_t e=0;e<ray.vertices.size();++e){const Vec3 start=ray.vertices[e],delta=ray.vertices[(e+1)%ray.vertices.size()]-start;
                const double length=delta.norm();if(length<1e-8)continue;
                const Vec3 center=start+delta*.5;
                bool physical=false;for(const auto& edge:features){const auto offset=center-edge.center;
                    if((offset-edge.tangent*edge.tangent.dot(offset)).norm()<1e-7 && std::abs(edge.tangent.dot(offset))<=edge.length*.5+1e-7){physical=true;break;}}
                if(physical || length*strength<=score)continue;
                const Vec3 phase=ray.d*rays.ri;
                best={{center,delta*(1/length),length},phase,ray.path+phase.dot(center),strength,depth,"internal_boundary_depth"+std::to_string(depth)};score=length*strength;
            }
        }
        if(score<=0)throw std::runtime_error("no interior reflected beam boundary at requested depth");
        sources.push_back(best);
    }
    std::vector<Candidate> result;
    for(int family=0;family<3;++family)for(const auto& source:sources)for(int pol=0;pol<2;++pol) {
        Specification s;s.center=source.line.center;s.tangent=source.line.tangent;s.phase_vector=source.phase;s.path=source.path;s.k=k;
        s.index=family==0?1.:family==1?rays.ri:source.phase.norm();s.family=family==2?Family::ShadowPlane:Family::EdgeCone;
        s.half_length=.5*source.line.length;s.width=.35*std::min(1.,source.line.length);s.core=std::min(.25*s.width,1/(4*k*s.index));
        HertzTrace check(s);result.push_back({s,pol,source.depth,source.label,source.strength});
    }
    return result;
}
}
