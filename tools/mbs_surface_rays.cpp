// Standalone CPU exporter linked against the frozen native MBS ray tracer.
// No diffraction, orientation averaging, or changes to the MBS executable.
#include "particle/Hexagonal.h"
#include "scattering/ScatteringConvex.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <memory>

// Use the native splitting/intersection routines while retaining boundary
// traces of both internal incident and reflected waves, including TIR.
class SurfaceTracer:public ScatteringConvex {
    double index_;
public:
    int tir_events=0;
    SurfaceTracer(Particle* p,Light* l,int depth,double index):ScatteringConvex(p,l,true,depth),index_(index){}
    bool trace(std::vector<Beam>& surface,std::vector<Beam>& outgoing) {
        if(!PrepareExternalBeams(outgoing))return false;
        for(const auto& seed:m_beamTree){Beam value=seed;value.opticalPath+=index_*value.front;surface.push_back(value);}
        while(m_treeSize){
            Beam beam=m_beamTree.back();m_beamTree.pop_back();m_treeSize=m_beamTree.size();
            if(IsTerminalAct(beam))continue;
            for(int f=0;f<m_particle->nFacets;++f)if(f!=beam.lastFacetId){
                Beam reflected;const size_t previous=outgoing.size();
                if(!SplitSecondaryBeams(beam,f,reflected,outgoing))continue;
                if(outgoing.size()==previous)++tir_events;
                Beam incident=beam;incident.SetPolygon(reflected);incident.lastFacetId=f;incident.nActs=beam.nActs+1;
                incident.opticalPath=beam.opticalPath+index_*beam.front;surface.push_back(incident);
                Beam trace=reflected;trace.lastFacetId=f;trace.nActs=beam.nActs+1;trace.opticalPath+=index_*trace.front;surface.push_back(trace);
                reflected.id=RecomputeTrackId(beam.id,f);reflected.locations=beam.locations;
                if(!IsTracePruned(reflected)&&!PushBeamToTree(reflected,f,beam.nActs+1,Location::In))return false;
            }
        }
        return true;
    }
};

int main(int argc,char** argv)try {
    if(argc!=7)throw std::runtime_error("Usage: mbs_surface_rays RI ASPECT DEPTH THETA_RAD PHI_RAD OUTPUT");
    const double ri=std::stod(argv[1]),aspect=std::stod(argv[2]);const int depth=std::stoi(argv[3]);
    const double theta=std::stod(argv[4]),phi=std::stod(argv[5]);
    if(ri<=0||aspect<=0||depth<1||depth>18)throw std::runtime_error("invalid parameters");
    const double pi=std::acos(-1.),width=std::cbrt(8*pi/(3*std::sqrt(3.)*aspect));
    std::unique_ptr<Hexagonal> particle(new Hexagonal(complex(ri,0),2*width/std::sqrt(3.),aspect*width));
    particle->RotateQuaternion(0,0,std::sin(pi/12),std::cos(pi/12));
    Light light;light.direction=Point3f(std::sin(theta)*std::cos(phi),std::sin(theta)*std::sin(phi),std::cos(theta));
    light.polarizationBasis=Point3f(-std::sin(phi),std::cos(phi),0);
    SurfaceTracer scatter(particle.get(),&light,depth,ri);
    scatter.restriction=0; // Convex intersections do not need fragment pruning.
    const auto start=std::chrono::steady_clock::now();std::vector<Beam> beams,outgoing;
    if(!scatter.trace(beams,outgoing))throw std::runtime_error("native trace failed");
    const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::ofstream out(argv[6]);if(!out)throw std::runtime_error("cannot open ray output");
    out<<std::setprecision(17)<<"MBS_SURFACE_RAYS_V2\n"<<ri<<' '<<aspect<<' '<<depth<<' '<<seconds<<'\n';
    const auto vector=[&](const Point3f& p){out<<p.cx<<' '<<p.cy<<' '<<p.cz<<' ';};
    vector(light.direction);vector(light.polarizationBasis);out<<'\n';
    out<<particle->nFacets<<' '<<beams.size()<<'\n';
    for(int f=0;f<particle->nFacets;++f){const auto& facet=particle->facets[f];
        vector(facet.ex_normal);vector(facet.arr[0]);out<<'\n';}
    for(const auto& beam:beams){
        out<<beam.lastFacetId<<' '<<beam.nActs<<' '<<beam.nVertices<<' '<<beam.opticalPath<<' ';
        vector(beam.direction);vector(beam.polarizationBasis);
        const complex j[]={beam.J.m11,beam.J.m12,beam.J.m21,beam.J.m22};
        for(auto z:j)out<<real(z)<<' '<<imag(z)<<' ';out<<'\n';
        for(int v=0;v<beam.nVertices;++v){vector(beam.arr[v]);out<<'\n';}
    }
    if(!out)throw std::runtime_error("ray output failed");
    std::cout<<"{\"surface_traces\":"<<beams.size()<<",\"outgoing_beams\":"<<outgoing.size()<<",\"tir_events\":"<<scatter.tir_events
             <<",\"trace_s\":"<<seconds<<",\"volume\":"<<particle->Volume()<<"}\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
