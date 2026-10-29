// CPU-only benchmark of streaming RHS assembly on the actual prism mesh.
// No operator, FMM engine, dense matrix or GPU context is created.
#include "tmatrix_rhs_assembly.h"
#include "quadrature.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sys/resource.h>
#include <omp.h>

using Complex=std::complex<double>;
using Clock=std::chrono::steady_clock;
struct CachedPoint {int element;MullerFrameSample sample;double weight;};

void legacy_add(const MullerP2Mesh& mesh,const TriQuad& rule,size_t begin,size_t count,
                const Complex* fields,std::vector<Complex>& rhs) {
    for(size_t local=0;local<count;++local) {
        const size_t index=begin+local;
        const int element=index/rule.npts,q=index%rule.npts;
        const auto sample=evaluate_muller_frame(mesh,element,rule.pts[q][0],rule.pts[q][1]);
        const double weight=.5*rule.wts[q]*sample.jacobian;
        const auto basis=evaluate_muller_basis(mesh,element,sample);
        const Complex* f=fields+local*6;
        for(int a=0;a<basis.count;++a) {
            const Vec3 t=basis.values[a].cross(sample.normal)*weight;
            rhs[basis.dofs[a]]-=t.x*f[0]+t.y*f[1]+t.z*f[2];
            rhs[mesh.current_dofs()+basis.dofs[a]]+=t.x*f[3]+t.y*f[4]+t.z*f[5];
        }
    }
}

int main(int argc,char** argv) try {
    const int ref=argc>1?std::stoi(argv[1]):4,repeats=argc>2?std::stoi(argv[2]):3;
    const int chunk_elements=argc>3?std::stoi(argv[3]):2048;
    if(ref<0 || ref>7 || repeats<1 || repeats>10 || chunk_elements<1 || chunk_elements>8192)
        throw std::runtime_error("usage: probe REF[0..7] REPEATS[1..10] CHUNK_ELEMENTS[1..8192]");
    MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
    const auto mesh=build_muller_p2_mesh(regular_prism(6,1.,ref,1.,0,false,true),options);
    const auto rule=tri_quadrature(13);
    std::vector<CachedPoint> cache;cache.reserve(mesh.elements.size()*rule.npts);
    std::vector<Vec3> positions;positions.reserve(cache.capacity());
    const auto setup=Clock::now();
    for(int e=0;e<int(mesh.elements.size());++e)for(int q=0;q<rule.npts;++q) {
        const auto frame=evaluate_muller_frame(mesh,e,rule.pts[q][0],rule.pts[q][1]);
        cache.push_back({e,frame,.5*rule.wts[q]*frame.jacobian});positions.push_back(frame.position);
    }
    std::cout<<std::setprecision(12)<<"{\"stage\":\"geometry_ready\",\"ref\":"<<ref
        <<",\"triangles\":"<<mesh.elements.size()<<",\"dofs\":"<<mesh.system_dofs()
        <<",\"rhs_points\":"<<cache.size()<<",\"cache_setup_s\":"
        <<std::chrono::duration<double>(Clock::now()-setup).count()<<"}"<<std::endl;
    bem_tmatrix::StreamingRhsAssembler assembler(rule.npts,chunk_elements,16);
    std::vector<Complex> fields(std::max(assembler.chunk_points(),size_t(4096))*6);
    std::vector<Complex> rhs[2]={std::vector<Complex>(mesh.system_dofs()),std::vector<Complex>(mesh.system_dofs())};
    std::vector<Complex> reference[2];
    // Field generation reads a separate position array, keeping the cached
    // frames cold. Its time is excluded from every assembly measurement.
    const auto fill=[&](size_t begin,size_t count,int pol) {
        for(size_t local=0;local<count;++local) {
            const Complex phase=std::exp(Complex(0,112.*positions[begin+local].z));
            for(int c=0;c<6;++c)fields[6*local+c]=0.;
            fields[6*local+pol]=phase;
            fields[6*local+(pol==0?4:3)]=(pol==0?1.:-1.)*phase;
        }
    };
    for(int trial=0;trial<repeats;++trial) {
        std::vector<int> variants={0,1,4,16};
        if(trial%2)std::reverse(variants.begin(),variants.end());
        for(int threads:variants) {
            omp_set_num_threads(std::max(1,threads));
            const size_t chunk=threads?assembler.chunk_points():4096;
            double elapsed=0;
            for(int pol=0;pol<2;++pol) {
                std::fill(rhs[pol].begin(),rhs[pol].end(),Complex(0));
                for(size_t begin=0;begin<cache.size();begin+=chunk) {
                    const size_t count=std::min(chunk,cache.size()-begin);fill(begin,count,pol);
                    const auto start=Clock::now();
                    if(threads)assembler.add(mesh,cache,begin,count,fields.data(),rhs[pol]);
                    else legacy_add(mesh,rule,begin,count,fields.data(),rhs[pol]);
                    elapsed+=std::chrono::duration<double>(Clock::now()-start).count();
                }
            }
            if(trial==0 && threads==0)for(int p=0;p<2;++p)reference[p]=rhs[p];
            double maximum_error=0;
            for(int p=0;p<2;++p) {
                double difference=0,length=0;
                for(size_t i=0;i<rhs[p].size();++i) {
                    difference+=std::norm(rhs[p][i]-reference[p][i]);length+=std::norm(reference[p][i]);
                }
                maximum_error=std::max(maximum_error,std::sqrt(difference/std::max(length,1e-300)));
            }
            if(!(maximum_error<2e-13))throw std::runtime_error("RHS assembly accuracy regression");
            rusage usage={};getrusage(RUSAGE_SELF,&usage);
            std::cout<<"{\"trial\":"<<trial<<",\"variant\":\""<<(threads?"cached_elements":"legacy")
                <<"\",\"threads\":"<<std::max(threads,1)<<",\"assembly_pair_s\":"<<elapsed
                <<",\"maximum_relative_rhs_error\":"<<maximum_error
                <<",\"chunk_points\":"<<chunk<<",\"extra_integral_workspace_bytes\":"<<(threads?assembler.workspace_bytes():0)
                <<",\"peak_rss_mib\":"<<usage.ru_maxrss/1024.<<"}"<<std::endl;
        }
    }
    return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<std::endl;return 1;}
