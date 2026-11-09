#include "host_fmm_band.h"
#include "host_shared_near_avx512.h"
#include <fstream>
#include <iomanip>
#include <numeric>
#include <omp.h>
using C=bem_cpu::C;using Six=bem_cpu::Six;using Twelve=bem_cpu::Twelve;
int main(int argc,char** argv)try{
    if(argc!=3)throw std::invalid_argument("usage: THREADS REPORT");
    const int threads=std::stoi(argv[1]);if(threads<1||threads>32)throw std::invalid_argument("threads");
#ifdef BEM_CPU_HAS_AVX512_P2P
    if(!__builtin_cpu_supports("avx512f")||!__builtin_cpu_supports("avx512vl"))throw std::runtime_error("AVX512 unavailable");
    omp_set_dynamic(0);omp_set_num_threads(threads);
    std::ofstream out(argv[2]);if(!out)throw std::runtime_error("cannot open report");
    out<<std::setprecision(17)<<"{\"GPU_used\":false,\"scope\":\"Synthetic mapped near-list kernel screen on shared CPU; not a full BEM operator or solve\",\"threads\":"<<threads<<",\"cases\":[";
    bool first=true;
    for(int near_count:{96,384,519})for(int batches:{1,2,6})for(double absorption:{0.,.2}){
        const int points=2048,targets=256;const C k(40,absorption);
        std::array<std::vector<double>,15> soa;for(auto& v:soa)v.resize(points);
        std::vector<Vec3> geometry(points);std::vector<std::vector<int>> ids(targets,std::vector<int>(near_count));
        for(int p=0;p<points;++p){geometry[p]=Vec3(std::sin(.731*p),std::cos(.531*p),std::sin(.193*p));
            soa[0][p]=geometry[p].x;soa[1][p]=geometry[p].y;soa[2][p]=geometry[p].z;}
        for(int p=0;p<targets;++p)for(int s=0;s<near_count;++s)ids[p][s]=(p+37*s)%points;
        std::vector<std::vector<Six>> charges(batches,std::vector<Six>(points));
        std::vector<std::vector<Twelve>> fields(batches,std::vector<Twelve>(targets)),reference=fields;
        const Six* inputs[8];for(int b=0;b<batches;++b){inputs[b]=charges[b].data();
            for(int p=0;p<points;++p)for(int c=0;c<6;++c)charges[b][p][c]=C(std::sin(.13*p+.2*c+.7*b),std::cos(.17*p+.3*c-.6*b));}
        const auto run=[&](bool shared){
            auto start=bem_cpu::Clock::now();for(auto& f:fields)std::fill(f.begin(),f.end(),Twelve{});
            if(shared){
#pragma omp parallel for schedule(dynamic,16)
                for(int p=0;p<targets;++p){Twelve* results[8];for(int b=0;b<batches;++b)results[b]=&fields[b][p];
                    bem_cpu::avx512::near_field_batch(soa,ids[p],geometry[p],k,inputs,results,batches);}
            }else for(int b=0;b<batches;++b){
#pragma omp parallel for schedule(static)
                for(int p=0;p<points;++p)for(int c=0;c<6;++c){soa[3+2*c][p]=charges[b][p][c].real();soa[4+2*c][p]=charges[b][p][c].imag();}
#pragma omp parallel for schedule(dynamic,16)
                for(int p=0;p<targets;++p)bem_cpu::avx512::near_field(soa,ids[p],geometry[p],k,fields[b][p]);
            }
            return bem_cpu::elapsed(start);
        };
        const double warm_original=run(false);reference=fields;const double warm_shared=run(true);double e=0,n=0;
        for(int b=0;b<batches;++b)for(int p=0;p<targets;++p)for(int c=0;c<12;++c){e+=std::norm(fields[b][p][c]-reference[b][p][c]);n+=std::norm(reference[b][p][c]);}
        const double relative=std::sqrt(e/n);if(!std::isfinite(relative)||relative>2e-12)throw std::runtime_error("kernel numerical control failed");
        std::vector<double> serial,shared;
        const int repetitions=std::min(128,std::max(1,int(std::ceil(.025/std::min(warm_original,warm_shared)))));
        const auto sample=[&](bool use_shared){double sum=0;for(int j=0;j<repetitions;++j)sum+=run(use_shared);return sum/repetitions;};
        for(int cycle=0;cycle<3;++cycle){serial.push_back(sample(false));shared.push_back(sample(true));shared.push_back(sample(true));serial.push_back(sample(false));}
        const auto median=[](std::vector<double> v){std::sort(v.begin(),v.end());return .5*(v[v.size()/2-1]+v[v.size()/2]);};
        out<<(first?"":",")<<"{\"near_sources_per_target\":"<<near_count<<",\"batches\":"<<batches<<",\"points\":"<<points<<",\"targets\":"<<targets
            <<",\"ka\":40,\"k_imag\":"<<absorption<<",\"repetitions_per_sample\":"<<repetitions<<",\"relative_error\":"<<relative<<",\"median_saved_percent\":"<<100*(1-median(shared)/median(serial));
        const auto array=[&](const char* name,const std::vector<double>& v){out<<",\""<<name<<"\":[";for(size_t i=0;i<v.size();++i)out<<(i?",":"")<<v[i];out<<"]";};
        array("original_s",serial);array("shared_s",shared);out<<"}";first=false;
    }
    out<<"],\"passed\":true}\n";std::puts("PASS CPU shared near timing and numerical controls");
#else
    throw std::runtime_error("AVX512 compile flags required");
#endif
}catch(const std::exception& e){std::fprintf(stderr,"shared near CPU probe: %s\n",e.what());return 1;}
