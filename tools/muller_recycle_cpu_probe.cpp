// Small dense BEM benchmark of host recycling; no CUDA context or FMM run.
#include "streaming_host_gmres.h"
#include "muller_dense.h"
#include "muller_mbj.h"
#include <iomanip>
#include <iostream>

int main(int argc,char** argv) try {
    using namespace bem_streaming;
    const int ref=argc>1?std::stoi(argv[1]):1;
    const double ka=argc>2?std::stod(argv[2]):6.;
    const int repeats=argc>3?std::stoi(argv[3]):1;
    if(repeats<1 || repeats>20)throw std::runtime_error("repeats must be in [1,20]");
    const bool compare_strict=std::getenv("BEM_PROBE_COMPARE_STRICT_INTERVAL")!=nullptr;
    const double tolerance=std::getenv("BEM_PROBE_TOL")?std::stod(std::getenv("BEM_PROBE_TOL")):1e-8;
    MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
    options.project_edge_nodes_to_sphere=false;
    const auto dense=assemble_muller_nodal_dense(regular_prism(6,1.,ref,1.,0,false,true),ka,Complex(1.3,0),options,7,6);
    const int n=dense.system_dofs;
    MullerMbjPreconditioner mbj;mbj.build(dense,12);
    const Action action=[&](const Complex* v,Complex* w) {
#pragma omp parallel for schedule(static)
        for(int i=0;i<n;++i) {
            Complex sum=0.;for(int j=0;j<n;++j)sum+=dense.matrix[size_t(i)*n+j]*v[j];w[i]=sum;
        }
    };
    const Action inverse=[&](const Complex* v,Complex* w){mbj.apply(v,w);};
    std::vector<std::vector<Complex>> rhs,reference;
    std::vector<Complex> x(n),residual(n);
    for(int k=0;k<8;++k) {
        const double theta=.17+.19*k,phi=.41*k;
        const Vec3 direction(std::sin(theta)*std::cos(phi),std::sin(theta)*std::sin(phi),std::cos(theta));
        const Vec3 polarization(std::cos(theta)*std::cos(phi),std::cos(theta)*std::sin(phi),-std::sin(theta));
        rhs.push_back(muller_nodal_planewave_rhs(dense.mesh,ka,polarization,direction,13));
    }
    setenv("BEM_MIXED_ADAPTIVE_INNER","1",1);
    std::cout<<std::setprecision(12);
    std::vector<std::pair<int,int>> configurations;
    if(compare_strict)configurations={{0,1},{0,4},{8,1},{8,4}};
    else configurations={{0,1},{8,1},{16,1},{32,1}};
    for(int trial=0;trial<repeats;++trial)for(size_t run=0;run<configurations.size();++run) {
        const auto config=configurations[trial%2?configurations.size()-1-run:run];
        const int rank=config.first,strict_interval=config.second;
        setenv("BEM_HOST_GMRES_STRICT_INTERVAL",std::to_string(strict_interval).c_str(),1);
        RecycleSpace space(n,rank);
        int iterations=0;double solve_s=0,max_residual=0,max_error=0,update_s=0;
        int strict_checks=0,deferred_checks=0;
        double action_s=0,strict_s=0;
        std::vector<int> counts;
        for(int k=0;k<int(rhs.size());++k) {
            const auto result=gmres(action,action,inverse,rhs[k].data(),x.data(),n,40,tolerance,1000,false,false,&space,k!=0);
            action(x.data(),residual.data());
            for(int i=0;i<n;++i)residual[i]-=rhs[k][i];
            const double strict=norm(residual.data(),n)/norm(rhs[k].data(),n);
            if(!(strict<=tolerance))throw std::runtime_error("physical BEM solve failed strict residual");
            max_residual=std::max(max_residual,strict);
            if(trial==0 && run==0)reference.push_back(x);
            else {
                for(int i=0;i<n;++i)residual[i]=x[i]-reference[k][i];
                max_error=std::max(max_error,norm(residual.data(),n)/norm(reference[k].data(),n));
            }
            iterations+=result.iterations;solve_s+=result.seconds;update_s+=result.recycle_seconds;counts.push_back(result.iterations);
            strict_checks+=result.strict_checks;deferred_checks+=result.deferred_strict_checks;
            action_s+=result.action_seconds;strict_s+=result.strict_seconds;
        }
        if(max_error>1e-6)throw std::runtime_error("physical solution changed");
        std::cout<<"{\"ref\":"<<ref<<",\"ka\":"<<ka<<",\"dofs\":"<<n<<",\"rank\":"<<rank
            <<",\"trial\":"<<trial<<",\"strict_interval\":"<<strict_interval<<",\"strict_checks\":"<<strict_checks
            <<",\"deferred_checks\":"<<deferred_checks<<",\"action_s\":"<<action_s<<",\"strict_s\":"<<strict_s
            <<",\"iterations\":"<<iterations<<",\"solve_s\":"<<solve_s<<",\"recycle_update_s\":"<<update_s
            <<",\"maximum_strict_residual\":"<<max_residual<<",\"maximum_solution_difference\":"<<max_error<<",\"per_rhs\":[";
        for(size_t i=0;i<counts.size();++i)std::cout<<(i?",":"")<<counts[i];
        std::cout<<"]}"<<std::endl;
    }
    return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<std::endl;return 1;}
