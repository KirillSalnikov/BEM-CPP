// Experimental residual corrections; all final solves use the original strict BEM action.
// For high ka, supply the validated production BEM_FMM_BANDED_* settings.
// A small FP64 iterative residual alone does not validate FMM truncation.
#include "mbs_surface_seed.h"
#include "muller_dense.h"
#include "muller_mbj.h"
#include "streaming_host_gmres.h"
#include "quadrature.h"
#include "maxwell_exterior_projector.h"
#include "streaming_image_space.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#ifdef BEM_MBS_FMM
#include "muller_fmm.h"
#include "prism_circulant_inverse.h"
#include "prism_symmetric_circulant.h"
#include "prism_sector_solver.h"
#include "maxwell_exterior_fmm.h"
#endif

int main(int argc,char** argv)try {
    using namespace bem_streaming;using V=std::vector<Complex>;using Clock=std::chrono::steady_clock;
    const auto now=[](){return Clock::now();};
    const auto elapsed=[&](Clock::time_point a){return std::chrono::duration<double>(now()-a).count();};
    if(argc!=5)throw std::runtime_error("Usage: muller_mbs_variants_probe REF KA RAY_FILE_LIST REPORT");
    const int ref=std::stoi(argv[1]);const double ka=std::stod(argv[2]),tol=2e-6;
    const char* packet_file=std::getenv("BEM_PROBE_RHS_FILE");
    const std::string selection=std::getenv("BEM_VARIANTS_T_METHODS")?std::getenv("BEM_VARIANTS_T_METHODS"):"";
    const auto wanted=[&](const std::string& name){return selection.empty()||(","+selection+",").find(","+name+",")!=std::string::npos;};
    const bool do_planes=std::getenv("BEM_VARIANTS_PLANES")==nullptr||std::stoi(std::getenv("BEM_VARIANTS_PLANES"))!=0;
    const bool ray_spaces=wanted("ray_two_level192")||wanted("ray_residual_two_level384")||wanted("ray_krylov_two_level576");
    const bool sector_solver=std::getenv("BEM_MBS_SYMMETRY")!=nullptr;
    const bool sector_coarse=std::getenv("BEM_MBS_COARSE_SECTORS")!=nullptr;
    if(!packet_file||!sector_solver||!sector_coarse)throw std::runtime_error("variants need packed RHS and both sector flags");
    if(sector_coarse&&!sector_solver)throw std::runtime_error("coarse sectors require symmetry");
#ifndef BEM_MBS_FMM
    if(sector_solver)throw std::runtime_error("symmetry probe requires FMM build");
#endif
    if(ref<0||ref>
#ifdef BEM_MBS_FMM
       5
#else
       2
#endif
       ||ka<=0)throw std::runtime_error("unsupported probe size");
    std::ifstream files(argv[3]);std::string name;std::vector<std::unique_ptr<MbsSurfaceSeed>> rays;
    while(std::getline(files,name))if(!name.empty())rays.emplace_back(new MbsSurfaceSeed(name));
    if(rays.size()<3)throw std::runtime_error("need two holdout directions and independent training directions");
    const double ri=rays[0]->ri,aspect=rays[0]->aspect;
    for(const auto& r:rays)if(std::abs(r->ri-ri)>1e-12||std::abs(r->aspect-aspect)>1e-12)throw std::runtime_error("mismatched ray cases");
    int mixed_calls=0,strict_calls=0;
    const auto start=now();MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
    const auto linear=regular_prism(6,aspect,ref,1.,0,false,true);
#ifdef BEM_MBS_FMM
    MullerFmmOperator system;
    const int max_leaf=std::getenv("BEM_TMATRIX_MAX_LEAF")?std::stoi(std::getenv("BEM_TMATRIX_MAX_LEAF")):64;
    if(max_leaf<1)throw std::runtime_error("invalid FMM leaf capacity");
    system.init(linear,ka,Complex(ri,0),options,7,6,5,max_leaf,false,2,0,.96,nullptr,3);
    system.set_fmm_near_fp32(true);
    const int n=system.system_dofs;MullerMbjPreconditioner mbj;mbj.build(system,50);
    const Action action=[&](const Complex* v,Complex* w){++mixed_calls;system.matvec(v,w);};
    const Action strict_action=[&](const Complex* v,Complex* w){++strict_calls;system.matvec_strict(v,w);};
    PrismCirculantInverse circulant;circulant.build(system.mesh,ref,action,[&](const Complex* v,Complex* w){mbj.apply(v,w);});
    bem_tmatrix::D6hOrbit orbit;bem_tmatrix::PrismSymmetrySectors sectors;
    PrismOrbitInverse orbit_inverse;PrismSymmetricCirculant symmetric;
    if(sector_solver){orbit.build(system.mesh);bem_tmatrix::PrismGroupMaps maps;maps.build(orbit);sectors.build(maps);
        orbit_inverse.build(system,maps,50);symmetric.build(orbit,circulant,orbit_inverse);
        if(!symmetric.prepare_full_fft_symmetry(system.mesh))throw std::runtime_error("FFT symmetry proof failed");}
    const Action local=[&](const Complex* v,Complex* w){if(sector_solver)symmetric.apply_full_fft_symmetry(v,w,.5);else circulant.apply(v,w,.5);};
    const double setup=elapsed(start);
#else
    MullerDenseSystem system;
    if(std::abs(ri-1)<1e-12) {
        // At equal indices all kernel differences vanish exactly; A has only
        // its two off-diagonal mass blocks. Avoid evaluating zero pair kernels.
        system.mesh=build_muller_p2_mesh(linear,options);system.current_dofs=system.mesh.current_dofs();system.system_dofs=system.mesh.system_dofs();
        const int n=system.system_dofs,m=system.current_dofs;system.matrix.resize(size_t(n)*n);
        const auto quad=tri_quadrature(13);
        for(size_t t=0;t<system.mesh.elements.size();++t)for(int q=0;q<quad.npts;++q){
            const auto frame=evaluate_muller_frame(system.mesh,t,quad.pts[q][0],quad.pts[q][1]);
            const auto basis=evaluate_muller_basis(system.mesh,t,frame);const double weight=.5*quad.wts[q]*frame.jacobian;
            for(int i=0;i<basis.count;++i)for(int j=0;j<basis.count;++j){const int a=basis.dofs[i],b=basis.dofs[j];
                const double v=weight*basis.values[i].dot(basis.values[j]);system.matrix[size_t(a)*n+m+b]+=v;system.matrix[size_t(m+a)*n+b]+=v;}
        }
    } else system=assemble_muller_nodal_dense(linear,ka,Complex(ri,0),options,7,6);
    const int n=system.system_dofs;MullerMbjPreconditioner mbj;mbj.build(system,12);
    const double setup=elapsed(start);
    const Action action=[&](const Complex* v,Complex* w){
#pragma omp parallel for schedule(static)
        for(int i=0;i<n;++i){Complex sum=0;for(int j=0;j<n;++j)sum+=system.matrix[size_t(i)*n+j]*v[j];w[i]=sum;}};
    const Action local=[&](const Complex* v,Complex* w){mbj.apply(v,w);};
    const Action strict_action=action;
#endif
    std::ofstream report(argv[4]);if(!report)throw std::runtime_error("cannot write variants report");
    report<<std::setprecision(17)<<"{\"diagnostic_only\":true,\"ref\":"<<ref<<",\"ka\":"<<ka<<",\"ri\":"<<ri
          <<",\"aspect\":"<<aspect<<",\"dofs\":"<<n<<",\"setup_s\":"<<setup<<",\"tolerance\":"<<tol<<"}\n"<<std::flush;
    const std::vector<int> offsets(sectors.offsets.begin(),sectors.offsets.end());
    std::vector<V> seeds,rhs_rays;std::vector<double> seed_cost;
    for(const auto& ray:rays)for(int pol=0;pol<2;++pol){const size_t j=seeds.size();const auto t=now();
        const bool needed=(do_planes&&(j==0||j==2))||(ray_spaces&&j>=4)||(wanted("mbs_trained_recycle16")&&j>=4&&j<8);
        seeds.push_back(needed?ray->project(system.mesh,ka,pol):V(n));
        seed_cost.push_back(needed?elapsed(t)+ray->trace_seconds*.5:0.);
        rhs_rays.push_back(muller_nodal_planewave_rhs(system.mesh,ka,ray->polarization(pol),ray->direction,13));}
    V warm(n);action(rhs_rays[0].data(),warm.data());strict_action(rhs_rays[0].data(),warm.data());
    const auto add=[&](ImageSpace& space,const V& v){V av(n),encoded_v(n),encoded_a(n);action(v.data(),av.data());
        sectors.encode(v.data(),encoded_v.data(),false);sectors.encode(av.data(),encoded_a.data(),true);
        space.add(std::move(encoded_v),std::move(encoded_a));};
    const auto coarse=[&](const ImageSpace& space,const V& rhs,V& guess,V& removed){V encoded(n),x,c;
        sectors.encode(rhs.data(),encoded.data(),true);space.apply(encoded,x,c);guess.resize(n);removed.resize(n);
        sectors.decode(x.data(),guess.data(),false);sectors.decode(c.data(),removed.data(),true);};
    const auto audit=[&](const char* name,const ImageSpace& space){V rhs(n),guess,image,actual(n);
        for(int i=0;i<n;++i)rhs[i]=Complex(std::sin(.71*i),std::cos(.37*i));
        coarse(space,rhs,guess,image);strict_action(guess.data(),actual.data());
        for(int i=0;i<n;++i)actual[i]-=image[i];
        const double error=norm(actual.data(),n)/std::max(norm(image.data(),n),1e-300);
        report<<"{\"space\":\""<<name<<"\",\"rank\":"<<space.rank()<<",\"storage_bytes\":"<<space.bytes()<<",\"strict_image_defect\":"<<error<<"}\n"<<std::flush;
        if(!std::isfinite(error)||error>1e-3)throw std::runtime_error("coarse operator/symmetry check failed");};
    // All final solutions are accepted using the original strict BEM action.
    const auto solve=[&](const std::string& family,int j,const std::string& name,const V& rhs,const V& initial,
                         const Action& inverse,double seed_s,V& reference,SectorRecycleSpace* recycle){
        const auto t=now();const int mixed_before=mixed_calls,strict_before=strict_calls;
        const double bn=norm(rhs.data(),n);V x=initial,residual=rhs,work(n),correction(n);
        if(norm(x.data(),n)>0){strict_action(x.data(),work.data());for(int i=0;i<n;++i)residual[i]-=work[i];}
        const double rn=norm(residual.data(),n);HostResult result;
        if(rn>tol*bn){result=bem_tmatrix::solve_prism_sectors(action,strict_action,inverse,sectors,residual.data(),correction.data(),64,tol*bn/rn,200,false,false,recycle);
            for(int i=0;i<n;++i)x[i]+=correction[i];}
        strict_action(x.data(),work.data());for(int i=0;i<n;++i)work[i]-=rhs[i];const double strict=norm(work.data(),n)/bn;
        const double solve_s=elapsed(t);if(reference.empty())reference=x;
        for(int i=0;i<n;++i)work[i]=x[i]-reference[i];
        const double difference=norm(work.data(),n)/norm(reference.data(),n);
        const bool accepted=std::isfinite(strict)&&std::isfinite(difference)&&strict<=tol&&difference<=1e-3;
        report<<"{\"family\":\""<<family<<"\",\"rhs\":"<<j<<",\"method\":\""<<name<<"\",\"iterations\":"<<result.iterations
              <<",\"solve_including_acceptance_s\":"<<solve_s<<",\"seed_s\":"<<seed_s<<",\"total_s\":"<<solve_s+seed_s
              <<",\"initial_residual\":"<<rn/bn<<",\"strict_residual\":"<<strict<<",\"relative_solution_difference\":"<<difference
              <<",\"mixed_calls\":"<<mixed_calls-mixed_before<<",\"strict_calls\":"<<strict_calls-strict_before
              <<",\"recycle_rank\":"<<(recycle?recycle->rank():0)<<",\"accepted\":"<<(accepted?"true":"false")<<"}\n"<<std::flush;
        std::cout<<family<<' '<<j<<' '<<name<<" iterations="<<result.iterations<<" seconds="<<solve_s+seed_s<<" residual="<<strict<<std::endl;
        if(!accepted&&name=="zero")throw std::runtime_error("baseline strict solve acceptance failed");
        return accepted;
    };
    const auto smooth=[&](const V& rhs,V x,int steps){V image(n),residual(rhs),initial_residual,correction,removed,z(n);
        if(norm(x.data(),n)>0){action(x.data(),image.data());for(int i=0;i<n;++i)residual[i]-=image[i];}
        initial_residual=residual;
        ImageSpace space(offsets);
        for(int k=0;k<steps;++k){local(residual.data(),z.data());add(space,z);coarse(space,initial_residual,correction,removed);
            for(int i=0;i<n;++i)residual[i]=initial_residual[i]-removed[i];}
        for(int i=0;i<n;++i)x[i]+=correction[i];
        return x;};
    if(do_planes){const auto pt=now();MaxwellExteriorProjector projection(system.mesh,ka,7,10,2.);MaxwellExteriorFmm fmm(system,projection);
        report<<"{\"stage\":\"plane_projection_setup\",\"seconds\":"<<elapsed(pt)<<",\"correction_bytes\":"<<projection.correction_bytes()<<"}\n"<<std::flush;
        for(int j:{0,2}){const V& rhs=rhs_rays[j];V reference,zero(n);solve("plane",j,"zero",rhs,zero,local,0,reference,nullptr);
            solve("plane",j,"raw_mbs",rhs,seeds[j],local,seed_cost[j],reference,nullptr);
            auto t=now();ImageSpace fit(offsets);add(fit,seeds[j]);V prhs(n),guess,removed;local(rhs.data(),prhs.data());add(fit,prhs);
            coarse(fit,rhs,guess,removed);const double fit_s=elapsed(t);
            solve("plane",j,"minres_raw_local",rhs,guess,local,seed_cost[j]+fit_s,reference,nullptr);
            t=now();V ext(n);projection.apply(seeds[j].data(),ext.data(),[&](const Complex*a,const Complex*b,Complex*p,Complex*g,Complex*q,Complex*h){fmm(a,b,p,g,q,h);});
            V incident=projection.incident(rhs);for(int i=0;i<n;++i)ext[i]+=incident[i];add(fit,ext);coarse(fit,rhs,guess,removed);
            const double ext_s=elapsed(t);
            solve("plane",j,"minres_raw_external_local",rhs,guess,local,seed_cost[j]+fit_s+ext_s,reference,nullptr);
            for(int steps:{4,8}){t=now();guess=smooth(rhs,seeds[j],steps);const double cost=elapsed(t);
                solve("plane",j,"mbs_gcr"+std::to_string(steps),rhs,guess,local,seed_cost[j]+cost,reference,nullptr);}
            t=now();guess=smooth(rhs,zero,4);const double cost=elapsed(t);
            solve("plane",j,"zero_gcr4",rhs,guess,local,cost,reference,nullptr);
        }
    }
    std::ifstream packet(packet_file,std::ios::binary);std::vector<V> rhs_T;
    while(true){V rhs(n);packet.read(reinterpret_cast<char*>(rhs.data()),n*sizeof(Complex));
        if(packet.gcount()==0)break;
        if(packet.gcount()!=std::streamsize(n*sizeof(Complex)))throw std::runtime_error("partial packed RHS");
        rhs_T.push_back(std::move(rhs));}
    if(rhs_T.empty()||rhs_T.size()>16)throw std::runtime_error("invalid packed RHS count");
    std::vector<V> references(rhs_T.size());
    const auto batch=[&](const std::string& name,const Action& inverse,SectorRecycleSpace* recycle){
        if(name!="zero"&&!wanted(name))return;
        for(size_t j=0;j<rhs_T.size();++j)
        if(!solve("T",j,name,rhs_T[j],V(n),inverse,0,references[j],recycle))break;};
    batch("zero",local,nullptr);SectorRecycleSpace plain_recycle(sectors.offsets,16);batch("base_recycle16",local,&plain_recycle);
    for(bool reuse:{false,true}){const std::string label=reuse?"staged_gcr4_recycle16":"staged_gcr4";
        if(!wanted(label))continue;
        SectorRecycleSpace space(sectors.offsets,16);
        for(size_t j=0;j<rhs_T.size();++j){const auto before=now();V guess=smooth(rhs_T[j],V(n),4);const double cost=elapsed(before);
            if(!solve("T",j,label,rhs_T[j],guess,local,cost,references[j],reuse?&space:nullptr))break;}}
    // Training excludes both plane-wave holdout directions. Extend the ray
    // space with dielectric equation residuals and one Krylov image.
    if(ray_spaces){ImageSpace raw(offsets);std::vector<V> residual_directions;double raw_cost=0;const auto training=now();
    for(size_t j=4;j<seeds.size();++j){add(raw,seeds[j]);raw_cost+=seed_cost[j];}
    raw_cost+=elapsed(training);audit("raw",raw);auto t=now();
    for(size_t j=4;j<seeds.size();++j){V image(n),residual=rhs_rays[j],z(n);action(seeds[j].data(),image.data());
        for(int i=0;i<n;++i)residual[i]-=image[i];
        local(residual.data(),z.data());residual_directions.push_back(std::move(z));}
    ImageSpace corrected=raw;for(const auto& z:residual_directions)add(corrected,z);const double residual_cost=elapsed(t);
    audit("raw_residual",corrected);t=now();ImageSpace expanded=corrected;
    for(const auto& z:residual_directions){V image(n),v(n);action(z.data(),image.data());local(image.data(),v.data());add(expanded,v);}
    const double krylov_cost=elapsed(t);audit("raw_residual_krylov",expanded);
    report<<"{\"stage\":\"ray_space_setup\",\"raw_s\":"<<raw_cost<<",\"residual_s\":"<<raw_cost+residual_cost
          <<",\"krylov_s\":"<<raw_cost+residual_cost+krylov_cost<<"}\n"<<std::flush;
    const auto space_batch=[&](const char* label,const ImageSpace& space){V guess,removed,fine(n),r(n);
        const Action inverse=[&](const Complex* b,Complex* x){std::copy(b,b+n,r.begin());coarse(space,r,guess,removed);
            for(int i=0;i<n;++i)r[i]=b[i]-removed[i];
            local(r.data(),fine.data());for(int i=0;i<n;++i)x[i]=guess[i]+fine[i];};
        batch(label,inverse,nullptr);};
    space_batch("ray_two_level192",raw);space_batch("ray_residual_two_level384",corrected);space_batch("ray_krylov_two_level576",expanded);
    }
    // These are fixed linear polynomials of A P. Extra FMM calls are timed.
    for(int degree:{2,3}){if(!wanted("polynomial"+std::to_string(degree)))continue;
        V r(n),z(n),az(n);const Action polynomial=[&](const Complex* b,Complex* x){std::copy(b,b+n,r.begin());std::fill(x,x+n,Complex(0));
            for(int k=0;k<degree;++k){local(r.data(),z.data());for(int i=0;i<n;++i)x[i]+=z[i];
                if(k+1<degree){action(z.data(),az.data());for(int i=0;i<n;++i)r[i]-=az[i];}}};
        batch("polynomial"+std::to_string(degree),polynomial,nullptr);}
    for(bool mbs:{false,true}){if(!wanted(mbs?"mbs_trained_recycle16":"incident_trained_recycle16"))continue;
        SectorRecycleSpace trained(sectors.offsets,16);const auto before=now();double source_cost=0;
        for(int j:{4,5,6,7}){V residual=rhs_rays[j],image(n),x(n);
            if(mbs){action(seeds[j].data(),image.data());for(int i=0;i<n;++i)residual[i]-=image[i];source_cost+=seed_cost[j];}
            // Incomplete solves collect Ritz directions only, never T columns.
            const auto learned=bem_tmatrix::solve_prism_sectors(action,strict_action,local,sectors,residual.data(),x.data(),24,tol,24,false,false,&trained);
            report<<"{\"stage\":\"recycle_training\",\"mbs\":"<<(mbs?"true":"false")<<",\"ray\":"<<j<<",\"iterations\":"<<learned.iterations
                  <<",\"training_residual\":"<<learned.residual<<",\"rank\":"<<trained.rank()<<"}\n"<<std::flush;
        }
        report<<"{\"stage\":\"recycle_training_setup\",\"mbs\":"<<(mbs?"true":"false")<<",\"seconds\":"<<elapsed(before)+source_cost<<"}\n"<<std::flush;
        batch(mbs?"mbs_trained_recycle16":"incident_trained_recycle16",local,&trained);
    }
    if(wanted("zero_repeat"))for(size_t j=0;j<rhs_T.size();++j)
        solve("T",j,"zero_repeat",rhs_T[j],V(n),local,0,references[j],nullptr);
    system.cleanup();return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
