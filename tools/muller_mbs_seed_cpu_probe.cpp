// CPU or FMM diagnostic; no production T or solver settings are modified.
// For high ka, supply the validated production BEM_FMM_BANDED_* settings.
// A small FP64 iterative residual alone does not validate FMM truncation.
#include "mbs_surface_seed.h"
#include "muller_dense.h"
#include "muller_mbj.h"
#include "streaming_host_gmres.h"
#include "quadrature.h"
#include "maxwell_exterior_projector.h"
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
    if(argc!=5)throw std::runtime_error("Usage: muller_mbs_seed_cpu_probe REF KA RAY_FILE_LIST REPORT");
    const int ref=std::stoi(argv[1]);const double ka=std::stod(argv[2]),tol=2e-6;
    const char* packet_file=std::getenv("BEM_PROBE_RHS_FILE");
    const bool sector_solver=std::getenv("BEM_MBS_SYMMETRY")!=nullptr;
    const bool sector_coarse=std::getenv("BEM_MBS_COARSE_SECTORS")!=nullptr;
    const bool project_mbs=std::getenv("BEM_MBS_CALDERON")!=nullptr;
    const int rhs_count=std::getenv("BEM_PROBE_RHS_COUNT")?std::stoi(std::getenv("BEM_PROBE_RHS_COUNT")):(packet_file?2:4);
    const int method_mask=std::getenv("BEM_MBS_METHOD_MASK")?std::stoi(std::getenv("BEM_MBS_METHOD_MASK")):15;
    if(rhs_count<1||rhs_count>(packet_file?2:4)||!(method_mask&1))throw std::runtime_error("invalid probe selection");
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
    if(!packet_file&&!(method_mask&8))rays.resize(2);
    const auto start=now();MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
    const auto linear=regular_prism(6,aspect,ref,1.,0,false,true);
#ifdef BEM_MBS_FMM
    MullerFmmOperator system;
    const int max_leaf=std::getenv("BEM_TMATRIX_MAX_LEAF")?std::stoi(std::getenv("BEM_TMATRIX_MAX_LEAF")):64;
    if(max_leaf<1)throw std::runtime_error("invalid FMM leaf capacity");
    system.init(linear,ka,Complex(ri,0),options,7,6,5,max_leaf,false,2,0,.96,nullptr,3);
    system.set_fmm_near_fp32(true);
    const int n=system.system_dofs;MullerMbjPreconditioner mbj;mbj.build(system,50);
    const Action action=[&](const Complex* v,Complex* w){system.matvec(v,w);};
    const Action strict_action=[&](const Complex* v,Complex* w){system.matvec_strict(v,w);};
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
    const auto projection_setup_start=now();std::unique_ptr<MaxwellExteriorProjector> projection;
    MaxwellExteriorProjector::ScalarPair scalar;
#ifdef BEM_MBS_FMM
    std::unique_ptr<MaxwellExteriorFmm> exterior_fmm;
#endif
    if(project_mbs){projection.reset(new MaxwellExteriorProjector(system.mesh,ka,7,10,2.));
#ifdef BEM_MBS_FMM
        exterior_fmm.reset(new MaxwellExteriorFmm(system,*projection));
        scalar=[&](const Complex*a,const Complex*b,Complex*p,Complex*g,Complex*q,Complex*h){(*exterior_fmm)(a,b,p,g,q,h);};
#else
        scalar=[&](const Complex*a,const Complex*b,Complex*p,Complex*g,Complex*q,Complex*h){projection->direct_pair(a,b,p,g,q,h);};
#endif
    }
    const double projection_setup=elapsed(projection_setup_start);
    std::ofstream report(argv[4]);if(!report)throw std::runtime_error("cannot write probe report");
    report<<std::setprecision(17)<<"{\"diagnostic_only\":true,\"ref\":"<<ref<<",\"ka\":"<<ka<<",\"ri\":"<<ri
          <<",\"aspect\":"<<aspect<<",\"dofs\":"<<n<<",\"setup_s\":"<<setup<<",\"tolerance\":"<<tol<<"}\n";report.flush();
    report<<"{\"sector_solver\":"<<(sector_solver?"true":"false")<<",\"rhs_kind\":\""<<(packet_file?"packed_T_first_codes":"plane_waves")<<"\"}\n";
    report<<"{\"calderon_seed\":"<<(project_mbs?"true":"false")<<",\"projection_setup_s\":"<<projection_setup
          <<",\"projection_correction_bytes\":"<<(projection?projection->correction_bytes():0)<<",\"rhs_count\":"<<rhs_count<<",\"method_mask\":"<<method_mask<<"}\n";
#ifdef BEM_MBS_FMM
    report<<"{\"fmm_max_leaf\":"<<max_leaf<<",\"banded_fmm\":"<<(system.banded_fmm?"true":"false")
          <<",\"fmm_order_exterior\":"<<system.fmm_exterior.p<<",\"fmm_order_interior\":"<<system.fmm_interior.p<<"}\n";
#endif
    std::vector<V> seeds;std::vector<double> costs;double field_num=0,field_den=0;
    for(const auto& r:rays)for(int pol=0;pol<2;++pol){const auto before=now();V seed=r->project(system.mesh,ka,pol);
        if(project_mbs){V projected(n);projection->apply(seed.data(),projected.data(),scalar);
            V incident=projection->incident(muller_nodal_planewave_rhs(system.mesh,ka,r->polarization(pol),r->direction,13));
            for(int i=0;i<n;++i)seed[i]=incident[i]+projected[i];}
        seeds.push_back(std::move(seed));costs.push_back(elapsed(before)+r->trace_seconds*.5);
        if(std::abs(ri-1)<1e-12){const auto quad=tri_quadrature(7);const Vec3 p=r->polarization(pol),h=r->direction.cross(p);
            for(size_t t=0;t<system.mesh.elements.size();++t)for(int q=0;q<quad.npts;++q){auto f=evaluate_muller_frame(system.mesh,t,quad.pts[q][0],quad.pts[q][1]);
                std::array<Complex,3> e,g;r->fields(r->face(f.normal,f.position),f.position,ka,pol,e,g);
                const Complex phase=std::exp(Complex(0,ka*r->direction.dot(f.position)));
                const double pv[]={p.x,p.y,p.z},hv[]={h.x,h.y,h.z};
                for(int j=0;j<3;++j){field_num+=std::norm(e[j]-phase*pv[j])+std::norm(g[j]-phase*hv[j]);field_den+=pv[j]*pv[j]+hv[j]*hv[j];}}}}
    if(std::abs(ri-1)<1e-12){const double error=std::sqrt(field_num/field_den);report<<"{\"no_contrast_field_relative_rms\":"<<error<<"}\n";report.flush();
        if(error>1e-8)throw std::runtime_error("MBS exported phase/polarization fails no-contrast field test");}
    // QR of A U, with matching updates of U, provides an operator-tested
    // frozen coarse correction; directions 0 and 1 are excluded from training.
    std::vector<int> offsets{0,n};
#ifdef BEM_MBS_FMM
    if(sector_coarse)offsets.assign(sectors.offsets.begin(),sectors.offsets.end());
#endif
    std::vector<V> u,c;double training_s=0;int coarse_rank=0;const auto training_start=now();
    for(size_t j=4;j<seeds.size();++j){V v=seeds[j],w(n);action(v.data(),w.data());const double original=norm(w.data(),n);
#ifdef BEM_MBS_FMM
        if(sector_coarse){V encoded(n);sectors.encode(v.data(),encoded.data(),false);v.swap(encoded);
            sectors.encode(w.data(),encoded.data(),true);w.swap(encoded);}
        if(sector_coarse&&j==4){V strict_full(n),strict_encoded(n);
            strict_action(seeds[j].data(),strict_full.data());sectors.encode(strict_full.data(),strict_encoded.data(),true);
            for(size_t s=0;s+1<offsets.size();++s){V projected(n),physical(n),expected(n),observed(n);
                std::copy(v.begin()+offsets[s],v.begin()+offsets[s+1],projected.begin()+offsets[s]);
                sectors.decode(projected.data(),physical.data(),false);
                for(int full=0;full<2;++full){std::fill(projected.begin(),projected.end(),Complex(0));const V& image=full?strict_encoded:w;
                    std::copy(image.begin()+offsets[s],image.begin()+offsets[s+1],projected.begin()+offsets[s]);
                    sectors.decode(projected.data(),expected.data(),true);(full?strict_action:action)(physical.data(),observed.data());
                    for(int i=0;i<n;++i)observed[i]-=expected[i];
                    const double error=norm(observed.data(),n)/std::max(norm(expected.data(),n),1e-300);
                    report<<"{\"symmetry_audit_sector\":"<<s<<",\"strict\":"<<(full?"true":"false")
                          <<",\"unorthogonalized_sector_action_error\":"<<error<<"}\n";
                }
            }report.flush();}
#endif
        bool any=false;
        for(size_t s=0;s+1<offsets.size();++s){const int begin=offsets[s],end=offsets[s+1],length=end-begin;
            for(int pass=0;pass<2;++pass)for(size_t a=0;a<c.size();++a){const auto alpha=dot(c[a].data()+begin,w.data()+begin,length);
                for(int i=begin;i<end;++i){w[i]-=alpha*c[a][i];v[i]-=alpha*u[a][i];}}
            const double wn=norm(w.data()+begin,length);
            if(wn>original*1e-9){for(int i=begin;i<end;++i){v[i]/=wn;w[i]/=wn;}++coarse_rank;any=true;}
            else for(int i=begin;i<end;++i){v[i]=0.;w[i]=0.;}}
        if(any){u.push_back(std::move(v));c.push_back(std::move(w));}
        training_s+=costs[j];}
    training_s+=elapsed(training_start);
    report<<"{\"coarse_rank\":"<<coarse_rank<<",\"coarse_sectors\":"<<(offsets.size()-1)
          <<",\"coarse_setup_including_MBS_s\":"<<training_s<<",\"holdout_directions\":2}\n";
    // Solution and equation coordinates have opposite mirror parity. Their
    // respective transforms align the sectors of U and A U correctly.
    const auto coarse=[&](const Complex* rhs,Complex* out,Complex* removed){V encoded(rhs,rhs+n),guess(n),projected(n);
#ifdef BEM_MBS_FMM
        if(sector_coarse)sectors.encode(rhs,encoded.data(),true);
#endif
        for(size_t s=0;s+1<offsets.size();++s){const int begin=offsets[s],end=offsets[s+1];
            for(size_t a=0;a<c.size();++a){const auto alpha=dot(c[a].data()+begin,encoded.data()+begin,end-begin);
                for(int i=begin;i<end;++i){guess[i]+=alpha*u[a][i];if(removed)projected[i]+=alpha*c[a][i];}}}
#ifdef BEM_MBS_FMM
        if(sector_coarse){sectors.decode(guess.data(),out,false);if(removed)sectors.decode(projected.data(),removed,true);return;}
#endif
        std::copy(guess.begin(),guess.end(),out);if(removed)std::copy(projected.begin(),projected.end(),removed);
    };
    V scratch(n),fine(n);
    const Action inverse=[&](const Complex* rhs,Complex* out){coarse(rhs,out,scratch.data());
        for(int i=0;i<n;++i)scratch[i]=rhs[i]-scratch[i];
        local(scratch.data(),fine.data());for(int i=0;i<n;++i)out[i]+=fine[i];};
    std::ifstream packet;if(packet_file)packet.open(packet_file,std::ios::binary);
    for(int j=0;j<rhs_count;++j){const auto& ray=*rays[j/2];V rhs=muller_nodal_planewave_rhs(system.mesh,ka,ray.polarization(j%2),ray.direction,13),predicted(n);
        if(packet_file){packet.read(reinterpret_cast<char*>(rhs.data()),n*sizeof(Complex));if(!packet)throw std::runtime_error("packet input size");
            const auto before=now();coarse(rhs.data(),seeds[j].data(),predicted.data());costs[j]=elapsed(before);}
        const double bn=norm(rhs.data(),n);V image(n),reference,delta(n);
        const auto scale_start=now();action(seeds[j].data(),image.data());
        const double image_norm=norm(image.data(),n);
        const Complex alpha=image_norm>0?dot(image.data(),rhs.data(),n)/(image_norm*image_norm):Complex(0);
        const double scale_seconds=elapsed(scale_start);
        if(packet_file){for(int i=0;i<n;++i)delta[i]=image[i]-predicted[i];
            const double error=norm(delta.data(),n)/std::max(norm(predicted.data(),n),1e-300);
            report<<"{\"rhs\":"<<j<<",\"coarse_operator_relative_error\":"<<error<<"}\n";
            if(!std::isfinite(error)||error>1e-3)throw std::runtime_error("coarse operator/symmetry consistency failed");}
        for(int i=0;i<n;++i)delta[i]=rhs[i]-image[i];
        const double initial=norm(delta.data(),n)/bn;
        for(int method=0;method<4;++method){if(!(method_mask&(1<<method)))continue;
            if(packet_file&&(method==2||(method==3&&!sector_coarse)))continue;
            V x(n);if(method==1||method==2){x=seeds[j];if(method==2)for(auto& value:x)value*=alpha;}
            const char* label=method==0?"zero":method==1?(packet_file?"mbs_coarse_guess":"mbs_raw"):method==2?"mbs_minres_scale":"mbs_two_level";
            HostResult result;
#ifdef BEM_MBS_FMM
            if(sector_solver && (method!=3||sector_coarse)){const auto solve_start=now();V correction(n),residual=rhs;
                double rn=bn;
                if(method==1||method==2){strict_action(x.data(),delta.data());for(int i=0;i<n;++i)residual[i]-=delta[i];rn=norm(residual.data(),n);}
                if(rn>tol*bn){result=bem_tmatrix::solve_prism_sectors(action,strict_action,method==3?inverse:local,sectors,residual.data(),correction.data(),64,tol*bn/rn,1000,false,false);
                    for(int i=0;i<n;++i)x[i]+=correction[i];}
                result.seconds=elapsed(solve_start);
            }else
#endif
            result=gmres(action,strict_action,method==3?inverse:local,rhs.data(),x.data(),n,64,tol,1000,false,method==1||method==2);
            strict_action(x.data(),delta.data());for(int i=0;i<n;++i)delta[i]-=rhs[i];const double strict=norm(delta.data(),n)/bn;
            if(method==0){reference=x;
                if(project_mbs){V projected(n);projection->apply(x.data(),projected.data(),scalar);V incident=projection->incident(rhs);
                    for(int i=0;i<n;++i)projected[i]+=incident[i]-x[i];
                    report<<"{\"rhs\":"<<j<<",\"converged_BEM_exterior_trace_consistency\":"<<norm(projected.data(),n)/norm(x.data(),n)
                          <<",\"projection_mass_residual\":"<<projection->last_mass_residual()<<"}\n";}}
            double seed_difference=0;
            for(int i=0;i<n;++i)delta[i]=seeds[j][i]-reference[i];
            seed_difference=norm(delta.data(),n)/norm(reference.data(),n);
            for(int i=0;i<n;++i)delta[i]=x[i]-reference[i];
            const double difference=norm(delta.data(),n)/norm(reference.data(),n);
            report<<"{\"rhs\":"<<j<<",\"method\":\""<<label<<"\",\"iterations\":"<<result.iterations
                  <<",\"solve_s\":"<<result.seconds<<",\"seed_s\":"<<((method==1||method==2)?costs[j]:0)
                  <<",\"scale_setup_s\":"<<(method==2?scale_seconds:0)
                  <<",\"raw_initial_residual\":"<<initial<<",\"strict_residual\":"<<strict
                  <<",\"raw_seed_relative_current_error\":"<<seed_difference
                  <<",\"relative_solution_difference\":"<<difference<<",\"scale_real\":"<<alpha.real()<<",\"scale_imag\":"<<alpha.imag()<<"}\n";report.flush();
            std::cout<<label<<" rhs="<<j<<" iterations="<<result.iterations<<" residual="<<strict<<" seconds="<<result.seconds<<std::endl;
            if(!std::isfinite(strict)||!std::isfinite(difference)||strict>tol||difference>1e-3)
                throw std::runtime_error("corrected BEM solution failed acceptance");
        }
    }
    if(packet_file&&rhs_count==2&&packet.peek()!=std::char_traits<char>::eof())throw std::runtime_error("unexpected trailing packet data");
#ifdef BEM_MBS_FMM
    system.cleanup();
#endif
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
