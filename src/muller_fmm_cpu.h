#ifndef BEM_MULLER_FMM_CPU_H
#define BEM_MULLER_FMM_CPU_H
#include "host_fmm_band.h"
#include "host_prism_face_fft.h"
namespace bem_cpu {
class MullerAction {
    const MullerFmmOperator& op;
    std::vector<Vec3> points;
    std::vector<MullerBasisSample> basis,mass_basis;
    struct Support { int point,local; };
    std::vector<std::vector<Support>> rows,mass_rows;
    std::vector<std::unique_ptr<FmmBand>> exterior,interior;
    std::unique_ptr<PrismFaceAction> face_exterior,face_interior;
    std::vector<Six> charges,mass_current;
    std::vector<Twelve> exterior_fields,interior_fields;
public:
    std::array<double,6> fmm_times{};
    double project_s=0,fields_s=0,test_s=0;
    double face_fft_s=0,face_cross_s=0,face_packing_s=0;
    double face_phase_s=0;
    size_t face_phase_evaluations=0;
    int merged_angular_bands=0;
    int avx512_near_bands=0;
    static size_t estimated_workspace_bytes(const MullerFmmOperator& op,size_t phase_budget,
                                            bool merge_equal_angles=true) {
        const size_t points=op.quadrature.size(),mass=op.mass_quadrature.size();
        size_t bytes=(points+mass)*(sizeof(Six)+sizeof(MullerBasisSample))+2*points*sizeof(Twelve);
        const bool merge=merge_equal_angles && op.banded_fmm_middle &&
            op.fmm_exterior.L==op.fmm_exterior_middle.L && op.fmm_interior.L==op.fmm_interior_middle.L;
        const auto band=[&](const HelmholtzFMM& f,bool forced_far=false) {
            bool far=forced_far;for(const auto& level:f.m2l_batches)far=far||level.n_pairs;
            if(far) {
                bytes+=2*size_t(f.n_nodes)*f.L*sizeof(Six);
                const size_t phases=points*size_t(f.L)*sizeof(C);if(phases<=phase_budget)bytes+=phases;
            }
            if(f.p2p_enabled)bytes+=15*points*sizeof(double);
        };
        band(op.fmm_exterior,merge);band(op.fmm_interior,merge);
        if(op.banded_fmm_middle&&!merge){band(op.fmm_exterior_middle);band(op.fmm_interior_middle);}
        if(op.banded_fmm){band(op.fmm_exterior_coarse);band(op.fmm_interior_coarse);}
        // Conservative allowance for support indices, near lists and vector
        // bookkeeping. Shared immutable operator storage is already resident.
        return bytes+points*1024+size_t(op.system_dofs)*128;
    }
    size_t workspace_bytes() const {
        size_t n=(charges.size()+mass_current.size())*sizeof(Six)
            +(exterior_fields.size()+interior_fields.size())*sizeof(Twelve)
            +(basis.size()+mass_basis.size())*sizeof(MullerBasisSample);
        for(const auto* bands:{&exterior,&interior})for(const auto& band:*bands)n+=band->workspace_bytes();
        if(face_exterior)n+=face_exterior->workspace_bytes();
        if(face_interior)n+=face_interior->workspace_bytes();
        return n; // excludes shared immutable operator metadata and support indices
    }
    // Phase budget is PER active band. Total cache use is reported by workspace_bytes().
    explicit MullerAction(const MullerFmmOperator& prepared,size_t phase_budget=64*1024*1024,
                          bool merge_equal_angles=true,bool use_face_fft=false,
                          bool batch_face_fft=false,int face_tile=32):op(prepared) {
        if(batch_face_fft && !use_face_fft)throw std::invalid_argument("CPU batched faces require face FFT");
        if(op.use_pfft || !op.fmm_exterior.initialized || !op.fmm_interior.initialized)
            throw std::runtime_error("CPU action needs retained FMM trees (use strict-only face FFT)");
        rows.resize(op.current_dofs);mass_rows.resize(op.current_dofs);
        auto prepare=[&](const std::vector<MullerFmmQuadraturePoint>& qs,
                         std::vector<MullerBasisSample>& bs,std::vector<std::vector<Support>>& supports) {
            bs.resize(qs.size());
            for(size_t p=0;p<qs.size();++p) {
                bs[p]=evaluate_muller_basis(op.mesh,qs[p].element,qs[p].sample);
                for(int j=0;j<bs[p].count;++j)supports[bs[p].dofs[j]].push_back({int(p),j});
            }
        };
        prepare(op.quadrature,basis,rows);prepare(op.mass_quadrature,mass_basis,mass_rows);
        for(const auto& q:op.quadrature)points.push_back(q.sample.position);
        charges.resize(points.size());mass_current.resize(mass_basis.size());
        exterior_fields.resize(points.size());interior_fields.resize(points.size());
        if(use_face_fft){
            if(!op.prism_fft_exterior||!op.prism_fft_interior)throw std::invalid_argument("CPU face FFT needs prepared cross-face geometry");
            face_exterior.reset(new PrismFaceAction(*op.prism_fft_exterior,points,phase_budget,batch_face_fft,face_tile));
            face_interior.reset(new PrismFaceAction(*op.prism_fft_interior,points,phase_budget,batch_face_fft,face_tile));
            return;
        }
        auto append=[&](std::vector<std::unique_ptr<FmmBand>>& bands,const HelmholtzFMM& f,const HelmholtzFMM* upper=nullptr) {
            bands.emplace_back(new FmmBand(f,points,phase_budget,upper));
            if(bands.back()->uses_avx512_near())++avx512_near_bands;
        };
        const bool merge=merge_equal_angles && op.banded_fmm_middle &&
            op.fmm_exterior.L==op.fmm_exterior_middle.L && op.fmm_interior.L==op.fmm_interior_middle.L;
        append(exterior,op.fmm_exterior,merge?&op.fmm_exterior_middle:nullptr);
        append(interior,op.fmm_interior,merge?&op.fmm_interior_middle:nullptr);
        if(merge)merged_angular_bands=2;
        if(op.banded_fmm_middle && !merge) {append(exterior,op.fmm_exterior_middle);append(interior,op.fmm_interior_middle);}
        if(op.banded_fmm) {append(exterior,op.fmm_exterior_coarse);append(interior,op.fmm_interior_coarse);}
    }
    void apply(const C* input,C* output) {
        auto start=Clock::now();const int n=op.current_dofs;
        auto project=[&](const std::vector<MullerBasisSample>& bs,
                         const std::vector<MullerFmmQuadraturePoint>& qs,std::vector<Six>& values) {
#pragma omp parallel for schedule(static)
            for(size_t p=0;p<bs.size();++p) {
                Six current{};
                for(int j=0;j<bs[p].count;++j) {
                    const Vec3& v=bs[p].values[j];const int dof=bs[p].dofs[j];
                    for(int c=0;c<2;++c) {
                        const C q=input[dof+c*n]*qs[p].weight;
                        current[3*c]+=v.x*q;current[3*c+1]+=v.y*q;current[3*c+2]+=v.z*q;
                    }
                }
                values[p]=current;
            }
        };
        project(basis,op.quadrature,charges);project(mass_basis,op.mass_quadrature,mass_current);
        project_s=elapsed(start);start=Clock::now();
        std::fill(exterior_fields.begin(),exterior_fields.end(),Twelve{});
        std::fill(interior_fields.begin(),interior_fields.end(),Twelve{});fmm_times.fill(0);
        face_fft_s=face_cross_s=face_packing_s=face_phase_s=0;face_phase_evaluations=0;
        for(int media=0;media<2;++media) {
            auto& bands=media?interior:exterior;auto& fields=media?interior_fields:exterior_fields;
            auto* face=media?face_interior.get():face_exterior.get();
            if(face){
                face->add(charges,fields);face_fft_s+=face->self_seconds;
                face_cross_s+=face->cross_seconds;face_packing_s+=face->packing_seconds;
                face_phase_s+=face->phase_seconds;face_phase_evaluations+=face->phase_evaluations;
                for(int j=0;j<6;++j)fmm_times[j]+=face->fmm_times[j];
                continue;
            }
            for(auto& band:bands) {
                band->add(charges,fields);
                for(int j=0;j<6;++j)fmm_times[j]+=band->times[j];
            }
        }
        fields_s=elapsed(start);start=Clock::now();
        // Copy protects in-place use; corrections also read the original current.
        std::vector<C> in(input,input+op.system_dofs);
#pragma omp parallel for schedule(static)
        for(int row=0;row<n;++row) {
            C mass[2]{},k1[2]{},k2e[2]{},k2m[2]{};
            for(const auto& s:mass_rows[row]) {
                const Vec3& v=mass_basis[s.point].values[s.local];const Six& q=mass_current[s.point];
                for(int c=0;c<2;++c)mass[c]+=v.x*q[3*c]+v.y*q[3*c+1]+v.z*q[3*c+2];
            }
            for(const auto& s:rows[row]) {
                const auto& point=op.quadrature[s.point];const Vec3& v=basis[s.point].values[s.local];
                const Vec3& normal=point.sample.normal;const Vec3 rotated=v.cross(normal);
                const double rot[3]={rotated.x,rotated.y,rotated.z};
                const double curl[3]={v.x*normal.y-normal.x*v.y,v.x*normal.z-normal.x*v.z,v.y*normal.z-normal.y*v.z};
                const auto& e=exterior_fields[s.point];const auto& i=interior_fields[s.point];
                for(int c=0;c<2;++c)for(int a=0;a<3;++a) {
                    k1[c]+=point.weight*rot[a]*(e[6*c+3+a]-i[6*c+3+a]);
                    k2e[c]+=point.weight*curl[a]*(op.epsilon_exterior*e[6*c+a]-op.epsilon_interior*i[6*c+a]);
                    k2m[c]+=point.weight*curl[a]*(op.mu_exterior*e[6*c+a]-op.mu_interior*i[6*c+a]);
                }
            }
            for(int j=op.correction.row_offsets[row];j<op.correction.row_offsets[row+1];++j) {
                const auto& entry=op.correction.entries[j];
                for(int c=0;c<2;++c) {
                    const C q=in[entry.column+c*n];k1[c]+=entry.k1*q;k2e[c]+=entry.k2_epsilon*q;k2m[c]+=entry.k2_mu*q;
                }
            }
            output[row]=C(0,1)/op.k_exterior*k1[0]+.5*(op.epsilon_interior+op.epsilon_exterior)*mass[1]+k2e[1];
            output[n+row]=.5*(op.mu_interior+op.mu_exterior)*mass[0]+k2m[0]-C(0,1)/op.k_exterior*k1[1];
        }
        test_s=elapsed(start);
    }
};
} // namespace bem_cpu
#endif
