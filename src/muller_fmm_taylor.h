#pragma once
#include "host_fmm_taylor.h"
#include "host_prism_face_taylor.h"
#include "muller_taylor_correction.h"

namespace bem_cpu {
// Complete matrix-free Muller moment action for a fixed dielectric and mesh.
// Output[j]=sum_{q=0}^j A_q input[j-q]. This CPU reference uses the prepared
// FMM bands or the explicitly selected same-face FFT/cross-FMM path.
// No dense Aq is stored.
template<size_t N> class MullerTaylorAction {
    using Band=TaylorFmmBand<N>;
    using Charge=typename Band::Charge;
    using Field=typename Band::Field;
    const MullerFmmOperator& op;
    const double k;
    const C index;
    std::vector<Vec3> points;
    std::vector<MullerBasisSample> basis,mass_basis;
    struct Support{int point,local;};
    std::vector<std::vector<Support>> rows,mass_rows;
    std::vector<std::unique_ptr<Band>> exterior,interior;
    std::unique_ptr<TaylorPrismFaceAction<N>> face_exterior,face_interior;
    std::vector<Charge> charges,mass_current;
    std::vector<Field> exterior_fields,interior_fields;
    static double checked_k(const MullerFmmOperator& op){
        if(op.use_pfft||!op.fmm_exterior.initialized||!op.fmm_interior.initialized||
           op.k_exterior.imag()!=0||op.k_exterior.real()<=0||op.epsilon_exterior!=C(1)||op.mu_exterior!=C(1)||op.mu_interior!=C(1)||
           std::abs(op.epsilon_interior-std::pow(op.k_interior/op.k_exterior,2))>1e-12*std::max(1.,std::abs(op.epsilon_interior)))
            throw std::invalid_argument("Taylor Muller requires prepared FMM and fixed nonmagnetic dielectric with real positive exterior k");
        return op.k_exterior.real();
    }
public:
    bem_taylor::SparseCorrection<N> correction;
    double central_correction_error=-1;
    double face_fft_seconds=0,face_cross_seconds=0,face_packing_seconds=0;
    explicit MullerTaylorAction(const MullerFmmOperator& prepared,int duffy,bool reuse=true,bool use_face_fft=false)
        :op(prepared),k(checked_k(prepared)),index(op.k_interior/op.k_exterior),correction(op.mesh,k,index,op.quadrature_order,duffy,reuse){
        rows.resize(op.current_dofs);mass_rows.resize(op.current_dofs);
        auto prepare=[&](const std::vector<MullerFmmQuadraturePoint>& qs,std::vector<MullerBasisSample>& bs,std::vector<std::vector<Support>>& support){
            bs.resize(qs.size());for(size_t p=0;p<qs.size();++p){bs[p]=evaluate_muller_basis(op.mesh,qs[p].element,qs[p].sample);
                for(int j=0;j<bs[p].count;++j)support[bs[p].dofs[j]].push_back({int(p),j});}
        };
        prepare(op.quadrature,basis,rows);prepare(op.mass_quadrature,mass_basis,mass_rows);
        for(const auto& q:op.quadrature)points.push_back(q.sample.position);
        charges.resize(points.size());mass_current.resize(mass_basis.size());exterior_fields.resize(points.size());interior_fields.resize(points.size());
        if(use_face_fft){
            if(!op.prism_fft_exterior||!op.prism_fft_interior)throw std::invalid_argument("Taylor face FFT needs prepared cross trees");
            face_exterior.reset(new TaylorPrismFaceAction<N>(*op.prism_fft_exterior,points,C(1)));
            face_interior.reset(new TaylorPrismFaceAction<N>(*op.prism_fft_interior,points,index));
        }else{
        auto append=[&](std::vector<std::unique_ptr<Band>>& list,const HelmholtzFMM& f,C slope){list.emplace_back(new Band(f,points,slope));};
        append(exterior,op.fmm_exterior,C(1));append(interior,op.fmm_interior,index);
        if(op.banded_fmm_middle){append(exterior,op.fmm_exterior_middle,C(1));append(interior,op.fmm_interior_middle,index);}
        if(op.banded_fmm){append(exterior,op.fmm_exterior_coarse,C(1));append(interior,op.fmm_interior_coarse,index);}
        }
        if(!op.correction.row_offsets.empty()){
            double delta=0,norm=0;for(int row=0;row<op.current_dofs;++row){std::map<int,std::array<C,3>> original;
                for(int q=op.correction.row_offsets[row];q<op.correction.row_offsets[row+1];++q){const auto& e=op.correction.entries[q];original[e.column]={{C(0,1)/k*e.k1,e.k2_epsilon,e.k2_mu}};}
                for(int q=correction.offsets[row];q<correction.offsets[row+1];++q){const auto& e=correction.entries[q];const auto ref=original[e.column];
                    delta+=std::norm(e.value.diagonal[0]-ref[0])+std::norm(e.value.epsilon[0]-ref[1])+std::norm(e.value.mu[0]-ref[2]);
                    for(C z:ref)norm+=std::norm(z);
                    original.erase(e.column);}
                for(auto item:original)for(C z:item.second){delta+=std::norm(z);norm+=std::norm(z);}
            }
            central_correction_error=std::sqrt(delta/std::max(norm,1e-280));
            if(!std::isfinite(central_correction_error)||central_correction_error>2e-8)throw std::runtime_error("Taylor correction differs from prepared Duffy operator");
        }
    }
    size_t workspace_bytes()const{
        size_t bytes=correction.storage_bytes()+(charges.size()+mass_current.size())*sizeof(Charge)+(exterior_fields.size()+interior_fields.size())*sizeof(Field);
        for(const auto* bands:{&exterior,&interior})for(const auto& b:*bands)bytes+=b->workspace_bytes();
        if(face_exterior)bytes+=face_exterior->workspace_bytes();
        if(face_interior)bytes+=face_interior->workspace_bytes();
        return bytes;
    }
    void apply(const std::vector<std::vector<C>>& inputs,std::vector<std::vector<C>>& output){
        if(inputs.size()!=N+1)throw std::invalid_argument("Muller Taylor moment count");
        for(const auto& v:inputs)if(int(v.size())!=op.system_dofs)throw std::invalid_argument("Muller Taylor vector dimensions");
        const int n=op.current_dofs;const auto input=inputs;
        auto project=[&](const std::vector<MullerBasisSample>& bs,const std::vector<MullerFmmQuadraturePoint>& qs,std::vector<Charge>& values){
#pragma omp parallel for schedule(static)
            for(size_t p=0;p<bs.size();++p){Charge current{};
                for(int a=0;a<bs[p].count;++a){const Vec3& v=bs[p].values[a];const int dof=bs[p].dofs[a];
                    for(size_t j=0;j<=N;++j)for(int c=0;c<2;++c){const C q=input[j][dof+c*n]*qs[p].weight;
                        current[j][3*c]+=v.x*q;current[j][3*c+1]+=v.y*q;current[j][3*c+2]+=v.z*q;}}
                values[p]=current;
            }
        };
        project(basis,op.quadrature,charges);project(mass_basis,op.mass_quadrature,mass_current);
        std::fill(exterior_fields.begin(),exterior_fields.end(),Field{});std::fill(interior_fields.begin(),interior_fields.end(),Field{});
        face_fft_seconds=face_cross_seconds=face_packing_seconds=0;
        if(face_exterior){
            face_exterior->add(charges,exterior_fields);face_interior->add(charges,interior_fields);
            for(const auto* face:{face_exterior.get(),face_interior.get()}){
                face_fft_seconds+=face->self_seconds;face_cross_seconds+=face->cross_seconds;face_packing_seconds+=face->packing_seconds;}
        }else{
            for(auto& band:exterior)band->add(charges,exterior_fields);
            for(auto& band:interior)band->add(charges,interior_fields);
        }
        output.assign(N+1,std::vector<C>(op.system_dofs));
#pragma omp parallel for schedule(static)
        for(int row=0;row<n;++row){
            C mass[N+1][2]{},k1[N+1][2]{},k2e[N+1][2]{},k2m[N+1][2]{};
            for(auto s:mass_rows[row]){const Vec3& v=mass_basis[s.point].values[s.local];
                for(size_t j=0;j<=N;++j)for(int c=0;c<2;++c){const auto& q=mass_current[s.point][j];mass[j][c]+=v.x*q[3*c]+v.y*q[3*c+1]+v.z*q[3*c+2];}}
            for(auto s:rows[row]){const auto& point=op.quadrature[s.point];const Vec3 v=basis[s.point].values[s.local],normal=point.sample.normal,rotated=v.cross(normal);
                const double rot[3]={rotated.x,rotated.y,rotated.z},curl[3]={v.x*normal.y-normal.x*v.y,v.x*normal.z-normal.x*v.z,v.y*normal.z-normal.y*v.z};
                for(size_t j=0;j<=N;++j){const auto& e=exterior_fields[s.point][j];const auto& i=interior_fields[s.point][j];
                    for(int c=0;c<2;++c)for(int a=0;a<3;++a){k1[j][c]+=point.weight*rot[a]*(e[6*c+3+a]-i[6*c+3+a]);
                        k2e[j][c]+=point.weight*curl[a]*(e[6*c+a]-op.epsilon_interior*i[6*c+a]);k2m[j][c]+=point.weight*curl[a]*(e[6*c+a]-i[6*c+a]);}}
            }
            C diagonal[N+1][2]{};
            for(size_t j=0;j<=N;++j){for(int c=0;c<2;++c)diagonal[j][c]=(C(0,1)*k1[j][c]-(j?diagonal[j-1][c]:C(0)))/k;
                output[j][row]=diagonal[j][0]+.5*(C(1)+op.epsilon_interior)*mass[j][1]+k2e[j][1];
                output[j][n+row]=mass[j][0]+k2m[j][0]-diagonal[j][1];}
            for(int e=correction.offsets[row];e<correction.offsets[row+1];++e){const auto& entry=correction.entries[e];const auto& v=entry.value;
                for(size_t j=0;j<=N;++j)for(size_t q=0;q<=j;++q){const C x=input[j-q][entry.column],y=input[j-q][n+entry.column];
                    output[j][row]+=v.diagonal[q]*x+v.epsilon[q]*y;output[j][n+row]+=v.mu[q]*x-v.diagonal[q]*y;}}
        }
    }
};
}
