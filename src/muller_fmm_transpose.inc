// Reverse Galerkin projection/assembly around the transposed FMM traversal.
void MullerFmmOperator::matvec_transpose(const cdouble* input,cdouble* output) {
    if(system_dofs<=0 || use_pfft || prism_fft_active())
        throw std::runtime_error("Muller transpose currently requires the FMM backend");
    const std::vector<cdouble> seed(input,input+system_dofs); // permits in-place use
    const int points=int(quadrature.size());
    const cdouble a=cdouble(0,1)/k_exterior;
    std::vector<cdouble> mass;
    apply_mass(seed.data()+current_dofs,mass);
    for(int j=0;j<current_dofs;++j)output[j]=.5*(mu_exterior+mu_interior)*mass[j];
    apply_mass(seed.data(),mass);
    for(int j=0;j<current_dofs;++j)output[current_dofs+j]=.5*(epsilon_exterior+epsilon_interior)*mass[j];
    std::vector<cdouble>* curl[2]={&curl_exterior,&curl_interior};
    std::vector<cdouble>* hess[2]={&hessian_action_exterior,&hessian_action_interior};
    std::array<std::vector<cdouble>,3> reverse[2];
    for(int medium=0;medium<2;++medium){curl[medium]->resize(size_t(points)*3);hess[medium]->resize(size_t(points)*3);
        for(int d=0;d<3;++d)reverse[medium][d].resize(points);}
    for(int current=0;current<2;++current) {
#pragma omp parallel for schedule(static)
        for(int q=0;q<points;++q) {
            const auto& point=quadrature[q];const auto basis=evaluate_muller_basis(mesh,point.element,point.sample);
            cdouble ch[3]={},cc[3]={};
            for(int j=0;j<basis.count;++j) {
                const Vec3 tangent=basis.values[j],normal=point.sample.normal,rot=tangent.cross(normal);
                const double cross[3]={rot.z,-rot.y,rot.x};
                const cdouble hs=(current?-a:a)*seed[(current?current_dofs:0)+basis.dofs[j]];
                const cdouble cs=seed[(current?0:current_dofs)+basis.dofs[j]];
                const double rotated[3]={rot.x,rot.y,rot.z};
                for(int d=0;d<3;++d){ch[d]+=hs*rotated[d];cc[d]+=cs*cross[d];}
            }
            for(int medium=0;medium<2;++medium) {
                const cdouble material=current?(medium?epsilon_interior:epsilon_exterior):(medium?mu_interior:mu_exterior);
                const double weight=point.weight*(medium?-1.:1.);
                for(int d=0;d<3;++d){(*hess[medium])[3*q+d]=weight*ch[d];(*curl[medium])[3*q+d]=weight*material*cc[d];}
            }
        }
        const char* flag=std::getenv("BEM_FMM_CONCURRENT_MEDIA");
        bool concurrent=false;
#ifdef BEM_FMM_CONCURRENT_MEDIA_DEFAULT
        concurrent=true;
#endif
        if(flag)concurrent=std::strcmp(flag,"0")!=0;
        std::exception_ptr errors[2];
#pragma omp parallel for num_threads(2) if(concurrent)
        for(int medium=0;medium<2;++medium)try {
            auto& engine=medium?fmm_interior:fmm_exterior;
            auto& middle=medium?fmm_interior_middle:fmm_exterior_middle;
            auto& coarse=medium?fmm_interior_coarse:fmm_exterior_coarse;
            const char* ml=std::getenv("BEM_FMM_MULTILEVEL");
            const bool multilevel=banded_fmm && ml && std::strcmp(ml,"0")!=0;
            engine.evaluate_vector_actions_transpose(curl[medium]->data(),hess[medium]->data(),
                reverse[medium][0].data(),reverse[medium][1].data(),reverse[medium][2].data(),
                multilevel && banded_fmm_middle?&middle:nullptr,multilevel?&coarse:nullptr);
            if(banded_fmm && !multilevel) {
                std::array<std::vector<cdouble>,3> tmp;for(auto& v:tmp)v.resize(points);
                for(auto* band:{banded_fmm_middle?&middle:nullptr,&coarse})if(band){
                    band->evaluate_vector_actions_transpose(curl[medium]->data(),hess[medium]->data(),tmp[0].data(),tmp[1].data(),tmp[2].data());
                    for(int d=0;d<3;++d)for(int q=0;q<points;++q)reverse[medium][d][q]+=tmp[d][q];}
            }
        }catch(...){errors[medium]=std::current_exception();}
        for(const auto& error:errors)if(error)std::rethrow_exception(error);
        for(const auto& color:assembly_colors) {
#pragma omp parallel for schedule(static)
            for(int index=0;index<int(color.size());++index) {
                const int first=color[index]*regular_points_per_element;
                for(int q=first;q<first+regular_points_per_element;++q) {
                    const auto& point=quadrature[q];const auto basis=evaluate_muller_basis(mesh,point.element,point.sample);
                    for(int j=0;j<basis.count;++j) {
                        const auto& b=basis.values[j];
                        output[current*current_dofs+basis.dofs[j]]+=point.weight*(
                            b.x*(reverse[0][0][q]+reverse[1][0][q])+b.y*(reverse[0][1][q]+reverse[1][1][q])+b.z*(reverse[0][2][q]+reverse[1][2][q]));
                    }
                }
            }
        }
    }
    if(transpose_correction_offsets.empty()) {
        transpose_correction_offsets.assign(current_dofs+1,0);
        for(const auto& e:correction.entries)++transpose_correction_offsets[e.column+1];
        for(int j=0;j<current_dofs;++j)transpose_correction_offsets[j+1]+=transpose_correction_offsets[j];
        auto cursor=transpose_correction_offsets;
        transpose_correction_rows.resize(correction.entries.size());transpose_correction_indices.resize(correction.entries.size());
        for(int row=0;row<current_dofs;++row)for(int j=correction.row_offsets[row];j<correction.row_offsets[row+1];++j) {
            const int dest=cursor[correction.entries[j].column]++;
            transpose_correction_rows[dest]=row;transpose_correction_indices[dest]=j;
        }
    }
#pragma omp parallel for schedule(static)
    for(int column=0;column<current_dofs;++column) {
        cdouble j=0,m=0;
        for(int q=transpose_correction_offsets[column];q<transpose_correction_offsets[column+1];++q) {
            const int row=transpose_correction_rows[q];const auto& e=correction.entries[transpose_correction_indices[q]];
            j+=a*e.k1*seed[row]+e.k2_mu*seed[current_dofs+row];
            m+=e.k2_epsilon*seed[row]-a*e.k1*seed[current_dofs+row];
        }
        output[column]+=j;output[current_dofs+column]+=m;
    }
}
void MullerFmmOperator::matvec_adjoint(const cdouble* input,cdouble* output) {
    std::vector<cdouble> conjugated(system_dofs);
    for(int j=0;j<system_dofs;++j)conjugated[j]=std::conj(input[j]);
    matvec_transpose(conjugated.data(),output);
    for(int j=0;j<system_dofs;++j)output[j]=std::conj(output[j]);
}
void MullerFmmOperator::matvec_transpose_strict(const cdouble* input,cdouble* output) {
    const bool previous=fmm_near_fp32;set_fmm_near_fp32(false);
    try{matvec_transpose(input,output);}catch(...){set_fmm_near_fp32(previous);throw;}
    set_fmm_near_fp32(previous);
}
void MullerFmmOperator::matvec_adjoint_strict(const cdouble* input,cdouble* output) {
    const bool previous=fmm_near_fp32;set_fmm_near_fp32(false);
    try{matvec_adjoint(input,output);}catch(...){set_fmm_near_fp32(previous);throw;}
    set_fmm_near_fp32(previous);
}
