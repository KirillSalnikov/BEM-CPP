#include "dirac_transmission.h"
#include "quadrature.h"
#include <string>
using namespace bem_dirac;
static thread_local std::string error;
extern "C" {
const char* dirac_error(){return error.c_str();}
void* dirac_create_geometry(int shape,int ref,double aspect,double k,double re,double im,int order,size_t max_bytes,int target_order,int degree,int near_subdiv){try{
    if((shape!=0&&shape!=1)||ref<0||ref>7)throw std::invalid_argument("bounded Dirac mesh controls");
    Mesh mesh=shape==0?icosphere(1.,ref):regular_prism(6,aspect,ref,1.,0,false,true);
    return new Transmission(mesh,k,C(re,im),order,13,2.,max_bytes,target_order,degree,false,near_subdiv,true);
}catch(const std::exception&e){error=e.what();return nullptr;}}
int dirac_build_rows(void*p,int n,const int*ids,size_t budget){try{static_cast<Transmission*>(p)->build_rows(n,ids,budget);return 0;}catch(const std::exception&e){error=e.what();return 1;}}
int dirac_build_near_rows(void*p,int n,const int*ids,size_t budget){try{static_cast<Transmission*>(p)->build_near_rows(n,ids,budget);return 0;}catch(const std::exception&e){error=e.what();return 1;}}
int dirac_build_near_rows_shared(void*p,int n,const int*ids,size_t budget){try{static_cast<Transmission*>(p)->build_near_rows(n,ids,budget,true);return 0;}catch(const std::exception&e){error=e.what();return 1;}}
size_t dirac_near_pair_count(void*p){return static_cast<Transmission*>(p)->near_pair_count();}
size_t dirac_near_pattern_count(void*p){return static_cast<Transmission*>(p)->near_pattern_count();}
int dirac_near_rows(void*p,int medium,const C*x,int n,const int*ids,C*y){try{static_cast<Transmission*>(p)->near_correction_rows(medium,x,n,ids,y);return 0;}catch(const std::exception&e){error=e.what();return 1;}}
int dirac_project_cauchy_rows(void*p,const C*x,int n,const int*ids,C*y){try{static_cast<Transmission*>(p)->project_cauchy_rows(x,n,ids,y);return 0;}catch(const std::exception&e){error=e.what();return 1;}}
int dirac_self_blocks_direct(void*p,int n,const int*ids,C*y){try{static_cast<Transmission*>(p)->self_blocks_direct(n,ids,y);return 0;}catch(const std::exception&e){error=e.what();return 1;}}
int dirac_direct_transformed_columns(void*p,int nt,const int*targets,int ns,const int*sources,const double*basis,const double*components,C*y){try{static_cast<Transmission*>(p)->direct_transformed_columns(nt,targets,ns,sources,basis,components,y);return 0;}catch(const std::exception&e){error=e.what();return 1;}}
int dirac_direct_panel_pairs(void*p,int n,const int*targets,const int*sources,C*y){try{static_cast<Transmission*>(p)->direct_panel_pairs(n,targets,sources,y);return 0;}catch(const std::exception&e){error=e.what();return 1;}}
int dirac_direct_transformed_columns_basis(void*p,int nt,const int*targets,int ns,const int*sources,int cb,const double*basis,const double*components,C*y){try{static_cast<Transmission*>(p)->direct_transformed_columns_basis(nt,targets,ns,sources,cb,basis,components,y);return 0;}catch(const std::exception&e){error=e.what();return 1;}}
int dirac_apply_rows_reference(void*p,const C*x,int n,const int*rows,C*y){try{static_cast<Transmission*>(p)->apply_rows_reference(x,n,rows,y);return 0;}catch(const std::exception&e){error=e.what();return 1;}}
int dirac_apply_rows_reference_batch(void*p,int count,const C*x,int n,const int*rows,C*y){try{static_cast<Transmission*>(p)->apply_rows_reference_batch(count,x,n,rows,y);return 0;}catch(const std::exception&e){error=e.what();return 1;}}
int dirac_apply_direct_panels(void*p,const C*x,int n,const int*ids,C*y){try{static_cast<Transmission*>(p)->apply_direct_panels(x,n,ids,y);return 0;}catch(const std::exception&e){error=e.what();return 1;}}
int dirac_self_blocks_selected(void*p,int n,const int*ids,C*y){try{static_cast<Transmission*>(p)->self_blocks_selected(n,ids,y);return 0;}catch(const std::exception&e){error=e.what();return 1;}}
int dirac_submatrix(void*p,int nr,const int*rows,int nc,const int*cols,C*out){try{static_cast<Transmission*>(p)->submatrix(nr,rows,nc,cols,out);return 0;}catch(const std::exception&e){error=e.what();return 1;}}
int dirac_transformed_columns(void*p,int nr,const int*rows,int ns,const int*ids,int cb,const double*basis,const double*components,C*out){try{static_cast<Transmission*>(p)->transformed_columns(nr,rows,ns,ids,cb,basis,components,out);return 0;}catch(const std::exception&e){error=e.what();return 1;}}
void* dirac_create_dense_ex(int shape,int ref,double aspect,double k,double re,double im,int order,size_t max_bytes,int target_order,int degree,int near_subdiv){try{
    if((shape!=0&&shape!=1)||ref<0||ref>5)throw std::invalid_argument("bounded Dirac mesh controls");
    Mesh mesh=shape==0?icosphere(1.,ref):regular_prism(6,aspect,ref,1.,0,false,true);
    return new Transmission(mesh,k,C(re,im),order,13,2.,max_bytes,target_order,degree,false,near_subdiv);
}catch(const std::exception&e){error=e.what();return nullptr;}}
void* dirac_create(int shape,int ref,double aspect,double k,double re,double im,int order,size_t max_bytes,int target_order,int degree){try{
    if((shape!=0&&shape!=1)||ref<0||ref>5)throw std::invalid_argument("bounded Dirac mesh controls");
    Mesh mesh=shape==0?icosphere(1.,ref):regular_prism(6,aspect,ref,1.,0,false,true);
    return new Transmission(mesh,k,C(re,im),order,13,2.,max_bytes,target_order,degree);
}catch(const std::exception&e){error=e.what();return nullptr;}}
void* dirac_create_near(int shape,int ref,double aspect,double k,double re,double im,int order,size_t max_bytes,int target_order,int degree){try{
    if((shape!=0&&shape!=1)||ref<0||ref>7)throw std::invalid_argument("Dirac near mesh controls");
    Mesh mesh=shape==0?icosphere(1.,ref):regular_prism(6,aspect,ref,1.,0,false,true);
    return new Transmission(mesh,k,C(re,im),order,13,2.,max_bytes,target_order,degree,true);
}catch(const std::exception&e){error=e.what();return nullptr;}}
void* dirac_create_near_ex(int shape,int ref,double aspect,double k,double re,double im,int order,size_t max_bytes,int target_order,int degree,int near_subdiv){try{
    if((shape!=0&&shape!=1)||ref<0||ref>7)throw std::invalid_argument("Dirac near mesh controls");
    Mesh mesh=shape==0?icosphere(1.,ref):regular_prism(6,aspect,ref,1.,0,false,true);
    return new Transmission(mesh,k,C(re,im),order,13,2.,max_bytes,target_order,degree,true,near_subdiv);
}catch(const std::exception&e){error=e.what();return nullptr;}}
void dirac_destroy(void*p){delete static_cast<Transmission*>(p);}
int dirac_size(void*p){return static_cast<Transmission*>(p)->size();}
int dirac_basis_count(void*p){return static_cast<Transmission*>(p)->basis_count();}
size_t dirac_bytes(void*p){return static_cast<Transmission*>(p)->kernel_bytes();}
void dirac_scaling(void*p,C*out){const auto&s=static_cast<Transmission*>(p)->coefficients();
    for(auto a:{s.p,s.np,s.n,s.pp})for(auto v:a)*out++=v;}
void dirac_geometry(void*p,double*out){for(const auto&t:static_cast<Transmission*>(p)->panels()){
    for(auto v:{t.center,t.normal,t.tangent,t.second}){*out++=v.x;*out++=v.y;*out++=v.z;}*out++=t.area;}}
void dirac_vertices(void*p,double*out){for(const auto&t:static_cast<Transmission*>(p)->panels())for(const auto&v:t.vertex){*out++=v.x;*out++=v.y;*out++=v.z;}}
void dirac_quadrature(void*p,double*out){auto q=tri_quadrature(13);for(const auto&t:static_cast<Transmission*>(p)->panels())for(int j=0;j<q.npts;++j){
    auto x=t.vertex[0]+(t.vertex[1]-t.vertex[0])*q.pts[j][0]+(t.vertex[2]-t.vertex[0])*q.pts[j][1];
    *out++=x.x;*out++=x.y;*out++=x.z;*out++=q.wts[j];}}
int dirac_apply(void*p,const C*x,C*y){try{static_cast<Transmission*>(p)->apply(x,y);return 0;}catch(const std::exception&e){error=e.what();return 1;}}
int dirac_apply_simd(void*p,const C*x,C*y){try{static_cast<Transmission*>(p)->apply_simd(x,y);return 0;}catch(const std::exception&e){error=e.what();return 1;}}
int dirac_apply_batch(void*p,int count,const C*x,C*y,int tile){try{static_cast<Transmission*>(p)->apply_batch(count,x,y,tile);return 0;}catch(const std::exception&e){error=e.what();return 1;}}
int dirac_apply_rows(void*p,int count,const C*x,int n,const int*rows,C*y,int tile){try{static_cast<Transmission*>(p)->apply_rows(count,x,n,rows,y,tile);return 0;}catch(const std::exception&e){error=e.what();return 1;}}
int dirac_self_blocks(void*p,C*y){try{static_cast<Transmission*>(p)->self_blocks(y);return 0;}catch(const std::exception&e){error=e.what();return 1;}}
int dirac_diagonal_blocks(void*p,int group,C*y){try{static_cast<Transmission*>(p)->diagonal_blocks(group,y);return 0;}catch(const std::exception&e){error=e.what();return 1;}}
int dirac_prepare_mixed(void*p,size_t budget){try{static_cast<Transmission*>(p)->prepare_mixed_cache(budget);return 0;}catch(const std::exception&e){error=e.what();return 1;}}
int dirac_apply_mixed(void*p,const C*x,C*y){try{static_cast<Transmission*>(p)->apply_simd(x,y,true);return 0;}catch(const std::exception&e){error=e.what();return 1;}}
void dirac_block_lu(int blocks,int width,const C*lu,const int*piv,const C*b,C*x){
#pragma omp parallel for schedule(static)
    for(int block=0;block<blocks;++block){const C*a=lu+size_t(block)*width*width;const int*p=piv+block*width;C*y=x+block*width;
        for(int i=0;i<width;++i)y[i]=b[block*width+i];
        for(int i=0;i<width;++i)if(p[i]!=i)std::swap(y[i],y[p[i]]);
        for(int i=0;i<width;++i)for(int j=0;j<i;++j)y[i]-=a[i*width+j]*y[j];
        for(int i=width-1;i>=0;--i){for(int j=i+1;j<width;++j)y[i]-=a[i*width+j]*y[j];y[i]/=a[i*width+i];}
    }
}
// Row-major FP64 coarse correction. OpenMP parallel rows, SIMD columns;
// retain the independent BLAS path in the Python screening implementation.
int dirac_dense_gemv_subtract(int rows,int cols,const C*matrix,const C*x,C*y){
    if(rows<1||cols<1){error="coarse GEMV dimensions";return 1;}
    const double*q=reinterpret_cast<const double*>(x);
#pragma omp parallel for schedule(static)
    for(int i=0;i<rows;++i){const double*a=reinterpret_cast<const double*>(matrix+size_t(i)*cols);double re=0,im=0;
#pragma omp simd reduction(+:re,im)
        for(int j=0;j<cols;++j){re+=a[2*j]*q[2*j]-a[2*j+1]*q[2*j+1];im+=a[2*j]*q[2*j+1]+a[2*j+1]*q[2*j];}
        y[i]-=C(re,im);}
    return 0;
}
// Column-major W is also the assembly layout. Avoid a second full W copy.
// A small row tile stays in L1 while contiguous coefficients feed SIMD.
int dirac_dense_gemv_subtract_f(int rows,int cols,const C*matrix,const C*x,C*y){
    if(rows<1||cols<1){error="coarse GEMV dimensions";return 1;}
    const int tile=64;
#pragma omp parallel for schedule(static)
    for(int begin=0;begin<rows;begin+=tile){
        const int width=std::min(tile,rows-begin);double re[tile]={},im[tile]={};
        for(int j=0;j<cols;++j){
            const double*a=reinterpret_cast<const double*>(matrix+size_t(j)*rows+begin);
            const double xr=x[j].real(),xi=x[j].imag();
#pragma omp simd
            for(int i=0;i<width;++i){re[i]+=a[2*i]*xr-a[2*i+1]*xi;im[i]+=a[2*i]*xi+a[2*i+1]*xr;}
        }
        for(int i=0;i<width;++i)y[begin+i]-=C(re[i],im[i]);
    }
    return 0;
}
// Compressed preconditioner only: FP32 coefficients, FP64 input/product/sum.
// The BEM operator and residual checks never call this entry point.
int dirac_dense_gemv_subtract_cf(int rows,int cols,const std::complex<float>*matrix,const C*x,C*y){
    if(rows<1||cols<1){error="compressed coarse GEMV dimensions";return 1;}
    const int tile=64;
#pragma omp parallel for schedule(static)
    for(int begin=0;begin<rows;begin+=tile){
        const int width=std::min(tile,rows-begin);double re[tile]={},im[tile]={};
        for(int j=0;j<cols;++j){
            const float*a=reinterpret_cast<const float*>(matrix+size_t(j)*rows+begin);
            const double xr=x[j].real(),xi=x[j].imag();
#pragma omp simd
            for(int i=0;i<width;++i){re[i]+=double(a[2*i])*xr-double(a[2*i+1])*xi;im[i]+=double(a[2*i])*xi+double(a[2*i+1])*xr;}
        }
        for(int i=0;i<width;++i)y[begin+i]-=C(re[i],im[i]);
    }
    return 0;
}
// Multiple currents in one symmetry sector share W. Read each coefficient
// tile once for the batch; inputs and all accumulation remain FP64.
// x/y are row-major [nrhs, cols/rows]; W is column-major [rows, cols].
int dirac_dense_gemm_subtract_cf(int rows,int cols,int nrhs,const std::complex<float>*matrix,const C*x,C*y){
    if(rows<1||cols<1||nrhs<1||nrhs>16){error="compressed coarse GEMM dimensions";return 1;}
    const int tile=64;
#pragma omp parallel for schedule(static)
    for(int begin=0;begin<rows;begin+=tile){
        const int width=std::min(tile,rows-begin);
        double re[16][tile]={},im[16][tile]={};
        for(int j=0;j<cols;++j){
            const float*a=reinterpret_cast<const float*>(matrix+size_t(j)*rows+begin);
            double ar[tile],ai[tile];
#pragma omp simd
            for(int i=0;i<width;++i){ar[i]=double(a[2*i]);ai[i]=double(a[2*i+1]);}
            for(int rhs=0;rhs<nrhs;++rhs){
                const C value=x[size_t(rhs)*cols+j];const double xr=value.real(),xi=value.imag();
#pragma omp simd
                for(int i=0;i<width;++i){re[rhs][i]+=ar[i]*xr-ai[i]*xi;im[rhs][i]+=ar[i]*xi+ai[i]*xr;}
            }
        }
        for(int rhs=0;rhs<nrhs;++rhs)for(int i=0;i<width;++i)
            y[size_t(rhs)*rows+begin+i]-=C(re[rhs][i],im[rhs][i]);
    }
    return 0;
}
int dirac_cauchy(void*p,int medium,const C*x,C*y){try{static_cast<Transmission*>(p)->cauchy(medium,x,y);return 0;}catch(const std::exception&e){error=e.what();return 1;}}
int dirac_near_correction(void*p,int medium,const C*x,C*y){try{static_cast<Transmission*>(p)->near_correction(medium,x,y);return 0;}catch(const std::exception&e){error=e.what();return 1;}}
void dirac_quadrature_charges(void*p,const C*x,C*y){static_cast<Transmission*>(p)->quadrature_charges(x,y);}
void dirac_project_cauchy(void*p,const C*x,C*y){static_cast<Transmission*>(p)->project_cauchy(x,y);}
void dirac_rhs(void*p,const C*x,C*y){static_cast<Transmission*>(p)->rhs(x,y);}
void dirac_project_fields(void*p,const C*x,C*y){static_cast<Transmission*>(p)->project_fields(x,y);}
int dirac_incident(void*p,const double*d,const double*e,C*out){try{
    static_cast<Transmission*>(p)->incident(Vec3(d[0],d[1],d[2]),Vec3(e[0],e[1],e[2]),out);return 0;
}catch(const std::exception&e){error=e.what();return 1;}}
int dirac_field(void*p,int medium,const C*h,int n,const double*x,C*out,int far){try{
    std::vector<Vec3> targets;for(int j=0;j<n;++j)targets.emplace_back(x[3*j],x[3*j+1],x[3*j+2]);
    if(far)static_cast<Transmission*>(p)->farfield(h,targets,out);else static_cast<Transmission*>(p)->field(medium,h,targets,out);return 0;
}catch(const std::exception&e){error=e.what();return 1;}}
}
