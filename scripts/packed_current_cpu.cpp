// Shared-index three-coefficient CSR. No fast-math or reduced precision.
#include <cstdint>
#include <omp.h>
static inline void apply_row(int row,int n,const int32_t* rows,const int32_t* cols,
    const float* coefficients,const float* x,float* y) {
    float jr=0,ji=0,mr=0,mi=0;
    for(int p=rows[row];p<rows[row+1];++p) {
        const int col=cols[p];const float* v=coefficients+6LL*p;
        const float ar=v[0],ai=v[1],br=v[2],bi=v[3],cr=v[4],ci=v[5];
        const float xr=x[2LL*col],xi=x[2LL*col+1],zr=x[2LL*(n+col)],zi=x[2LL*(n+col)+1];
        jr+=(ar*xr-ai*xi)+(br*zr-bi*zi);ji+=(ar*xi+ai*xr)+(br*zi+bi*zr);
        mr+=(cr*xr-ci*xi)-(ar*zr-ai*zi);mi+=(cr*xi+ci*xr)-(ar*zi+ai*zr);
    }
    y[2LL*row]=jr;y[2LL*row+1]=ji;y[2LL*(n+row)]=mr;y[2LL*(n+row)+1]=mi;
}
extern "C" void bem_packed_current_cpu(int n,const int32_t* rows,const int32_t* cols,
    const float* coefficients,const float* x,float* y,int threads) {
#pragma omp parallel for schedule(static) num_threads(threads)
    for(int row=0;row<n;++row)apply_row(row,n,rows,cols,coefficients,x,y);
}
extern "C" void bem_packed_current_rotated(int n,const int32_t* rows,const int32_t* cols,
    const float* coefficients,const float* x,float* y,int threads,
    const int32_t* representatives,int count,const int32_t* owner,const uint8_t* phase_id,const float* phases) {
#pragma omp parallel num_threads(threads)
    {
#pragma omp for schedule(static)
        for(int i=0;i<count;++i)apply_row(representatives[i],n,rows,cols,coefficients,x,y);
        // The first loop completes all representative rows before expansion.
#pragma omp for schedule(static)
        for(int row=0;row<n;++row) {
            const int source=owner[row];
            if(source==row)continue;
            const float pr=phases[2*phase_id[row]],pi=phases[2*phase_id[row]+1];
            for(int j=0;j<2;++j) {
                const float* z=y+2LL*(source+j*n);
                float* target=y+2LL*(row+j*n);
                target[0]=pr*z[0]-pi*z[1];
                target[1]=pr*z[1]+pi*z[0];
            }
        }
    }
}
