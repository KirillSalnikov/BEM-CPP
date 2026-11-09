// Shared-index three-coefficient CSR. No fast-math or reduced precision.
#include <cstdint>
#include <omp.h>
extern "C" void bem_packed_current_cpu(int n,const int32_t* rows,const int32_t* cols,
    const float* coefficients,const float* x,float* y,int threads) {
#pragma omp parallel for schedule(static) num_threads(threads)
    for(int row=0;row<n;++row) {
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
}
