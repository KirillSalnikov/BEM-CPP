#include "muller_mbj.h"
#include "krylov_deflation.h"
#include <cstdio>
using C=std::complex<double>;
int main()try {
    for(bool inverse:{false,true})for(bool overlap:{false,true})for(bool coarse:{false,true}) {
        MullerMbjPreconditioner p;p.system_dofs=9;p.stores_inverse=inverse;
        for(int j=0;j<3;++j){MullerMbjBlock b;
            const int first=overlap?std::max(0,3*j-1):3*j,last=overlap?std::min(9,3*j+4):3*j+3;
            for(int k=first;k<last;++k)b.dofs.push_back(k);const int n=int(b.dofs.size());b.lu.resize(n*n);
            for(int r=0;r<n;++r)for(int c=0;c<n;++c)b.lu[r*n+c]=C(r==c?3.:.2*std::sin(r+2*c+1),.1*std::cos(3*r+c));
            // Force a pivot and test inverse permutation order.
            if(n>2){for(int c=0;c<n;++c)std::swap(b.lu[c],b.lu[(n-1)*n+c]);}
            if(!inverse && !bem_krylov::dense_lu_factor(b.lu,n,b.pivots))return 1;
            b.core_dof_begin=3*j-first;b.core_dof_end=3*j+3-first;p.blocks.push_back(b);
        }
        if(coarse){p.coarse_rank=2;p.coarse_action.resize(18);p.coarse_update.resize(18);
            for(int j=0;j<18;++j){p.coarse_action[j]=C(.1*j,.2*std::sin(j));p.coarse_update[j]=C(.3*std::cos(j),.05*j);}
            p.coarse_gram_lu={C(.1,.2),C(2,.3),C(3,-.1),C(.4,.8)};
            if(!bem_krylov::dense_lu_factor(p.coarse_gram_lu,2,p.coarse_gram_pivots))return 2;}
        std::vector<C>x(9),y(9),ax(9),aty(9),ahy(9);for(int i=0;i<9;++i){x[i]=C(std::sin(i),std::cos(.3*i));y[i]=C(std::cos(.7*i),std::sin(.8*i));}
        p.apply(x.data(),ax.data());p.apply_transpose(y.data(),aty.data());p.apply_adjoint(y.data(),ahy.data());
        C t=0,h=0;double scale=0;
        for(int i=0;i<9;++i){t+=y[i]*ax[i]-aty[i]*x[i];h+=std::conj(y[i])*ax[i]-std::conj(ahy[i])*x[i];scale+=std::abs(y[i]*ax[i]);}
        if(std::abs(t)/scale>1e-13 || std::abs(h)/scale>1e-13)return 3;
    }
    std::puts("MBJ transpose/adjoint: LU pivots, stored inverse, restricted overlap and coarse correction passed");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 9;}
