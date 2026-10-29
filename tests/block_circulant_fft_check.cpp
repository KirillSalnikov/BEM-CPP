#include "block_circulant_fft.h"
#include <cstdio>
#include <random>
#include <stdexcept>
int main()try {
    using C=std::complex<double>;
    std::mt19937 rng(513);std::normal_distribution<double> normal;
    for(int c:{2,12})for(int nx:{3,4}) {
        const int ny=5,g=nx*ny,batches=2;
        std::vector<C> kernel(c*c*g),x(batches*c*g),rhs(x.size()),out(x.size());
        for(auto& z:kernel)z=C(normal(rng),normal(rng))*.002;
        for(int i=0;i<c;++i)kernel[(i*c+i)*g]+=2.;
        for(auto& z:x)z=C(normal(rng),normal(rng));
        for(int b=0;b<batches;++b)for(int r=0;r<c;++r)for(int i=0;i<nx;++i)for(int j=0;j<ny;++j)
            for(int s=0;s<c;++s)for(int u=0;u<nx;++u)for(int v=0;v<ny;++v)
                rhs[(b*c+r)*g+i*ny+j]+=kernel[(r*c+s)*g+(i-u+nx)%nx*ny+(j-v+ny)%ny]*x[(b*c+s)*g+u*ny+v];
        BlockCirculantFFT inverse;inverse.init(nx,ny,c,batches,kernel);inverse.apply(rhs.data(),out.data());
        double error=0,norm=0;for(size_t i=0;i<x.size();++i){error+=std::norm(out[i]-x[i]);norm+=std::norm(x[i]);}
        error=std::sqrt(error/norm);
        std::printf("channels=%d grid=%dx%d relative_inverse_error=%.3e\n",c,nx,ny,error);
        if(error>2e-12 || inverse.fallback_count())throw std::runtime_error("periodic inverse failed");
        std::vector<C> seed(x.size()),ax(x.size()),ah(x.size()),at(x.size());
        for(auto& z:seed)z=C(normal(rng),normal(rng));
        inverse.apply(x.data(),ax.data());inverse.apply_adjoint(seed.data(),ah.data());inverse.apply_transpose(seed.data(),at.data());
        C h=0,t=0;double scale=0;
        for(size_t j=0;j<x.size();++j){h+=std::conj(seed[j])*ax[j]-std::conj(ah[j])*x[j];t+=seed[j]*ax[j]-at[j]*x[j];scale+=std::abs(seed[j]*ax[j]);}
        if(std::abs(h)/scale>2e-13 || std::abs(t)/scale>2e-13)throw std::runtime_error("periodic inverse adjoint failed");
        // Independent direct adjoint convolution of C^H seed must recover seed.
        std::vector<C> recovered(x.size());
        for(int b=0;b<batches;++b)for(int r=0;r<c;++r)for(int i=0;i<nx;++i)for(int j=0;j<ny;++j)
            for(int s=0;s<c;++s)for(int u=0;u<nx;++u)for(int v=0;v<ny;++v)
                recovered[(b*c+s)*g+u*ny+v]+=std::conj(kernel[(r*c+s)*g+(i-u+nx)%nx*ny+(j-v+ny)%ny])*ah[(b*c+r)*g+i*ny+j];
        error=norm=0;for(size_t j=0;j<x.size();++j){error+=std::norm(recovered[j]-seed[j]);norm+=std::norm(seed[j]);}
        if(std::sqrt(error/norm)>2e-12)throw std::runtime_error("direct adjoint inverse failed");
    }
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
