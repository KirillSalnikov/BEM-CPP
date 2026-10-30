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
        BlockCirculantFFT restored;restored.init_inverse_kernel(nx,ny,c,batches,inverse.inverse_kernel());
        restored.apply(x.data(),out.data());
        error=norm=0;for(size_t j=0;j<x.size();++j){error+=std::norm(out[j]-ax[j]);norm+=std::norm(ax[j]);}
        if(std::sqrt(error/norm)>2e-13)throw std::runtime_error("inverse kernel roundtrip failed");
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
    // Independent full 2c block inverse versus the two c-channel parity
    // inverses, for noncommuting, complex D/X kernels and multiple pairs.
    for(int c:{2,12}) {
        const int nx=3,ny=4,g=nx*ny,batches=2,channels=2*c;
        const size_t size=size_t(c)*g;
        std::vector<C> d(c*c*g),cross(d.size()),plus(d.size()),minus(d.size()),full(4*d.size());
        for(auto& z:d)z=C(normal(rng),normal(rng))*.01;
        for(auto& z:cross)z=C(normal(rng),normal(rng))*.03;
        for(int j=0;j<c;++j)d[(j*c+j)*g]+=2.;
        for(size_t j=0;j<d.size();++j){plus[j]=d[j]+cross[j];minus[j]=d[j]-cross[j];}
        for(int a=0;a<2;++a)for(int b=0;b<2;++b)for(int r=0;r<c;++r)for(int s=0;s<c;++s)for(int f=0;f<g;++f)
            full[((a*c+r)*channels+b*c+s)*g+f]=(a==b?d:cross)[(r*c+s)*g+f];
        BlockCirculantFFT direct,positive,negative;
        direct.init(nx,ny,channels,batches,full);positive.init(nx,ny,c,batches,plus);negative.init(nx,ny,c,batches,minus);
        std::vector<C> seed(2*batches*size),expected(seed.size()),result(seed.size()),even(batches*size),odd(even.size()),ye(even.size()),yo(even.size());
        for(auto& z:seed)z=C(normal(rng),normal(rng));
        const double scale=1/std::sqrt(2.);
        for(int adj=0;adj<2;++adj) {
            for(int b=0;b<batches;++b)for(size_t j=0;j<size;++j) {
                even[b*size+j]=scale*(seed[2*b*size+j]+seed[(2*b+1)*size+j]);
                odd[b*size+j]=scale*(seed[2*b*size+j]-seed[(2*b+1)*size+j]);
            }
            if(adj){direct.apply_adjoint(seed.data(),expected.data());positive.apply_adjoint(even.data(),ye.data());negative.apply_adjoint(odd.data(),yo.data());}
            else {direct.apply(seed.data(),expected.data());positive.apply(even.data(),ye.data());negative.apply(odd.data(),yo.data());}
            for(int b=0;b<batches;++b)for(size_t j=0;j<size;++j) {
                result[2*b*size+j]=scale*(ye[b*size+j]+yo[b*size+j]);result[(2*b+1)*size+j]=scale*(ye[b*size+j]-yo[b*size+j]);
            }
            double error=0,norm=0;for(size_t j=0;j<seed.size();++j){error+=std::norm(result[j]-expected[j]);norm+=std::norm(expected[j]);}
            std::printf("opposite_channels=%d adjoint=%d parity_vs_full_inverse=%.3e\n",channels,adj,std::sqrt(error/norm));
            if(std::sqrt(error/norm)>2e-12)throw std::runtime_error("opposite parity inverse failed");
        }
    }
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
