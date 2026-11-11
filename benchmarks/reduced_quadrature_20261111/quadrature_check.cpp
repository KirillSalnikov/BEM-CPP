#include "quadrature.h"
#include <iostream>
#include <cmath>
int main(){for(int q:{1,3,7}){auto r=tri_quadrature(q);int degree=q==1?1:q==3?2:5;double worst=0;for(int a=0;a<=degree;++a)for(int b=0;b<=degree-a;++b){double sum=0;for(int j=0;j<r.npts;++j)sum+=r.wts[j]*std::pow(r.pts[j][0],a)*std::pow(r.pts[j][1],b);double exact=2*std::tgamma(a+1)*std::tgamma(b+1)/std::tgamma(a+b+3);worst=std::max(worst,std::abs(sum-exact));}std::cout<<q<<" "<<r.npts<<" "<<worst<<std::endl;if(r.npts!=q||worst>1e-12)return 1;}}