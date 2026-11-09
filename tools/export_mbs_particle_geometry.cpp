// Optional independent oracle linked to an external MBS-fast source tree.
// Exports the seven fixed test geometries; not part of the BEM runtime.
#include "Hexagonal.h"
#include "Bullet.h"
#include "BulletRosette.h"
#include "Droxtal.h"
#include "ConcaveHexagonal.h"
#include "HexagonalAggregate.h"
#include "CertainAggregate.h"
#include <memory>
#include <cstdlib>
#include <cstdio>
int main(int argc,char**argv){
 if(argc!=3)return 2;int kind=std::atoi(argv[1]);complex ri(1.3116,0.);std::unique_ptr<Particle> p;
 double cap=sqrt(3.)*tan(62.*M_PI/180.)/4.;
 if(kind==1)p.reset(new Hexagonal(ri,1,2));
 if(kind==2)p.reset(new Bullet(ri,1,2,cap));
 if(kind==3)p.reset(new BulletRosette(ri,1,2,cap));
 if(kind==4)p.reset(new Droxtal(ri,32.35*M_PI/180.,71.81*M_PI/180.,1));
 if(kind==10)p.reset(new ConcaveHexagonal(ri,1,2,20));
 if(kind==12)p.reset(new HexagonalAggregate(ri,1,2,2));
 if(kind==999)p.reset(new CertainAggregate(ri,1));
 if(!p)return 3;p->Output(argv[2]);printf("volume %.17g dmax %.17g\n",p->Volume(),p->MaximalDimention());
}
