! Derived from MSTM by Daniel Mackowski, MIT license; see LICENSE.
! Upstream commit a0c982121cf9ac352531f4816639a07d814385bd.
#ifndef BEM_SO3_STORAGE_KIND
#define BEM_SO3_STORAGE_KIND 8
#endif
module bem_mstm_average
use bem_mstm_special
implicit none
integer,parameter::run_print_unit=6
contains
subroutine coefficients(nodr,tc,sm,smcf,cyclic_order)
         implicit none
         logical :: symmetrical
         integer :: nodrw,nodr2,m,n,p,k,l,q,t,v,u,w,nblk,kl,mn,nn1,tn, &
                    lmax,ll1,tvl,ku,k1,ns,ik,ik1,m1,nu,n1s,n1e,nu1,p1,n1max, &
                    in,n1,i,kt,nodrt,nodrrhs,mnm,klm,ikm, &
                    rank,numprocs, &
                    nblkrhs,numprocscalc,orig_group,new_group, &
                    new_comm,new_rank,nblkw,wv,sizedm,sizetm,nread,mpicomm
         integer, allocatable :: windex(:),vindex(:),wvindex(:),wvnum(:),group_list(:)
         integer,intent(in)::nodr,cyclic_order
         complex(BEM_SO3_STORAGE_KIND),intent(inout)::tc(2,nodr*(nodr+2),2,nodr*(nodr+2))
         real(8) :: sm(4,4,0:2*nodr),fl,xv,fl2,cbeam,gbn,wvperproc,sum, &
                    time1,time2,smcf(4,4,0:2*nodr),qextt,qscatt
         real(8), allocatable :: vc(:)

         complex(8) :: ci,cin,a,tct
         complex(8), allocatable :: aw(:,:,:),bw(:,:,:),cw(:), &
                       dw(:),pp(:,:,:),bm(:,:,:), &
                       am(:,:,:),fm(:,:,:,:,:),bmcf(:,:,:), &
                       amcf(:,:,:),fmcf(:,:,:,:,:),awcf(:,:,:), &
                       bwcf(:,:,:),cwcf(:),dwcf(:)
         complex(8), allocatable :: dm(:,:,:,:,:,:),dmcf(:,:,:,:,:,:)

         character*30 :: tmatrixfile
         data ci/(0.d0,1.d0)/

rank=0;new_rank=0;numprocscalc=1;numprocs=1
nodrrhs=nodr;nodrt=nodr;nodr2=2*nodr;nodrw=nodr2
nblk=nodr*(nodr+2);nblkrhs=nblk
cbeam=0d0;xv=1d0;sm=0d0;smcf=0d0
call cpu_time(time1)
            allocate(vc(0:4*nodr+2))
            allocate(aw(0:2,-1:1,0:nodrw),bw(0:2,-1:1,0:nodrw),cw(0:nodrw), &
                 dw(0:nodrw),pp(nodr,2,2),bm(2,nodr*(nodr+2),2), &
                 am(2,nodrrhs+1,2),fm(3,nodr,2,nodr,2),bmcf(2,nodr*(nodr+2),2), &
                 amcf(2,nodrrhs+1,2),fmcf(3,nodr,2,nodr,2),awcf(0:2,-1:1,0:nodrw), &
                 bwcf(0:2,-1:1,0:nodrw),cwcf(0:nodrw),dwcf(0:nodrw))
            allocate(dm(-nodr-1:nodr+1,3,nodr,2,nodr,2),dmcf(-nodr-1:nodr+1,3,nodr,2,nodr,2))
            if(new_rank.eq.0) then
call cpu_time(time2)
time2=time2-time1
call cpu_time(time1)
            endif
            dm=(0.d0,0.d0)
            dmcf=(0.d0,0.d0)
            sizedm=size(dm)
            call init(nodr2)
!
!  compute the GB modified T matrix
!
            do n=1,nodrrhs
               gbn=dexp(-((dble(n)+.5d0)*cbeam)**2.)
               cin=ci**(n+1)
               pp(n,1,1) =-.5d0*cin*fnr(n+n+1)*gbn
               pp(n,2,1) =-pp(n,1,1)
               pp(n,1,2)=-pp(n,1,1)
               pp(n,2,2)=pp(n,2,1)
            enddo
            do n=1,nodr
               nn1=n*(n+1)
               do m=-n,n
                  mn=nn1+m
                  do p=1,2
                     do l=1,nodrrhs
                        do k=-l,l
                           kl=l*(l+1)+k
                           a=tc(p,mn,1,kl)
                           tc(p,mn,1,kl)=tc(p,mn,1,kl)*pp(l,1,1)&
                              +tc(p,mn,2,kl)*pp(l,1,2)
                           tc(p,mn,2,kl)=a*pp(l,2,1)+tc(p,mn,2,kl)*pp(l,2,2)
                        enddo
                     enddo
                  enddo
               enddo
            enddo
!
!  determine the distribution of work load among the processors
!
            nblkw=nodr2*(nodr2+2)+1
            allocate(windex(nblkw),vindex(nblkw),wvindex(0:numprocscalc-1), &
               wvnum(0:numprocscalc-1))
            w=0
            do n=0,nodr2
               do m=-n,n
                  w=w+1
                  windex(w)=n
                  vindex(w)=m
               enddo
            enddo
            wvperproc=dble(nblkw)/dble(numprocscalc)
            sum=0.
            do i=0,numprocscalc-1
               wvindex(i)=floor(sum)
               sum=sum+wvperproc
            enddo
            do i=0,numprocscalc-2
               wvnum(i)=wvindex(i+1)-wvindex(i)
            enddo
            wvnum(numprocscalc-1)=nblkw-wvindex(numprocscalc-1)
            if(new_rank.eq.0) then
               write(run_print_unit,'('' d matrix calculation, order+degree per proc.:'',f9.2)') &
                   wvperproc
               call flush(run_print_unit)
            endif
!
!  the big loop
!
            vc=0d0;am=0d0;amcf=0d0;bm=0d0;bmcf=0d0
!$omp parallel do default(shared) schedule(dynamic) reduction(+:dm,dmcf) &
!$omp firstprivate(vc,am,amcf,bm,bmcf) &
!$omp private(w,v,n,nn1,l,t,tn,lmax,ll1,tvl,k,p,m,mn,u,ku,k1,ns,ik,ik1,m1,n1s,n1e,n1,nu,nu1,cin,fl,p1,a,kt,tct)
            do wv=wvindex(rank)+1,wvindex(rank)+wvnum(rank)
               w=windex(wv)
               v=vindex(wv)
               if(mod(v,cyclic_order).ne.0)cycle
               bm=(0.d0,0.d0)
               bmcf=(0.d0,0.d0)
               do n=1,nodr
                  nn1=n*(n+1)
                  do l=max(1,abs(w-n)),min(nodrrhs,w+n)
                     am(1,l,1)=0.d0
                     am(1,l,2)=0.d0
                     am(2,l,1)=0.d0
                     am(2,l,2)=0.d0
                     amcf(1,l,1)=0.d0
                     amcf(1,l,2)=0.d0
                     amcf(2,l,1)=0.d0
                     amcf(2,l,2)=0.d0
                  enddo
                  do t=-n,n
                     tn=nn1+t
                     lmax=min(nodrrhs,w+n)
                     call vcfunc(v,w,-t,n,vc)
                     do l=max(1,abs(v-t),abs(n-w)),lmax
                        ll1=l*(l+1)
                        tvl=ll1+t-v
                        do k=1,2
                           do p=1,2
                              am(k,l,p)=am(k,l,p)+vc(l)*tc(p,tn,k,tvl)
                              if(l.eq.n.and.v.eq.0) then
! may 2019: orientation averaged T matrix is azimuthal independent
                                 tct=0.d0
                                 do kt=-n,n
                                    tct=tct+tc(p,nn1+kt,k,nn1+kt)
                                 enddo
                                 amcf(k,l,p)=amcf(k,l,p)+vc(l)*tct/dble(n+n+1)
!                                 amcf(k,l,p)=amcf(k,l,p)+vc(l)*tcm(p,n)
!                                 amcf(k,l,p)=amcf(k,l,p)+vc(l)*tc(p,tn,k,tvl)
                              endif
                           enddo
                        enddo
                     enddo
                  enddo
                  do m=-n,n
                     mn=nn1+m
                     do k=1,2
                        u=m-(-3+2*k)
                        if(abs(u).le.w) then
                           lmax=min(nodrrhs,w+n)
                           call vcfunc(-u,w,m,n,vc)
                           do l=max(1,abs(w-n)),lmax
                              fl=-(-1)**l*vc(l)/dble(l+l+1)
                              do p=1,2
                                 bm(k,mn,p)=bm(k,mn,p)+am(k,l,p)*fl
                                 if(v.eq.0) then
                                    bmcf(k,mn,p)=bmcf(k,mn,p)+amcf(k,l,p)*fl
                                 endif
                              enddo
                           enddo
                        endif
                     enddo
                  enddo
               enddo

!               do k=1,2
!                  do n=1,nodr*(nodr+2)
!                     do p=1,2
!                        bmcf(k,n,p)=bm(k,n,p)-bmcf(k,n,p)
!                     enddo
!                  enddo
!               enddo

               do u=-min(w,nodr+1),min(w,nodr+1)
                  do ku=1,3
                     if(ku.eq.1) then
                        k=-1
                        k1=-1
                     elseif(ku.eq.2) then
                        k=1
                        k1=1
                     else
                        k=1
                        k1=-1
                     endif
                     m=u+k
                     ns=max(1,abs(m))
                     ik=(k+1)/2+1
                     ik1=(k1+1)/2+1
                     m1=u+k1
                     do n=ns,nodr
                        nu=n*(n+1)+m
                        n1s=max(1,abs(m1),n-nodrw)
                        n1e=min(nodr,n+nodrw)
                        do n1=n1s,n1e
                           cin=ci**(n-n1)
                           nu1=n1*(n1+1)+m1
                           fl=-fnr(n+n+1)*fnr(n1+n1+1)*dble(w+w+1)
                           do p=1,2
                              do p1=1,2
                                 a=bm(ik,nu,p)*cin*fl*conjg(bm(ik1,nu1,p1))
                                 dm(u,ku,n,p,n1,p1)=dm(u,ku,n,p,n1,p1)+a
                                 if(v.eq.0) then
                                    a=bmcf(ik,nu,p)*cin*fl*conjg(bmcf(ik1,nu1,p1))
                                    dmcf(u,ku,n,p,n1,p1)=dmcf(u,ku,n,p,n1,p1)+a
                                 endif
                              enddo
                           enddo
                        enddo
                     enddo
                  enddo
               enddo
            enddo

!$omp end parallel do
            if(new_rank.eq.0) then
call cpu_time(time2)
time2=time2-time1
call cpu_time(time1)
            endif
!
!  compute the expansion coefficients
!
            aw=0.d0
            bw=0.d0
            cw=0.d0
            dw=0.d0
            awcf=0.d0
            bwcf=0.d0
            cwcf=0.d0
            dwcf=0.d0
            do w=0,nodrw
               do n=1,nodr
                  n1s=max(1,abs(n-w))
                  n1e=min(nodr,n+w)
                  do n1=n1s,n1e
                     do k=1,3
                        do p=1,2
                           do p1=1,2
                              fm(k,n,p,n1,p1)=0.
                              fmcf(k,n,p,n1,p1)=0.
                           enddo
                        enddo
                     enddo
                  enddo
               enddo
               do u=-nodr-1,nodr+1
                  do k=-1,1,2
                     m=u+k
                     ik=(k+1)/2+1
                     ns=max(1,abs(m))
                     do n=ns,nodr
                        n1max=min(w+n,nodr)
                        call vcfunc(m,n,0,w,vc)
                        do n1=ns,nodr
                           if((n+n1.lt.w).or.(abs(n-n1).gt.w)) cycle
                           fl=-(-1)**n*vc(n1)*fnr(w+w+1)/fnr(n1+n1+1)
                           do p=1,2
                              do p1=1,2
                                 fm(ik,n,p,n1,p1)=fm(ik,n,p,n1,p1) &
                                    +dm(u,ik,n,p,n1,p1)*fl
                                 fmcf(ik,n,p,n1,p1)=fmcf(ik,n,p,n1,p1) &
                                    +dmcf(u,ik,n,p,n1,p1)*fl
                              enddo
                           enddo
                        enddo
                     enddo
                  enddo
                  if(w.lt.2) cycle
                  m=u+1
                  m1=u-1
                  ns=max(1,abs(m))
                  n1s=max(1,abs(m1))
                  do n=ns,nodr
                     n1max=min(w+n,nodr)
                     call vcfunc(m,n,-2,w,vc)
                     do n1=n1s,nodr
                        if((n+n1.lt.w).or.(abs(n-n1).gt.w)) cycle
                        fl=-(-1)**n*vc(n1)*fnr(w+w+1)/fnr(n1+n1+1)
                        do p=1,2
                           do p1=1,2
                              fm(3,n,p,n1,p1)=fm(3,n,p,n1,p1) &
                                 +dm(u,3,n,p,n1,p1)*fl
                              fmcf(3,n,p,n1,p1)=fmcf(3,n,p,n1,p1) &
                                 +dmcf(u,3,n,p,n1,p1)*fl
                           enddo
                        enddo
                     enddo
                  enddo
               enddo
               do n=1,nodr
                  n1s=max(1,abs(n-w))
                  n1e=min(nodr,n+w)
                  in=(-1)**n
                  n1max=min(w+n,nodr)
                  call vcfunc(1,n,0,w,vc)
                  do n1=n1s,n1e
                     fl=2.d0*in*vc(n1)*fnr(w+w+1)/fnr(n1+n1+1)
                     i=mod(n+n1-w,2)+1
                     do p=1,2
                        p1=(2-i)*p+(i-1)*(3-p)
                        do k=-1,1,2
                           ik=(k+1)/2+1
                           aw(0,k,w)=aw(0,k,w)+fm(ik,n,p,n1,p1)*fl
                           bw(0,k,w)=bw(0,k,w)+fm(ik,n,p,n1,3-p1)*fl
                           awcf(0,k,w)=awcf(0,k,w)+fmcf(ik,n,p,n1,p1)*fl
                           bwcf(0,k,w)=bwcf(0,k,w)+fmcf(ik,n,p,n1,3-p1)*fl
                        enddo
                        bw(2,0,w)=bw(2,0,w)+fm(3,n,p,n1,3-p1)*fl
                        aw(2,0,w)=aw(2,0,w)+fm(3,n,p,n1,p1)*fl
                        bwcf(2,0,w)=bwcf(2,0,w)+fmcf(3,n,p,n1,3-p1)*fl
                        awcf(2,0,w)=awcf(2,0,w)+fmcf(3,n,p,n1,p1)*fl
                     enddo
                  enddo
                  if(w.lt.2) cycle
                  call vcfunc(1,n,-2,w,vc)
                  do n1=n1s,n1e
                     fl=2.d0*in*vc(n1)*fnr(w+w+1)/fnr(n1+n1+1)
                     i=mod(n+n1-w,2)+1
                     do p=1,2
                        p1=(2-i)*p+(i-1)*(3-p)
                        do k=-1,1,2
                           ik=(k+1)/2+1
                           aw(2,k,w)=aw(2,k,w)+fm(ik,n,p,n1,p1)*fl*(-1)**p1
                           bw(2,k,w)=bw(2,k,w)+fm(ik,n,p,n1,3-p1)*fl*(-1)**(3-p1)
                           awcf(2,k,w)=awcf(2,k,w)+fmcf(ik,n,p,n1,p1)*fl*(-1)**p1
                           bwcf(2,k,w)=bwcf(2,k,w) &
                              +fmcf(ik,n,p,n1,3-p1)*fl*(-1)**(3-p1)
                        enddo
                     enddo
                     fl2=2.*(-1)**(n1+w)*vc(n1)*fnr(w+w+1)/fnr(n1+n1+1)
                     do p=1,2
                        do p1=1,2
                           cw(w)=cw(w)+fm(3,n,p,n1,p1)*fl*(-1)**p1
                           dw(w)=dw(w)+fm(3,n,p,n1,p1)*fl2*(-1)**p
                           cwcf(w)=cwcf(w)+fmcf(3,n,p,n1,p1)*fl*(-1)**p1
                           dwcf(w)=dwcf(w)+fmcf(3,n,p,n1,p1)*fl2*(-1)**p
                        enddo
                     enddo
                  enddo
               enddo
            enddo
!            do w=0,nodrw
!               do k=-1,1
!                  do i=0,2
!                     aw(i,k,w)=aw(i,k,w)*2./xv/xv
!                     bw(i,k,w)=bw(i,k,w)*2./xv/xv
!                     awcf(i,k,w)=awcf(i,k,w)*2./xv/xv
!                     bwcf(i,k,w)=bwcf(i,k,w)*2./xv/xv
!                  enddo
!               enddo
!               cw(w)=cw(w)*2./xv/xv
!               dw(w)=dw(w)*2./xv/xv
!               cwcf(w)=cwcf(w)*2./xv/xv
!               dwcf(w)=dwcf(w)*2./xv/xv
!            enddo
            do n=0,nodrw
               sm(1,1,n)=aw(0,-1,n)+aw(0,1,n)
               sm(1,2,n)=aw(2,-1,n)+aw(2,1,n)
               sm(1,3,n)=2.d0*dimag(aw(2,0,n))
               sm(1,4,n)=aw(0,1,n)-aw(0,-1,n)
               sm(2,2,n)=dw(n)
               sm(2,3,n)=dimag(dw(n))
               sm(2,4,n)=aw(2,1,n)-aw(2,-1,n)
               sm(3,1,n)=-dimag(bw(2,-1,n)+bw(2,1,n))
               sm(3,2,n)=dimag(cw(n))
               sm(3,3,n)=cw(n)
               sm(3,4,n)=dimag(bw(2,-1,n)-bw(2,1,n))
               sm(4,1,n)=bw(0,-1,n)+bw(0,1,n)
               sm(4,2,n)=2.*bw(2,0,n)
               sm(4,4,n)=bw(0,1,n)-bw(0,-1,n)

               smcf(1,1,n)=awcf(0,-1,n)+awcf(0,1,n)
               smcf(1,2,n)=awcf(2,-1,n)+awcf(2,1,n)
               smcf(1,3,n)=2.d0*dimag(awcf(2,0,n))
               smcf(1,4,n)=awcf(0,1,n)-awcf(0,-1,n)
               smcf(2,2,n)=dwcf(n)
               smcf(2,3,n)=dimag(dwcf(n))
               smcf(2,4,n)=awcf(2,1,n)-awcf(2,-1,n)
               smcf(3,1,n)=-dimag(bwcf(2,-1,n)+bwcf(2,1,n))
               smcf(3,2,n)=dimag(cwcf(n))
               smcf(3,3,n)=cwcf(n)
               smcf(3,4,n)=dimag(bwcf(2,-1,n)-bwcf(2,1,n))
               smcf(4,1,n)=bwcf(0,-1,n)+bwcf(0,1,n)
               smcf(4,2,n)=2.*bwcf(2,0,n)
               smcf(4,4,n)=bwcf(0,1,n)-bwcf(0,-1,n)

            enddo
!
!  normalization
!
!            qsca0=sm(1,1,0)
!            do n=0,nodrw
!               do i=1,4
!                  do j=1,4
!                     sm(i,j,n)=sm(i,j,n)/qsca0
!                     smcf(i,j,n)=smcf(i,j,n)/qsca0
!                  enddo
!               enddo
!            enddo
            if(new_rank.eq.0) then
call cpu_time(time2)
time2=time2-time1
            endif
            deallocate(windex,vindex,wvindex,wvnum,dm)
            deallocate(vc)
            deallocate(aw,bw,cw,dw,pp,bm,am,fm,bmcf,amcf,fmcf,awcf, &
                 bwcf,cwcf,dwcf)

end subroutine coefficients
         subroutine ranorienscatmatrixcalc(ct,smc,nodrexp,sm)
         implicit none
         integer :: nodrexp,n,nn0,nnp2,nnm2
         real(8) :: smc(4,4,0:nodrexp),sm(4,4),dc(-2:2,0:nodrexp*(nodrexp+2)), &
                    ct
!
!     dc is the normalized generalized spherical function
!     dc(k,n*(n+1)+m) = ((n-k)!(n+m)!/(n+k)!/(n-m)!)^(1/2) D^k_{mn},
!     where D^k_{mn} is defined in M&M JOSA 96
!
         call rotcoef(ct,2,nodrexp,dc)
         sm=0.d0
         do n=0,nodrexp
            nn0=n*(n+1)
            nnp2=nn0+2
            nnm2=nn0-2
            sm(1,1)=sm(1,1)+dc(0,nn0)*smc(1,1,n)
            sm(1,4)=sm(1,4)+dc(0,nn0)*smc(1,4,n)
            sm(4,1)=sm(4,1)+dc(0,nn0)*smc(4,1,n)
            sm(4,4)=sm(4,4)+dc(0,nn0)*smc(4,4,n)
            if(n.ge.2) then
               sm(1,2)=sm(1,2)+dc(2,nn0)*smc(1,2,n)
               sm(2,4)=sm(2,4)+dc(2,nn0)*smc(2,4,n)
               sm(3,4)=sm(3,4)+dc(2,nn0)*smc(3,4,n)
               sm(1,3)=sm(1,3)+dc(2,nn0)*smc(1,3,n)
               sm(3,1)=sm(3,1)+dc(2,nn0)*smc(3,1,n)
               sm(4,2)=sm(4,2)+dc(2,nn0)*smc(4,2,n)
               sm(2,2)=sm(2,2)+dc(2,nnm2)*smc(2,2,n)+dc(2,nnp2)*smc(3,3,n)
               sm(2,3)=sm(2,3)+dc(2,nnm2)*smc(2,3,n)+dc(2,nnp2)*smc(3,2,n)
               sm(3,3)=sm(3,3)-dc(2,nnm2)*smc(2,2,n)+dc(2,nnp2)*smc(3,3,n)
               sm(3,2)=sm(3,2)+dc(2,nnm2)*smc(2,3,n)-dc(2,nnp2)*smc(3,2,n)
            endif
         enddo
         sm(2,1)=sm(1,2)
         sm(4,3)=-sm(3,4)
!
!  discontiued scaling option: now done in main program
!
!            if(iscale.eq.1) then
!               do j=1,4
!                  do k=j,4
!                     if(j.ne.1.or.k.ne.1) then
!                        sm(j,k,i)=sm(j,k,i)/sm(1,1,i)
!                     endif
!                  enddo
!               enddo
!            endif
!
!    here are the VV and HH differential cross sections
!
!            gvv=.25*(sm(1,1)+sm(2,2)-2.*sm(1,2))
!            ghh=.25*(sm(1,1)+sm(2,2)+2.*sm(1,2))
!
         return
         end subroutine ranorienscatmatrixcalc

end module
