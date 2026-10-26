! Derived from MSTM by Daniel Mackowski, MIT license; see LICENSE.
! Upstream commit a0c982121cf9ac352531f4816639a07d814385bd.
module bem_mstm_constants
implicit none
real(8), allocatable :: bcof(:,:),fnr(:)
integer :: initialized_order=0
contains
subroutine init(n)
integer,intent(in)::n
integer::nbc,i,j
if(n<=initialized_order)return
initialized_order=n
nbc=2*n+4
if(allocated(bcof))deallocate(bcof,fnr)
allocate(bcof(0:nbc,0:nbc),fnr(0:2*nbc))
fnr(0)=0d0
do i=1,2*nbc
 fnr(i)=sqrt(dble(i))
enddo
bcof(0,0)=1d0
do i=0,nbc-1
 do j=i+1,nbc
  bcof(i,j)=fnr(i+j)*bcof(i,j-1)/fnr(j)
  bcof(j,i)=bcof(i,j)
 enddo
 bcof(i+1,i+1)=fnr(2*i+2)*fnr(2*i+1)*bcof(i,i)/fnr(i+1)/fnr(i+1)
enddo
end subroutine
end module
module bem_mstm_special
use bem_mstm_constants
implicit none
contains
         subroutine vcfunc(m,n,k,l,vcn)
         use bem_mstm_constants
         implicit none
         integer :: m,n,k,l,wmax,wmin,w,mk
         real(8) :: vcn(0:n+l),t1,t2,t3,vcmax,vctest,rat
         vcn=0.d0
         wmax=n+l
         wmin=max(abs(n-l),abs(m+k))
         vcn(wmax)=bcof(n+m,l+k)*bcof(n-m,l-k)/bcof(n+n,l+l)
         if(wmin.eq.wmax) return
         vcn(wmax-1)=vcn(wmax)*(l*m-k*n)*fnr(2*(l+n)-1)/fnr(l)/fnr(n)&
        &  /fnr(n+l+m+k)/fnr(n+l-m-k)
         if(wmin.eq.wmax-1) return
         mk=m+k
         vcmax=abs(vcn(wmax))+abs(vcn(wmax-1))
!
!  a downwards recurrence is used initially
!
         do w=wmax,wmin+2,-1
            t1=2*w*fnr(w+w+1)*fnr(w+w-1)/(fnr(w+mk)*fnr(w-mk)&
        &     *fnr(n-l+w)*fnr(l-n+w)*fnr(n+l-w+1)*fnr(n+l+w+1))
            t2=dble((m-k)*w*(w-1)-mk*n*(n+1)+mk*l*(l+1))&
        &    /dble(2*w*(w-1))
            t3=fnr(w-mk-1)*fnr(w+mk-1)*fnr(l-n+w-1)*fnr(n-l+w-1)&
        &     *fnr(n+l-w+2)*fnr(n+l+w)/(dble(2*(w-1))*fnr(2*w-3)&
        &     *fnr(2*w-1))
            vcn(w-2)=(t2*vcn(w-1)-vcn(w)/t1)/t3
            if(mod(wmax-w,2).eq.1) then
               vctest=abs(vcn(w-2))+abs(vcn(w-1))
               vcmax=max(vcmax,vctest)
               rat=vctest/vcmax
!
!  if/when the coefficients start to decrease in magnitude, an upwards recurrence takes over
!
               if(rat.lt.0.01d0) exit
            endif
         enddo
         if(w-2.gt.wmin) then
            wmax=w-3
            call vcfuncuprec(m,n,k,l,wmax,vcn)
         endif
         end subroutine vcfunc

         subroutine vcfuncuprec(m,n,k,l,wmax,vcn)
         use bem_mstm_constants
         implicit none
         integer :: m,n,k,l,wmax,w,mk,nl,m1,n1,l1,k1,w1,w2
         real(8) :: vcn(0:n+l),t1,t2,t3,vc1
         mk=abs(m+k)
         nl=abs(n-l)
         if(nl.ge.mk) then
            w=nl
            if(n.ge.l) then
               m1=m
               n1=n
               l1=l
               k1=k
            else
               m1=k
               n1=l
               k1=m
               l1=n
            endif
            vc1=(-1)**(k1+l1)*bcof(l1+k1,w-m1-k1) &
               *bcof(l1-k1,w+m1+k1)/bcof(l1+l1,w+w+1)
         else
            w=mk
            if(m+k.ge.0) then
               vc1=(-1)**(n+m)*bcof(n-l+w,l-k)*bcof(l-n+w,n-m) &
                  /bcof(w+w+1,n+l-w)
            else
               vc1=(-1)**(l+k)*bcof(n-l+w,l+k)*bcof(l-n+w,n+m) &
                 /bcof(w+w+1,n+l-w)
            endif
         endif
         w1=w
         vcn(w)=vc1
         w=w1+1
         mk=m+k
         w2=min(wmax,n+l)
         if(w2.gt.w1) then
            t1=2*w*fnr(w+w+1)*fnr(w+w-1)/(fnr(w+mk)*fnr(w-mk) &
              *fnr(n-l+w)*fnr(l-n+w)*fnr(n+l-w+1)*fnr(n+l+w+1))
            if(w1.eq.0) then
               t2=.5*dble(m-k)
            else
               t2=dble((m-k)*w*(w-1)-mk*n*(n+1)+mk*l*(l+1)) &
                 /dble(2*w*(w-1))
            endif
            vcn(w)=t1*t2*vcn(w1)
         endif
         do w=w1+2,w2
            t1=2*w*fnr(w+w+1)*fnr(w+w-1)/(fnr(w+mk)*fnr(w-mk) &
              *fnr(n-l+w)*fnr(l-n+w)*fnr(n+l-w+1)*fnr(n+l+w+1))
            t2=dble((m-k)*w*(w-1)-mk*n*(n+1)+mk*l*(l+1)) &
             /dble(2*w*(w-1))
            t3=fnr(w-mk-1)*fnr(w+mk-1)*fnr(l-n+w-1)*fnr(n-l+w-1) &
              *fnr(n+l-w+2)*fnr(n+l+w)/(dble(2*(w-1))*fnr(2*w-3) &
              *fnr(2*w-1))
            vcn(w)=t1*(t2*vcn(w-1)-t3*vcn(w-2))
         enddo
         end subroutine vcfuncuprec

         subroutine rotcoef(cbe,kmax,nmax,dc)
         use bem_mstm_constants
         implicit none
         integer :: kmax,nmax,k,m,sin,n,knmax,nn1,kn,im,m1
         real(8) :: cbe,sbe,dc(-kmax:kmax,0:nmax*(nmax+2)),cbe2,sbe2,dk0(-nmax-1:nmax+1),&
                    dk01(-nmax-1:nmax+1),sben,dkt,fmn,dkm0,dkm1,dkn1
!if(light_up) then
!write(*,'('' rot1 '',3es13.5)') cbe
!call flush(6)
!endif
         dc=0.d0
         if(abs(cbe).ge.1.d0) then
            sbe=0.d0
         else
            sbe=dsqrt(abs((1.d0+cbe)*(1.d0-cbe)))
         endif
         cbe2=.5d0*(1.d0+cbe)
         sbe2=.5d0*(1.d0-cbe)
         sin=1
         dk0(0)=1.d0
         sben=1.d0
         dc(0,0)=1.d0
         dk01(0)=0.d0
!if(light_up) then
!write(*,'('' rot2 '',3es13.5)') cbe,sbe
!call flush(6)
!endif
         do n=1,nmax
            knmax=min(n,kmax)
            nn1=n*(n+1)
            sin=-sin
            sben=sben*sbe/2.d0
            if(sben.lt.1.d-30) sben=0.d0
            dk0(n)=sin*sben*bcof(n,n)
            dk0(-n)=sin*dk0(n)
            dk01(n)=0.d0
            dk01(-n)=0.d0
            dc(0,nn1+n)=dk0(n)
            dc(0,nn1-n)=dk0(-n)
!if(light_up.and.abs(cbe).gt.0.99999d0) then
!write(*,'('' rot2b '',i5,3es13.5)') n,sben
!call flush(6)
!endif
            do k=-n+1,n-1
               kn=nn1+k
               dkt=dk01(k)
               dk01(k)=dk0(k)
               dk0(k)=(cbe*dble(n+n-1)*dk01(k)-fnr(n-k-1)*fnr(n+k-1)*dkt)&
                     /(fnr(n+k)*fnr(n-k))
               dc(0,kn)=dk0(k)
            enddo
            im=1
            do m=1,knmax
               im=-im
               fmn=1.d0/fnr(n-m+1)/fnr(n+m)
               m1=m-1
               dkm0=0.d0
               do k=-n,n
                  kn=nn1+k
                  dkm1=dkm0
                  dkm0=dc(m1,kn)
                  if(k.eq.n) then
                     dkn1=0.d0
                  else
                     dkn1=dc(m1,kn+1)
                  endif
                  dc(m,kn)=(fnr(n+k)*fnr(n-k+1)*cbe2*dkm1 &
                          -fnr(n-k)*fnr(n+k+1)*sbe2*dkn1  &
                          -dble(k)*sbe*dc(m1,kn))*fmn
                  dc(-m,nn1-k)=dc(m,kn)*(-1)**(k)*im
               enddo
            enddo
         enddo
!if(light_up) then
!write(*,'('' rot3 '',3es13.5)') cbe
!call flush(6)
!endif
         end subroutine rotcoef

end module
