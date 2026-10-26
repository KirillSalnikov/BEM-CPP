#ifndef BEM_SO3_STORAGE_KIND
#define BEM_SO3_STORAGE_KIND 8
#endif
program bem_so3
use bem_mstm_average
implicit none
integer::n,nt,blk,j,cyclic
character(4096)::file,arg
complex(BEM_SO3_STORAGE_KIND),allocatable::tc(:,:,:,:)
real(8),allocatable::theta(:),smc(:,:,:),smcf(:,:,:),mueller(:,:,:)
call get_command_argument(1,arg);read(arg,*)n
call get_command_argument(2,arg);read(arg,*)nt
blk=n*(n+2)
if(n<1.or.n>256.or.nt<1)error stop 'invalid order/angular count'
allocate(tc(2,blk,2,blk),theta(nt),smc(4,4,0:2*n),smcf(4,4,0:2*n),mueller(4,4,nt))
call get_command_argument(3,file)
open(10,file=trim(file),access='stream',form='unformatted',status='old',action='read')
read(10)tc
close(10)
call get_command_argument(4,file)
open(10,file=trim(file),access='stream',form='unformatted',status='old',action='read')
read(10)theta
close(10)
call get_command_argument(6,arg);read(arg,*)cyclic
if(cyclic<1)error stop "invalid symmetry order"
call coefficients(n,tc,smc,smcf,cyclic)
do j=1,nt
 call ranorienscatmatrixcalc(cos(theta(j)),smc,2*n,mueller(:,:,j))
enddo
call get_command_argument(5,file)
open(10,file=trim(file),access='stream',form='unformatted',status='replace')
write(10)mueller
close(10)
end program
