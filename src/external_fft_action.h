#ifndef BEM_EXTERNAL_FFT_ACTION_H
#define BEM_EXTERNAL_FFT_ACTION_H
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cerrno>
#include <complex>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

// Optional host-vector action supplied by the validated compressed FFT service.
// The worker retains its own physical FMM residual and its existing C6 solver.
// A closed or incompatible service is an error, never an implicit zero action.
class ExternalFFTAction {
    int fd_=-1;
    size_t n_=0;
    uint64_t sequence_=0;
    std::vector<std::complex<float>> wire_;
    static const uint64_t magic_=0x42454d4646543031ULL;
    static void transfer(int fd,void* data,size_t size,bool sending) {
        char* p=static_cast<char*>(data);
        while(size) {
            ssize_t count=sending?send(fd,p,size,MSG_NOSIGNAL):recv(fd,p,size,0);
            if(count<0 && errno==EINTR)continue;
            if(count<=0)throw std::runtime_error("compressed FFT service closed or timed out; inspect external_fft.log");
            p+=count;size-=size_t(count);
        }
    }
public:
    ExternalFFTAction(const std::string& path,size_t n,int ref,double ka,
                      std::complex<double> ri,double aspect,int quadrature,int duffy):n_(n) {
        sockaddr_un address={};address.sun_family=AF_UNIX;
        if(path.empty() || path.size()>=sizeof(address.sun_path))throw std::invalid_argument("compressed FFT socket path");
        std::memcpy(address.sun_path,path.c_str(),path.size()+1);
        fd_=socket(AF_UNIX,SOCK_STREAM,0);
        try {
            if(fd_<0 || connect(fd_,reinterpret_cast<sockaddr*>(&address),sizeof(address)))
                throw std::runtime_error("cannot connect to compressed FFT service");
            timeval timeout={600,0};
            if(setsockopt(fd_,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout)) ||
               setsockopt(fd_,SOL_SOCKET,SO_SNDTIMEO,&timeout,sizeof(timeout)))
                throw std::runtime_error("cannot configure compressed FFT socket timeout");
            uint64_t magic=magic_;
            double parameters[9]={1.,double(n),double(ref),ka,ri.real(),ri.imag(),aspect,double(quadrature),double(duffy)};
            transfer(fd_,&magic,sizeof(magic),true);transfer(fd_,parameters,sizeof(parameters),true);
            uint64_t ready[2];transfer(fd_,ready,sizeof(ready),false);
            if(ready[0]!=magic_ || ready[1]!=n_)throw std::runtime_error("compressed FFT handshake mismatch");
        }catch(...){if(fd_>=0)close(fd_);fd_=-1;throw;}
    }
    ~ExternalFFTAction(){
        if(fd_>=0){uint64_t quit[2]={0,sequence_+1};send(fd_,quit,sizeof(quit),MSG_NOSIGNAL);close(fd_);}
    }
    ExternalFFTAction(const ExternalFFTAction&)=delete;
    ExternalFFTAction& operator=(const ExternalFFTAction&)=delete;
    void apply(const std::complex<double>* input,std::complex<double>* output) {
        const char* setting=std::getenv("BEM_TMATRIX_COMPRESSED_FFT_WIRE_FP32");
        const bool compact=setting && std::strcmp(setting,"0")!=0;
        // The service already casts every input to complex64 before the FFT.
        // Sending that representation avoids double-width transfers/casts.
        if(compact) {
            wire_.resize(n_);
            for(size_t i=0;i<n_;++i)wire_[i]=std::complex<float>(input[i]);
        }
        uint64_t request[2]={compact?2ULL:1ULL,++sequence_};
        transfer(fd_,request,sizeof(request),true);
        if(compact)transfer(fd_,wire_.data(),n_*sizeof(wire_[0]),true);
        else transfer(fd_,const_cast<std::complex<double>*>(input),n_*sizeof(*input),true);
        uint64_t response[2];transfer(fd_,response,sizeof(response),false);
        if(response[0]!=sequence_ || response[1]!=0)throw std::runtime_error("compressed FFT response mismatch");
        if(compact) {
            transfer(fd_,wire_.data(),n_*sizeof(wire_[0]),false);
            for(size_t i=0;i<n_;++i)output[i]=std::complex<double>(wire_[i]);
        }else transfer(fd_,output,n_*sizeof(*output),false);
    }
    uint64_t calls()const{return sequence_;}
};
#endif
