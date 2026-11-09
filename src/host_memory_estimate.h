#ifndef BEM_HOST_MEMORY_ESTIMATE_H
#define BEM_HOST_MEMORY_ESTIMATE_H
#include <cstddef>
#include <limits>
#include <stdexcept>
namespace bem_cpu { namespace memory {
inline size_t add(size_t a,size_t b){
    if(b>std::numeric_limits<size_t>::max()-a)throw std::overflow_error("CPU workspace estimate overflow");
    return a+b;
}
inline size_t mul(size_t a,size_t b){
    if(a && b>std::numeric_limits<size_t>::max()/a)throw std::overflow_error("CPU workspace estimate overflow");
    return a*b;
}
} }
#endif
