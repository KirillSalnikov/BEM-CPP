#pragma once

// glibc 2.43 adds GNU math declarations that conflict with CUDA 13.1's
// declarations. The nvcc build disables GNU extensions, then restores the
// pthread declarations expected by libstdc++ configured with those extensions.
#if defined(__linux__) && defined(__cplusplus)
#include <pthread.h>
#include <time.h>
#if defined(__GLIBC__) && __GLIBC_PREREQ(2, 30) && !defined(__USE_GNU)
extern "C" int pthread_cond_clockwait(
    pthread_cond_t*, pthread_mutex_t*, clockid_t, const struct timespec*) noexcept;
extern "C" int pthread_mutex_clocklock(
    pthread_mutex_t*, clockid_t, const struct timespec*) noexcept;
#endif
#endif
