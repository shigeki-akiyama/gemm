#pragma once

#ifdef __INTELLISENSE__
#undef __AVX__
#define __AVX__ 1
#undef __AVX512F__
#define __AVX512F__ 1
#endif

#ifdef __AVX__
#define USE_AVX
#include <immintrin.h>
#endif

#ifdef __AVX512F__
#define USE_AVX512

#if 0
#undef  M_CACHE
#define M_CACHE  135
#endif

#include <immintrin.h>
#endif


//#define PACK_CYCLES


struct blis_arch {
    template <int MC, int NC, int KC, int THREADS>
    struct make_arch {
        enum : int {
            M_CACHE = MC,
            N_CACHE = NC,
            K_CACHE = KC,
            THREADS_PER_CORE = THREADS,
        };
    };

    struct haswell : make_arch<144, 4080, 256, 2> {};
    struct knl_9x3 : make_arch<144, 4080, 256, 4> {};
    struct knl_5x5 : make_arch<135, 4080, 256, 4> {};
    struct knl_28x1 : make_arch<140, 4080, 256, 4> {};
};

