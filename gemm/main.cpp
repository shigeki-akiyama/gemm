
#include "00naive.h"
#include "01register.h"
#include "02cache.h"
#include "03blis.h"
#include "04blis.h"
#include "util.h"

#ifdef USE_AVX512
#include "01register512.h"
#endif

#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cstdio>

#if defined(USE_MKL)
#include <mkl_cblas.h>
#define USE_CBLAS
#define CBLAS_IMPL "MKL"
#elif defined(USE_BLIS)
#include <blis/cblas.h>
#define USE_CBLAS
#define CBLAS_IMPL "BLIS"
#endif

using elem_type = float;


#ifdef USE_CBLAS
template <class T>
void cblas_gemm(
    int M, int N, int K, T alpha, T *A, int lda,
    T *B, int ldb, T beta, T *C, int ldc);

template <>
void cblas_gemm<float>(
    int M, int N, int K, float alpha, float *A, int lda,
    float *B, int ldb, float beta, float *C, int ldc)
{
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
        alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
void cblas_gemm<double>(
    int M, int N, int K, double alpha, double *A, int lda,
    double *B, int ldb, double beta, double *C, int ldc)
{
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
        alpha, A, lda, B, ldb, beta, C, ldc);
}
#endif

template <class T>
static void ref_gemm(
    int M, int N, int K, T alpha, T *A, int lda,
    T *B, int ldb, T beta, T *C, int ldc)
{
#ifdef USE_CBLAS
    cblas_gemm(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
#else
    naive::gemm(
    //cache_blocking_L3::gemm(
        M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
}

int main(int argc, char *argv[])
{
#if 0
    int size = (argc >= 2) ? atoi(argv[1]) : 48;

    if (0 && size % 32 != 0) {
        std::fprintf(stderr,
            "error: matrix size must be a multiple of 32\n");
        return 1;
    }

    int M = size;
    int N = size;
    int K = size;
#else
    int M = (argc >= 2) ? atoi(argv[1]) : 48;
    int N = (argc >= 3) ? atoi(argv[2]) : 48;
    int K = (argc >= 4) ? atoi(argv[3]) : 48;
#endif

#if 0
    M = 6;
    N = 32;
    K = 76;
#endif

    int lda = K;
    int ldb = N;
    int ldc = N;

    auto alpha = elem_type(1.0); // elem_type(4.0);
    auto beta = elem_type(1.0); // elem_type(0.25);

    int align = 4096; // sizeof(__m256);
    auto A = make_aligned_array<elem_type>(M * lda, align, elem_type(2.0));
    auto B = make_aligned_array<elem_type>(K * ldb, align, elem_type(0.5));
    auto C = make_aligned_array<elem_type>(M * ldc, align, elem_type(0.0));
    auto result_C = make_aligned_array<elem_type>(M * ldc, align, elem_type(0.0));
    
    auto buf_size = 16 * 1024 * 1024 / sizeof(elem_type);
    auto buf = make_aligned_array<elem_type>(buf_size, align, 0.1);

    fill_random(A.get(), M * lda, 0);
    fill_random(B.get(), K * ldb, 1);
//    fill_random(C.get(), M * ldc, 2);

#if 0
    for (int i = 0; i < M; i++)
        for (int j = 0; j < K; j++)
            A[lda * i + j] = K * i + j;

    for (int i = 0; i < K; i++)
        for (int j = 0; j < N; j++)
            B[ldb * i + j] = N * i + j;
#endif

    bench_params<elem_type> bp = {
        M, N, K, alpha, A.get(), lda, B.get(), ldb, beta, 
        C.get(), ldc, result_C.get(), buf.get(), int(buf_size),
    };

#if 0
    register_bench::performL1(bp);
    return 0;
#endif

    // make result_C
    ref_gemm(M, N, K, alpha, A.get(), lda, B.get(), ldb, beta, result_C.get(), ldc);

#ifdef USE_CBLAS
    // MKL warmup
    cblas_gemm(M, N, K, alpha, A.get(), lda, B.get(), ldb, beta, C.get(), ldc);

    // MKL
    benchmark(CBLAS_IMPL, bp, cblas_gemm<elem_type>);
#endif

#if 1
    if (M <= 512) {
        // 0-0. Naive implementation
        //benchmark("naive", bp, naive::gemm<elem_type>);
    }
#endif

#ifdef USE_AVX
#if 0
    if (size <= 768) {
        // 0-1. Naive AVX implementation
        //benchmark("naive_avx", bp, naive_avx::gemm);

        // 1-0. Register blocking (but spilled) AVX implementation
        benchmark("register_avx_0", bp, register_avx_0::gemm);

        // 1-1. Register blocking AVX implementation
        benchmark("register_avx_1", bp, register_avx_1::gemm);

        // 1-2. Register blocking AVX implementation (blocking with K)
        benchmark("register_avx_2", bp, register_avx_2::gemm);
    }
#endif

#if 1
    // 2-1. cache-oblivious implementation with 1-2.
    benchmark("cache_oblivious", bp, cache_oblivious::gemm);
#endif

#if 0
    // 2-2. L1-cache blocking implementation with 1-2.
    benchmark("L1 blocking", bp, cache_blocking_L1::gemm);

    // 2-3. L2-cache blocking implementation with 2-2.
    benchmark("L2 blocking", bp, cache_blocking_L2::gemm);
#endif
    // 2-3. L2-cache blocking implementation with 2-2.
    //benchmark("L3 blocking", bp, cache_blocking_L3::gemm);
#endif
    using option = blis_opt<elem_type>;

#if 1
    // 3-1. BLIS-based implementation
    {
        using blis = blis_copy<register_avx_3_6x2, option::naive>;
        blis::initialize();
        benchmark("blis_naive_6x2", bp, blis::gemm);
    }
#endif

#if 1
    // 3-2. BLIS-based implementation w/ copy optimization on L1
    {
        using blis = blis_copy<register_avx_3_6x2, option::copyL1>;
        blis::initialize();
        benchmark("blis_copyL1_6x2", bp, blis::gemm);
    }
#endif

#if 1
    // 3-3. BLIS-based implementation w/ copy optimization on L1/L2
    {
        using blis = blis_copy<register_avx_3_6x2, option::copyL2>;
        blis::initialize();
        benchmark("blis_copyL2_6x2", bp, blis::gemm);
    }

    // 3-4. BLIS-based implementation w/ copy optimization on L1/L2/L3
    {
        using blis = blis_copy<register_avx_3_6x2, option::copyL3>;
        blis::initialize();
        benchmark("blis_copyL3_6x2", bp, blis::gemm);
    }
#endif
#if 1
    // 3-1'. BLIS-based implementation
    {
        using blis = blis_copy<register_avx_3_4x3, option::naive>;
        blis::initialize();
        benchmark("blis_naive_4x3", bp, blis::gemm);
    }

    // 3-2'. BLIS-based implementation w/ copy optimization on L1
    {
        using blis = blis_copy<register_avx_3_4x3, option::copyL1>;
        blis::initialize();
        benchmark("blis_copyL1_4x3", bp, blis::gemm);
    }

    // 3-3'. BLIS-based implementation w/ copy optimization on L1/L2
    {
        using blis = blis_copy<register_avx_3_4x3, option::copyL2>;
        blis::initialize();
        benchmark("blis_copyL2_4x3", bp, blis::gemm);
    }

    // 3-4'. BLIS-based implementation w/ copy optimization on L1/L2/L3
    {
        using blis = blis_copy<register_avx_3_4x3, option::copyL3>;
        blis::initialize();
        benchmark("blis_copyL3_4x3", bp, blis::gemm);
    }
#endif
#if 1
    /*
    // 4-1. BLIS-based implementation w/ copy with stride format on L1
    {
        using blis = blis<register_avx_3_6x2, option::packL1>;
        blis::intiialize();
        benchmark("blis_packL1_6x2", bp, blis::gemm);
    }
    */
    /*
    // 4-2. BLIS-based implementation w/ copy with stride format on L1/L2
    {
        using blis = blis_<register_avx_3_6x2, option::packL2>;
        blis::intiialize();
        benchmark("blis_packL2_6x2", bp, blis::gemm);
    }
    */
    // 4-3. BLIS-based implementation w/ copy with stride format on L1/L2/L3
    {
        using blis = blis<register_avx_3_6x2, option::packL3>;
        blis::intiialize();
        benchmark("blis_packL3_6x2", bp, blis::gemm);
    }
    
    {
        using blis = blis<register_avx_3_6x2, option::packL3>;
        blis::intiialize();
        benchmark("blis_packL3_4x3", bp, blis::gemm);
    }
#endif
#ifdef USE_AVX512
    // 5-1. AVX512 implementation based on naive BLIS
    {
        using blis = blis_copy<register_avx512_9x3, option::naive>;
        blis::initialize();
        benchmark("blis512_naive_9x3", bp, blis::gemm);
    }

    // 5-2. AVX512 implementation based on L1-BLIS
    {
        using blis = blis_copy<register_avx512_9x3, option::copyL1>;
        blis::initialize();
        benchmark("blis512_copyL1_9x3", bp, blis::gemm);
    }

    // 5-3. AVX512 implementation based on L2-BLIS
    {
        using blis = blis_copy<register_avx512_9x3, option::copyL2>;
        blis::initialize();
        benchmark("blis512_copyL2_9x3", bp, blis::gemm);
    }

    // 5-4. AVX512 implementation based on L3-BLIS
    {
        using blis = blis_copy<register_avx512_9x3, option::copyL3>;
        blis::initialize();
        benchmark("blis512_copyL3_9x3", bp, blis::gemm);
    }
#endif

    return 0;
}
