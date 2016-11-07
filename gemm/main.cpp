
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
#include <tuple>
#include <sstream>
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
    //naive::gemm(
    cache_blocking_L3::gemm(
        M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
}

using blas_function = void(*)(
    int M, int N, int K, float alpha, float *A, int lda,
    float *B, int ldb, float beta, float *C, int ldc);
using bench_pair = std::tuple<const char *, blas_function>;

static std::vector<bench_pair> make_benchmarks(int M, int N, int K)
{
    std::vector<bench_pair> pairs;

    auto push = [&](const char * name, blas_function func) {
        pairs.push_back(std::make_tuple(name, func));
    };

#ifdef USE_CBLAS
    // MKL
    push(CBLAS_IMPL, cblas_gemm<elem_type>);
#endif

#if 0
    if (M * N * K <= 512 * 512 * 512) {
        // 0-0. Naive implementation
        push("naive", naive::gemm<elem_type>);
    }
#endif

#if 0
    if (M * N * K <= 768 * 768 * 768) {
        // 0-1. Naive AVX implementation
        push("naive_avx", naive_avx::gemm);

        // 1-0. Register blocking (but spilled) AVX implementation
        push("register_avx_0", register_avx_0::gemm);

        // 1-1. Register blocking AVX implementation
        push("register_avx_1", register_avx_1::gemm);

        // 1-2. Register blocking AVX implementation (blocking with K)
        push("register_avx_2", register_avx_2::gemm);
    }
#endif

#if 1
    // 2-1. cache-oblivious implementation with 1-2.
    push("cache_oblivious", cache_oblivious::gemm);
#endif

#if 1
    // 2-2. L1-cache blocking implementation with 1-2.
    push("L1_blocking", cache_blocking_L1::gemm);

    // 2-3. L2-cache blocking implementation with 2-2.
    push("L2_blocking", cache_blocking_L2::gemm);

    // 2-3. L2-cache blocking implementation with 2-2.
    push("L3_blocking", cache_blocking_L3::gemm);
#endif

    using option = blis_opt<elem_type>;

#if 1
    // 3-1. BLIS-based implementation
    {
        using blis = blis_copy<register_avx_3_6x2, option::naive>;
        blis::initialize();
        push("blis_naive_6x2", blis::gemm);
    }
#endif

#if 1
    // 3-2. BLIS-based implementation w/ copy optimization on L1
    {
        using blis = blis_copy<register_avx_3_6x2, option::copyL1>;
        blis::initialize();
        push("blis_copyL1_6x2", blis::gemm);
    }
#endif

#if 1
    // 3-3. BLIS-based implementation w/ copy optimization on L1/L2
    {
        using blis = blis_copy<register_avx_3_6x2, option::copyL2>;
        blis::initialize();
        push("blis_copyL2_6x2", blis::gemm);
    }

    // 3-4. BLIS-based implementation w/ copy optimization on L1/L2/L3
    {
        using blis = blis_copy<register_avx_3_6x2, option::copyL3>;
        blis::initialize();
        push("blis_copyL3_6x2", blis::gemm);
    }
#endif
#if 1
    // 3-1'. BLIS-based implementation
    {
        using blis = blis_copy<register_avx_3_4x3, option::naive>;
        blis::initialize();
        push("blis_naive_4x3", blis::gemm);
    }

    // 3-2'. BLIS-based implementation w/ copy optimization on L1
    {
        using blis = blis_copy<register_avx_3_4x3, option::copyL1>;
        blis::initialize();
        push("blis_copyL1_4x3", blis::gemm);
    }

    // 3-3'. BLIS-based implementation w/ copy optimization on L1/L2
    {
        using blis = blis_copy<register_avx_3_4x3, option::copyL2>;
        blis::initialize();
        push("blis_copyL2_4x3", blis::gemm);
    }

    // 3-4'. BLIS-based implementation w/ copy optimization on L1/L2/L3
    {
        using blis = blis_copy<register_avx_3_4x3, option::copyL3>;
        blis::initialize();
        push("blis_copyL3_4x3", blis::gemm);
    }
#endif
#if 1
    // 4-1. BLIS-based implementation w/ copy with stride format on L1/L2
    {
        using blis = blisL2<register_avx_3_6x2>;
        blis::intiialize();
        push("blis_packL2_6x2", blis::gemm);
    }
    {
        using blis = blisL2<register_avx_3_4x3>;
        blis::intiialize();
        push("blis_packL2_4x3", blis::gemm);
    }

    // 4-1'. BLIS-based implementation w/ copy with stride format on L1/L2/L3
    {
        using blis = blis<register_avx_3_6x2>;
        blis::intiialize();
        push("blis_packL3_6x2", blis::gemm);
    }
    {
        using blis = blis<register_avx_3_4x3>;
        blis::intiialize();
        push("blis_packL3_4x3", blis::gemm);
    }
#endif
#ifdef USE_AVX512
    // 5-1. AVX512 implementation based on naive BLIS
    {
        using blis = blis_copy<register_avx512_9x3, option::naive>;
        blis::initialize();
        push("blis512_naive_9x3", blis::gemm);
    }

    // 5-2. AVX512 implementation based on L1-BLIS
    {
        using blis = blis_copy<register_avx512_9x3, option::copyL1>;
        blis::initialize();
        push("blis512_copyL1_9x3", blis::gemm);
    }

    // 5-3. AVX512 implementation based on L2-BLIS
    {
        using blis = blis_copy<register_avx512_9x3, option::copyL2>;
        blis::initialize();
        push("blis512_copyL2_9x3", blis::gemm);
    }

    // 5-4. AVX512 implementation based on L3-BLIS
    {
        using blis = blis_copy<register_avx512_9x3, option::copyL3>;
        blis::initialize();
        push("blis512_copyL3_9x3", blis::gemm);
    }
#endif

    return pairs;
}

static int real_main(
    int M, int N, int K, const std::vector<std::string>& fnames, bool verify)
{
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

#ifdef USE_CBLAS
    // MKL warmup
    cblas_gemm(M, N, K, alpha, A.get(), lda, B.get(), ldb, beta, C.get(), ldc);
#endif

    // make result_C
    if (verify)
        ref_gemm(M, N, K, alpha, A.get(), lda, B.get(), ldb, beta, result_C.get(), ldc);

    auto all_benchs = make_benchmarks(M, N, K);

    std::vector<bench_pair> benchs;
    {
        if (fnames.size() == 0) {
            benchs = all_benchs;
        } else {
            for (auto& fname : fnames) {
                auto it = std::find_if(all_benchs.begin(), all_benchs.end(), 
                    [&] (const bench_pair& pair) {
                        return fname == std::get<0>(pair);
                    });

                if (it != all_benchs.end())
                    benchs.push_back(*it);
            }
        }
    }

    papix papi;
    for (auto& bench : benchs) {
        auto fname = std::get<0>(bench);
        auto f = std::get<1>(bench);
         benchmark(papi, fname, bp, f, verify);
    }

    papi.print_results();

    return 0;
}

std::vector<std::string> parse_select(const char * s)
{
    std::vector<std::string> fnames;

    std::stringstream ss(s);
    std::string fname;
    while (std::getline(ss, fname, ',')) {
        fnames.push_back(fname);
    }

    return fnames;
}

int main(int argc, char ** argv)
{
    // Skip argv[0]
    argc -= 1;
    argv += 1;

    // Parse select arguments (-s fname0,fname1,...)
    std::vector<std::string> fnames;
    if (argc >= 2 && std::strcmp(argv[0], "-select") == 0) {
        fnames = parse_select(argv[1]);
        argc -= 2;
        argv += 2;
    }

    // Parse verify argument (specify to execute the verify code)
    bool verify = false;
    if (argc >= 1 && std::strcmp(argv[0], "-verify") == 0) {
        verify = true;
        argc -= 1;
        argv += 1;
    }

    // Parse matrix size
    int M, N, K;
    M = N = K = 48;     // default value
    if (argc == 1) {
        M = N = K = std::atoi(argv[0]);
        argc -= 1;
        argv += 1;
    } else if (argc == 3) {
        M = std::atoi(argv[0]);
        N = std::atoi(argv[1]);
        K = std::atoi(argv[2]);
        argc -= 3;
        argv += 3;
    }

#if 0
    M = 6;
    N = 32;
    K = 76;
#endif

#if 0
    M = N = K = 384;
#endif
    {
        std::stringstream ss;
        if (fnames.size() >= 1) {
            ss << fnames[0];
            for (auto it = fnames.begin() + 1; it != fnames.end(); ++it)
                ss << ", " << *it;
        }

        printf("Arguments: fnames = { %s }, M = %d, N = %d, K = %d\n",
            ss.str().c_str(), M, N, K);
    }

    return real_main(M, N, K, fnames, verify);
}
