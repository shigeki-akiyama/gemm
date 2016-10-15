
#include "00naive.h"
#include "util.h"
#include <vector>
#include <algorithm>
#include <cstdlib>

#define USE_MKL

#ifdef USE_MKL
#include <mkl_blas.h>
#endif

using elem_type = float;
using elem_vector = std::vector<elem_type>;

template <class T>
struct bench_params {
    int M, N, K;
    T alpha;
    T *A;
    int lda;
    T *B;
    int ldb;
    T beta;
    T *C;
    int ldc;
    T *result_C;
};

template <class T>
static bool verify_results(T *C, T *result_C, int M, int N)
{
    size_t err_count = 0;
    for (auto i = 0; i < M; i++) {
        for (auto j = 0; j < N; j++) {
            T c0 = C[M * i + j];
            T c1 = result_C[M * i + j];

            if (c0 != c1) {
                std::printf(
                    "error: gemm result does not match at "
                    "[%5d, %5d] (%f != %f)\n",
                    i, j, c0, c1);

                if (++err_count > 8) {
                    return false;
                }
            }
        }
    }
    
    return err_count == 0;
}

template <class T, class F>
static void benchmark(
    const char *name, bench_params<T>& bp, F f, 
    bool verify=true)
{
    size_t n_times = 10;

    auto preprocess = [&] {
        std::fill_n(bp.C, bp.M * bp.N, elem_type(0.0));
    };

    auto r = measure_ntimes(n_times, [&] {
        f(bp.M, bp.N, bp.K, bp.alpha, bp.A, bp.lda,
            bp.B, bp.ldb, bp.beta, bp.C, bp.ldc);
    }, preprocess);

    std::printf("%-20s\t%10.6f\t%10.6f\t%10.6f\t%6zu\n",
        name, r.avg_time, r.min_time, r.max_time, n_times);
    std::fflush(stdout);

    if (verify) {
        verify_results(bp.C, bp.result_C, bp.M, bp.N);
    }
}


static void mkl_gemm(
    int M, int N, int K, float alpha, float *A, int lda,
    float *B, int ldb, float beta, float *C, int ldc)
{
    sgemm(
        "N", "N", &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta,
        C, &ldc);
}

static void mkl_gemm(
    int M, int N, int K, double alpha, double *A, int lda,
    double *B, int ldb, double beta, double *C, int ldc)
{
    dgemm(
        "N", "N", &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta,
        C, &ldc);
}

template <class T>
static void ref_gemm(
    int M, int N, int K, T alpha, T *A, int lda,
    T *B, int ldb, T beta, T *C, int ldc)
{
#ifdef USE_MKL
    mkl_gemm(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
#else
    naive::gemm(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
}

int main(int argc, char *argv[])
{
    int size = (argc >= 2) ? atoi(argv[1]) : 32;

    if (size % 8 != 0) {
        std::fprintf(stderr,
            "error: matrix size must be a multiple of 8\n");
        return 1;
    }

    int M = size;
    int N = size;
    int K = size;
    int lda = K;
    int ldb = N;
    int ldc = K;
    auto alpha = elem_type(4.0);
    auto beta  = elem_type(0.25);

    elem_vector A(M * K);
    elem_vector B(K * N);
    elem_vector C(M * N);
    elem_vector result_C(M * N);

    std::fill(A.begin(), A.end(), elem_type(2.0));
    std::fill(B.begin(), B.end(), elem_type(0.5));

    bench_params<elem_type> bp = {
        M, N, K, alpha, A.data(), lda, B.data(), ldb, beta, 
        C.data(), ldc, result_C.data(),
    };

    // Reference implementation
    benchmark("ref_gemm", bp, ref_gemm<elem_type>, false);
    std::copy(C.begin(), C.end(), result_C.begin());

    // 0-0. Naive implementation
    benchmark("naive", bp, naive::gemm<elem_type>);

    // 0-1. Naive AVX implementation
    benchmark("naive_avx", bp, naive_avx::gemm);

    return 0;
}
