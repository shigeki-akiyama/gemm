#pragma once

#include "util.h"
#include <algorithm>
#include <limits>
#include <type_traits>
#include <memory>

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
#include <immintrin.h>
#endif

template <class T>
struct aligned_deleter { 
    void operator()(T *p) { _mm_free(static_cast<void *>(p)); }
};
template <class T>
struct aligned_deleter<T[]> {
    void operator()(T *p) { _mm_free(static_cast<void *>(p));  }
};

template <class T>
using aligned_ptr = std::unique_ptr<T, aligned_deleter<T>>;

template <class T, class... Args>
static aligned_ptr<T> make_aligned_ptr(int align, Args... args)
{
    void *p = _mm_malloc(sizeof(T), align);
    T *obj = new(p) T(args...);

    return aligned_ptr<T>(obj, aligned_deleter<T>());
}

template <class T>  // require U[]
static aligned_ptr<T[]> make_aligned_array(size_t size, size_t align)
{
    void *p = _mm_malloc(sizeof(T) * size, align);
    T *obj = new(p) T[size];

    return aligned_ptr<T[]>(obj, aligned_deleter<T[]>());
}

template <class T>  // require U[]
static aligned_ptr<T[]> make_aligned_array(size_t size, size_t align, const T& init)
{
    auto p = make_aligned_array<T>(size, align);
    std::fill_n(p.get(), size, init);

    return p;
}

template <class T>
static void copy2d(const T *A, int lda, int M, int N, T *B, int ldb)
{
    for (int i = 0; i < M; i++) {
        std::copy_n(A + lda * i, N, B + ldb * i);
    }
}

template <class T, int RS, int CS>
static void pack2d(const T *A, int lda, int M, int N, T *B, int ldb)
{
    if (RS > 0) {           // for matrix A
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                int k = RS * N * (i / RS) + RS * j + i % RS;
                B[k] = A[lda * i + j];
            }
        }
    } else if (CS > 0) {    // for matrix B
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                int k = CS * M * (j / CS) + CS * i + j % CS;
                B[k] = A[lda * i + j];
            }
        }
    }
}

static void scale_matrix(
    float *A, int lda, int M, int N, float coeff)
{
    if (coeff == 1.0f)
        return;

    __m256 vcoeff = _mm256_set1_ps(coeff);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j += 8) {
            auto *pa = A + lda * i + j;
            auto va = _mm256_load_ps(pa);
            va = _mm256_mul_ps(vcoeff, va);
            _mm256_store_ps(pa, va);
        }
    }
}


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
    T *buf;
    int buf_size;
};

template <class T>
static bool verify_results(T *C, T *result_C, int ldc, int M, int N)
{
    size_t err_count = 0;
    for (auto i = 0; i < M; i++) {
        for (auto j = 0; j < N; j++) {
            T c0 = C[ldc * i + j];
            T c1 = result_C[ldc * i + j];

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

template <class T>
static void flush_all_cachelines(T *buf, int buf_size)
{
    for (int i = 0; i < buf_size; i++) {
        buf[i] += T(0.1);
    }
}

template <class T>
static void ref_gemm(
    int M, int N, int K, T alpha, T *A, int lda,
    T *B, int ldb, T beta, T *C, int ldc);

template <class T, class F>
static void benchmark(
    const char *name, bench_params<T>& bp, F f, 
    bool verify = true, bool on_cache = false, size_t n_times = 10)
{
    int M = bp.M, N = bp.N, K = bp.K;

    auto preprocess = [&] {
        if (on_cache) {
            f(bp.M, bp.N, bp.K, bp.alpha, bp.A, bp.lda,
                bp.B, bp.ldb, bp.beta, bp.C, bp.ldc);
            std::fill_n(bp.C, M * bp.ldc, T(0.0));
            std::fill_n(bp.A, M * bp.lda, T(2.0));
            std::fill_n(bp.B, K * bp.ldb, T(0.5));
        } else {
            std::fill_n(bp.C, M * bp.ldc, T(0.0));
            flush_all_cachelines(bp.buf, bp.buf_size);
        }
    };

    auto r = measure_ntimes(n_times, [&] {
        f(bp.M, bp.N, bp.K, bp.alpha, bp.A, bp.lda,
            bp.B, bp.ldb, bp.beta, bp.C, bp.ldc);
    }, preprocess);

    auto flop = 2.0 * M * N * K; // +3.0 * M * N;
#if 1
    auto gflops = flop / (1e9 * r.min_time);
#else
    auto gflops = flop / (1e9 * r.avg_time);
#endif
    std::printf("%-20s\t%10.3f\t%10.9f\t%10.9f\t%10.9f\t%6zu\n",
        name, gflops, r.avg_time, r.min_time, r.max_time,
        n_times);
    std::fflush(stdout);

    if (verify) {
        verify_results(bp.C, bp.result_C, bp.ldc, M, N);
    }
}
