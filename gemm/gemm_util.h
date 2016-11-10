#pragma once

#include "config.h"
#include "util.h"
#include <algorithm>
#include <limits>
#include <type_traits>
#include <memory>
#include <cstring>
#include <cassert>

#ifdef USE_AVX
#include <immintrin.h>
#endif

#ifdef USE_AVX512
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
static NOINLINE void copy2d(const T *A, int lda, int M, int N, T *B, int ldb)
{
    for (int i = 0; i < M; i++) {
        //std::copy_n(A + lda * i, N, B + ldb * i);
        std::memcpy(
            static_cast<void *>(B + ldb * i),
            static_cast<const void *>(A + lda * i),
            sizeof(T) * N);
    }
}

static void transpose_matrix_4x16(const float *A, int lda, float *B)
{
    // a00 a10: a0 b0 c0 d0 e0 f0 g0 h0 | i0 j0 k0 l0 m0 n0 o0 p0
    // a01 a11: a1 b1 c1 d1 e1 f1 g1 h1 | i1 j1 k1 l1 m1 n1 o1 p1
    // a02 a12: a2 b2 c2 d1 e2 f2 g2 h2 | i2 j2 k2 l2 m2 n2 o2 p2
    // a03 a13: a3 b3 c3 d1 e3 f3 g3 h3 | i2 j3 k2 l2 m2 n3 o3 p3
    auto a00 = _mm256_load_ps(A + lda * 0 + 8 * 0);
    auto a10 = _mm256_load_ps(A + lda * 0 + 8 * 1);
    auto a01 = _mm256_load_ps(A + lda * 1 + 8 * 0);
    auto a11 = _mm256_load_ps(A + lda * 1 + 8 * 1);
    auto a02 = _mm256_load_ps(A + lda * 2 + 8 * 0);
    auto a12 = _mm256_load_ps(A + lda * 2 + 8 * 1);
    auto a03 = _mm256_load_ps(A + lda * 3 + 8 * 0);
    auto a13 = _mm256_load_ps(A + lda * 3 + 8 * 1);

    // t00 t10: a0 a1 b0 b1 e0 e1 f0 f1 | i0 i1 j0 j1 m0 m1 n0 n1
    // t01 t11: c0 c1 d0 d1 g0 g1 h0 h1 | k0 k1 l0 l1 m2 m3 n2 n3
    // t02 t12: a2 a3 b2 b3 e2 e3 f2 f3 | i2 i3 j2 j3 o0 o1 p0 p1
    // t03 t13: c2 c3 c2 c3 g2 g3 h2 h3 | k2 k3 l2 l3 o2 o3 p2 p3
    auto t00 = _mm256_unpacklo_ps(a00, a01);
    auto t01 = _mm256_unpackhi_ps(a00, a01);
    auto t02 = _mm256_unpacklo_ps(a02, a03);
    auto t03 = _mm256_unpackhi_ps(a02, a03);
    auto t10 = _mm256_unpacklo_ps(a10, a11);
    auto t11 = _mm256_unpackhi_ps(a10, a11);
    auto t12 = _mm256_unpacklo_ps(a12, a13);
    auto t13 = _mm256_unpackhi_ps(a12, a13);

    // u00 u10: a0 a1 a2 a3 e0 e1 e2 e3 | i0 i1 i2 i3 m0 m1 m2 m3
    // u01 u11: b0 b1 b2 b3 f0 f1 f2 f3 | j0 j1 j2 j3 n0 n1 n2 n3
    // u02 u12: c0 c1 c2 c3 g0 g1 g2 g3 | k0 k1 k2 k3 o0 o1 o2 o3
    // u03 u13: d0 d1 d2 d3 h0 h1 h2 h3 | l0 l1 l2 l3 p0 p1 p2 p3
    auto u00 = _mm256_shuffle_ps(t00, t02, _MM_SHUFFLE(1, 0, 1, 0));
    auto u01 = _mm256_shuffle_ps(t00, t02, _MM_SHUFFLE(3, 2, 3, 2));
    auto u02 = _mm256_shuffle_ps(t01, t03, _MM_SHUFFLE(1, 0, 1, 0));
    auto u03 = _mm256_shuffle_ps(t01, t03, _MM_SHUFFLE(3, 2, 3, 2));
    auto u10 = _mm256_shuffle_ps(t10, t12, _MM_SHUFFLE(1, 0, 1, 0));
    auto u11 = _mm256_shuffle_ps(t10, t12, _MM_SHUFFLE(3, 2, 3, 2));
    auto u12 = _mm256_shuffle_ps(t11, t13, _MM_SHUFFLE(1, 0, 1, 0));
    auto u13 = _mm256_shuffle_ps(t11, t13, _MM_SHUFFLE(3, 2, 3, 2));

    // b00L: a0 a1 a2 a3
    // b00H: b0 b1 b2 b3
    // b01L: c0 c1 c2 c3
    // b01H: d0 d1 d2 d3
    // ...
    auto b00 = _mm256_permute2f128_ps(u00, u01, 0x20);
    auto b01 = _mm256_permute2f128_ps(u02, u03, 0x20);
    auto b02 = _mm256_permute2f128_ps(u00, u01, 0x31);
    auto b03 = _mm256_permute2f128_ps(u02, u03, 0x31);
    auto b10 = _mm256_permute2f128_ps(u10, u11, 0x20);
    auto b11 = _mm256_permute2f128_ps(u12, u13, 0x20);
    auto b12 = _mm256_permute2f128_ps(u10, u11, 0x31); 
    auto b13 = _mm256_permute2f128_ps(u12, u13, 0x31);

    _mm256_store_ps(B + 8 * 0, b00);
    _mm256_store_ps(B + 8 * 1, b01);
    _mm256_store_ps(B + 8 * 2, b02);
    _mm256_store_ps(B + 8 * 3, b03);
    _mm256_store_ps(B + 8 * 4, b10);
    _mm256_store_ps(B + 8 * 5, b11);
    _mm256_store_ps(B + 8 * 6, b12);
    _mm256_store_ps(B + 8 * 7, b13);
}

static void transpose_matrix_4xN(const float *A, int lda, int N, float *B)
{
    assert(N % 16 == 0);

    for (int i = 0; i < N / 16; i++) {
        auto Ab = A + 16 * i;
        auto Bb = B + (4 * 16) * i;
        transpose_matrix_4x16(Ab, lda, Bb);
    }
}

template <class T>
static void transpose_matrix(const T *A, int lda, int M, int N, T *B)
{
#if 1
    if (M == 4) {
        transpose_matrix_4xN(A, lda, N, B);
        return;
    }
#endif

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            B[M * j + i] = A[lda * i + j];
        }
    }
}

template <class T, int RS, int CS>
static NOINLINE void pack2d(const T *A, int lda, int M, int N, T *B)
{
    if (RS > 0) {           // for matrix A
        int b_idx = 0;
        for (int i = 0; i < M; i += RS) {
            auto Ab = A + lda * i;
            auto Bb = B + b_idx;

            transpose_matrix(Ab, lda, RS, N, Bb);

            b_idx += RS * N;
        }
    } else if (CS > 0) {    // for matrix B
        int b_idx = 0;
        for (int i = 0; i < N; i += CS) {
            auto Ab = A + i;
            auto Bb = B + b_idx;

            copy2d(Ab, lda, M, CS, Bb, CS);

            b_idx += M * CS;
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
    papix& papi, const char *name, bench_params<T>& bp, F f, 
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

    auto r = measure_ntimes(papi, name, n_times, preprocess, [&] {
        f(bp.M, bp.N, bp.K, bp.alpha, bp.A, bp.lda,
            bp.B, bp.ldb, bp.beta, bp.C, bp.ldc);
    });

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
