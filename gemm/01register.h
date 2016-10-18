#pragma once

#include "util.h"

#ifdef USE_AVX
#include <immintrin.h>

namespace register_avx_0 {

    static void gemm(
        int M, int N, int K, float alpha, float *A, int lda,
        float *B, int ldb, float beta, float *C, int ldc)
    {
        auto valpha = _mm256_set1_ps(alpha);
        auto vbeta = _mm256_set1_ps(beta);

        for (int i = 0; i < M; i += 8) {
            for (int j = 0; j < N; j += 8) {
#if 0
                alignas(32) float vab[64] = { 0.0f };
                for (int k = 0; k < K; k += 8) {
                    auto pa = &A[lda * i + k];
                    auto pb = &B[ldb * k + j];
                    for (int ii = 0; ii < 8; ii++) {
                        for (int jj = 0; jj < 8; jj++) {
                            float c = 0.0f;
                            for (int kk = 0; kk < 8; kk++) {
                                c += pa[lda * ii + kk] * pb[ldb * kk + jj];
                            }
                            vab[8 * ii + jj] += c;
                        }
                    }
                }

                auto vab0 = _mm256_load_ps(vab + 8 * 0);
                auto vab1 = _mm256_load_ps(vab + 8 * 1);
                auto vab2 = _mm256_load_ps(vab + 8 * 2);
                auto vab3 = _mm256_load_ps(vab + 8 * 3);
                auto vab4 = _mm256_load_ps(vab + 8 * 4);
                auto vab5 = _mm256_load_ps(vab + 8 * 5);
                auto vab6 = _mm256_load_ps(vab + 8 * 6);
                auto vab7 = _mm256_load_ps(vab + 8 * 7);

#else
                auto vab0 = _mm256_setzero_ps();
                auto vab1 = _mm256_setzero_ps();
                auto vab2 = _mm256_setzero_ps();
                auto vab3 = _mm256_setzero_ps();
                auto vab4 = _mm256_setzero_ps();
                auto vab5 = _mm256_setzero_ps();
                auto vab6 = _mm256_setzero_ps();
                auto vab7 = _mm256_setzero_ps();

                for (int k = 0; k < K; k += 8) {
                    auto pa = &A[lda * i + k];
                    auto pb = &B[ldb * k + j];

                    auto vb0 = _mm256_load_ps(pb + ldb * 0);
                    auto vb1 = _mm256_load_ps(pb + ldb * 1);
                    auto vb2 = _mm256_load_ps(pb + ldb * 2);
                    auto vb3 = _mm256_load_ps(pb + ldb * 3);
                    auto vb4 = _mm256_load_ps(pb + ldb * 4);
                    auto vb5 = _mm256_load_ps(pb + ldb * 5);
                    auto vb6 = _mm256_load_ps(pb + ldb * 6);
                    auto vb7 = _mm256_load_ps(pb + ldb * 7);

                    // dot product
                    //__m256 va[] = { va0, va1, va2, va3, va4, va5, va6, va7 };
                    __m256 vb[] = { vb0, vb1, vb2, vb3, vb4, vb5, vb6, vb7 };
                    __m256 *vab[] = { &vab0, &vab1, &vab2, &vab3, &vab4, &vab5, &vab6, &vab7 };
                    for (int ii = 0; ii < 8; ii++) {
                        float *vaii = pa + lda * ii; // (float *)&va[ii];
                        for (int kk = 0; kk < 8; kk++) {
                            //auto v = _mm256_set1_ps(vaii[kk]);
                            auto v = _mm256_broadcast_ss(&vaii[kk]);
                            *(vab[ii]) = _mm256_fmadd_ps(v, vb[kk], *(vab[ii]));
                        }
                    }
                }
#endif

                auto pc = &C[ldc * i + j];
                auto vc0 = _mm256_load_ps(pc + ldc * 0);
                auto vc1 = _mm256_load_ps(pc + ldc * 1);
                auto vc2 = _mm256_load_ps(pc + ldc * 2);
                auto vc3 = _mm256_load_ps(pc + ldc * 3);
                auto vc4 = _mm256_load_ps(pc + ldc * 4);
                auto vc5 = _mm256_load_ps(pc + ldc * 5);
                auto vc6 = _mm256_load_ps(pc + ldc * 6);
                auto vc7 = _mm256_load_ps(pc + ldc * 7);

                vab0 = _mm256_mul_ps(valpha, vab0);
                vab1 = _mm256_mul_ps(valpha, vab1);
                vab2 = _mm256_mul_ps(valpha, vab2);
                vab3 = _mm256_mul_ps(valpha, vab3);
                vab4 = _mm256_mul_ps(valpha, vab4);
                vab5 = _mm256_mul_ps(valpha, vab5);
                vab6 = _mm256_mul_ps(valpha, vab6);
                vab7 = _mm256_mul_ps(valpha, vab7);

                vc0 = _mm256_fmadd_ps(vbeta, vc0, vab0);
                vc1 = _mm256_fmadd_ps(vbeta, vc1, vab1);
                vc2 = _mm256_fmadd_ps(vbeta, vc2, vab2);
                vc3 = _mm256_fmadd_ps(vbeta, vc3, vab3);
                vc4 = _mm256_fmadd_ps(vbeta, vc4, vab4);
                vc5 = _mm256_fmadd_ps(vbeta, vc5, vab5);
                vc6 = _mm256_fmadd_ps(vbeta, vc6, vab6);
                vc7 = _mm256_fmadd_ps(vbeta, vc7, vab7);

                _mm256_store_ps(pc + ldc * 0, vc0);
                _mm256_store_ps(pc + ldc * 1, vc1);
                _mm256_store_ps(pc + ldc * 2, vc2);
                _mm256_store_ps(pc + ldc * 3, vc3);
                _mm256_store_ps(pc + ldc * 4, vc4);
                _mm256_store_ps(pc + ldc * 5, vc5);
                _mm256_store_ps(pc + ldc * 6, vc6);
                _mm256_store_ps(pc + ldc * 7, vc7);
            }
        }
    }

}

namespace register_avx_1 {
    
    constexpr int TILE_M = 2;
    constexpr int TILE_N = 8 * 4;
    constexpr int TILE_SIZE = TILE_M * TILE_N;

    static void gemm_register(
        int K, float alpha, float *A, int lda, float *B, int ldb,
        float beta, float *C, int ldc, int i, int j)
    {
#if 0
        alignas(32) float vab[TILE_M][TILE_N] = {};
        for (int k = 0; k < K; k++) {
            auto pa = &A[lda * i + k];
            auto pb = &B[ldb * k + j];
            for (int ii = 0; ii < TILE_M; ii++) {
                for (int jj = 0; jj < TILE_N; jj++) {
                    vab[ii][jj] += pa[lda * ii] * pb[jj];
                }
            }
        }
        
        // vab0 vab1 vab2 vab3
        // vab4 vab5 vab6 vab7
        auto vab0 = _mm256_load_ps(&vab[0][8 * 0]);
        auto vab1 = _mm256_load_ps(&vab[0][8 * 1]);
        auto vab2 = _mm256_load_ps(&vab[0][8 * 2]);
        auto vab3 = _mm256_load_ps(&vab[0][8 * 3]);
        auto vab4 = _mm256_load_ps(&vab[1][8 * 0]);
        auto vab5 = _mm256_load_ps(&vab[1][8 * 1]);
        auto vab6 = _mm256_load_ps(&vab[1][8 * 2]);
        auto vab7 = _mm256_load_ps(&vab[1][8 * 3]);

#else
        // vab0 vab1 vab2 vab3
        // vab4 vab5 vab6 vab7
        auto vab0 = _mm256_setzero_ps();
        auto vab1 = _mm256_setzero_ps();
        auto vab2 = _mm256_setzero_ps();
        auto vab3 = _mm256_setzero_ps();
        auto vab4 = _mm256_setzero_ps();
        auto vab5 = _mm256_setzero_ps();
        auto vab6 = _mm256_setzero_ps();
        auto vab7 = _mm256_setzero_ps();

        for (int k = 0; k < K; k++) {
            auto pa = &A[lda * i + k];
            auto pb = &B[ldb * k + j];

            auto vb0 = _mm256_load_ps(pb + 8 * 0);
            auto vb1 = _mm256_load_ps(pb + 8 * 1);
            auto vb2 = _mm256_load_ps(pb + 8 * 2);
            auto vb3 = _mm256_load_ps(pb + 8 * 3);
            auto vb4 = _mm256_load_ps(pb + 8 * 3);

            __m256 va0 = _mm256_broadcast_ss(&pa[lda * 0]);
            __m256 va1 = _mm256_broadcast_ss(&pa[lda * 1]);

            // dot product (ii/jj loop)
            vab0 = _mm256_fmadd_ps(va0, vb0, vab0);
            vab1 = _mm256_fmadd_ps(va0, vb1, vab1);
            vab2 = _mm256_fmadd_ps(va0, vb2, vab2);
            vab3 = _mm256_fmadd_ps(va0, vb3, vab3);
            vab4 = _mm256_fmadd_ps(va1, vb0, vab4);
            vab5 = _mm256_fmadd_ps(va1, vb1, vab5);
            vab6 = _mm256_fmadd_ps(va1, vb2, vab6);
            vab7 = _mm256_fmadd_ps(va1, vb3, vab7);
        }
#endif

        auto pc = &C[ldc * i + j];
        auto vc0 = _mm256_load_ps(pc + ldc * 0 + 8 * 0);
        auto vc1 = _mm256_load_ps(pc + ldc * 0 + 8 * 1);
        auto vc2 = _mm256_load_ps(pc + ldc * 0 + 8 * 2);
        auto vc3 = _mm256_load_ps(pc + ldc * 0 + 8 * 3);
        auto vc4 = _mm256_load_ps(pc + ldc * 1 + 8 * 0);
        auto vc5 = _mm256_load_ps(pc + ldc * 1 + 8 * 1);
        auto vc6 = _mm256_load_ps(pc + ldc * 1 + 8 * 2);
        auto vc7 = _mm256_load_ps(pc + ldc * 1 + 8 * 3);

        auto valpha = _mm256_set1_ps(alpha);
        auto vbeta = _mm256_set1_ps(beta);

        vab0 = _mm256_mul_ps(valpha, vab0);
        vab1 = _mm256_mul_ps(valpha, vab1);
        vab2 = _mm256_mul_ps(valpha, vab2);
        vab3 = _mm256_mul_ps(valpha, vab3);
        vab4 = _mm256_mul_ps(valpha, vab4);
        vab5 = _mm256_mul_ps(valpha, vab5);
        vab6 = _mm256_mul_ps(valpha, vab6);
        vab7 = _mm256_mul_ps(valpha, vab7);

        vc0 = _mm256_fmadd_ps(vbeta, vc0, vab0);
        vc1 = _mm256_fmadd_ps(vbeta, vc1, vab1);
        vc2 = _mm256_fmadd_ps(vbeta, vc2, vab2);
        vc3 = _mm256_fmadd_ps(vbeta, vc3, vab3);
        vc4 = _mm256_fmadd_ps(vbeta, vc4, vab4);
        vc5 = _mm256_fmadd_ps(vbeta, vc5, vab5);
        vc6 = _mm256_fmadd_ps(vbeta, vc6, vab6);
        vc7 = _mm256_fmadd_ps(vbeta, vc7, vab7);

        _mm256_store_ps(pc + ldc * 0 + 8 * 0, vc0);
        _mm256_store_ps(pc + ldc * 0 + 8 * 1, vc1);
        _mm256_store_ps(pc + ldc * 0 + 8 * 2, vc2);
        _mm256_store_ps(pc + ldc * 0 + 8 * 3, vc3);
        _mm256_store_ps(pc + ldc * 1 + 8 * 0, vc4);
        _mm256_store_ps(pc + ldc * 1 + 8 * 1, vc5);
        _mm256_store_ps(pc + ldc * 1 + 8 * 2, vc6);
        _mm256_store_ps(pc + ldc * 1 + 8 * 3, vc7);

    }

    static void gemm(
        int M, int N, int K, float alpha, float *A, int lda,
        float *B, int ldb, float beta, float *C, int ldc)
    {
        for (int i = 0; i < M; i += TILE_M) {
            for (int j = 0; j < N; j += TILE_N) {
                gemm_register(
                    K, alpha, A, lda, B, ldb, beta, C, ldc, i, j);
            }
        }
    }

}

namespace register_avx_2 {

    constexpr int TILE_M = 2;
    constexpr int TILE_N = 8 * 4;
    constexpr int TILE_K = 32;
    constexpr int TILE_SIZE = TILE_M * TILE_N;

    static void matmul_register(
        float *A, int lda, float *B, int ldb, float *C, int ldc)
    {
#if 0
        alignas(32) float vab[TILE_M][TILE_N] = {};
        for (int k = 0; k < TILE_K; k++) {
            auto pa = A + k;
            auto pb = B + ldb * k;
            for (int ii = 0; ii < TILE_M; ii++) {
                for (int jj = 0; jj < TILE_N; jj++) {
                    vab[ii][jj] += pa[lda * ii] * pb[jj];
                }
            }
        }

        // vab0 vab1 vab2 vab3
        // vab4 vab5 vab6 vab7
        auto vab0 = _mm256_load_ps(&vab[0][8 * 0]);
        auto vab1 = _mm256_load_ps(&vab[0][8 * 1]);
        auto vab2 = _mm256_load_ps(&vab[0][8 * 2]);
        auto vab3 = _mm256_load_ps(&vab[0][8 * 3]);
        auto vab4 = _mm256_load_ps(&vab[1][8 * 0]);
        auto vab5 = _mm256_load_ps(&vab[1][8 * 1]);
        auto vab6 = _mm256_load_ps(&vab[1][8 * 2]);
        auto vab7 = _mm256_load_ps(&vab[1][8 * 3]);

#else
        // vab0 vab1 vab2 vab3
        // vab4 vab5 vab6 vab7
        auto vab0 = _mm256_setzero_ps();
        auto vab1 = _mm256_setzero_ps();
        auto vab2 = _mm256_setzero_ps();
        auto vab3 = _mm256_setzero_ps();
        auto vab4 = _mm256_setzero_ps();
        auto vab5 = _mm256_setzero_ps();
        auto vab6 = _mm256_setzero_ps();
        auto vab7 = _mm256_setzero_ps();

        for (int k = 0; k < TILE_K; k++) {
            auto pa = A + k;
            auto pb = B + ldb * k;

            auto vb0 = _mm256_load_ps(pb + 8 * 0);
            auto vb1 = _mm256_load_ps(pb + 8 * 1);
            auto vb2 = _mm256_load_ps(pb + 8 * 2);
            auto vb3 = _mm256_load_ps(pb + 8 * 3);
            auto vb4 = _mm256_load_ps(pb + 8 * 3);

            __m256 va0 = _mm256_broadcast_ss(&pa[lda * 0]);
            __m256 va1 = _mm256_broadcast_ss(&pa[lda * 1]);

            vab0 = _mm256_fmadd_ps(va0, vb0, vab0);
            vab1 = _mm256_fmadd_ps(va0, vb1, vab1);
            vab2 = _mm256_fmadd_ps(va0, vb2, vab2);
            vab3 = _mm256_fmadd_ps(va0, vb3, vab3);
            vab4 = _mm256_fmadd_ps(va1, vb0, vab4);
            vab5 = _mm256_fmadd_ps(va1, vb1, vab5);
            vab6 = _mm256_fmadd_ps(va1, vb2, vab6);
            vab7 = _mm256_fmadd_ps(va1, vb3, vab7);
        }
#endif

        auto vc0 = _mm256_load_ps(C + ldc * 0 + 8 * 0);
        auto vc1 = _mm256_load_ps(C + ldc * 0 + 8 * 1);
        auto vc2 = _mm256_load_ps(C + ldc * 0 + 8 * 2);
        auto vc3 = _mm256_load_ps(C + ldc * 0 + 8 * 3);
        auto vc4 = _mm256_load_ps(C + ldc * 1 + 8 * 0);
        auto vc5 = _mm256_load_ps(C + ldc * 1 + 8 * 1);
        auto vc6 = _mm256_load_ps(C + ldc * 1 + 8 * 2);
        auto vc7 = _mm256_load_ps(C + ldc * 1 + 8 * 3);

        vc0 = _mm256_add_ps(vab0, vc0);
        vc1 = _mm256_add_ps(vab1, vc1);
        vc2 = _mm256_add_ps(vab2, vc2);
        vc3 = _mm256_add_ps(vab3, vc3);
        vc4 = _mm256_add_ps(vab4, vc4);
        vc5 = _mm256_add_ps(vab5, vc5);
        vc6 = _mm256_add_ps(vab6, vc6);
        vc7 = _mm256_add_ps(vab7, vc7);
        
        _mm256_store_ps(C + ldc * 0 + 8 * 0, vc0);
        _mm256_store_ps(C + ldc * 0 + 8 * 1, vc1);
        _mm256_store_ps(C + ldc * 0 + 8 * 2, vc2);
        _mm256_store_ps(C + ldc * 0 + 8 * 3, vc3);
        _mm256_store_ps(C + ldc * 1 + 8 * 0, vc4);
        _mm256_store_ps(C + ldc * 1 + 8 * 1, vc5);
        _mm256_store_ps(C + ldc * 1 + 8 * 2, vc6);
        _mm256_store_ps(C + ldc * 1 + 8 * 3, vc7);
    }

    static void matmul(
        int M, int N, int K, float *A, int lda,
        float *B, int ldb, float *C, int ldc)
    {
        for (int i = 0; i < M; i += TILE_M) {
            for (int j = 0; j < N; j += TILE_N) {
                for (int k = 0; k < K; k += TILE_K) {
                    auto Ar = A + lda * i + k;
                    auto Br = B + ldb * k + j;
                    auto Cr = C + ldc * i + j;
                    matmul_register(Ar, lda, Br, ldb, Cr, ldc);
                }
            }
        }
    }

    static void gemm(
        int M, int N, int K, float alpha, float *A, int lda,
        float *B, int ldb, float beta, float *C, int ldc)
    {
#if 1
        scale_matrix(A, lda, M, K, alpha);
        scale_matrix(C, ldc, M, N, beta);
        
        matmul(M, N, K, A, lda, B, ldb, C, ldc);
#else
        auto r = measure_ntimes(30, [&] {
            scale_matrix(A, lda, M, K, alpha);
            scale_matrix(C, ldc, M, N, beta);
        }, [] {});

        printf("SCALE: avg = %f, min = %f, max = %f\n",
            r.avg_time, r.min_time, r.max_time);

        r = measure_ntimes(30, [&] {
            matmul(M, N, K, A, lda, B, ldb, C, ldc);
        }, [] {});

        printf("MATMUL: avg = %f, min = %f, max = %f\n",
            r.avg_time, r.min_time, r.max_time);
#endif
    }

}

#endif
