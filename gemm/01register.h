#pragma once

#include "config.h"
#include "gemm_util.h"

#include <cstring>
#include <cassert>
#undef NODEBUG

#ifdef USE_AVX
#include <immintrin.h>

#define _mm256_stream_load_ps(p) \
    _mm256_castsi256_ps(_mm256_stream_load_si256((__m256i *)(p)))

struct register_avx_0 {

    static void gemm(
        int M, int N, int K, float alpha, float *A, int lda,
        float *B, int ldb, float beta, float *C, int ldc)
    {
        auto valpha = _mm256_set1_ps(alpha);
        auto vbeta = _mm256_set1_ps(beta);

        for (int i = 0; i < M; i += 8) {
            for (int j = 0; j < N; j += 8) {
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
                        float *vaii = pa + lda * ii;
                        for (int kk = 0; kk < 8; kk++) {
                            auto v = _mm256_broadcast_ss(&vaii[kk]);
                            *(vab[ii]) = _mm256_fmadd_ps(v, vb[kk], *(vab[ii]));
                        }
                    }
                }

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

};

struct register_avx_1 {
    
    enum : int {
        BLOCK_M = 2,
        BLOCK_N = 8 * 4,
    };

    static void gemm_register(
        int M, int N, int K, float alpha, float *A, int lda, float *B, int ldb,
        float beta, float *C, int ldc, int i, int j)
    {
        assert(M % BLOCK_M == 0);
        assert(N % BLOCK_N == 0);

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

            auto vb0 = _mm256_loadu_ps(pb + 8 * 0);
            auto vb1 = _mm256_loadu_ps(pb + 8 * 1);
            auto vb2 = _mm256_loadu_ps(pb + 8 * 2);
            auto vb3 = _mm256_loadu_ps(pb + 8 * 3);

            auto va0 = _mm256_broadcast_ss(&pa[lda * 0]);
            auto va1 = _mm256_broadcast_ss(&pa[lda * 1]);

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

        auto pc = &C[ldc * i + j];
        auto vc0 = _mm256_loadu_ps(pc + ldc * 0 + 8 * 0);
        auto vc1 = _mm256_loadu_ps(pc + ldc * 0 + 8 * 1);
        auto vc2 = _mm256_loadu_ps(pc + ldc * 0 + 8 * 2);
        auto vc3 = _mm256_loadu_ps(pc + ldc * 0 + 8 * 3);
        auto vc4 = _mm256_loadu_ps(pc + ldc * 1 + 8 * 0);
        auto vc5 = _mm256_loadu_ps(pc + ldc * 1 + 8 * 1);
        auto vc6 = _mm256_loadu_ps(pc + ldc * 1 + 8 * 2);
        auto vc7 = _mm256_loadu_ps(pc + ldc * 1 + 8 * 3);

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

        _mm256_storeu_ps(pc + ldc * 0 + 8 * 0, vc0);
        _mm256_storeu_ps(pc + ldc * 0 + 8 * 1, vc1);
        _mm256_storeu_ps(pc + ldc * 0 + 8 * 2, vc2);
        _mm256_storeu_ps(pc + ldc * 0 + 8 * 3, vc3);
        _mm256_storeu_ps(pc + ldc * 1 + 8 * 0, vc4);
        _mm256_storeu_ps(pc + ldc * 1 + 8 * 1, vc5);
        _mm256_storeu_ps(pc + ldc * 1 + 8 * 2, vc6);
        _mm256_storeu_ps(pc + ldc * 1 + 8 * 3, vc7);
    }

    static void gemm(
        int M, int N, int K, float alpha, float *A, int lda,
        float *B, int ldb, float beta, float *C, int ldc)
    {
        assert(M % BLOCK_M == 0);
        assert(N % BLOCK_N == 0);

        for (int i = 0; i < M; i += BLOCK_M) {
            for (int j = 0; j < N; j += BLOCK_N) {
                gemm_register(
                    BLOCK_M, BLOCK_N, K, alpha, A, lda, B, ldb, beta, C, ldc, i, j);
            }
        }
    }
};

struct register_avx_2 {

    enum : int {
        BLOCK_M = 2,
        BLOCK_N = 8 * 4,
        BLOCK_K = 32,
    };

    static void matmul_register(
        int M, int N, int K, float *A, int lda, float *B, int ldb, float *C, int ldc)
    {
        assert(M % BLOCK_M == 0);
        assert(N % BLOCK_N == 0);
        assert(K % BLOCK_K == 0);

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

        for (int k = 0; k < BLOCK_K; k++) {
            auto pa = A + k;
            auto pb = B + ldb * k;

            auto vb0 = _mm256_loadu_ps(pb + 8 * 0);
            auto va0 = _mm256_broadcast_ss(&pa[lda * 0]);
            vab0 = _mm256_fmadd_ps(va0, vb0, vab0);

            auto va1 = _mm256_broadcast_ss(&pa[lda * 1]);
            auto vb1 = _mm256_loadu_ps(pb + 8 * 1);
            vab1 = _mm256_fmadd_ps(va0, vb1, vab1);
            vab5 = _mm256_fmadd_ps(va1, vb1, vab5);
            
            auto vb2 = _mm256_loadu_ps(pb + 8 * 2);
            vab2 = _mm256_fmadd_ps(va0, vb2, vab2);
            vab6 = _mm256_fmadd_ps(va1, vb2, vab6);

            auto vb3 = _mm256_loadu_ps(pb + 8 * 3);
            vab3 = _mm256_fmadd_ps(va0, vb3, vab3);
            vab7 = _mm256_fmadd_ps(va1, vb3, vab7);

            vab4 = _mm256_fmadd_ps(va1, vb0, vab4);
        }

        auto vc0 = _mm256_loadu_ps(C + ldc * 0 + 8 * 0);
        auto vc1 = _mm256_loadu_ps(C + ldc * 0 + 8 * 1);
        auto vc2 = _mm256_loadu_ps(C + ldc * 0 + 8 * 2);
        auto vc3 = _mm256_loadu_ps(C + ldc * 0 + 8 * 3);
        auto vc4 = _mm256_loadu_ps(C + ldc * 1 + 8 * 0);
        auto vc5 = _mm256_loadu_ps(C + ldc * 1 + 8 * 1);
        auto vc6 = _mm256_loadu_ps(C + ldc * 1 + 8 * 2);
        auto vc7 = _mm256_loadu_ps(C + ldc * 1 + 8 * 3);

        vc0 = _mm256_add_ps(vab0, vc0);
        vc1 = _mm256_add_ps(vab1, vc1);
        vc2 = _mm256_add_ps(vab2, vc2);
        vc3 = _mm256_add_ps(vab3, vc3);
        vc4 = _mm256_add_ps(vab4, vc4);
        vc5 = _mm256_add_ps(vab5, vc5);
        vc6 = _mm256_add_ps(vab6, vc6);
        vc7 = _mm256_add_ps(vab7, vc7);
        
        _mm256_storeu_ps(C + ldc * 0 + 8 * 0, vc0);
        _mm256_storeu_ps(C + ldc * 0 + 8 * 1, vc1);
        _mm256_storeu_ps(C + ldc * 0 + 8 * 2, vc2);
        _mm256_storeu_ps(C + ldc * 0 + 8 * 3, vc3);
        _mm256_storeu_ps(C + ldc * 1 + 8 * 0, vc4);
        _mm256_storeu_ps(C + ldc * 1 + 8 * 1, vc5);
        _mm256_storeu_ps(C + ldc * 1 + 8 * 2, vc6);
        _mm256_storeu_ps(C + ldc * 1 + 8 * 3, vc7);
    }

    static void matmul(
        int M, int N, int K, float *A, int lda,
        float *B, int ldb, float *C, int ldc)
    {
        assert(M % BLOCK_M == 0);
        assert(N % BLOCK_N == 0);
        assert(K % BLOCK_K == 0);

        for (int i = 0; i < M; i += BLOCK_M) {
            for (int j = 0; j < N; j += BLOCK_N) {
                for (int k = 0; k < K; k += BLOCK_K) {
                    auto Ar = A + lda * i + k;
                    auto Br = B + ldb * k + j;
                    auto Cr = C + ldc * i + j;
                    matmul_register(
                        BLOCK_M, BLOCK_N, BLOCK_K, Ar, lda, Br, ldb, Cr, ldc);
                }
            }
        }
    }

    static void gemm(
        int M, int N, int K, float alpha, float *A, int lda,
        float *B, int ldb, float beta, float *C, int ldc)
    {
        scale_matrix(A, lda, M, K, alpha);
        scale_matrix(C, ldc, M, N, beta);
        
        matmul(M, N, K, A, lda, B, ldb, C, ldc);
    }

};

struct register_avx_3_6x2 {

    enum : int {
        BLOCK_M = 6,
        BLOCK_N = 8 * 2,
        BLOCK_K = 32,
    };

    // C += A * B for 6x16 matrix
    static NOINLINE void matmul_register(
        int M, int N, int K, float *A, int lda, float *B, int ldb,
        float *C, int ldc)
    {
        assert(M % BLOCK_M == 0);
        assert(N % BLOCK_N == 0);

        //       b0    b1
        // a0 vab00 vab01
        // a1 vab02 vab03
        // a2 vab04 vab05
        // a3 vab06 vab07
        // a4 vab08 vab09
        // a5 vab10 vab11
        auto vab00 = _mm256_setzero_ps();
        auto vab01 = _mm256_setzero_ps();
        auto vab02 = _mm256_setzero_ps();
        auto vab03 = _mm256_setzero_ps();
        auto vab04 = _mm256_setzero_ps();
        auto vab05 = _mm256_setzero_ps();
        auto vab06 = _mm256_setzero_ps();
        auto vab07 = _mm256_setzero_ps();
        auto vab08 = _mm256_setzero_ps();
        auto vab09 = _mm256_setzero_ps();
        auto vab10 = _mm256_setzero_ps();
        auto vab11 = _mm256_setzero_ps();

#define N_UNROLLS  1    

        for (int k = 0; k < K; k += N_UNROLLS) {
#define MM(i) \
            auto pa##i = &A[lda * 0 + (k + i)]; \
            auto pb##i = &B[ldb * (k + i) + 0]; \
            \
            auto vb0##i = _mm256_load_ps(pb##i + 8 * 0); \
            auto vb1##i = _mm256_load_ps(pb##i + 8 * 1); \
            \
            auto va0##i = _mm256_broadcast_ss(&pa##i[lda * 0]); \
            auto va1##i = _mm256_broadcast_ss(&pa##i[lda * 1]); \
            vab00 = _mm256_fmadd_ps(va0##i, vb0##i, vab00); \
            vab01 = _mm256_fmadd_ps(va0##i, vb1##i, vab01); \
            vab02 = _mm256_fmadd_ps(va1##i, vb0##i, vab02); \
            vab03 = _mm256_fmadd_ps(va1##i, vb1##i, vab03); \
            \
            auto va2##i = _mm256_broadcast_ss(&pa##i[lda * 2]); \
            auto va3##i = _mm256_broadcast_ss(&pa##i[lda * 3]); \
            vab04 = _mm256_fmadd_ps(va2##i, vb0##i, vab04); \
            vab05 = _mm256_fmadd_ps(va2##i, vb1##i, vab05); \
            vab06 = _mm256_fmadd_ps(va3##i, vb0##i, vab06); \
            vab07 = _mm256_fmadd_ps(va3##i, vb1##i, vab07); \
            \
            auto va4##i = _mm256_broadcast_ss(&pa##i[lda * 4]); \
            auto va5##i = _mm256_broadcast_ss(&pa##i[lda * 5]); \
            vab08 = _mm256_fmadd_ps(va4##i, vb0##i, vab08); \
            vab09 = _mm256_fmadd_ps(va4##i, vb1##i, vab09); \
            vab10 = _mm256_fmadd_ps(va5##i, vb0##i, vab10); \
            vab11 = _mm256_fmadd_ps(va5##i, vb1##i, vab11)

            MM(0);
#if N_UNROLLS >= 2
            MM(1);
#if N_UNROLLS >= 4
            MM(2);
            MM(3);
#if N_UNROLLS >= 8
            MM(4);
            MM(5);
            MM(6);
            MM(7);
#endif
#endif
#endif
#undef MM
#undef N_UNROLLS
        }

        auto vc00 = _mm256_load_ps(C + ldc * 0 + 8 * 0);
        auto vc01 = _mm256_load_ps(C + ldc * 0 + 8 * 1);
        auto vc02 = _mm256_load_ps(C + ldc * 1 + 8 * 0);
        auto vc03 = _mm256_load_ps(C + ldc * 1 + 8 * 1);
        auto vc04 = _mm256_load_ps(C + ldc * 2 + 8 * 0);
        auto vc05 = _mm256_load_ps(C + ldc * 2 + 8 * 1);
        auto vc06 = _mm256_load_ps(C + ldc * 3 + 8 * 0);
        auto vc07 = _mm256_load_ps(C + ldc * 3 + 8 * 1);
        auto vc08 = _mm256_load_ps(C + ldc * 4 + 8 * 0);
        auto vc09 = _mm256_load_ps(C + ldc * 4 + 8 * 1);
        auto vc10 = _mm256_load_ps(C + ldc * 5 + 8 * 0);
        auto vc11 = _mm256_load_ps(C + ldc * 5 + 8 * 1);

        vc00 = _mm256_add_ps(vc00, vab00);
        vc01 = _mm256_add_ps(vc01, vab01);
        vc02 = _mm256_add_ps(vc02, vab02);
        vc03 = _mm256_add_ps(vc03, vab03);
        vc04 = _mm256_add_ps(vc04, vab04);
        vc05 = _mm256_add_ps(vc05, vab05);
        vc06 = _mm256_add_ps(vc06, vab06);
        vc07 = _mm256_add_ps(vc07, vab07);
        vc08 = _mm256_add_ps(vc08, vab08);
        vc09 = _mm256_add_ps(vc09, vab09);
        vc10 = _mm256_add_ps(vc10, vab10);
        vc11 = _mm256_add_ps(vc11, vab11);

        _mm256_store_ps(C + ldc * 0 + 8 * 0, vc00);
        _mm256_store_ps(C + ldc * 0 + 8 * 1, vc01);
        _mm256_store_ps(C + ldc * 1 + 8 * 0, vc02);
        _mm256_store_ps(C + ldc * 1 + 8 * 1, vc03);
        _mm256_store_ps(C + ldc * 2 + 8 * 0, vc04);
        _mm256_store_ps(C + ldc * 2 + 8 * 1, vc05);
        _mm256_store_ps(C + ldc * 3 + 8 * 0, vc06);
        _mm256_store_ps(C + ldc * 3 + 8 * 1, vc07);
        _mm256_store_ps(C + ldc * 4 + 8 * 0, vc08);
        _mm256_store_ps(C + ldc * 4 + 8 * 1, vc09);
        _mm256_store_ps(C + ldc * 5 + 8 * 0, vc10);
        _mm256_store_ps(C + ldc * 5 + 8 * 1, vc11);
    }

    // C += packed(A) * packed(B) for 6x16 matrix
    static NOINLINE void matmul_register_packed(
        int M, int N, int K, float *A, int lda, float *B, int ldb,
        float *C, int ldc)
    {
        assert(M % BLOCK_M == 0);
        assert(N % BLOCK_N == 0);
        assert(lda == BLOCK_M);
        assert(ldb == BLOCK_N);

        lda = BLOCK_M;
        ldb = BLOCK_N;

        //       b0    b1
        // a0 vab00 vab01
        // a1 vab02 vab03
        // a2 vab04 vab05
        // a3 vab06 vab07
        // a4 vab08 vab09
        // a5 vab10 vab11
        auto vab00 = _mm256_setzero_ps();
        auto vab01 = _mm256_setzero_ps();
        auto vab02 = _mm256_setzero_ps();
        auto vab03 = _mm256_setzero_ps();
        auto vab04 = _mm256_setzero_ps();
        auto vab05 = _mm256_setzero_ps();
        auto vab06 = _mm256_setzero_ps();
        auto vab07 = _mm256_setzero_ps();
        auto vab08 = _mm256_setzero_ps();
        auto vab09 = _mm256_setzero_ps();
        auto vab10 = _mm256_setzero_ps();
        auto vab11 = _mm256_setzero_ps();

#define N_UNROLLS  1

        for (int k = 0; k < K; k += N_UNROLLS) {

            int i = 0;
            auto pa = &A[lda * (k + i) + 0];
            auto pb = &B[ldb * (k + i) + 0];
           
            auto vb0 = _mm256_load_ps(pb + 8 * 0);
            auto vb1 = _mm256_load_ps(pb + 8 * 1);
           
            auto va0 = _mm256_broadcast_ss(&pa[8 * i + 0]);
            auto va1 = _mm256_broadcast_ss(&pa[8 * i + 1]);
            vab00 = _mm256_fmadd_ps(va0, vb0, vab00);
            vab01 = _mm256_fmadd_ps(va0, vb1, vab01);
            vab02 = _mm256_fmadd_ps(va1, vb0, vab02);
            vab03 = _mm256_fmadd_ps(va1, vb1, vab03);
           
            auto va2 = _mm256_broadcast_ss(&pa[8 * i + 2]);
            auto va3 = _mm256_broadcast_ss(&pa[8 * i + 3]);
            vab04 = _mm256_fmadd_ps(va2, vb0, vab04);
            vab05 = _mm256_fmadd_ps(va2, vb1, vab05);
            vab06 = _mm256_fmadd_ps(va3, vb0, vab06);
            vab07 = _mm256_fmadd_ps(va3, vb1, vab07);
           
            auto va4 = _mm256_broadcast_ss(&pa[8 * i + 4]);
            auto va5 = _mm256_broadcast_ss(&pa[8 * i + 5]);
            vab08 = _mm256_fmadd_ps(va4, vb0, vab08);
            vab09 = _mm256_fmadd_ps(va4, vb1, vab09);
            vab10 = _mm256_fmadd_ps(va5, vb0, vab10);
            vab11 = _mm256_fmadd_ps(va5, vb1, vab11);

#define MM(i) \
            auto pa##i = &A[lda * (k + i) + 0]; \
            auto pb##i = &B[ldb * (k + i) + 0]; \
            \
            auto vb0##i = _mm256_load_ps(pb##i + 8 * 0); \
            auto vb1##i = _mm256_load_ps(pb##i + 8 * 1); \
            \
            auto va0##i = _mm256_broadcast_ss(&pa##i[8 * i + 0]); \
            auto va1##i = _mm256_broadcast_ss(&pa##i[8 * i + 1]); \
            vab00 = _mm256_fmadd_ps(va0##i, vb0##i, vab00); \
            vab01 = _mm256_fmadd_ps(va0##i, vb1##i, vab01); \
            vab02 = _mm256_fmadd_ps(va1##i, vb0##i, vab02); \
            vab03 = _mm256_fmadd_ps(va1##i, vb1##i, vab03); \
            \
            auto va2##i = _mm256_broadcast_ss(&pa##i[8 * i + 2]); \
            auto va3##i = _mm256_broadcast_ss(&pa##i[8 * i + 3]); \
            vab04 = _mm256_fmadd_ps(va2##i, vb0##i, vab04); \
            vab05 = _mm256_fmadd_ps(va2##i, vb1##i, vab05); \
            vab06 = _mm256_fmadd_ps(va3##i, vb0##i, vab06); \
            vab07 = _mm256_fmadd_ps(va3##i, vb1##i, vab07); \
            \
            auto va4##i = _mm256_broadcast_ss(&pa##i[8 * i + 4]); \
            auto va5##i = _mm256_broadcast_ss(&pa##i[8 * i + 5]); \
            vab08 = _mm256_fmadd_ps(va4##i, vb0##i, vab08); \
            vab09 = _mm256_fmadd_ps(va4##i, vb1##i, vab09); \
            vab10 = _mm256_fmadd_ps(va5##i, vb0##i, vab10); \
            vab11 = _mm256_fmadd_ps(va5##i, vb1##i, vab11)

            //MM(0);
#if N_UNROLLS >= 2
            MM(1);
#if N_UNROLLS >= 4
            MM(2);
            MM(3);
#if N_UNROLLS >= 8
            MM(4);
            MM(5);
            MM(6);
            MM(7);
#endif
#endif
#endif
#undef MM
#undef N_UNROLLS
        }

        auto vc00 = _mm256_load_ps(C + ldc * 0 + 8 * 0);
        auto vc01 = _mm256_load_ps(C + ldc * 0 + 8 * 1);
        auto vc02 = _mm256_load_ps(C + ldc * 1 + 8 * 0);
        auto vc03 = _mm256_load_ps(C + ldc * 1 + 8 * 1);
        auto vc04 = _mm256_load_ps(C + ldc * 2 + 8 * 0);
        auto vc05 = _mm256_load_ps(C + ldc * 2 + 8 * 1);
        auto vc06 = _mm256_load_ps(C + ldc * 3 + 8 * 0);
        auto vc07 = _mm256_load_ps(C + ldc * 3 + 8 * 1);
        auto vc08 = _mm256_load_ps(C + ldc * 4 + 8 * 0);
        auto vc09 = _mm256_load_ps(C + ldc * 4 + 8 * 1);
        auto vc10 = _mm256_load_ps(C + ldc * 5 + 8 * 0);
        auto vc11 = _mm256_load_ps(C + ldc * 5 + 8 * 1);

        vc00 = _mm256_add_ps(vc00, vab00);
        vc01 = _mm256_add_ps(vc01, vab01);
        vc02 = _mm256_add_ps(vc02, vab02);
        vc03 = _mm256_add_ps(vc03, vab03);
        vc04 = _mm256_add_ps(vc04, vab04);
        vc05 = _mm256_add_ps(vc05, vab05);
        vc06 = _mm256_add_ps(vc06, vab06);
        vc07 = _mm256_add_ps(vc07, vab07);
        vc08 = _mm256_add_ps(vc08, vab08);
        vc09 = _mm256_add_ps(vc09, vab09);
        vc10 = _mm256_add_ps(vc10, vab10);
        vc11 = _mm256_add_ps(vc11, vab11);

        _mm256_store_ps(C + ldc * 0 + 8 * 0, vc00);
        _mm256_store_ps(C + ldc * 0 + 8 * 1, vc01);
        _mm256_store_ps(C + ldc * 1 + 8 * 0, vc02);
        _mm256_store_ps(C + ldc * 1 + 8 * 1, vc03);
        _mm256_store_ps(C + ldc * 2 + 8 * 0, vc04);
        _mm256_store_ps(C + ldc * 2 + 8 * 1, vc05);
        _mm256_store_ps(C + ldc * 3 + 8 * 0, vc06);
        _mm256_store_ps(C + ldc * 3 + 8 * 1, vc07);
        _mm256_store_ps(C + ldc * 4 + 8 * 0, vc08);
        _mm256_store_ps(C + ldc * 4 + 8 * 1, vc09);
        _mm256_store_ps(C + ldc * 5 + 8 * 0, vc10);
        _mm256_store_ps(C + ldc * 5 + 8 * 1, vc11);
    }

    template <bool PACKED = false>
    static void matmul(
        int M, int N, int K, float *A, int lda,
        float *B, int ldb, float *C, int ldc)
    {
        assert(M % BLOCK_M == 0);
        assert(N % BLOCK_N == 0);

        for (int i = 0; i < M; i += BLOCK_M) {
            for (int j = 0; j < N; j += BLOCK_N) {
                auto Ar = A + lda * i + 0;
                auto Br = B + ldb * 0 + j;
                auto Cr = C + ldc * i + j;
                if (PACKED)
                    matmul_register_packed(
                        BLOCK_M, BLOCK_N, K, Ar, lda, Br, ldb, Cr, ldc);
                else
                    matmul_register(
                        BLOCK_M, BLOCK_N, K, Ar, lda, Br, ldb, Cr, ldc);
            }
        }
    }

    template <bool PACKED = false>
    static void gemm(
        int M, int N, int K, float alpha, float *A, int lda,
        float *B, int ldb, float beta, float *C, int ldc)
    {
        scale_matrix(A, lda, M, K, alpha);
        scale_matrix(C, ldc, M, N, beta);

        matmul<PACKED>(M, N, K, A, lda, B, ldb, C, ldc);
    }
};

struct register_avx_3_4x3 {

    enum : int {
        BLOCK_M = 4,
        BLOCK_N = 8 * 3,
        BLOCK_K = 32,
    };

    // for 6x16 matrix
    static NOINLINE void matmul_register(
        int M, int N, int K, float *A, int lda, float *B, int ldb,
        float *C, int ldc)
    {
        assert(M % BLOCK_M == 0);
        assert(N % BLOCK_N == 0);

        //       b0    b1    b2
        // a0 vab00 vab01 vab02
        // a1 vab03 vab04 vab05
        // a2 vab06 vab07 vab08
        // a3 vab09 vab10 vab11
        auto vab00 = _mm256_setzero_ps();
        auto vab01 = _mm256_setzero_ps();
        auto vab02 = _mm256_setzero_ps();
        auto vab03 = _mm256_setzero_ps();
        auto vab04 = _mm256_setzero_ps();
        auto vab05 = _mm256_setzero_ps();
        auto vab06 = _mm256_setzero_ps();
        auto vab07 = _mm256_setzero_ps();
        auto vab08 = _mm256_setzero_ps();
        auto vab09 = _mm256_setzero_ps();
        auto vab10 = _mm256_setzero_ps();
        auto vab11 = _mm256_setzero_ps();
        
        for (int k = 0; k < K; k++) {
            auto pa = &A[lda * 0 + k];
            auto pb = &B[ldb * k + 0];

            auto vb0 = _mm256_load_ps(pb + 8 * 0);
            auto vb1 = _mm256_load_ps(pb + 8 * 1);
            auto vb2 = _mm256_load_ps(pb + 8 * 2);

            auto va0 = _mm256_broadcast_ss(&pa[lda * 0]);
            auto va1 = _mm256_broadcast_ss(&pa[lda * 1]);
            vab00 = _mm256_fmadd_ps(va0, vb0, vab00);
            vab01 = _mm256_fmadd_ps(va0, vb1, vab01);
            vab02 = _mm256_fmadd_ps(va0, vb2, vab02);
            vab03 = _mm256_fmadd_ps(va1, vb0, vab03);
            vab04 = _mm256_fmadd_ps(va1, vb1, vab04);
            vab05 = _mm256_fmadd_ps(va1, vb2, vab05);

            auto va2 = _mm256_broadcast_ss(&pa[lda * 2]);
            auto va3 = _mm256_broadcast_ss(&pa[lda * 3]);
            vab06 = _mm256_fmadd_ps(va2, vb0, vab06);
            vab07 = _mm256_fmadd_ps(va2, vb1, vab07);
            vab08 = _mm256_fmadd_ps(va2, vb2, vab08);
            vab09 = _mm256_fmadd_ps(va3, vb0, vab09);
            vab10 = _mm256_fmadd_ps(va3, vb1, vab10);
            vab11 = _mm256_fmadd_ps(va3, vb2, vab11);
        }

        auto vc00 = _mm256_load_ps(C + ldc * 0 + 8 * 0);
        auto vc01 = _mm256_load_ps(C + ldc * 0 + 8 * 1);
        auto vc02 = _mm256_load_ps(C + ldc * 0 + 8 * 2);
        auto vc03 = _mm256_load_ps(C + ldc * 1 + 8 * 0);
        auto vc04 = _mm256_load_ps(C + ldc * 1 + 8 * 1);
        auto vc05 = _mm256_load_ps(C + ldc * 1 + 8 * 2);
        auto vc06 = _mm256_load_ps(C + ldc * 2 + 8 * 0);
        auto vc07 = _mm256_load_ps(C + ldc * 2 + 8 * 1);
        auto vc08 = _mm256_load_ps(C + ldc * 2 + 8 * 2);
        auto vc09 = _mm256_load_ps(C + ldc * 3 + 8 * 0);
        auto vc10 = _mm256_load_ps(C + ldc * 3 + 8 * 1);
        auto vc11 = _mm256_load_ps(C + ldc * 3 + 8 * 2);

        vc00 = _mm256_add_ps(vc00, vab00);
        vc01 = _mm256_add_ps(vc01, vab01);
        vc02 = _mm256_add_ps(vc02, vab02);
        vc03 = _mm256_add_ps(vc03, vab03);
        vc04 = _mm256_add_ps(vc04, vab04);
        vc05 = _mm256_add_ps(vc05, vab05);
        vc06 = _mm256_add_ps(vc06, vab06);
        vc07 = _mm256_add_ps(vc07, vab07);
        vc08 = _mm256_add_ps(vc08, vab08);
        vc09 = _mm256_add_ps(vc09, vab09);
        vc10 = _mm256_add_ps(vc10, vab10);
        vc11 = _mm256_add_ps(vc11, vab11);

        _mm256_store_ps(C + ldc * 0 + 8 * 0, vc00);
        _mm256_store_ps(C + ldc * 0 + 8 * 1, vc01);
        _mm256_store_ps(C + ldc * 0 + 8 * 2, vc02);
        _mm256_store_ps(C + ldc * 1 + 8 * 0, vc03);
        _mm256_store_ps(C + ldc * 1 + 8 * 1, vc04);
        _mm256_store_ps(C + ldc * 1 + 8 * 2, vc05);
        _mm256_store_ps(C + ldc * 2 + 8 * 0, vc06);
        _mm256_store_ps(C + ldc * 2 + 8 * 1, vc07);
        _mm256_store_ps(C + ldc * 2 + 8 * 2, vc08);
        _mm256_store_ps(C + ldc * 3 + 8 * 0, vc09);
        _mm256_store_ps(C + ldc * 3 + 8 * 1, vc10);
        _mm256_store_ps(C + ldc * 3 + 8 * 2, vc11);
    }

    static void NOINLINE matmul_register_packed(
        int M, int N, int K, float *A, int lda, float *B, int ldb,
        float *C, int ldc)
    {
        assert(M % BLOCK_M == 0);
        assert(N % BLOCK_N == 0);
        assert(lda == BLOCK_M);
        assert(ldb == BLOCK_N);

        lda = BLOCK_M;
        ldb = BLOCK_N;

        //       b0    b1    b2
        // a0 vab00 vab01 vab02
        // a1 vab03 vab04 vab05
        // a2 vab06 vab07 vab08
        // a3 vab09 vab10 vab11
        auto vab00 = _mm256_setzero_ps();
        auto vab01 = _mm256_setzero_ps();
        auto vab02 = _mm256_setzero_ps();
        auto vab03 = _mm256_setzero_ps();
        auto vab04 = _mm256_setzero_ps();
        auto vab05 = _mm256_setzero_ps();
        auto vab06 = _mm256_setzero_ps();
        auto vab07 = _mm256_setzero_ps();
        auto vab08 = _mm256_setzero_ps();
        auto vab09 = _mm256_setzero_ps();
        auto vab10 = _mm256_setzero_ps();
        auto vab11 = _mm256_setzero_ps();

        for (int k = 0; k < K; k++) {
            auto pa = &A[lda * k + 0];
            auto pb = &B[ldb * k + 0];

            auto vb0 = _mm256_load_ps(pb + 8 * 0);
            auto vb1 = _mm256_load_ps(pb + 8 * 1);
            auto vb2 = _mm256_load_ps(pb + 8 * 2);

            auto va0 = _mm256_broadcast_ss(&pa[0]);
            auto va1 = _mm256_broadcast_ss(&pa[1]);
            vab00 = _mm256_fmadd_ps(va0, vb0, vab00);
            vab01 = _mm256_fmadd_ps(va0, vb1, vab01);
            vab02 = _mm256_fmadd_ps(va0, vb2, vab02);
            vab03 = _mm256_fmadd_ps(va1, vb0, vab03);
            vab04 = _mm256_fmadd_ps(va1, vb1, vab04);
            vab05 = _mm256_fmadd_ps(va1, vb2, vab05);

            auto va2 = _mm256_broadcast_ss(&pa[2]);
            auto va3 = _mm256_broadcast_ss(&pa[3]);
            vab06 = _mm256_fmadd_ps(va2, vb0, vab06);
            vab07 = _mm256_fmadd_ps(va2, vb1, vab07);
            vab08 = _mm256_fmadd_ps(va2, vb2, vab08);
            vab09 = _mm256_fmadd_ps(va3, vb0, vab09);
            vab10 = _mm256_fmadd_ps(va3, vb1, vab10);
            vab11 = _mm256_fmadd_ps(va3, vb2, vab11);
        }

        auto vc00 = _mm256_load_ps(C + ldc * 0 + 8 * 0);
        auto vc01 = _mm256_load_ps(C + ldc * 0 + 8 * 1);
        auto vc02 = _mm256_load_ps(C + ldc * 0 + 8 * 2);
        auto vc03 = _mm256_load_ps(C + ldc * 1 + 8 * 0);
        auto vc04 = _mm256_load_ps(C + ldc * 1 + 8 * 1);
        auto vc05 = _mm256_load_ps(C + ldc * 1 + 8 * 2);
        auto vc06 = _mm256_load_ps(C + ldc * 2 + 8 * 0);
        auto vc07 = _mm256_load_ps(C + ldc * 2 + 8 * 1);
        auto vc08 = _mm256_load_ps(C + ldc * 2 + 8 * 2);
        auto vc09 = _mm256_load_ps(C + ldc * 3 + 8 * 0);
        auto vc10 = _mm256_load_ps(C + ldc * 3 + 8 * 1);
        auto vc11 = _mm256_load_ps(C + ldc * 3 + 8 * 2);

        vc00 = _mm256_add_ps(vc00, vab00);
        vc01 = _mm256_add_ps(vc01, vab01);
        vc02 = _mm256_add_ps(vc02, vab02);
        vc03 = _mm256_add_ps(vc03, vab03);
        vc04 = _mm256_add_ps(vc04, vab04);
        vc05 = _mm256_add_ps(vc05, vab05);
        vc06 = _mm256_add_ps(vc06, vab06);
        vc07 = _mm256_add_ps(vc07, vab07);
        vc08 = _mm256_add_ps(vc08, vab08);
        vc09 = _mm256_add_ps(vc09, vab09);
        vc10 = _mm256_add_ps(vc10, vab10);
        vc11 = _mm256_add_ps(vc11, vab11);

        _mm256_store_ps(C + ldc * 0 + 8 * 0, vc00);
        _mm256_store_ps(C + ldc * 0 + 8 * 1, vc01);
        _mm256_store_ps(C + ldc * 0 + 8 * 2, vc02);
        _mm256_store_ps(C + ldc * 1 + 8 * 0, vc03);
        _mm256_store_ps(C + ldc * 1 + 8 * 1, vc04);
        _mm256_store_ps(C + ldc * 1 + 8 * 2, vc05);
        _mm256_store_ps(C + ldc * 2 + 8 * 0, vc06);
        _mm256_store_ps(C + ldc * 2 + 8 * 1, vc07);
        _mm256_store_ps(C + ldc * 2 + 8 * 2, vc08);
        _mm256_store_ps(C + ldc * 3 + 8 * 0, vc09);
        _mm256_store_ps(C + ldc * 3 + 8 * 1, vc10);
        _mm256_store_ps(C + ldc * 3 + 8 * 2, vc11);
    }

    template <bool PACKED = false>
    static void matmul(
        int M, int N, int K, float *A, int lda,
        float *B, int ldb, float *C, int ldc)
    {
        assert(M % BLOCK_M == 0);
        assert(N % BLOCK_N == 0);

        for (int i = 0; i < M; i += BLOCK_M) {
            for (int j = 0; j < N; j += BLOCK_N) {
                auto Ar = A + lda * i + 0;
                auto Br = B + ldb * 0 + j;
                auto Cr = C + ldc * i + j;
                if (PACKED)
                    matmul_register_packed(
                        BLOCK_M, BLOCK_N, K, Ar, lda, Br, ldb, Cr, ldc);
                else
                    matmul_register(
                        BLOCK_M, BLOCK_N, K, Ar, lda, Br, ldb, Cr, ldc);
            }
        }
    }

    template <int PACKED = false>
    static void gemm(
        int M, int N, int K, float alpha, float *A, int lda,
        float *B, int ldb, float beta, float *C, int ldc)
    {
        scale_matrix(A, lda, M, K, alpha);
        scale_matrix(C, ldc, M, N, beta);

        matmul<PACKED>(M, N, K, A, lda, B, ldb, C, ldc);
    }
};

struct register_avx_3_4x3asm {

    enum : int {
        BLOCK_M = 4,
        BLOCK_N = 8 * 3,
        BLOCK_K = 32,
    };

    // for 6x16 matrix
    static NOINLINE void matmul_register(
        int M, int N, int K, float *A, int lda, float *B, int ldb,
        float *C, int ldc)
    {
        assert(M % BLOCK_M == 0);
        assert(N % BLOCK_N == 0);

        register_avx_3_4x3::matmul_register(M, N, K, A, lda, B, ldb, C, ldc);
    }

    static void NOINLINE matmul_register_packed(
        int M, int N, int K, float *A, int lda, float *B, int ldb,
        float *C, int ldc)
    {
#if 0
        assert(M % BLOCK_M == 0);
        assert(N % BLOCK_N == 0);
        assert(lda == BLOCK_M);
        assert(ldb == BLOCK_N);
#endif

#ifdef _MSC_VER
        register_avx_3_4x3::matmul_register_packed(
            M, N, K, A, lda, B, ldb, C, ldc);
#else
        int64_t K64 = K;
        int64_t ldc64 = ldc;

        __asm__ __volatile__ (
            "                                                   \n\t"
            "vzeroall                                           \n\t"
            "                                                   \n\t"
            "mov                %0, %%rax                       \n\t" // A
            "mov                %1, %%rbx                       \n\t" // B
            "mov                %2, %%rcx                       \n\t" // C
            "mov                %3, %%rsi                       \n\t" // K
            "mov                %4, %%rdi                       \n\t" // ldc
            "                                                   \n\t"
            "lea                (,%%rsi,4), %%r8                \n\t"
            "lea                (%%rax,%%r8,4), %%r9            \n\t"
            "                                                   \n\t"
            ".KLOOP:                                            \n\t"
            "                                                   \n\t"
            "cmp                %%rax, %%r9                     \n\t"
            "je                 .KLOOP_EXIT                     \n\t"
            "                                                   \n\t"
            "vmovaps            0 * 32(%%rbx), %%ymm12          \n\t"
            "vmovaps            1 * 32(%%rbx), %%ymm13          \n\t"
            "vmovaps            2 * 32(%%rbx), %%ymm14          \n\t"
            "                                                   \n\t"
            "vbroadcastss       0 * 4(%%rax), %%ymm15           \n\t"
            "vfmadd231ps        %%ymm15, %%ymm12, %%ymm0        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm13, %%ymm1        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm14, %%ymm2        \n\t"
            "                                                   \n\t"
            "vbroadcastss       1 * 4(%%rax), %%ymm15           \n\t"
            "vfmadd231ps        %%ymm15, %%ymm12, %%ymm3        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm13, %%ymm4        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm14, %%ymm5        \n\t"
            "                                                   \n\t"
            "vbroadcastss       2 * 4(%%rax), %%ymm15           \n\t"
            "vfmadd231ps        %%ymm15, %%ymm12, %%ymm6        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm13, %%ymm7        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm14, %%ymm8        \n\t"
            "                                                   \n\t"
            "vbroadcastss       3 * 4(%%rax), %%ymm15           \n\t"
            "vfmadd231ps        %%ymm15, %%ymm12, %%ymm9        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm13, %%ymm10       \n\t"
            "vfmadd231ps        %%ymm15, %%ymm14, %%ymm11       \n\t"
            "                                                   \n\t"
            "vmovaps            3 * 32(%%rbx), %%ymm12          \n\t"
            "vmovaps            4 * 32(%%rbx), %%ymm13          \n\t"
            "vmovaps            5 * 32(%%rbx), %%ymm14          \n\t"
            "                                                   \n\t"
            "vbroadcastss       4 * 4(%%rax), %%ymm15           \n\t"
            "vfmadd231ps        %%ymm15, %%ymm12, %%ymm0        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm13, %%ymm1        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm14, %%ymm2        \n\t"
            "                                                   \n\t"
            "vbroadcastss       5 * 4(%%rax), %%ymm15           \n\t"
            "vfmadd231ps        %%ymm15, %%ymm12, %%ymm3        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm13, %%ymm4        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm14, %%ymm5        \n\t"
            "                                                   \n\t"
            "vbroadcastss       6 * 4(%%rax), %%ymm15           \n\t"
            "vfmadd231ps        %%ymm15, %%ymm12, %%ymm6        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm13, %%ymm7        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm14, %%ymm8        \n\t"
            "                                                   \n\t"
            "vbroadcastss       7 * 4(%%rax), %%ymm15           \n\t"
            "vfmadd231ps        %%ymm15, %%ymm12, %%ymm9        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm13, %%ymm10       \n\t"
            "vfmadd231ps        %%ymm15, %%ymm14, %%ymm11       \n\t"
            "                                                   \n\t"
            "vmovaps            6 * 32(%%rbx), %%ymm12          \n\t"
            "vmovaps            7 * 32(%%rbx), %%ymm13          \n\t"
            "vmovaps            8 * 32(%%rbx), %%ymm14          \n\t"
            "                                                   \n\t"
            "vbroadcastss       8 * 4(%%rax), %%ymm15           \n\t"
            "vfmadd231ps        %%ymm15, %%ymm12, %%ymm0        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm13, %%ymm1        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm14, %%ymm2        \n\t"
            "                                                   \n\t"
            "vbroadcastss       9 * 4(%%rax), %%ymm15           \n\t"
            "vfmadd231ps        %%ymm15, %%ymm12, %%ymm3        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm13, %%ymm4        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm14, %%ymm5        \n\t"
            "                                                   \n\t"
            "vbroadcastss       10 * 4(%%rax), %%ymm15          \n\t"
            "vfmadd231ps        %%ymm15, %%ymm12, %%ymm6        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm13, %%ymm7        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm14, %%ymm8        \n\t"
            "                                                   \n\t"
            "vbroadcastss       11 * 4(%%rax), %%ymm15          \n\t"
            "vfmadd231ps        %%ymm15, %%ymm12, %%ymm9        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm13, %%ymm10       \n\t"
            "vfmadd231ps        %%ymm15, %%ymm14, %%ymm11       \n\t"
            "                                                   \n\t"
            "vmovaps             9 * 32(%%rbx), %%ymm12         \n\t"
            "vmovaps            10 * 32(%%rbx), %%ymm13         \n\t"
            "vmovaps            11 * 32(%%rbx), %%ymm14         \n\t"
            "                                                   \n\t"
            "vbroadcastss       12 * 4(%%rax), %%ymm15          \n\t"
            "vfmadd231ps        %%ymm15, %%ymm12, %%ymm0        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm13, %%ymm1        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm14, %%ymm2        \n\t"
            "                                                   \n\t"
            "vbroadcastss       13 * 4(%%rax), %%ymm15          \n\t"
            "vfmadd231ps        %%ymm15, %%ymm12, %%ymm3        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm13, %%ymm4        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm14, %%ymm5        \n\t"
            "                                                   \n\t"
            "vbroadcastss       14 * 4(%%rax), %%ymm15          \n\t"
            "vfmadd231ps        %%ymm15, %%ymm12, %%ymm6        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm13, %%ymm7        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm14, %%ymm8        \n\t"
            "                                                   \n\t"
            "vbroadcastss       15 * 4(%%rax), %%ymm15          \n\t"
            "vfmadd231ps        %%ymm15, %%ymm12, %%ymm9        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm13, %%ymm10       \n\t"
            "vfmadd231ps        %%ymm15, %%ymm14, %%ymm11       \n\t"
            "                                                   \n\t"
            "add                $64, %%rax                      \n\t"
            "add                $384, %%rbx                     \n\t"
            "jmp                .KLOOP                          \n\t"
            "                                                   \n\t"
            ".KLOOP_EXIT:                                       \n\t"
            "                                                   \n\t"
            "lea                (,%%rdi,2), %%r10               \n\t"
            "lea                (%%r10,%%rdi), %%r11            \n\t"
            "vaddps              0(%%rcx), %%ymm0, %%ymm0       \n\t"
            "vaddps             32(%%rcx), %%ymm1, %%ymm1       \n\t"
            "vaddps             64(%%rcx), %%ymm2, %%ymm2       \n\t"
            "vaddps              0(%%rcx,%%rdi,4), %%ymm3, %%ymm3       \n\t"
            "vaddps             32(%%rcx,%%rdi,4), %%ymm4, %%ymm4       \n\t"
            "vaddps             64(%%rcx,%%rdi,4), %%ymm5, %%ymm5       \n\t"
            "vaddps              0(%%rcx,%%r10,4), %%ymm6, %%ymm6       \n\t"
            "vaddps             32(%%rcx,%%r10,4), %%ymm7, %%ymm7       \n\t"
            "vaddps             64(%%rcx,%%r10,4), %%ymm8, %%ymm8       \n\t"
            "vaddps              0(%%rcx,%%r11,4), %%ymm9, %%ymm9       \n\t"
            "vaddps             32(%%rcx,%%r11,4), %%ymm10, %%ymm10     \n\t"
            "vaddps             64(%%rcx,%%r11,4), %%ymm11, %%ymm11     \n\t"
            "                                                   \n\t"
            "vmovaps            %%ymm0,   0(%%rcx)              \n\t"
            "vmovaps            %%ymm1,  32(%%rcx)              \n\t"
            "vmovaps            %%ymm2,  64(%%rcx)              \n\t"
            "vmovaps            %%ymm3,   0(%%rcx,%%rdi,4)      \n\t"
            "vmovaps            %%ymm4,  32(%%rcx,%%rdi,4)      \n\t"
            "vmovaps            %%ymm5,  64(%%rcx,%%rdi,4)      \n\t"
            "vmovaps            %%ymm6,   0(%%rcx,%%r10,4)      \n\t"
            "vmovaps            %%ymm7,  32(%%rcx,%%r10,4)      \n\t"
            "vmovaps            %%ymm8,  64(%%rcx,%%r10,4)      \n\t"
            "vmovaps            %%ymm9,   0(%%rcx,%%r11,4)      \n\t"
            "vmovaps            %%ymm10, 32(%%rcx,%%r11,4)      \n\t"
            "vmovaps            %%ymm11, 64(%%rcx,%%r11,4)      \n\t"
            "                                                   \n\t"
            : // output operands
            : // input operands
              "m" (A)
            , "m" (B)
            , "m" (C)
            , "m" (K64)
            , "m" (ldc64)
            : // clobbered registers
              "rax", "rbx", "rcx", "rdx", "rsi", "rdi"
            , "r8", "r9", "r10", "r11" //, "r12", "r13", "r14", "r15"
            , "xmm0", "xmm1", "xmm2", "xmm3", "xmm4"
            , "xmm5", "xmm6", "xmm7", "xmm8", "xmm9"
            , "xmm10", "xmm11", "xmm12", "xmm13", "xmm14"
            , "memory"
        );
#endif
    }

    template <bool PACKED = false>
    static void matmul(
        int M, int N, int K, float *A, int lda,
        float *B, int ldb, float *C, int ldc)
    {
        assert(M % BLOCK_M == 0);
        assert(N % BLOCK_N == 0);

        for (int i = 0; i < M; i += BLOCK_M) {
            for (int j = 0; j < N; j += BLOCK_N) {
                auto Ar = A + lda * i + 0;
                auto Br = B + ldb * 0 + j;
                auto Cr = C + ldc * i + j;
                if (PACKED)
                    matmul_register_packed(
                        BLOCK_M, BLOCK_N, K, Ar, lda, Br, ldb, Cr, ldc);
                else
                    matmul_register(
                        BLOCK_M, BLOCK_N, K, Ar, lda, Br, ldb, Cr, ldc);
            }
        }
    }

    template <int PACKED = false>
    static void gemm(
        int M, int N, int K, float alpha, float *A, int lda,
        float *B, int ldb, float beta, float *C, int ldc)
    {
        scale_matrix(A, lda, M, K, alpha);
        scale_matrix(C, ldc, M, N, beta);

        matmul<PACKED>(M, N, K, A, lda, B, ldb, C, ldc);
    }
};

struct register_avx_3_4x3asmpf {

    enum : int {
        BLOCK_M = 4,
        BLOCK_N = 8 * 3,
        BLOCK_K = 32,
    };

    // for 6x16 matrix
    static NOINLINE void matmul_register(
        int M, int N, int K, float *A, int lda, float *B, int ldb,
        float *C, int ldc)
    {
        assert(M % BLOCK_M == 0);
        assert(N % BLOCK_N == 0);

        register_avx_3_4x3::matmul_register(M, N, K, A, lda, B, ldb, C, ldc);
    }

    static void NOINLINE matmul_register_packed(
        int M, int N, int K, float *A, int lda, float *B, int ldb,
        float *C, int ldc)
    {
#if 0
        assert(M % BLOCK_M == 0);
        assert(N % BLOCK_N == 0);
        assert(lda == BLOCK_M);
        assert(ldb == BLOCK_N);
#endif

#ifdef _MSC_VER
        register_avx_3_4x3::matmul_register_packed(
            M, N, K, A, lda, B, ldb, C, ldc);
#else
        int64_t K64 = K;
        int64_t ldc64 = ldc;

        __asm__ __volatile__ (
            "                                                   \n\t"
            "vzeroall                                           \n\t"
            "                                                   \n\t"
            "mov                %0, %%rax                       \n\t" // A
            "mov                %1, %%rbx                       \n\t" // B
            "mov                %2, %%rcx                       \n\t" // C
            "mov                %3, %%rsi                       \n\t" // K
            "mov                %4, %%rdi                       \n\t" // ldc
            "                                                   \n\t"
            "lea                (,%%rdi,2), %%r10               \n\t" // rdi*2
            "lea                (%%r10,%%rdi), %%r11            \n\t" // rdi*3
            "prefetcht0          0(%%rcx)                       \n\t"
            "prefetcht0         64(%%rcx)                       \n\t"
            "prefetcht0          0(%%rcx,%%rdi,4)               \n\t"
            "prefetcht0         64(%%rcx,%%rdi,4)               \n\t"
            "prefetcht0          0(%%rcx,%%r10,4)               \n\t"
            "prefetcht0         64(%%rcx,%%r10,4)               \n\t"
            "prefetcht0          0(%%rcx,%%r11,4)               \n\t"
            "prefetcht0         64(%%rcx,%%r11,4)               \n\t"
            "                                                   \n\t"
            "lea                (,%%rsi,4), %%r8                \n\t"
            "lea                (%%rax,%%r8,4), %%r9            \n\t"
            "                                                   \n\t"
            ".KLOOP_4x3asmpf:                                   \n\t"
            "                                                   \n\t"
            "cmp                %%rax, %%r9                     \n\t"
            "je                 .KLOOP_4x3asmpf_EXIT            \n\t"
            "                                                   \n\t"
            "vmovaps            0 * 32(%%rbx), %%ymm12          \n\t"
            "vmovaps            1 * 32(%%rbx), %%ymm13          \n\t"
            "vmovaps            2 * 32(%%rbx), %%ymm14          \n\t"
            "                                                   \n\t"
            "vbroadcastss       0 * 4(%%rax), %%ymm15           \n\t"
            "vfmadd231ps        %%ymm15, %%ymm12, %%ymm0        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm13, %%ymm1        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm14, %%ymm2        \n\t"
            "                                                   \n\t"
            "vbroadcastss       1 * 4(%%rax), %%ymm15           \n\t"
            "vfmadd231ps        %%ymm15, %%ymm12, %%ymm3        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm13, %%ymm4        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm14, %%ymm5        \n\t"
            "                                                   \n\t"
            "vbroadcastss       2 * 4(%%rax), %%ymm15           \n\t"
            "vfmadd231ps        %%ymm15, %%ymm12, %%ymm6        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm13, %%ymm7        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm14, %%ymm8        \n\t"
            "                                                   \n\t"
            "vbroadcastss       3 * 4(%%rax), %%ymm15           \n\t"
            "vfmadd231ps        %%ymm15, %%ymm12, %%ymm9        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm13, %%ymm10       \n\t"
            "vfmadd231ps        %%ymm15, %%ymm14, %%ymm11       \n\t"
            "                                                   \n\t"
            "vmovaps            3 * 32(%%rbx), %%ymm12          \n\t"
            "vmovaps            4 * 32(%%rbx), %%ymm13          \n\t"
            "vmovaps            5 * 32(%%rbx), %%ymm14          \n\t"
            "                                                   \n\t"
            "vbroadcastss       4 * 4(%%rax), %%ymm15           \n\t"
            "vfmadd231ps        %%ymm15, %%ymm12, %%ymm0        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm13, %%ymm1        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm14, %%ymm2        \n\t"
            "                                                   \n\t"
            "vbroadcastss       5 * 4(%%rax), %%ymm15           \n\t"
            "vfmadd231ps        %%ymm15, %%ymm12, %%ymm3        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm13, %%ymm4        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm14, %%ymm5        \n\t"
            "                                                   \n\t"
            "vbroadcastss       6 * 4(%%rax), %%ymm15           \n\t"
            "vfmadd231ps        %%ymm15, %%ymm12, %%ymm6        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm13, %%ymm7        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm14, %%ymm8        \n\t"
            "                                                   \n\t"
            "vbroadcastss       7 * 4(%%rax), %%ymm15           \n\t"
            "vfmadd231ps        %%ymm15, %%ymm12, %%ymm9        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm13, %%ymm10       \n\t"
            "vfmadd231ps        %%ymm15, %%ymm14, %%ymm11       \n\t"
            "                                                   \n\t"
            "vmovaps            6 * 32(%%rbx), %%ymm12          \n\t"
            "vmovaps            7 * 32(%%rbx), %%ymm13          \n\t"
            "vmovaps            8 * 32(%%rbx), %%ymm14          \n\t"
            "                                                   \n\t"
            "vbroadcastss       8 * 4(%%rax), %%ymm15           \n\t"
            "vfmadd231ps        %%ymm15, %%ymm12, %%ymm0        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm13, %%ymm1        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm14, %%ymm2        \n\t"
            "                                                   \n\t"
            "vbroadcastss       9 * 4(%%rax), %%ymm15           \n\t"
            "vfmadd231ps        %%ymm15, %%ymm12, %%ymm3        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm13, %%ymm4        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm14, %%ymm5        \n\t"
            "                                                   \n\t"
            "vbroadcastss       10 * 4(%%rax), %%ymm15          \n\t"
            "vfmadd231ps        %%ymm15, %%ymm12, %%ymm6        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm13, %%ymm7        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm14, %%ymm8        \n\t"
            "                                                   \n\t"
            "vbroadcastss       11 * 4(%%rax), %%ymm15          \n\t"
            "vfmadd231ps        %%ymm15, %%ymm12, %%ymm9        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm13, %%ymm10       \n\t"
            "vfmadd231ps        %%ymm15, %%ymm14, %%ymm11       \n\t"
            "                                                   \n\t"
            "vmovaps             9 * 32(%%rbx), %%ymm12         \n\t"
            "vmovaps            10 * 32(%%rbx), %%ymm13         \n\t"
            "vmovaps            11 * 32(%%rbx), %%ymm14         \n\t"
            "                                                   \n\t"
            "vbroadcastss       12 * 4(%%rax), %%ymm15          \n\t"
            "vfmadd231ps        %%ymm15, %%ymm12, %%ymm0        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm13, %%ymm1        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm14, %%ymm2        \n\t"
            "                                                   \n\t"
            "vbroadcastss       13 * 4(%%rax), %%ymm15          \n\t"
            "vfmadd231ps        %%ymm15, %%ymm12, %%ymm3        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm13, %%ymm4        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm14, %%ymm5        \n\t"
            "                                                   \n\t"
            "vbroadcastss       14 * 4(%%rax), %%ymm15          \n\t"
            "vfmadd231ps        %%ymm15, %%ymm12, %%ymm6        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm13, %%ymm7        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm14, %%ymm8        \n\t"
            "                                                   \n\t"
            "vbroadcastss       15 * 4(%%rax), %%ymm15          \n\t"
            "vfmadd231ps        %%ymm15, %%ymm12, %%ymm9        \n\t"
            "vfmadd231ps        %%ymm15, %%ymm13, %%ymm10       \n\t"
            "vfmadd231ps        %%ymm15, %%ymm14, %%ymm11       \n\t"
            "                                                   \n\t"
            "add                $64, %%rax                      \n\t"
            "add                $384, %%rbx                     \n\t"
            "jmp                .KLOOP_4x3asmpf                 \n\t"
            "                                                   \n\t"
            ".KLOOP_4x3asmpf_EXIT:                              \n\t"
            "                                                   \n\t"
            "vaddps              0(%%rcx), %%ymm0, %%ymm0       \n\t"
            "vaddps             32(%%rcx), %%ymm1, %%ymm1       \n\t"
            "vaddps             64(%%rcx), %%ymm2, %%ymm2       \n\t"
            "vaddps              0(%%rcx,%%rdi,4), %%ymm3, %%ymm3       \n\t"
            "vaddps             32(%%rcx,%%rdi,4), %%ymm4, %%ymm4       \n\t"
            "vaddps             64(%%rcx,%%rdi,4), %%ymm5, %%ymm5       \n\t"
            "vaddps              0(%%rcx,%%r10,4), %%ymm6, %%ymm6       \n\t"
            "vaddps             32(%%rcx,%%r10,4), %%ymm7, %%ymm7       \n\t"
            "vaddps             64(%%rcx,%%r10,4), %%ymm8, %%ymm8       \n\t"
            "vaddps              0(%%rcx,%%r11,4), %%ymm9, %%ymm9       \n\t"
            "vaddps             32(%%rcx,%%r11,4), %%ymm10, %%ymm10     \n\t"
            "vaddps             64(%%rcx,%%r11,4), %%ymm11, %%ymm11     \n\t"
            "                                                   \n\t"
            "vmovaps            %%ymm0,   0(%%rcx)              \n\t"
            "vmovaps            %%ymm1,  32(%%rcx)              \n\t"
            "vmovaps            %%ymm2,  64(%%rcx)              \n\t"
            "vmovaps            %%ymm3,   0(%%rcx,%%rdi,4)      \n\t"
            "vmovaps            %%ymm4,  32(%%rcx,%%rdi,4)      \n\t"
            "vmovaps            %%ymm5,  64(%%rcx,%%rdi,4)      \n\t"
            "vmovaps            %%ymm6,   0(%%rcx,%%r10,4)      \n\t"
            "vmovaps            %%ymm7,  32(%%rcx,%%r10,4)      \n\t"
            "vmovaps            %%ymm8,  64(%%rcx,%%r10,4)      \n\t"
            "vmovaps            %%ymm9,   0(%%rcx,%%r11,4)      \n\t"
            "vmovaps            %%ymm10, 32(%%rcx,%%r11,4)      \n\t"
            "vmovaps            %%ymm11, 64(%%rcx,%%r11,4)      \n\t"
            "                                                   \n\t"
            : // output operands
            : // input operands
              "m" (A)
            , "m" (B)
            , "m" (C)
            , "m" (K64)
            , "m" (ldc64)
            : // clobbered registers
              "rax", "rbx", "rcx", "rdx", "rsi", "rdi"
            , "r8", "r9", "r10", "r11" //, "r12", "r13", "r14", "r15"
            , "xmm0", "xmm1", "xmm2", "xmm3", "xmm4"
            , "xmm5", "xmm6", "xmm7", "xmm8", "xmm9"
            , "xmm10", "xmm11", "xmm12", "xmm13", "xmm14"
            , "memory"
        );
#endif
    }

    template <bool PACKED = false>
    static void matmul(
        int M, int N, int K, float *A, int lda,
        float *B, int ldb, float *C, int ldc)
    {
        assert(M % BLOCK_M == 0);
        assert(N % BLOCK_N == 0);

        for (int i = 0; i < M; i += BLOCK_M) {
            for (int j = 0; j < N; j += BLOCK_N) {
                auto Ar = A + lda * i + 0;
                auto Br = B + ldb * 0 + j;
                auto Cr = C + ldc * i + j;
                if (PACKED)
                    matmul_register_packed(
                        BLOCK_M, BLOCK_N, K, Ar, lda, Br, ldb, Cr, ldc);
                else
                    matmul_register(
                        BLOCK_M, BLOCK_N, K, Ar, lda, Br, ldb, Cr, ldc);
            }
        }
    }

    template <int PACKED = false>
    static void gemm(
        int M, int N, int K, float alpha, float *A, int lda,
        float *B, int ldb, float beta, float *C, int ldc)
    {
        scale_matrix(A, lda, M, K, alpha);
        scale_matrix(C, ldc, M, N, beta);

        matmul<PACKED>(M, N, K, A, lda, B, ldb, C, ldc);
    }
};

#include "01register512.h"

struct register_bench {

    template <class T>
    static void performL1(bench_params<T>& bp_)
    {
        int n_times = 1000 * 1000;

        struct matmul_buffer {
            float A[256 * 256];  // 4 * 128 * 128 = 16KB
            char padding0[LINE_SIZE * 1];
            float B[256 * 256];
            char padding1[LINE_SIZE * 2];
            float C[256 * 256];
        };
        auto buf_ = make_aligned_ptr<matmul_buffer>(PAGE_SIZE);
        auto& buf = *buf_.get();
        memset(&buf, 0, sizeof(matmul_buffer));

        auto make_bp = [&](int M, int N, int K) {
            bench_params<T> bp = bp_;
            bp.A = buf.A;
            bp.B = buf.B;
            bp.C = buf.C;
            bp.M = M;
            bp.N = bp.ldb = bp.ldc = N;
            bp.K = bp.lda = K;
            return bp;
        };

#if 0
        {
            auto bp = make_bp(6, 32, 256);
            benchmark("register_avx_1", bp, register_avx_1::gemm, false, true, n_times);
            benchmark("register_avx_2", bp, register_avx_2::gemm, false, true, n_times);
        }
#endif
        {
            auto bp = make_bp(6, 16, 256);
            benchmark("register_avx_3_6x2", bp, register_avx_3_6x2::gemm<false>, false, true, n_times);

            pack2d<T, 6, 1>(bp.A, bp.lda, bp.M, bp.K, bp.C, 1);
            std::copy_n(bp.C, bp.M * bp.K, bp.A);
            pack2d<T, 1, 16>(bp.B, bp.ldb, bp.K, bp.N, bp.C, 1);
            std::copy_n(bp.C, bp.K * bp.N, bp.B);
            std::fill_n(bp.C, bp.M * bp.ldc, T(0.0));
            bp.lda = register_avx_3_6x2::BLOCK_M;
            bp.ldb = register_avx_3_6x2::BLOCK_N;

            benchmark("register_avx_3_6x2p", bp, register_avx_3_6x2::gemm<true>, false, true, n_times);
        }

#if 1
        {
            auto bp = make_bp(4, 24, 256);
            benchmark("register_avx_3_4x3", bp, register_avx_3_4x3::gemm<false>, false, true, n_times);

            pack2d<T, 4, 1>(bp.A, bp.lda, bp.M, bp.K, bp.C, 1);
            std::copy_n(bp.C, bp.M * bp.K, bp.A);
            pack2d<T, 1, 24>(bp.B, bp.ldb, bp.K, bp.N, bp.C, 1);
            std::copy_n(bp.C, bp.K * bp.N, bp.B);
            std::fill_n(bp.C, bp.M * bp.ldc, T(0.0));
            bp.lda = register_avx_3_4x3::BLOCK_M;
            bp.ldb = register_avx_3_4x3::BLOCK_N;

            benchmark("register_avx_3_4x3p", bp, register_avx_3_4x3::gemm<true>, false, true, n_times);
        }
#endif


#ifdef USE_AVX512
        {
            auto bp = make_bp(9, 48, 32);
            benchmark("register_avx512_9x3", bp, register_avx512_9x3::gemm<false>, false, true, n_times);
        }
#endif
    }

};

#endif
