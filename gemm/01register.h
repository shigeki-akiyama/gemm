#pragma once

#include "util.h"

#include <cassert>
#undef NODEBUG

#ifdef USE_AVX
#include <immintrin.h>

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
                        float *vaii = pa + lda * ii; // (float *)&va[ii];
                        for (int kk = 0; kk < 8; kk++) {
                            //auto v = _mm256_set1_ps(vaii[kk]);
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
    };

    // C += A * B for 6x16 matrix
    static void matmul_register(
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

        for (int k = 0; k < K; k++) {
            auto pa = &A[lda * 0 + k];
            auto pb = &B[ldb * k + 0];

            auto vb0 = _mm256_load_ps(pb + 8 * 0);
            auto vb1 = _mm256_load_ps(pb + 8 * 1);

            auto va0 = _mm256_broadcast_ss(&pa[lda * 0]);
            auto va1 = _mm256_broadcast_ss(&pa[lda * 1]);
            vab00 = _mm256_fmadd_ps(va0, vb0, vab00);
            vab01 = _mm256_fmadd_ps(va0, vb1, vab01);
            vab02 = _mm256_fmadd_ps(va1, vb0, vab02);
            vab03 = _mm256_fmadd_ps(va1, vb1, vab03);

            auto va2 = _mm256_broadcast_ss(&pa[lda * 2]);
            auto va3 = _mm256_broadcast_ss(&pa[lda * 3]);
            vab04 = _mm256_fmadd_ps(va2, vb0, vab04);
            vab05 = _mm256_fmadd_ps(va2, vb1, vab05);
            vab06 = _mm256_fmadd_ps(va3, vb0, vab06);
            vab07 = _mm256_fmadd_ps(va3, vb1, vab07);

            auto va4 = _mm256_broadcast_ss(&pa[lda * 4]);
            auto va5 = _mm256_broadcast_ss(&pa[lda * 5]);
            vab08 = _mm256_fmadd_ps(va4, vb0, vab08);
            vab09 = _mm256_fmadd_ps(va4, vb1, vab09);
            vab10 = _mm256_fmadd_ps(va5, vb0, vab10);
            vab11 = _mm256_fmadd_ps(va5, vb1, vab11);
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
                matmul_register(
                    BLOCK_M, BLOCK_N, K, Ar, lda, Br, ldb, Cr, ldc);
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

struct register_avx_3_4x3 {

    enum : int {
        BLOCK_M = 4,
        BLOCK_N = 8 * 3,
    };

    // for 6x16 matrix
    static void matmul_register(
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
                matmul_register(BLOCK_M, BLOCK_N, K, Ar, lda, Br, ldb, Cr, ldc);
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

#endif
