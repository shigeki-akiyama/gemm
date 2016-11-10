#pragma once

#include "gemm_util.h"
#include <cassert>
#include <immintrin.h>

#ifdef USE_AVX512

#define _mm512_broadcast_ss(p) \
    _mm512_broadcast_f32x4(_mm_broadcast_ss(p))

struct register_avx512_9x3 {

    enum : int {
        BLOCK_M = 9,
        BLOCK_N = 16 * 3,
    };

    // 9x48 matrix multiplication
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
        // a4 vab12 vab13 vab14
        // a5 vab15 vab16 vab17
        // a6 vab18 vab19 vab20
        // a7 vab21 vab22 vab23
        // a8 vab24 vab25 vab26
        
        auto vab00 = _mm512_setzero_ps(); auto vab01 = _mm512_setzero_ps();
        auto vab02 = _mm512_setzero_ps(); auto vab03 = _mm512_setzero_ps();
        auto vab04 = _mm512_setzero_ps(); auto vab05 = _mm512_setzero_ps();
        auto vab06 = _mm512_setzero_ps(); auto vab07 = _mm512_setzero_ps();
        auto vab08 = _mm512_setzero_ps(); auto vab09 = _mm512_setzero_ps();
        auto vab10 = _mm512_setzero_ps(); auto vab11 = _mm512_setzero_ps();
        auto vab12 = _mm512_setzero_ps(); auto vab13 = _mm512_setzero_ps();
        auto vab14 = _mm512_setzero_ps(); auto vab15 = _mm512_setzero_ps();
        auto vab16 = _mm512_setzero_ps(); auto vab17 = _mm512_setzero_ps();
        auto vab18 = _mm512_setzero_ps(); auto vab19 = _mm512_setzero_ps();
        auto vab20 = _mm512_setzero_ps(); auto vab21 = _mm512_setzero_ps();
        auto vab22 = _mm512_setzero_ps(); auto vab23 = _mm512_setzero_ps();
        auto vab24 = _mm512_setzero_ps(); auto vab25 = _mm512_setzero_ps();
        auto vab26 = _mm512_setzero_ps();

        for (int k = 0; k < K; k++) {
            auto pa = &A[lda * 0 + k];
            auto pb = &B[ldb * k + 0];

            auto vb0 = _mm512_load_ps(pb + 16 * 0);
            auto vb1 = _mm512_load_ps(pb + 16 * 1);
            auto vb2 = _mm512_load_ps(pb + 16 * 2);

            auto va0 = _mm512_broadcast_ss(&pa[lda * 0]);
            auto va1 = _mm512_broadcast_ss(&pa[lda * 1]);
            vab00 = _mm512_fmadd_ps(va0, vb0, vab00);
            vab01 = _mm512_fmadd_ps(va0, vb1, vab01);
            vab02 = _mm512_fmadd_ps(va0, vb2, vab02);
            vab03 = _mm512_fmadd_ps(va1, vb0, vab03);
            vab04 = _mm512_fmadd_ps(va1, vb1, vab04);
            vab05 = _mm512_fmadd_ps(va1, vb2, vab05);

            auto va2 = _mm512_broadcast_ss(&pa[lda * 2]);
            auto va3 = _mm512_broadcast_ss(&pa[lda * 3]);
            vab06 = _mm512_fmadd_ps(va2, vb0, vab06);
            vab07 = _mm512_fmadd_ps(va2, vb1, vab07);
            vab08 = _mm512_fmadd_ps(va2, vb2, vab08);
            vab09 = _mm512_fmadd_ps(va3, vb0, vab09);
            vab10 = _mm512_fmadd_ps(va3, vb1, vab10);
            vab11 = _mm512_fmadd_ps(va3, vb2, vab11);

            auto va4 = _mm512_broadcast_ss(&pa[lda * 4]);
            auto va5 = _mm512_broadcast_ss(&pa[lda * 5]);
            vab12 = _mm512_fmadd_ps(va4, vb0, vab12);
            vab13 = _mm512_fmadd_ps(va4, vb1, vab13);
            vab14 = _mm512_fmadd_ps(va4, vb2, vab14);
            vab15 = _mm512_fmadd_ps(va5, vb0, vab15);
            vab16 = _mm512_fmadd_ps(va5, vb1, vab16);
            vab17 = _mm512_fmadd_ps(va5, vb2, vab17);

            auto va6 = _mm512_broadcast_ss(&pa[lda * 6]);
            auto va7 = _mm512_broadcast_ss(&pa[lda * 7]);
            vab18 = _mm512_fmadd_ps(va6, vb0, vab18);
            vab19 = _mm512_fmadd_ps(va6, vb1, vab19);
            vab20 = _mm512_fmadd_ps(va6, vb2, vab20);
            vab21 = _mm512_fmadd_ps(va7, vb0, vab21);
            vab22 = _mm512_fmadd_ps(va7, vb1, vab22);
            vab23 = _mm512_fmadd_ps(va7, vb2, vab23);

            auto va8 = _mm512_broadcast_ss(&pa[lda * 8]);
            vab24 = _mm512_fmadd_ps(va8, vb0, vab24);
            vab25 = _mm512_fmadd_ps(va8, vb1, vab25);
            vab26 = _mm512_fmadd_ps(va8, vb2, vab26);
        }

        auto vc00 = _mm512_load_ps(C + ldc * 0 + 16 * 0);
        auto vc01 = _mm512_load_ps(C + ldc * 0 + 16 * 1);
        auto vc02 = _mm512_load_ps(C + ldc * 0 + 16 * 2);
        auto vc03 = _mm512_load_ps(C + ldc * 1 + 16 * 0);
        auto vc04 = _mm512_load_ps(C + ldc * 1 + 16 * 1);
        auto vc05 = _mm512_load_ps(C + ldc * 1 + 16 * 2);
        auto vc06 = _mm512_load_ps(C + ldc * 2 + 16 * 0);
        auto vc07 = _mm512_load_ps(C + ldc * 2 + 16 * 1);
        auto vc08 = _mm512_load_ps(C + ldc * 2 + 16 * 2);
        auto vc09 = _mm512_load_ps(C + ldc * 3 + 16 * 0);
        auto vc10 = _mm512_load_ps(C + ldc * 3 + 16 * 1);
        auto vc11 = _mm512_load_ps(C + ldc * 3 + 16 * 2);
        auto vc12 = _mm512_load_ps(C + ldc * 4 + 16 * 0);
        auto vc13 = _mm512_load_ps(C + ldc * 4 + 16 * 1);
        auto vc14 = _mm512_load_ps(C + ldc * 4 + 16 * 2);
        auto vc15 = _mm512_load_ps(C + ldc * 5 + 16 * 0);
        auto vc16 = _mm512_load_ps(C + ldc * 5 + 16 * 1);
        auto vc17 = _mm512_load_ps(C + ldc * 5 + 16 * 2);
        auto vc18 = _mm512_load_ps(C + ldc * 6 + 16 * 0);
        auto vc19 = _mm512_load_ps(C + ldc * 6 + 16 * 1);
        auto vc20 = _mm512_load_ps(C + ldc * 6 + 16 * 2);
        auto vc21 = _mm512_load_ps(C + ldc * 7 + 16 * 0);
        auto vc22 = _mm512_load_ps(C + ldc * 7 + 16 * 1);
        auto vc23 = _mm512_load_ps(C + ldc * 7 + 16 * 2);
        auto vc24 = _mm512_load_ps(C + ldc * 8 + 16 * 0);
        auto vc25 = _mm512_load_ps(C + ldc * 8 + 16 * 1);
        auto vc26 = _mm512_load_ps(C + ldc * 8 + 16 * 2);

        vc00 = _mm512_add_ps(vc00, vab00);
        vc01 = _mm512_add_ps(vc01, vab01);
        vc02 = _mm512_add_ps(vc02, vab02);
        vc03 = _mm512_add_ps(vc03, vab03);
        vc04 = _mm512_add_ps(vc04, vab04);
        vc05 = _mm512_add_ps(vc05, vab05);
        vc06 = _mm512_add_ps(vc06, vab06);
        vc07 = _mm512_add_ps(vc07, vab07);
        vc08 = _mm512_add_ps(vc08, vab08);
        vc09 = _mm512_add_ps(vc09, vab09);
        vc10 = _mm512_add_ps(vc10, vab10);
        vc11 = _mm512_add_ps(vc11, vab11);
        vc12 = _mm512_add_ps(vc12, vab12);
        vc13 = _mm512_add_ps(vc13, vab13);
        vc14 = _mm512_add_ps(vc14, vab14);
        vc15 = _mm512_add_ps(vc15, vab15);
        vc16 = _mm512_add_ps(vc16, vab16);
        vc17 = _mm512_add_ps(vc17, vab17);
        vc18 = _mm512_add_ps(vc18, vab18);
        vc19 = _mm512_add_ps(vc19, vab19);
        vc20 = _mm512_add_ps(vc20, vab20);
        vc21 = _mm512_add_ps(vc21, vab21);
        vc22 = _mm512_add_ps(vc22, vab22);
        vc23 = _mm512_add_ps(vc23, vab23);
        vc24 = _mm512_add_ps(vc24, vab24);
        vc25 = _mm512_add_ps(vc25, vab25);
        vc26 = _mm512_add_ps(vc26, vab26);


        _mm512_store_ps(C + ldc * 0 + 16 * 0, vc00);
        _mm512_store_ps(C + ldc * 0 + 16 * 1, vc01);
        _mm512_store_ps(C + ldc * 0 + 16 * 2, vc02);
        _mm512_store_ps(C + ldc * 1 + 16 * 0, vc03);
        _mm512_store_ps(C + ldc * 1 + 16 * 1, vc04);
        _mm512_store_ps(C + ldc * 1 + 16 * 2, vc05);
        _mm512_store_ps(C + ldc * 2 + 16 * 0, vc06);
        _mm512_store_ps(C + ldc * 2 + 16 * 1, vc07);
        _mm512_store_ps(C + ldc * 2 + 16 * 2, vc08);
        _mm512_store_ps(C + ldc * 3 + 16 * 0, vc09);
        _mm512_store_ps(C + ldc * 3 + 16 * 1, vc10);
        _mm512_store_ps(C + ldc * 3 + 16 * 2, vc11);
        _mm512_store_ps(C + ldc * 4 + 16 * 0, vc12);
        _mm512_store_ps(C + ldc * 4 + 16 * 1, vc13);
        _mm512_store_ps(C + ldc * 4 + 16 * 2, vc14);
        _mm512_store_ps(C + ldc * 5 + 16 * 0, vc15);
        _mm512_store_ps(C + ldc * 5 + 16 * 1, vc16);
        _mm512_store_ps(C + ldc * 5 + 16 * 2, vc17);
        _mm512_store_ps(C + ldc * 6 + 16 * 0, vc18);
        _mm512_store_ps(C + ldc * 6 + 16 * 1, vc19);
        _mm512_store_ps(C + ldc * 6 + 16 * 2, vc20);
        _mm512_store_ps(C + ldc * 7 + 16 * 0, vc21);
        _mm512_store_ps(C + ldc * 7 + 16 * 1, vc22);
        _mm512_store_ps(C + ldc * 7 + 16 * 2, vc23);
        _mm512_store_ps(C + ldc * 8 + 16 * 0, vc24);
        _mm512_store_ps(C + ldc * 8 + 16 * 1, vc25);
        _mm512_store_ps(C + ldc * 8 + 16 * 2, vc26);
    }

    static void matmul_register_packed(
        int M, int N, int K, float *A, int lda, float *B, int ldb,
        float *C, int ldc)
    {
        assert(0);
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

#endif
