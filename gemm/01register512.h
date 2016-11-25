#pragma once

#include "gemm_util.h"
#include <cassert>
#include <immintrin.h>

#ifdef USE_AVX512

#define _mm512_broadcast_ss(p) _mm512_set1_ps(*(p))

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
            auto va2 = _mm512_broadcast_ss(&pa[lda * 2]);
            vab00 = _mm512_fmadd_ps(va0, vb0, vab00);
            vab01 = _mm512_fmadd_ps(va0, vb1, vab01);
            vab02 = _mm512_fmadd_ps(va0, vb2, vab02);
            vab03 = _mm512_fmadd_ps(va1, vb0, vab03);
            vab04 = _mm512_fmadd_ps(va1, vb1, vab04);
            vab05 = _mm512_fmadd_ps(va1, vb2, vab05);
            vab06 = _mm512_fmadd_ps(va2, vb0, vab06);
            vab07 = _mm512_fmadd_ps(va2, vb1, vab07);
            vab08 = _mm512_fmadd_ps(va2, vb2, vab08);

            auto va3 = _mm512_broadcast_ss(&pa[lda * 3]);
            auto va4 = _mm512_broadcast_ss(&pa[lda * 4]);
            auto va5 = _mm512_broadcast_ss(&pa[lda * 5]);
            vab09 = _mm512_fmadd_ps(va3, vb0, vab09);
            vab10 = _mm512_fmadd_ps(va3, vb1, vab10);
            vab11 = _mm512_fmadd_ps(va3, vb2, vab11);
            vab12 = _mm512_fmadd_ps(va4, vb0, vab12);
            vab13 = _mm512_fmadd_ps(va4, vb1, vab13);
            vab14 = _mm512_fmadd_ps(va4, vb2, vab14);
            vab15 = _mm512_fmadd_ps(va5, vb0, vab15);
            vab16 = _mm512_fmadd_ps(va5, vb1, vab16);
            vab17 = _mm512_fmadd_ps(va5, vb2, vab17);

            auto va6 = _mm512_broadcast_ss(&pa[lda * 6]);
            auto va7 = _mm512_broadcast_ss(&pa[lda * 7]);
            auto va8 = _mm512_broadcast_ss(&pa[lda * 8]);
            vab18 = _mm512_fmadd_ps(va6, vb0, vab18);
            vab19 = _mm512_fmadd_ps(va6, vb1, vab19);
            vab20 = _mm512_fmadd_ps(va6, vb2, vab20);
            vab21 = _mm512_fmadd_ps(va7, vb0, vab21);
            vab22 = _mm512_fmadd_ps(va7, vb1, vab22);
            vab23 = _mm512_fmadd_ps(va7, vb2, vab23);
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
            auto pa = &A[lda * k + 0];
            auto pb = &B[ldb * k + 0];
            auto vb0 = _mm512_load_ps(pb + 16 * 0);
            auto vb1 = _mm512_load_ps(pb + 16 * 1);
            auto vb2 = _mm512_load_ps(pb + 16 * 2);

            auto va0 = _mm512_broadcast_ss(pa + 0);
            auto va1 = _mm512_broadcast_ss(pa + 1);
            auto va2 = _mm512_broadcast_ss(pa + 2);
            vab00 = _mm512_fmadd_ps(va0, vb0, vab00);
            vab01 = _mm512_fmadd_ps(va0, vb1, vab01);
            vab02 = _mm512_fmadd_ps(va0, vb2, vab02);
            vab03 = _mm512_fmadd_ps(va1, vb0, vab03);
            vab04 = _mm512_fmadd_ps(va1, vb1, vab04);
            vab05 = _mm512_fmadd_ps(va1, vb2, vab05);
            vab06 = _mm512_fmadd_ps(va2, vb0, vab06);
            vab07 = _mm512_fmadd_ps(va2, vb1, vab07);
            vab08 = _mm512_fmadd_ps(va2, vb2, vab08);

            auto va3 = _mm512_broadcast_ss(pa + 3);
            auto va4 = _mm512_broadcast_ss(pa + 4);
            auto va5 = _mm512_broadcast_ss(pa + 5);
            vab09 = _mm512_fmadd_ps(va3, vb0, vab09);
            vab10 = _mm512_fmadd_ps(va3, vb1, vab10);
            vab11 = _mm512_fmadd_ps(va3, vb2, vab11);
            vab12 = _mm512_fmadd_ps(va4, vb0, vab12);
            vab13 = _mm512_fmadd_ps(va4, vb1, vab13);
            vab14 = _mm512_fmadd_ps(va4, vb2, vab14);
            vab15 = _mm512_fmadd_ps(va5, vb0, vab15);
            vab16 = _mm512_fmadd_ps(va5, vb1, vab16);
            vab17 = _mm512_fmadd_ps(va5, vb2, vab17);

            auto va6 = _mm512_broadcast_ss(pa + 6);
            auto va7 = _mm512_broadcast_ss(pa + 7);
            auto va8 = _mm512_broadcast_ss(pa + 8);
            vab18 = _mm512_fmadd_ps(va6, vb0, vab18);
            vab19 = _mm512_fmadd_ps(va6, vb1, vab19);
            vab20 = _mm512_fmadd_ps(va6, vb2, vab20);
            vab21 = _mm512_fmadd_ps(va7, vb0, vab21);
            vab22 = _mm512_fmadd_ps(va7, vb1, vab22);
            vab23 = _mm512_fmadd_ps(va7, vb2, vab23);
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

struct register_avx512_5x5 {

    enum : int {
        BLOCK_M = 5,
        BLOCK_N = 16 * 5,
    };

    // 5x80 matrix multiplication
    static void matmul_register(
        int M, int N, int K, float *A, int lda, float *B, int ldb,
        float *C, int ldc)
    {
        assert(M % BLOCK_M == 0);
        assert(N % BLOCK_N == 0);

        //       b0    b1    b2    b3    b4
        // a0 vab00 vab01 vab02 vab03 vab04 
        // a1 vab05 vab06 vab07 vab08 vab09
        // a2 vab10 vab11 vab12 vab13 vab14
        // a3 vab15 vab16 vab17 vab18 vab19
        // a4 vab20 vab21 vab22 vab23 vab24
        
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
        auto vab24 = _mm512_setzero_ps();

        for (int k = 0; k < K; k++) {
            auto pa = &A[lda * 0 + k];
            auto pb = &B[ldb * k + 0];

            auto vb0 = _mm512_load_ps(pb + 16 * 0);
            auto vb1 = _mm512_load_ps(pb + 16 * 1);
            auto vb2 = _mm512_load_ps(pb + 16 * 2);
            auto vb3 = _mm512_load_ps(pb + 16 * 3);
            auto vb4 = _mm512_load_ps(pb + 16 * 4);

            auto va0 = _mm512_broadcast_ss(pa + lda * 0);
            vab00 = _mm512_fmadd_ps(va0, vb0, vab00);
            vab01 = _mm512_fmadd_ps(va0, vb1, vab01);
            vab02 = _mm512_fmadd_ps(va0, vb2, vab02);
            vab03 = _mm512_fmadd_ps(va0, vb3, vab03);
            vab04 = _mm512_fmadd_ps(va0, vb4, vab04);

            auto va1 = _mm512_broadcast_ss(pa + lda * 1);
            vab05 = _mm512_fmadd_ps(va1, vb0, vab05);
            vab06 = _mm512_fmadd_ps(va1, vb1, vab06);
            vab07 = _mm512_fmadd_ps(va1, vb2, vab07);
            vab08 = _mm512_fmadd_ps(va1, vb3, vab08);
            vab09 = _mm512_fmadd_ps(va1, vb4, vab09);

            auto va2 = _mm512_broadcast_ss(pa + lda * 2);
            vab10 = _mm512_fmadd_ps(va2, vb0, vab10);
            vab11 = _mm512_fmadd_ps(va2, vb1, vab11);
            vab12 = _mm512_fmadd_ps(va2, vb2, vab12);
            vab13 = _mm512_fmadd_ps(va2, vb3, vab13);
            vab14 = _mm512_fmadd_ps(va2, vb4, vab14);

            auto va3 = _mm512_broadcast_ss(pa + lda * 3);
            vab15 = _mm512_fmadd_ps(va3, vb0, vab15);
            vab16 = _mm512_fmadd_ps(va3, vb1, vab16);
            vab17 = _mm512_fmadd_ps(va3, vb2, vab17);
            vab18 = _mm512_fmadd_ps(va3, vb3, vab18);
            vab19 = _mm512_fmadd_ps(va3, vb4, vab19);

            auto va4 = _mm512_broadcast_ss(pa + lda * 4);
            vab20 = _mm512_fmadd_ps(va4, vb0, vab20);
            vab21 = _mm512_fmadd_ps(va4, vb1, vab21);
            vab22 = _mm512_fmadd_ps(va4, vb2, vab22);
            vab23 = _mm512_fmadd_ps(va4, vb3, vab23);
            vab24 = _mm512_fmadd_ps(va4, vb4, vab24);
        }

        auto vc00 = _mm512_load_ps(C + ldc * 0 + 16 * 0);
        auto vc01 = _mm512_load_ps(C + ldc * 0 + 16 * 1);
        auto vc02 = _mm512_load_ps(C + ldc * 0 + 16 * 2);
        auto vc03 = _mm512_load_ps(C + ldc * 0 + 16 * 3);
        auto vc04 = _mm512_load_ps(C + ldc * 0 + 16 * 4);
        auto vc05 = _mm512_load_ps(C + ldc * 1 + 16 * 0);
        auto vc06 = _mm512_load_ps(C + ldc * 1 + 16 * 1);
        auto vc07 = _mm512_load_ps(C + ldc * 1 + 16 * 2);
        auto vc08 = _mm512_load_ps(C + ldc * 1 + 16 * 3);
        auto vc09 = _mm512_load_ps(C + ldc * 1 + 16 * 4);
        auto vc10 = _mm512_load_ps(C + ldc * 2 + 16 * 0);
        auto vc11 = _mm512_load_ps(C + ldc * 2 + 16 * 1);
        auto vc12 = _mm512_load_ps(C + ldc * 2 + 16 * 2);
        auto vc13 = _mm512_load_ps(C + ldc * 2 + 16 * 3);
        auto vc14 = _mm512_load_ps(C + ldc * 2 + 16 * 4);
        auto vc15 = _mm512_load_ps(C + ldc * 3 + 16 * 0);
        auto vc16 = _mm512_load_ps(C + ldc * 3 + 16 * 1);
        auto vc17 = _mm512_load_ps(C + ldc * 3 + 16 * 2);
        auto vc18 = _mm512_load_ps(C + ldc * 3 + 16 * 3);
        auto vc19 = _mm512_load_ps(C + ldc * 3 + 16 * 4);
        auto vc20 = _mm512_load_ps(C + ldc * 4 + 16 * 0);
        auto vc21 = _mm512_load_ps(C + ldc * 4 + 16 * 1);
        auto vc22 = _mm512_load_ps(C + ldc * 4 + 16 * 2);
        auto vc23 = _mm512_load_ps(C + ldc * 4 + 16 * 3);
        auto vc24 = _mm512_load_ps(C + ldc * 4 + 16 * 4);

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

        _mm512_store_ps(C + ldc * 0 + 16 * 0, vc00);
        _mm512_store_ps(C + ldc * 0 + 16 * 1, vc01);
        _mm512_store_ps(C + ldc * 0 + 16 * 2, vc02);
        _mm512_store_ps(C + ldc * 0 + 16 * 3, vc03);
        _mm512_store_ps(C + ldc * 0 + 16 * 4, vc04);
        _mm512_store_ps(C + ldc * 1 + 16 * 0, vc05);
        _mm512_store_ps(C + ldc * 1 + 16 * 1, vc06);
        _mm512_store_ps(C + ldc * 1 + 16 * 2, vc07);
        _mm512_store_ps(C + ldc * 1 + 16 * 3, vc08);
        _mm512_store_ps(C + ldc * 1 + 16 * 4, vc09);
        _mm512_store_ps(C + ldc * 2 + 16 * 0, vc10);
        _mm512_store_ps(C + ldc * 2 + 16 * 1, vc11);
        _mm512_store_ps(C + ldc * 2 + 16 * 2, vc12);
        _mm512_store_ps(C + ldc * 2 + 16 * 3, vc13);
        _mm512_store_ps(C + ldc * 2 + 16 * 4, vc14);
        _mm512_store_ps(C + ldc * 3 + 16 * 0, vc15);
        _mm512_store_ps(C + ldc * 3 + 16 * 1, vc16);
        _mm512_store_ps(C + ldc * 3 + 16 * 2, vc17);
        _mm512_store_ps(C + ldc * 3 + 16 * 3, vc18);
        _mm512_store_ps(C + ldc * 3 + 16 * 4, vc19);
        _mm512_store_ps(C + ldc * 4 + 16 * 0, vc20);
        _mm512_store_ps(C + ldc * 4 + 16 * 1, vc21);
        _mm512_store_ps(C + ldc * 4 + 16 * 2, vc22);
        _mm512_store_ps(C + ldc * 4 + 16 * 3, vc23);
        _mm512_store_ps(C + ldc * 4 + 16 * 4, vc24);
    }

    static void matmul_register_packed(
        int M, int N, int K, float *A, int lda, float *B, int ldb,
        float *C, int ldc)
    {
        assert(M % BLOCK_M == 0);
        assert(N % BLOCK_N == 0);
        assert(lda == BLOCK_M);
        assert(ldb == BLOCK_N);

        lda = BLOCK_M;
        ldb = BLOCK_N;

        //       b0    b1    b2    b3    b4
        // a0 vab00 vab01 vab02 vab03 vab04 
        // a1 vab05 vab06 vab07 vab08 vab09
        // a2 vab10 vab11 vab12 vab13 vab14
        // a3 vab15 vab16 vab17 vab18 vab19
        // a4 vab20 vab21 vab22 vab23 vab24
        
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
        auto vab24 = _mm512_setzero_ps();

        for (int k = 0; k < K; k++) {
            auto pa = A + lda * k;
            auto pb = B + ldb * k;

            auto vb0 = _mm512_load_ps(pb + 16 * 0);
            auto vb1 = _mm512_load_ps(pb + 16 * 1);
            auto vb2 = _mm512_load_ps(pb + 16 * 2);
            auto vb3 = _mm512_load_ps(pb + 16 * 3);
            auto vb4 = _mm512_load_ps(pb + 16 * 4);

            auto va0 = _mm512_broadcast_ss(pa + 0);
            vab00 = _mm512_fmadd_ps(va0, vb0, vab00);
            vab01 = _mm512_fmadd_ps(va0, vb1, vab01);
            vab02 = _mm512_fmadd_ps(va0, vb2, vab02);
            vab03 = _mm512_fmadd_ps(va0, vb3, vab03);
            vab04 = _mm512_fmadd_ps(va0, vb4, vab04);

            auto va1 = _mm512_broadcast_ss(pa + 1);
            vab05 = _mm512_fmadd_ps(va1, vb0, vab05);
            vab06 = _mm512_fmadd_ps(va1, vb1, vab06);
            vab07 = _mm512_fmadd_ps(va1, vb2, vab07);
            vab08 = _mm512_fmadd_ps(va1, vb3, vab08);
            vab09 = _mm512_fmadd_ps(va1, vb4, vab09);

            auto va2 = _mm512_broadcast_ss(pa + 2);
            vab10 = _mm512_fmadd_ps(va2, vb0, vab10);
            vab11 = _mm512_fmadd_ps(va2, vb1, vab11);
            vab12 = _mm512_fmadd_ps(va2, vb2, vab12);
            vab13 = _mm512_fmadd_ps(va2, vb3, vab13);
            vab14 = _mm512_fmadd_ps(va2, vb4, vab14);

            auto va3 = _mm512_broadcast_ss(pa + 3);
            vab15 = _mm512_fmadd_ps(va3, vb0, vab15);
            vab16 = _mm512_fmadd_ps(va3, vb1, vab16);
            vab17 = _mm512_fmadd_ps(va3, vb2, vab17);
            vab18 = _mm512_fmadd_ps(va3, vb3, vab18);
            vab19 = _mm512_fmadd_ps(va3, vb4, vab19);

            auto va4 = _mm512_broadcast_ss(pa + 4);
            vab20 = _mm512_fmadd_ps(va4, vb0, vab20);
            vab21 = _mm512_fmadd_ps(va4, vb1, vab21);
            vab22 = _mm512_fmadd_ps(va4, vb2, vab22);
            vab23 = _mm512_fmadd_ps(va4, vb3, vab23);
            vab24 = _mm512_fmadd_ps(va4, vb4, vab24);
        }

        auto vc00 = _mm512_load_ps(C + ldc * 0 + 16 * 0);
        auto vc01 = _mm512_load_ps(C + ldc * 0 + 16 * 1);
        auto vc02 = _mm512_load_ps(C + ldc * 0 + 16 * 2);
        auto vc03 = _mm512_load_ps(C + ldc * 0 + 16 * 3);
        auto vc04 = _mm512_load_ps(C + ldc * 0 + 16 * 4);
        auto vc05 = _mm512_load_ps(C + ldc * 1 + 16 * 0);
        auto vc06 = _mm512_load_ps(C + ldc * 1 + 16 * 1);
        auto vc07 = _mm512_load_ps(C + ldc * 1 + 16 * 2);
        auto vc08 = _mm512_load_ps(C + ldc * 1 + 16 * 3);
        auto vc09 = _mm512_load_ps(C + ldc * 1 + 16 * 4);
        auto vc10 = _mm512_load_ps(C + ldc * 2 + 16 * 0);
        auto vc11 = _mm512_load_ps(C + ldc * 2 + 16 * 1);
        auto vc12 = _mm512_load_ps(C + ldc * 2 + 16 * 2);
        auto vc13 = _mm512_load_ps(C + ldc * 2 + 16 * 3);
        auto vc14 = _mm512_load_ps(C + ldc * 2 + 16 * 4);
        auto vc15 = _mm512_load_ps(C + ldc * 3 + 16 * 0);
        auto vc16 = _mm512_load_ps(C + ldc * 3 + 16 * 1);
        auto vc17 = _mm512_load_ps(C + ldc * 3 + 16 * 2);
        auto vc18 = _mm512_load_ps(C + ldc * 3 + 16 * 3);
        auto vc19 = _mm512_load_ps(C + ldc * 3 + 16 * 4);
        auto vc20 = _mm512_load_ps(C + ldc * 4 + 16 * 0);
        auto vc21 = _mm512_load_ps(C + ldc * 4 + 16 * 1);
        auto vc22 = _mm512_load_ps(C + ldc * 4 + 16 * 2);
        auto vc23 = _mm512_load_ps(C + ldc * 4 + 16 * 3);
        auto vc24 = _mm512_load_ps(C + ldc * 4 + 16 * 4);

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

        _mm512_store_ps(C + ldc * 0 + 16 * 0, vc00);
        _mm512_store_ps(C + ldc * 0 + 16 * 1, vc01);
        _mm512_store_ps(C + ldc * 0 + 16 * 2, vc02);
        _mm512_store_ps(C + ldc * 0 + 16 * 3, vc03);
        _mm512_store_ps(C + ldc * 0 + 16 * 4, vc04);
        _mm512_store_ps(C + ldc * 1 + 16 * 0, vc05);
        _mm512_store_ps(C + ldc * 1 + 16 * 1, vc06);
        _mm512_store_ps(C + ldc * 1 + 16 * 2, vc07);
        _mm512_store_ps(C + ldc * 1 + 16 * 3, vc08);
        _mm512_store_ps(C + ldc * 1 + 16 * 4, vc09);
        _mm512_store_ps(C + ldc * 2 + 16 * 0, vc10);
        _mm512_store_ps(C + ldc * 2 + 16 * 1, vc11);
        _mm512_store_ps(C + ldc * 2 + 16 * 2, vc12);
        _mm512_store_ps(C + ldc * 2 + 16 * 3, vc13);
        _mm512_store_ps(C + ldc * 2 + 16 * 4, vc14);
        _mm512_store_ps(C + ldc * 3 + 16 * 0, vc15);
        _mm512_store_ps(C + ldc * 3 + 16 * 1, vc16);
        _mm512_store_ps(C + ldc * 3 + 16 * 2, vc17);
        _mm512_store_ps(C + ldc * 3 + 16 * 3, vc18);
        _mm512_store_ps(C + ldc * 3 + 16 * 4, vc19);
        _mm512_store_ps(C + ldc * 4 + 16 * 0, vc20);
        _mm512_store_ps(C + ldc * 4 + 16 * 1, vc21);
        _mm512_store_ps(C + ldc * 4 + 16 * 2, vc22);
        _mm512_store_ps(C + ldc * 4 + 16 * 3, vc23);
        _mm512_store_ps(C + ldc * 4 + 16 * 4, vc24);
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

struct register_avx512_5x5asm {

    enum : int {
        BLOCK_M = 5,
        BLOCK_N = 16 * 5,
    };

    // 5x80 matrix multiplication
    static void matmul_register(
        int M, int N, int K, float *A, int lda, float *B, int ldb,
        float *C, int ldc)
    {
        register_avx512_5x5::matmul_register(
            M, N, K, A, lda, B, ldb, C, ldc);
    }

    static NOINLINE void matmul_register_packed(
        int M, int N, int K, float *A, int lda, float *B, int ldb,
        float *C, int ldc)
    {
        assert(M % BLOCK_M == 0);
        assert(N % BLOCK_N == 0);
        assert(lda == BLOCK_M);
        assert(ldb == BLOCK_N);

#ifdef _MSC_VER
        register_avx512_5x5::matmul_register_packed(
            M, N, K, A, lda, B, ldb, C, ldc);
#else
        int64_t K64 = K;
        int64_t ldc64 = ldc;

        __asm__ __volatile__ (
            "                                                   \n\t"
            "vzeroall                                           \n\t"
            "vpxord             %%zmm0, %%zmm0, %%zmm0          \n\t"
            "vpxord             %%zmm1, %%zmm1, %%zmm1          \n\t"
            "vpxord             %%zmm2, %%zmm2, %%zmm2          \n\t"
            "vpxord             %%zmm3, %%zmm3, %%zmm3          \n\t"
            "vpxord             %%zmm4, %%zmm4, %%zmm4          \n\t"
            "vpxord             %%zmm5, %%zmm5, %%zmm5          \n\t"
            "vpxord             %%zmm6, %%zmm6, %%zmm6          \n\t"
            "vpxord             %%zmm7, %%zmm7, %%zmm7          \n\t"
            "vpxord             %%zmm8, %%zmm8, %%zmm8          \n\t"
            "vpxord             %%zmm9, %%zmm9, %%zmm9          \n\t"
            "vpxord             %%zmm10, %%zmm10, %%zmm10          \n\t"
            "vpxord             %%zmm11, %%zmm11, %%zmm11          \n\t"
            "vpxord             %%zmm12, %%zmm12, %%zmm12          \n\t"
            "vpxord             %%zmm13, %%zmm13, %%zmm13          \n\t"
            "vpxord             %%zmm14, %%zmm14, %%zmm14          \n\t"
            "vpxord             %%zmm15, %%zmm15, %%zmm15          \n\t"
            "vpxord             %%zmm16, %%zmm16, %%zmm16          \n\t"
            "vpxord             %%zmm17, %%zmm17, %%zmm17          \n\t"
            "vpxord             %%zmm18, %%zmm18, %%zmm18          \n\t"
            "vpxord             %%zmm19, %%zmm19, %%zmm19          \n\t"
            "vpxord             %%zmm20, %%zmm20, %%zmm20          \n\t"
            "vpxord             %%zmm21, %%zmm21, %%zmm21          \n\t"
            "vpxord             %%zmm22, %%zmm22, %%zmm22          \n\t"
            "vpxord             %%zmm23, %%zmm23, %%zmm23          \n\t"
            "vpxord             %%zmm24, %%zmm24, %%zmm24          \n\t"
            "vpxord             %%zmm25, %%zmm25, %%zmm25          \n\t"
            "vpxord             %%zmm26, %%zmm26, %%zmm26          \n\t"
            "vpxord             %%zmm27, %%zmm27, %%zmm27          \n\t"
            "vpxord             %%zmm28, %%zmm28, %%zmm28          \n\t"
            "vpxord             %%zmm29, %%zmm29, %%zmm29          \n\t"
            "vpxord             %%zmm30, %%zmm30, %%zmm30          \n\t"
            "vpxord             %%zmm31, %%zmm31, %%zmm31          \n\t"
            "                                                   \n\t"
            "mov                %0, %%rax                       \n\t" // A
            "mov                %1, %%rbx                       \n\t" // B
            "mov                %2, %%rcx                       \n\t" // C
            "mov                %3, %%rsi                       \n\t" // K
            "mov                %4, %%rdi                       \n\t" // ldc
            "                                                   \n\t"
            "lea                (,%%rsi,4), %%r8                \n\t" //  4*rsi
            "lea                (%%rax,%%r8,4), %%r9            \n\t" // 16*rsi
            "add                %%r8, %%r9                      \n\t" // 20*rsi
            "                                                   \n\t"
            ".KLOOP_5x5:                                        \n\t"
            "                                                   \n\t"
            "cmp                %%rax, %%r9                     \n\t"
            "je                 .KLOOP_EXIT_5x5                 \n\t"
            "                                                   \n\t"
            "vmovaps            0 * 64(%%rbx), %%zmm25          \n\t"
            "vmovaps            1 * 64(%%rbx), %%zmm26          \n\t"
            "vmovaps            2 * 64(%%rbx), %%zmm27          \n\t"
            "vmovaps            3 * 64(%%rbx), %%zmm28          \n\t"
            "vmovaps            4 * 64(%%rbx), %%zmm29          \n\t"
            "                                                   \n\t"
            "vbroadcastss       0 * 4(%%rax), %%zmm30           \n\t"
            "vfmadd231ps        %%zmm30, %%zmm25, %%zmm0        \n\t"
            "vfmadd231ps        %%zmm30, %%zmm26, %%zmm1        \n\t"
            "vfmadd231ps        %%zmm30, %%zmm27, %%zmm2        \n\t"
            "vfmadd231ps        %%zmm30, %%zmm28, %%zmm3        \n\t"
            "vfmadd231ps        %%zmm30, %%zmm29, %%zmm4        \n\t"
            "                                                   \n\t"
            "vbroadcastss       1 * 4(%%rax), %%zmm31           \n\t"
            "vfmadd231ps        %%zmm31, %%zmm25, %%zmm5        \n\t"
            "vfmadd231ps        %%zmm31, %%zmm26, %%zmm6        \n\t"
            "vfmadd231ps        %%zmm31, %%zmm27, %%zmm7        \n\t"
            "vfmadd231ps        %%zmm31, %%zmm28, %%zmm8        \n\t"
            "vfmadd231ps        %%zmm31, %%zmm29, %%zmm9        \n\t"
            "                                                   \n\t"
            "vbroadcastss       2 * 4(%%rax), %%zmm30           \n\t"
            "vfmadd231ps        %%zmm30, %%zmm25, %%zmm10       \n\t"
            "vfmadd231ps        %%zmm30, %%zmm26, %%zmm11       \n\t"
            "vfmadd231ps        %%zmm30, %%zmm27, %%zmm12       \n\t"
            "vfmadd231ps        %%zmm30, %%zmm28, %%zmm13       \n\t"
            "vfmadd231ps        %%zmm30, %%zmm29, %%zmm14       \n\t"
            "                                                   \n\t"
            "vbroadcastss       3 * 4(%%rax), %%zmm31           \n\t"
            "vfmadd231ps        %%zmm31, %%zmm25, %%zmm15       \n\t"
            "vfmadd231ps        %%zmm31, %%zmm26, %%zmm16       \n\t"
            "vfmadd231ps        %%zmm31, %%zmm27, %%zmm17       \n\t"
            "vfmadd231ps        %%zmm31, %%zmm28, %%zmm18       \n\t"
            "vfmadd231ps        %%zmm31, %%zmm29, %%zmm19       \n\t"
            "                                                   \n\t"
            "vbroadcastss       4 * 4(%%rax), %%zmm30           \n\t"
            "vfmadd231ps        %%zmm30, %%zmm25, %%zmm20       \n\t"
            "vfmadd231ps        %%zmm30, %%zmm26, %%zmm21       \n\t"
            "vfmadd231ps        %%zmm30, %%zmm27, %%zmm22       \n\t"
            "vfmadd231ps        %%zmm30, %%zmm28, %%zmm23       \n\t"
            "vfmadd231ps        %%zmm30, %%zmm29, %%zmm24       \n\t"
            "                                                   \n\t"
            "add                $20, %%rax                      \n\t"
            "add                $320, %%rbx                     \n\t"
            "jmp                .KLOOP_5x5                      \n\t"
            "                                                   \n\t"
            ".KLOOP_EXIT_5x5:                                   \n\t"
            "                                                   \n\t"
            "lea                (,%%rdi,2), %%r10               \n\t" // ldc*2
            "lea                (%%r10,%%rdi), %%r11            \n\t" // ldc*3
            "lea                (,%%rdi,4), %%r12               \n\t" // ldc*4
            "                                                   \n\t"
            "vaddps               0(%%rcx), %%zmm0, %%zmm0      \n\t"
            "vaddps              64(%%rcx), %%zmm1, %%zmm1      \n\t"
            "vaddps             128(%%rcx), %%zmm2, %%zmm2      \n\t"
            "vaddps             192(%%rcx), %%zmm3, %%zmm3      \n\t"
            "vaddps             256(%%rcx), %%zmm4, %%zmm4      \n\t"
            "vaddps               0(%%rcx,%%rdi,4), %%zmm5, %%zmm5      \n\t"
            "vaddps              64(%%rcx,%%rdi,4), %%zmm6, %%zmm6      \n\t"
            "vaddps             128(%%rcx,%%rdi,4), %%zmm7, %%zmm7      \n\t"
            "vaddps             192(%%rcx,%%rdi,4), %%zmm8, %%zmm8      \n\t"
            "vaddps             256(%%rcx,%%rdi,4), %%zmm9, %%zmm9      \n\t"
            "vaddps               0(%%rcx,%%r10,4), %%zmm10, %%zmm10    \n\t"
            "vaddps              64(%%rcx,%%r10,4), %%zmm11, %%zmm11    \n\t"
            "vaddps             128(%%rcx,%%r10,4), %%zmm12, %%zmm12    \n\t"
            "vaddps             192(%%rcx,%%r10,4), %%zmm13, %%zmm13    \n\t"
            "vaddps             256(%%rcx,%%r10,4), %%zmm14, %%zmm14    \n\t"
            "vaddps               0(%%rcx,%%r11,4), %%zmm15, %%zmm15    \n\t"
            "vaddps              64(%%rcx,%%r11,4), %%zmm16, %%zmm16    \n\t"
            "vaddps             128(%%rcx,%%r11,4), %%zmm17, %%zmm17    \n\t"
            "vaddps             192(%%rcx,%%r11,4), %%zmm18, %%zmm18    \n\t"
            "vaddps             256(%%rcx,%%r11,4), %%zmm19, %%zmm19    \n\t"
            "vaddps               0(%%rcx,%%r12,4), %%zmm20, %%zmm20    \n\t"
            "vaddps              64(%%rcx,%%r12,4), %%zmm21, %%zmm21    \n\t"
            "vaddps             128(%%rcx,%%r12,4), %%zmm22, %%zmm22    \n\t"
            "vaddps             192(%%rcx,%%r12,4), %%zmm23, %%zmm23    \n\t"
            "vaddps             256(%%rcx,%%r12,4), %%zmm24, %%zmm24    \n\t"
            "                                                   \n\t"
            "vmovaps            %%zmm0,    0(%%rcx)             \n\t"
            "vmovaps            %%zmm1,   64(%%rcx)             \n\t"
            "vmovaps            %%zmm2,  128(%%rcx)             \n\t"
            "vmovaps            %%zmm3,  192(%%rcx)             \n\t"
            "vmovaps            %%zmm4,  256(%%rcx)             \n\t"
            "vmovaps            %%zmm5,    0(%%rcx,%%rdi,4)     \n\t"
            "vmovaps            %%zmm6,   64(%%rcx,%%rdi,4)     \n\t"
            "vmovaps            %%zmm7,  128(%%rcx,%%rdi,4)     \n\t"
            "vmovaps            %%zmm8,  192(%%rcx,%%rdi,4)     \n\t"
            "vmovaps            %%zmm9,  256(%%rcx,%%rdi,4)     \n\t"
            "vmovaps            %%zmm10,   0(%%rcx,%%r10,4)     \n\t"
            "vmovaps            %%zmm11,  64(%%rcx,%%r10,4)     \n\t"
            "vmovaps            %%zmm12, 128(%%rcx,%%r10,4)     \n\t"
            "vmovaps            %%zmm13, 192(%%rcx,%%r10,4)     \n\t"
            "vmovaps            %%zmm14, 256(%%rcx,%%r10,4)     \n\t"
            "vmovaps            %%zmm15,   0(%%rcx,%%r11,4)     \n\t"
            "vmovaps            %%zmm16,  64(%%rcx,%%r11,4)     \n\t"
            "vmovaps            %%zmm17, 128(%%rcx,%%r11,4)     \n\t"
            "vmovaps            %%zmm18, 192(%%rcx,%%r11,4)     \n\t"
            "vmovaps            %%zmm19, 256(%%rcx,%%r11,4)     \n\t"
            "vmovaps            %%zmm20,   0(%%rcx,%%r12,4)     \n\t"
            "vmovaps            %%zmm21,  64(%%rcx,%%r12,4)     \n\t"
            "vmovaps            %%zmm22, 128(%%rcx,%%r12,4)     \n\t"
            "vmovaps            %%zmm23, 192(%%rcx,%%r12,4)     \n\t"
            "vmovaps            %%zmm24, 256(%%rcx,%%r12,4)     \n\t"
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
            , "r8", "r9", "r10", "r11", "r12" //, "r13", "r14", "r15"
            , "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"
            , "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14"
            , "xmm15", "xmm16", "xmm17", "xmm18", "xmm19", "xmm20", "xmm21"
            , "xmm22", "xmm23", "xmm24", "xmm25", "xmm26", "xmm27", "xmm28"
            , "xmm29", "xmm30", "xmm31"
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

struct register_avx512_5x5asm_unroll {

    enum : int {
        BLOCK_M = 5,
        BLOCK_N = 16 * 5,
    };

    // 5x80 matrix multiplication
    static void matmul_register(
        int M, int N, int K, float *A, int lda, float *B, int ldb,
        float *C, int ldc)
    {
        register_avx512_5x5::matmul_register(
            M, N, K, A, lda, B, ldb, C, ldc);
    }

    static NOINLINE void matmul_register_packed(
        int M, int N, int K, float *A, int lda, float *B, int ldb,
        float *C, int ldc)
    {
        assert(M % BLOCK_M == 0);
        assert(N % BLOCK_N == 0);
        assert(lda == BLOCK_M);
        assert(ldb == BLOCK_N);

#ifdef _MSC_VER
        register_avx512_5x5::matmul_register_packed(
            M, N, K, A, lda, B, ldb, C, ldc);
#else
        int64_t K64 = K;
        int64_t ldc64 = ldc;

        __asm__ __volatile__ (
            "                                                   \n\t"
            "vzeroall                                           \n\t"
            "vpxord             %%zmm0, %%zmm0, %%zmm0          \n\t"
            "vpxord             %%zmm1, %%zmm1, %%zmm1          \n\t"
            "vpxord             %%zmm2, %%zmm2, %%zmm2          \n\t"
            "vpxord             %%zmm3, %%zmm3, %%zmm3          \n\t"
            "vpxord             %%zmm4, %%zmm4, %%zmm4          \n\t"
            "vpxord             %%zmm5, %%zmm5, %%zmm5          \n\t"
            "vpxord             %%zmm6, %%zmm6, %%zmm6          \n\t"
            "vpxord             %%zmm7, %%zmm7, %%zmm7          \n\t"
            "vpxord             %%zmm8, %%zmm8, %%zmm8          \n\t"
            "vpxord             %%zmm9, %%zmm9, %%zmm9          \n\t"
            "vpxord             %%zmm10, %%zmm10, %%zmm10          \n\t"
            "vpxord             %%zmm11, %%zmm11, %%zmm11          \n\t"
            "vpxord             %%zmm12, %%zmm12, %%zmm12          \n\t"
            "vpxord             %%zmm13, %%zmm13, %%zmm13          \n\t"
            "vpxord             %%zmm14, %%zmm14, %%zmm14          \n\t"
            "vpxord             %%zmm15, %%zmm15, %%zmm15          \n\t"
            "vpxord             %%zmm16, %%zmm16, %%zmm16          \n\t"
            "vpxord             %%zmm17, %%zmm17, %%zmm17          \n\t"
            "vpxord             %%zmm18, %%zmm18, %%zmm18          \n\t"
            "vpxord             %%zmm19, %%zmm19, %%zmm19          \n\t"
            "vpxord             %%zmm20, %%zmm20, %%zmm20          \n\t"
            "vpxord             %%zmm21, %%zmm21, %%zmm21          \n\t"
            "vpxord             %%zmm22, %%zmm22, %%zmm22          \n\t"
            "vpxord             %%zmm23, %%zmm23, %%zmm23          \n\t"
            "vpxord             %%zmm24, %%zmm24, %%zmm24          \n\t"
            "vpxord             %%zmm25, %%zmm25, %%zmm25          \n\t"
            "vpxord             %%zmm26, %%zmm26, %%zmm26          \n\t"
            "vpxord             %%zmm27, %%zmm27, %%zmm27          \n\t"
            "vpxord             %%zmm28, %%zmm28, %%zmm28          \n\t"
            "vpxord             %%zmm29, %%zmm29, %%zmm29          \n\t"
            "vpxord             %%zmm30, %%zmm30, %%zmm30          \n\t"
            "vpxord             %%zmm31, %%zmm31, %%zmm31          \n\t"
            "                                                   \n\t"
            "mov                %0, %%rax                       \n\t" // A
            "mov                %1, %%rbx                       \n\t" // B
            "mov                %2, %%rcx                       \n\t" // C
            "mov                %3, %%rsi                       \n\t" // K
            "mov                %4, %%rdi                       \n\t" // ldc
            "                                                   \n\t"
            "lea                (,%%rsi,4), %%r8                \n\t" //  4*rsi
            "lea                (%%rax,%%r8,4), %%r9            \n\t" // 16*rsi
            "add                %%r8, %%r9                      \n\t" // 20*rsi
            "                                                   \n\t"
            ".KLOOP_5x5_UNROLL:                                 \n\t"
            "                                                   \n\t"
            "cmp                %%rax, %%r9                     \n\t"
            "je                 .KLOOP_EXIT_5x5_UNROLL          \n\t"
            "                                                   \n\t"
            "vmovaps            0 * 64(%%rbx), %%zmm25          \n\t"
            "vmovaps            1 * 64(%%rbx), %%zmm26          \n\t"
            "vmovaps            2 * 64(%%rbx), %%zmm27          \n\t"
            "vmovaps            3 * 64(%%rbx), %%zmm28          \n\t"
            "vmovaps            4 * 64(%%rbx), %%zmm29          \n\t"
            "                                                   \n\t"
            "vbroadcastss       0 * 4(%%rax), %%zmm30           \n\t"
            "vbroadcastss       1 * 4(%%rax), %%zmm31           \n\t"
            "vfmadd231ps        %%zmm30, %%zmm25, %%zmm0        \n\t"
            "vfmadd231ps        %%zmm30, %%zmm26, %%zmm1        \n\t"
            "vfmadd231ps        %%zmm30, %%zmm27, %%zmm2        \n\t"
            "vfmadd231ps        %%zmm30, %%zmm28, %%zmm3        \n\t"
            "vfmadd231ps        %%zmm30, %%zmm29, %%zmm4        \n\t"
            "vfmadd231ps        %%zmm31, %%zmm25, %%zmm5        \n\t"
            "vfmadd231ps        %%zmm31, %%zmm26, %%zmm6        \n\t"
            "vfmadd231ps        %%zmm31, %%zmm27, %%zmm7        \n\t"
            "vfmadd231ps        %%zmm31, %%zmm28, %%zmm8        \n\t"
            "vfmadd231ps        %%zmm31, %%zmm29, %%zmm9        \n\t"
            "                                                   \n\t"
            "vbroadcastss       2 * 4(%%rax), %%zmm30           \n\t"
            "vbroadcastss       3 * 4(%%rax), %%zmm31           \n\t"
            "vfmadd231ps        %%zmm30, %%zmm25, %%zmm10       \n\t"
            "vfmadd231ps        %%zmm30, %%zmm26, %%zmm11       \n\t"
            "vfmadd231ps        %%zmm30, %%zmm27, %%zmm12       \n\t"
            "vfmadd231ps        %%zmm30, %%zmm28, %%zmm13       \n\t"
            "vfmadd231ps        %%zmm30, %%zmm29, %%zmm14       \n\t"
            "vfmadd231ps        %%zmm31, %%zmm25, %%zmm15       \n\t"
            "vfmadd231ps        %%zmm31, %%zmm26, %%zmm16       \n\t"
            "vfmadd231ps        %%zmm31, %%zmm27, %%zmm17       \n\t"
            "vfmadd231ps        %%zmm31, %%zmm28, %%zmm18       \n\t"
            "vfmadd231ps        %%zmm31, %%zmm29, %%zmm19       \n\t"
            "                                                   \n\t"
            "vbroadcastss       4 * 4(%%rax), %%zmm30           \n\t"
            "vbroadcastss       5 * 4(%%rax), %%zmm31           \n\t"
            "vfmadd231ps        %%zmm30, %%zmm25, %%zmm20       \n\t"
            "vfmadd231ps        %%zmm30, %%zmm26, %%zmm21       \n\t"
            "vfmadd231ps        %%zmm30, %%zmm27, %%zmm22       \n\t"
            "vfmadd231ps        %%zmm30, %%zmm28, %%zmm23       \n\t"
            "vfmadd231ps        %%zmm30, %%zmm29, %%zmm24       \n\t"
            "                                                   \n\t"
            "vmovaps            5 * 64(%%rbx), %%zmm25          \n\t"
            "vmovaps            6 * 64(%%rbx), %%zmm26          \n\t"
            "vmovaps            7 * 64(%%rbx), %%zmm27          \n\t"
            "vmovaps            8 * 64(%%rbx), %%zmm28          \n\t"
            "vmovaps            9 * 64(%%rbx), %%zmm29          \n\t"
            "                                                   \n\t"
            "vfmadd231ps        %%zmm31, %%zmm25, %%zmm0        \n\t"
            "vfmadd231ps        %%zmm31, %%zmm26, %%zmm1        \n\t"
            "vfmadd231ps        %%zmm31, %%zmm27, %%zmm2        \n\t"
            "vfmadd231ps        %%zmm31, %%zmm28, %%zmm3        \n\t"
            "vfmadd231ps        %%zmm31, %%zmm29, %%zmm4        \n\t"
            "                                                   \n\t"
            "vbroadcastss       6 * 4(%%rax), %%zmm30           \n\t"
            "vbroadcastss       7 * 4(%%rax), %%zmm31           \n\t"
            "vfmadd231ps        %%zmm30, %%zmm25, %%zmm5        \n\t"
            "vfmadd231ps        %%zmm30, %%zmm26, %%zmm6        \n\t"
            "vfmadd231ps        %%zmm30, %%zmm27, %%zmm7        \n\t"
            "vfmadd231ps        %%zmm30, %%zmm28, %%zmm8        \n\t"
            "vfmadd231ps        %%zmm30, %%zmm29, %%zmm9        \n\t"
            "vfmadd231ps        %%zmm31, %%zmm25, %%zmm10       \n\t"
            "vfmadd231ps        %%zmm31, %%zmm26, %%zmm11       \n\t"
            "vfmadd231ps        %%zmm31, %%zmm27, %%zmm12       \n\t"
            "vfmadd231ps        %%zmm31, %%zmm28, %%zmm13       \n\t"
            "vfmadd231ps        %%zmm31, %%zmm29, %%zmm14       \n\t"
            "                                                   \n\t"
            "vbroadcastss       8 * 4(%%rax), %%zmm30           \n\t"
            "vbroadcastss       9 * 4(%%rax), %%zmm31           \n\t"
            "vfmadd231ps        %%zmm30, %%zmm25, %%zmm15       \n\t"
            "vfmadd231ps        %%zmm30, %%zmm26, %%zmm16       \n\t"
            "vfmadd231ps        %%zmm30, %%zmm27, %%zmm17       \n\t"
            "vfmadd231ps        %%zmm30, %%zmm28, %%zmm18       \n\t"
            "vfmadd231ps        %%zmm30, %%zmm29, %%zmm19       \n\t"
            "vfmadd231ps        %%zmm31, %%zmm25, %%zmm20       \n\t"
            "vfmadd231ps        %%zmm31, %%zmm26, %%zmm21       \n\t"
            "vfmadd231ps        %%zmm31, %%zmm27, %%zmm22       \n\t"
            "vfmadd231ps        %%zmm31, %%zmm28, %%zmm23       \n\t"
            "vfmadd231ps        %%zmm31, %%zmm29, %%zmm24       \n\t"
            "                                                   \n\t"
            "add                $40, %%rax                      \n\t"
            "add                $640, %%rbx                     \n\t"
            "jmp                .KLOOP_5x5_UNROLL               \n\t"
            "                                                   \n\t"
            ".KLOOP_EXIT_5x5_UNROLL:                            \n\t"
            "                                                   \n\t"
            "lea                (,%%rdi,2), %%r10               \n\t" // ldc*2
            "lea                (%%r10,%%rdi), %%r11            \n\t" // ldc*3
            "lea                (,%%rdi,4), %%r12               \n\t" // ldc*4
            "                                                   \n\t"
            "vaddps               0(%%rcx), %%zmm0, %%zmm0      \n\t"
            "vaddps              64(%%rcx), %%zmm1, %%zmm1      \n\t"
            "vaddps             128(%%rcx), %%zmm2, %%zmm2      \n\t"
            "vaddps             192(%%rcx), %%zmm3, %%zmm3      \n\t"
            "vaddps             256(%%rcx), %%zmm4, %%zmm4      \n\t"
            "vaddps               0(%%rcx,%%rdi,4), %%zmm5, %%zmm5      \n\t"
            "vaddps              64(%%rcx,%%rdi,4), %%zmm6, %%zmm6      \n\t"
            "vaddps             128(%%rcx,%%rdi,4), %%zmm7, %%zmm7      \n\t"
            "vaddps             192(%%rcx,%%rdi,4), %%zmm8, %%zmm8      \n\t"
            "vaddps             256(%%rcx,%%rdi,4), %%zmm9, %%zmm9      \n\t"
            "vaddps               0(%%rcx,%%r10,4), %%zmm10, %%zmm10    \n\t"
            "vaddps              64(%%rcx,%%r10,4), %%zmm11, %%zmm11    \n\t"
            "vaddps             128(%%rcx,%%r10,4), %%zmm12, %%zmm12    \n\t"
            "vaddps             192(%%rcx,%%r10,4), %%zmm13, %%zmm13    \n\t"
            "vaddps             256(%%rcx,%%r10,4), %%zmm14, %%zmm14    \n\t"
            "vaddps               0(%%rcx,%%r11,4), %%zmm15, %%zmm15    \n\t"
            "vaddps              64(%%rcx,%%r11,4), %%zmm16, %%zmm16    \n\t"
            "vaddps             128(%%rcx,%%r11,4), %%zmm17, %%zmm17    \n\t"
            "vaddps             192(%%rcx,%%r11,4), %%zmm18, %%zmm18    \n\t"
            "vaddps             256(%%rcx,%%r11,4), %%zmm19, %%zmm19    \n\t"
            "vaddps               0(%%rcx,%%r12,4), %%zmm20, %%zmm20    \n\t"
            "vaddps              64(%%rcx,%%r12,4), %%zmm21, %%zmm21    \n\t"
            "vaddps             128(%%rcx,%%r12,4), %%zmm22, %%zmm22    \n\t"
            "vaddps             192(%%rcx,%%r12,4), %%zmm23, %%zmm23    \n\t"
            "vaddps             256(%%rcx,%%r12,4), %%zmm24, %%zmm24    \n\t"
            "                                                   \n\t"
            "vmovaps            %%zmm0,    0(%%rcx)             \n\t"
            "vmovaps            %%zmm1,   64(%%rcx)             \n\t"
            "vmovaps            %%zmm2,  128(%%rcx)             \n\t"
            "vmovaps            %%zmm3,  192(%%rcx)             \n\t"
            "vmovaps            %%zmm4,  256(%%rcx)             \n\t"
            "vmovaps            %%zmm5,    0(%%rcx,%%rdi,4)     \n\t"
            "vmovaps            %%zmm6,   64(%%rcx,%%rdi,4)     \n\t"
            "vmovaps            %%zmm7,  128(%%rcx,%%rdi,4)     \n\t"
            "vmovaps            %%zmm8,  192(%%rcx,%%rdi,4)     \n\t"
            "vmovaps            %%zmm9,  256(%%rcx,%%rdi,4)     \n\t"
            "vmovaps            %%zmm10,   0(%%rcx,%%r10,4)     \n\t"
            "vmovaps            %%zmm11,  64(%%rcx,%%r10,4)     \n\t"
            "vmovaps            %%zmm12, 128(%%rcx,%%r10,4)     \n\t"
            "vmovaps            %%zmm13, 192(%%rcx,%%r10,4)     \n\t"
            "vmovaps            %%zmm14, 256(%%rcx,%%r10,4)     \n\t"
            "vmovaps            %%zmm15,   0(%%rcx,%%r11,4)     \n\t"
            "vmovaps            %%zmm16,  64(%%rcx,%%r11,4)     \n\t"
            "vmovaps            %%zmm17, 128(%%rcx,%%r11,4)     \n\t"
            "vmovaps            %%zmm18, 192(%%rcx,%%r11,4)     \n\t"
            "vmovaps            %%zmm19, 256(%%rcx,%%r11,4)     \n\t"
            "vmovaps            %%zmm20,   0(%%rcx,%%r12,4)     \n\t"
            "vmovaps            %%zmm21,  64(%%rcx,%%r12,4)     \n\t"
            "vmovaps            %%zmm22, 128(%%rcx,%%r12,4)     \n\t"
            "vmovaps            %%zmm23, 192(%%rcx,%%r12,4)     \n\t"
            "vmovaps            %%zmm24, 256(%%rcx,%%r12,4)     \n\t"
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
            , "r8", "r9", "r10", "r11", "r12" //, "r13", "r14", "r15"
            , "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"
            , "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14"
            , "xmm15", "xmm16", "xmm17", "xmm18", "xmm19", "xmm20", "xmm21"
            , "xmm22", "xmm23", "xmm24", "xmm25", "xmm26", "xmm27", "xmm28"
            , "xmm29", "xmm30", "xmm31"
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

struct register_avx512_5x5asmpf_unroll { 
    enum : int {
        BLOCK_M = 5,
        BLOCK_N = 16 * 5,
    };

    // 5x80 matrix multiplication
    static void matmul_register(
        int M, int N, int K, float *A, int lda, float *B, int ldb,
        float *C, int ldc)
    {
        register_avx512_5x5::matmul_register(
            M, N, K, A, lda, B, ldb, C, ldc);
    }

    static NOINLINE void matmul_register_packed(
        int M, int N, int K, float *A, int lda, float *B, int ldb,
        float *C, int ldc)
    {
        assert(M % BLOCK_M == 0);
        assert(N % BLOCK_N == 0);
        assert(lda == BLOCK_M);
        assert(ldb == BLOCK_N);

#ifdef _MSC_VER
        register_avx512_5x5::matmul_register_packed(
            M, N, K, A, lda, B, ldb, C, ldc);
#else
        int64_t K64 = K;
        int64_t ldc64 = ldc;

        __asm__ __volatile__ (
            "                                                   \n\t"
            "vzeroall                                           \n\t"
            "vpxord             %%zmm16, %%zmm16, %%zmm16       \n\t"
            "vpxord             %%zmm17, %%zmm17, %%zmm17       \n\t"
            "vpxord             %%zmm18, %%zmm18, %%zmm18       \n\t"
            "vpxord             %%zmm19, %%zmm19, %%zmm19       \n\t"
            "vpxord             %%zmm20, %%zmm20, %%zmm20       \n\t"
            "vpxord             %%zmm21, %%zmm21, %%zmm21       \n\t"
            "vpxord             %%zmm22, %%zmm22, %%zmm22       \n\t"
            "vpxord             %%zmm23, %%zmm23, %%zmm23       \n\t"
            "vpxord             %%zmm24, %%zmm24, %%zmm24       \n\t"
            "vpxord             %%zmm25, %%zmm25, %%zmm25       \n\t"
            "vpxord             %%zmm26, %%zmm26, %%zmm26       \n\t"
            "vpxord             %%zmm27, %%zmm27, %%zmm27       \n\t"
            "vpxord             %%zmm28, %%zmm28, %%zmm28       \n\t"
            "vpxord             %%zmm29, %%zmm29, %%zmm29       \n\t"
            "vpxord             %%zmm30, %%zmm30, %%zmm30       \n\t"
            "vpxord             %%zmm31, %%zmm31, %%zmm31       \n\t"
            "                                                   \n\t"
            "mov                %0, %%rax                       \n\t" // A
            "mov                %1, %%rbx                       \n\t" // B
            "mov                %2, %%rcx                       \n\t" // C
            "mov                %3, %%rsi                       \n\t" // K
            "mov                %4, %%rdi                       \n\t" // ldc
            "                                                   \n\t"
            "lea                (,%%rdi,2), %%r10               \n\t" // ldc*2
            "lea                (%%r10,%%rdi), %%r11            \n\t" // ldc*3
            "lea                (,%%rdi,4), %%r12               \n\t" // ldc*4
            "                                                   \n\t"
            "prefetcht0           0(%%rcx)                      \n\t"
            "prefetcht0          64(%%rcx)                      \n\t"
            "prefetcht0         128(%%rcx)                      \n\t"
            "prefetcht0         192(%%rcx)                      \n\t"
            "prefetcht0         256(%%rcx)                      \n\t"
            "prefetcht0           0(%%rcx,%%rdi,4)              \n\t"
            "prefetcht0          64(%%rcx,%%rdi,4)              \n\t"
            "prefetcht0         128(%%rcx,%%rdi,4)              \n\t"
            "prefetcht0         192(%%rcx,%%rdi,4)              \n\t"
            "prefetcht0         256(%%rcx,%%rdi,4)              \n\t"
            "prefetcht0           0(%%rcx,%%r10,4)              \n\t"
            "prefetcht0          64(%%rcx,%%r10,4)              \n\t"
            "prefetcht0         128(%%rcx,%%r10,4)              \n\t"
            "prefetcht0         192(%%rcx,%%r10,4)              \n\t"
            "prefetcht0         256(%%rcx,%%r10,4)              \n\t"
            "prefetcht0           0(%%rcx,%%r11,4)              \n\t"
            "prefetcht0          64(%%rcx,%%r11,4)              \n\t"
            "prefetcht0         128(%%rcx,%%r11,4)              \n\t"
            "prefetcht0         192(%%rcx,%%r11,4)              \n\t"
            "prefetcht0         256(%%rcx,%%r11,4)              \n\t"
            "prefetcht0           0(%%rcx,%%r12,4)              \n\t"
            "prefetcht0          64(%%rcx,%%r12,4)              \n\t"
            "prefetcht0         128(%%rcx,%%r12,4)              \n\t"
            "prefetcht0         192(%%rcx,%%r12,4)              \n\t"
            "prefetcht0         256(%%rcx,%%r12,4)              \n\t"
            "                                                   \n\t"
            "lea                (,%%rsi,4), %%r8                \n\t" //  4*rsi
            "lea                (%%rax,%%r8,4), %%r9            \n\t" // 16*rsi
            "add                %%r8, %%r9                      \n\t" // 20*rsi
            "                                                   \n\t"
            ".KLOOP_5x5PF_UNROLL:                               \n\t"
            "                                                   \n\t"
            "cmp                %%rax, %%r9                     \n\t"
            "je                 .KLOOP_EXIT_5x5PF_UNROLL        \n\t"
            "                                                   \n\t"
            "vmovaps            0 * 64(%%rbx), %%zmm25          \n\t"
            "vmovaps            1 * 64(%%rbx), %%zmm26          \n\t"
            "vmovaps            2 * 64(%%rbx), %%zmm27          \n\t"
            "vmovaps            3 * 64(%%rbx), %%zmm28          \n\t"
            "vmovaps            4 * 64(%%rbx), %%zmm29          \n\t"
            "                                                   \n\t"
            "vbroadcastss       0 * 4(%%rax), %%zmm30           \n\t"
            "vbroadcastss       1 * 4(%%rax), %%zmm31           \n\t"
            "vfmadd231ps        %%zmm30, %%zmm25, %%zmm0        \n\t"
            "vfmadd231ps        %%zmm30, %%zmm26, %%zmm1        \n\t"
            "vfmadd231ps        %%zmm30, %%zmm27, %%zmm2        \n\t"
            "vfmadd231ps        %%zmm30, %%zmm28, %%zmm3        \n\t"
            "vfmadd231ps        %%zmm30, %%zmm29, %%zmm4        \n\t"
            "vfmadd231ps        %%zmm31, %%zmm25, %%zmm5        \n\t"
            "vfmadd231ps        %%zmm31, %%zmm26, %%zmm6        \n\t"
            "vfmadd231ps        %%zmm31, %%zmm27, %%zmm7        \n\t"
            "vfmadd231ps        %%zmm31, %%zmm28, %%zmm8        \n\t"
            "vfmadd231ps        %%zmm31, %%zmm29, %%zmm9        \n\t"
            "                                                   \n\t"
            "vbroadcastss       2 * 4(%%rax), %%zmm30           \n\t"
            "vbroadcastss       3 * 4(%%rax), %%zmm31           \n\t"
            "vfmadd231ps        %%zmm30, %%zmm25, %%zmm10       \n\t"
            "vfmadd231ps        %%zmm30, %%zmm26, %%zmm11       \n\t"
            "vfmadd231ps        %%zmm30, %%zmm27, %%zmm12       \n\t"
            "vfmadd231ps        %%zmm30, %%zmm28, %%zmm13       \n\t"
            "vfmadd231ps        %%zmm30, %%zmm29, %%zmm14       \n\t"
            "vfmadd231ps        %%zmm31, %%zmm25, %%zmm15       \n\t"
            "vfmadd231ps        %%zmm31, %%zmm26, %%zmm16       \n\t"
            "vfmadd231ps        %%zmm31, %%zmm27, %%zmm17       \n\t"
            "vfmadd231ps        %%zmm31, %%zmm28, %%zmm18       \n\t"
            "vfmadd231ps        %%zmm31, %%zmm29, %%zmm19       \n\t"
            "                                                   \n\t"
            "vbroadcastss       4 * 4(%%rax), %%zmm30           \n\t"
            "vbroadcastss       5 * 4(%%rax), %%zmm31           \n\t"
            "vfmadd231ps        %%zmm30, %%zmm25, %%zmm20       \n\t"
            "vfmadd231ps        %%zmm30, %%zmm26, %%zmm21       \n\t"
            "vfmadd231ps        %%zmm30, %%zmm27, %%zmm22       \n\t"
            "vfmadd231ps        %%zmm30, %%zmm28, %%zmm23       \n\t"
            "vfmadd231ps        %%zmm30, %%zmm29, %%zmm24       \n\t"
            "                                                   \n\t"
            "vmovaps            5 * 64(%%rbx), %%zmm25          \n\t"
            "vmovaps            6 * 64(%%rbx), %%zmm26          \n\t"
            "vmovaps            7 * 64(%%rbx), %%zmm27          \n\t"
            "vmovaps            8 * 64(%%rbx), %%zmm28          \n\t"
            "vmovaps            9 * 64(%%rbx), %%zmm29          \n\t"
            "                                                   \n\t"
            "vfmadd231ps        %%zmm31, %%zmm25, %%zmm0        \n\t"
            "vfmadd231ps        %%zmm31, %%zmm26, %%zmm1        \n\t"
            "vfmadd231ps        %%zmm31, %%zmm27, %%zmm2        \n\t"
            "vfmadd231ps        %%zmm31, %%zmm28, %%zmm3        \n\t"
            "vfmadd231ps        %%zmm31, %%zmm29, %%zmm4        \n\t"
            "                                                   \n\t"
            "vbroadcastss       6 * 4(%%rax), %%zmm30           \n\t"
            "vbroadcastss       7 * 4(%%rax), %%zmm31           \n\t"
            "vfmadd231ps        %%zmm30, %%zmm25, %%zmm5        \n\t"
            "vfmadd231ps        %%zmm30, %%zmm26, %%zmm6        \n\t"
            "vfmadd231ps        %%zmm30, %%zmm27, %%zmm7        \n\t"
            "vfmadd231ps        %%zmm30, %%zmm28, %%zmm8        \n\t"
            "vfmadd231ps        %%zmm30, %%zmm29, %%zmm9        \n\t"
            "vfmadd231ps        %%zmm31, %%zmm25, %%zmm10       \n\t"
            "vfmadd231ps        %%zmm31, %%zmm26, %%zmm11       \n\t"
            "vfmadd231ps        %%zmm31, %%zmm27, %%zmm12       \n\t"
            "vfmadd231ps        %%zmm31, %%zmm28, %%zmm13       \n\t"
            "vfmadd231ps        %%zmm31, %%zmm29, %%zmm14       \n\t"
            "                                                   \n\t"
            "vbroadcastss       8 * 4(%%rax), %%zmm30           \n\t"
            "vbroadcastss       9 * 4(%%rax), %%zmm31           \n\t"
            "vfmadd231ps        %%zmm30, %%zmm25, %%zmm15       \n\t"
            "vfmadd231ps        %%zmm30, %%zmm26, %%zmm16       \n\t"
            "vfmadd231ps        %%zmm30, %%zmm27, %%zmm17       \n\t"
            "vfmadd231ps        %%zmm30, %%zmm28, %%zmm18       \n\t"
            "vfmadd231ps        %%zmm30, %%zmm29, %%zmm19       \n\t"
            "vfmadd231ps        %%zmm31, %%zmm25, %%zmm20       \n\t"
            "vfmadd231ps        %%zmm31, %%zmm26, %%zmm21       \n\t"
            "vfmadd231ps        %%zmm31, %%zmm27, %%zmm22       \n\t"
            "vfmadd231ps        %%zmm31, %%zmm28, %%zmm23       \n\t"
            "vfmadd231ps        %%zmm31, %%zmm29, %%zmm24       \n\t"
            "                                                   \n\t"
            "add                $40, %%rax                      \n\t"
            "add                $640, %%rbx                     \n\t"
            "jmp                .KLOOP_5x5PF_UNROLL             \n\t"
            "                                                   \n\t"
            ".KLOOP_EXIT_5x5PF_UNROLL:                          \n\t"
            "                                                   \n\t"
            "vaddps               0(%%rcx), %%zmm0, %%zmm0      \n\t"
            "vaddps              64(%%rcx), %%zmm1, %%zmm1      \n\t"
            "vaddps             128(%%rcx), %%zmm2, %%zmm2      \n\t"
            "vaddps             192(%%rcx), %%zmm3, %%zmm3      \n\t"
            "vaddps             256(%%rcx), %%zmm4, %%zmm4      \n\t"
            "vaddps               0(%%rcx,%%rdi,4), %%zmm5, %%zmm5      \n\t"
            "vaddps              64(%%rcx,%%rdi,4), %%zmm6, %%zmm6      \n\t"
            "vaddps             128(%%rcx,%%rdi,4), %%zmm7, %%zmm7      \n\t"
            "vaddps             192(%%rcx,%%rdi,4), %%zmm8, %%zmm8      \n\t"
            "vaddps             256(%%rcx,%%rdi,4), %%zmm9, %%zmm9      \n\t"
            "vaddps               0(%%rcx,%%r10,4), %%zmm10, %%zmm10    \n\t"
            "vaddps              64(%%rcx,%%r10,4), %%zmm11, %%zmm11    \n\t"
            "vaddps             128(%%rcx,%%r10,4), %%zmm12, %%zmm12    \n\t"
            "vaddps             192(%%rcx,%%r10,4), %%zmm13, %%zmm13    \n\t"
            "vaddps             256(%%rcx,%%r10,4), %%zmm14, %%zmm14    \n\t"
            "vaddps               0(%%rcx,%%r11,4), %%zmm15, %%zmm15    \n\t"
            "vaddps              64(%%rcx,%%r11,4), %%zmm16, %%zmm16    \n\t"
            "vaddps             128(%%rcx,%%r11,4), %%zmm17, %%zmm17    \n\t"
            "vaddps             192(%%rcx,%%r11,4), %%zmm18, %%zmm18    \n\t"
            "vaddps             256(%%rcx,%%r11,4), %%zmm19, %%zmm19    \n\t"
            "vaddps               0(%%rcx,%%r12,4), %%zmm20, %%zmm20    \n\t"
            "vaddps              64(%%rcx,%%r12,4), %%zmm21, %%zmm21    \n\t"
            "vaddps             128(%%rcx,%%r12,4), %%zmm22, %%zmm22    \n\t"
            "vaddps             192(%%rcx,%%r12,4), %%zmm23, %%zmm23    \n\t"
            "vaddps             256(%%rcx,%%r12,4), %%zmm24, %%zmm24    \n\t"
            "                                                   \n\t"
            "vmovaps            %%zmm0,    0(%%rcx)             \n\t"
            "vmovaps            %%zmm1,   64(%%rcx)             \n\t"
            "vmovaps            %%zmm2,  128(%%rcx)             \n\t"
            "vmovaps            %%zmm3,  192(%%rcx)             \n\t"
            "vmovaps            %%zmm4,  256(%%rcx)             \n\t"
            "vmovaps            %%zmm5,    0(%%rcx,%%rdi,4)     \n\t"
            "vmovaps            %%zmm6,   64(%%rcx,%%rdi,4)     \n\t"
            "vmovaps            %%zmm7,  128(%%rcx,%%rdi,4)     \n\t"
            "vmovaps            %%zmm8,  192(%%rcx,%%rdi,4)     \n\t"
            "vmovaps            %%zmm9,  256(%%rcx,%%rdi,4)     \n\t"
            "vmovaps            %%zmm10,   0(%%rcx,%%r10,4)     \n\t"
            "vmovaps            %%zmm11,  64(%%rcx,%%r10,4)     \n\t"
            "vmovaps            %%zmm12, 128(%%rcx,%%r10,4)     \n\t"
            "vmovaps            %%zmm13, 192(%%rcx,%%r10,4)     \n\t"
            "vmovaps            %%zmm14, 256(%%rcx,%%r10,4)     \n\t"
            "vmovaps            %%zmm15,   0(%%rcx,%%r11,4)     \n\t"
            "vmovaps            %%zmm16,  64(%%rcx,%%r11,4)     \n\t"
            "vmovaps            %%zmm17, 128(%%rcx,%%r11,4)     \n\t"
            "vmovaps            %%zmm18, 192(%%rcx,%%r11,4)     \n\t"
            "vmovaps            %%zmm19, 256(%%rcx,%%r11,4)     \n\t"
            "vmovaps            %%zmm20,   0(%%rcx,%%r12,4)     \n\t"
            "vmovaps            %%zmm21,  64(%%rcx,%%r12,4)     \n\t"
            "vmovaps            %%zmm22, 128(%%rcx,%%r12,4)     \n\t"
            "vmovaps            %%zmm23, 192(%%rcx,%%r12,4)     \n\t"
            "vmovaps            %%zmm24, 256(%%rcx,%%r12,4)     \n\t"
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
            , "r8", "r9", "r10", "r11", "r12" //, "r13", "r14", "r15"
            , "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"
            , "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14"
            , "xmm15", "xmm16", "xmm17", "xmm18", "xmm19", "xmm20", "xmm21"
            , "xmm22", "xmm23", "xmm24", "xmm25", "xmm26", "xmm27", "xmm28"
            , "xmm29", "xmm30", "xmm31"
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

struct register_avx512_5x5asmpf_ebcast { 
    enum : int {
        BLOCK_M = 5,
        BLOCK_N = 16 * 5,
    };

    // 5x80 matrix multiplication
    static void matmul_register(
        int M, int N, int K, float *A, int lda, float *B, int ldb,
        float *C, int ldc)
    {
        register_avx512_5x5::matmul_register(
            M, N, K, A, lda, B, ldb, C, ldc);
    }

    static NOINLINE void matmul_register_packed(
        int M, int N, int K, float *A, int lda, float *B, int ldb,
        float *C, int ldc)
    {
        assert(M % BLOCK_M == 0);
        assert(N % BLOCK_N == 0);
        assert(lda == BLOCK_M);
        assert(ldb == BLOCK_N);

#ifdef _MSC_VER
        register_avx512_5x5::matmul_register_packed(
            M, N, K, A, lda, B, ldb, C, ldc);
#else
        int64_t K64 = K;
        int64_t ldc64 = ldc;

        __asm__ __volatile__ (
            "                                                   \n\t"
            "vzeroall                                           \n\t"
            "vpxord             %%zmm16, %%zmm16, %%zmm16       \n\t"
            "vpxord             %%zmm17, %%zmm17, %%zmm17       \n\t"
            "vpxord             %%zmm18, %%zmm18, %%zmm18       \n\t"
            "vpxord             %%zmm19, %%zmm19, %%zmm19       \n\t"
            "vpxord             %%zmm20, %%zmm20, %%zmm20       \n\t"
            "vpxord             %%zmm21, %%zmm21, %%zmm21       \n\t"
            "vpxord             %%zmm22, %%zmm22, %%zmm22       \n\t"
            "vpxord             %%zmm23, %%zmm23, %%zmm23       \n\t"
            "vpxord             %%zmm24, %%zmm24, %%zmm24       \n\t"
            "vpxord             %%zmm25, %%zmm25, %%zmm25       \n\t"
            "vpxord             %%zmm26, %%zmm26, %%zmm26       \n\t"
            "vpxord             %%zmm27, %%zmm27, %%zmm27       \n\t"
            "vpxord             %%zmm28, %%zmm28, %%zmm28       \n\t"
            "vpxord             %%zmm29, %%zmm29, %%zmm29       \n\t"
            "vpxord             %%zmm30, %%zmm30, %%zmm30       \n\t"
            "vpxord             %%zmm31, %%zmm31, %%zmm31       \n\t"
            "                                                   \n\t"
            "mov                %0, %%rax                       \n\t" // A
            "mov                %1, %%rbx                       \n\t" // B
            "mov                %2, %%rcx                       \n\t" // C
            "mov                %3, %%rsi                       \n\t" // K
            "mov                %4, %%rdi                       \n\t" // ldc
            "                                                   \n\t"
            "lea                (,%%rdi,2), %%r10               \n\t" // ldc*2
            "lea                (%%r10,%%rdi), %%r11            \n\t" // ldc*3
            "lea                (,%%rdi,4), %%r12               \n\t" // ldc*4
            "                                                   \n\t"
            "prefetcht0           0(%%rcx)                      \n\t"
            "prefetcht0          64(%%rcx)                      \n\t"
            "prefetcht0         128(%%rcx)                      \n\t"
            "prefetcht0         192(%%rcx)                      \n\t"
            "prefetcht0         256(%%rcx)                      \n\t"
            "prefetcht0           0(%%rcx,%%rdi,4)              \n\t"
            "prefetcht0          64(%%rcx,%%rdi,4)              \n\t"
            "prefetcht0         128(%%rcx,%%rdi,4)              \n\t"
            "prefetcht0         192(%%rcx,%%rdi,4)              \n\t"
            "prefetcht0         256(%%rcx,%%rdi,4)              \n\t"
            "prefetcht0           0(%%rcx,%%r10,4)              \n\t"
            "prefetcht0          64(%%rcx,%%r10,4)              \n\t"
            "prefetcht0         128(%%rcx,%%r10,4)              \n\t"
            "prefetcht0         192(%%rcx,%%r10,4)              \n\t"
            "prefetcht0         256(%%rcx,%%r10,4)              \n\t"
            "prefetcht0           0(%%rcx,%%r11,4)              \n\t"
            "prefetcht0          64(%%rcx,%%r11,4)              \n\t"
            "prefetcht0         128(%%rcx,%%r11,4)              \n\t"
            "prefetcht0         192(%%rcx,%%r11,4)              \n\t"
            "prefetcht0         256(%%rcx,%%r11,4)              \n\t"
            "prefetcht0           0(%%rcx,%%r12,4)              \n\t"
            "prefetcht0          64(%%rcx,%%r12,4)              \n\t"
            "prefetcht0         128(%%rcx,%%r12,4)              \n\t"
            "prefetcht0         192(%%rcx,%%r12,4)              \n\t"
            "prefetcht0         256(%%rcx,%%r12,4)              \n\t"
            "                                                   \n\t"
            "lea                (,%%rsi,4), %%r8                \n\t" //  4*rsi
            "lea                (%%rax,%%r8,4), %%r9            \n\t" // 16*rsi
            "add                %%r8, %%r9                      \n\t" // 20*rsi
            "                                                   \n\t"
            "vmovaps            0 * 64(%%rbx), %%zmm25          \n\t"
            "vmovaps            1 * 64(%%rbx), %%zmm26          \n\t"
            "vmovaps            2 * 64(%%rbx), %%zmm27          \n\t"
            "vmovaps            3 * 64(%%rbx), %%zmm28          \n\t"
            "                                                   \n\t"
            ".KLOOP_5x5PF_EBCAST:                               \n\t"
            "                                                   \n\t"
            "cmp                %%rax, %%r9                     \n\t"
            "je                 .KLOOP_EXIT_5x5PF_EBCAST        \n\t"
            "                                                   \n\t"
            "vmovaps            4 * 64(%%rbx), %%zmm29          \n\t"
            "vfmadd231ps        0 * 4(%%rax)%{1to16%}, %%zmm25, %%zmm0  \n\t"
            "vfmadd231ps        1 * 4(%%rax)%{1to16%}, %%zmm25, %%zmm5  \n\t"
            "vfmadd231ps        2 * 4(%%rax)%{1to16%}, %%zmm25, %%zmm10 \n\t"
            "vfmadd231ps        3 * 4(%%rax)%{1to16%}, %%zmm25, %%zmm15 \n\t"
            "vfmadd231ps        4 * 4(%%rax)%{1to16%}, %%zmm25, %%zmm20 \n\t"
            
            "vmovaps            5 * 64(%%rbx), %%zmm25                  \n\t"
            "vfmadd231ps        0 * 4(%%rax)%{1to16%}, %%zmm26, %%zmm1  \n\t"
            "vfmadd231ps        1 * 4(%%rax)%{1to16%}, %%zmm26, %%zmm6  \n\t"
            "vfmadd231ps        2 * 4(%%rax)%{1to16%}, %%zmm26, %%zmm11 \n\t"
            "vfmadd231ps        3 * 4(%%rax)%{1to16%}, %%zmm26, %%zmm16 \n\t"
            "vfmadd231ps        4 * 4(%%rax)%{1to16%}, %%zmm26, %%zmm21 \n\t"
            "                                                           \n\t"
            "vmovaps            6 * 64(%%rbx), %%zmm26                  \n\t"
            "vfmadd231ps        0 * 4(%%rax)%{1to16%}, %%zmm27, %%zmm2  \n\t"
            "vfmadd231ps        1 * 4(%%rax)%{1to16%}, %%zmm27, %%zmm7  \n\t"
            "vfmadd231ps        2 * 4(%%rax)%{1to16%}, %%zmm27, %%zmm12 \n\t"
            "vfmadd231ps        3 * 4(%%rax)%{1to16%}, %%zmm27, %%zmm17 \n\t"
            "vfmadd231ps        4 * 4(%%rax)%{1to16%}, %%zmm27, %%zmm22 \n\t"
            "                                                           \n\t"
            "vmovaps            7 * 64(%%rbx), %%zmm27                  \n\t"
            "vfmadd231ps        0 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm3  \n\t"
            "vfmadd231ps        1 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm8  \n\t"
            "vfmadd231ps        2 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm13 \n\t"
            "vfmadd231ps        3 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm18 \n\t"
            "vfmadd231ps        4 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm23 \n\t"
            "                                                           \n\t"
            "vmovaps            8 * 64(%%rbx), %%zmm28                  \n\t"
            "vfmadd231ps        0 * 4(%%rax)%{1to16%}, %%zmm29, %%zmm4  \n\t"
            "vfmadd231ps        1 * 4(%%rax)%{1to16%}, %%zmm29, %%zmm9  \n\t"
            "vfmadd231ps        2 * 4(%%rax)%{1to16%}, %%zmm29, %%zmm14 \n\t"
            "vfmadd231ps        3 * 4(%%rax)%{1to16%}, %%zmm29, %%zmm19 \n\t"
            "vfmadd231ps        4 * 4(%%rax)%{1to16%}, %%zmm29, %%zmm24 \n\t"
            "                                                           \n\t"
            "vmovaps            9 * 64(%%rbx), %%zmm29                  \n\t"
            "vfmadd231ps        5 * 4(%%rax)%{1to16%}, %%zmm25, %%zmm0  \n\t"
            "vfmadd231ps        6 * 4(%%rax)%{1to16%}, %%zmm25, %%zmm5  \n\t"
            "vfmadd231ps        7 * 4(%%rax)%{1to16%}, %%zmm25, %%zmm10 \n\t"
            "vfmadd231ps        8 * 4(%%rax)%{1to16%}, %%zmm25, %%zmm15 \n\t"
            "vfmadd231ps        9 * 4(%%rax)%{1to16%}, %%zmm25, %%zmm20 \n\t"
            "                                                           \n\t"
            "vmovaps            10 * 64(%%rbx), %%zmm25                 \n\t"
            "vfmadd231ps        5 * 4(%%rax)%{1to16%}, %%zmm26, %%zmm1  \n\t"
            "vfmadd231ps        6 * 4(%%rax)%{1to16%}, %%zmm26, %%zmm6  \n\t"
            "vfmadd231ps        7 * 4(%%rax)%{1to16%}, %%zmm26, %%zmm11 \n\t"
            "vfmadd231ps        8 * 4(%%rax)%{1to16%}, %%zmm26, %%zmm16 \n\t"
            "vfmadd231ps        9 * 4(%%rax)%{1to16%}, %%zmm26, %%zmm21 \n\t"
            "                                                           \n\t"
            "vmovaps            11 * 64(%%rbx), %%zmm26                 \n\t"
            "vfmadd231ps        5 * 4(%%rax)%{1to16%}, %%zmm27, %%zmm2  \n\t"
            "vfmadd231ps        6 * 4(%%rax)%{1to16%}, %%zmm27, %%zmm7  \n\t"
            "vfmadd231ps        7 * 4(%%rax)%{1to16%}, %%zmm27, %%zmm12 \n\t"
            "vfmadd231ps        8 * 4(%%rax)%{1to16%}, %%zmm27, %%zmm17 \n\t"
            "vfmadd231ps        9 * 4(%%rax)%{1to16%}, %%zmm27, %%zmm22 \n\t"
            "                                                           \n\t"
            "vmovaps            12 * 64(%%rbx), %%zmm27                 \n\t"
            "vfmadd231ps        5 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm3  \n\t"
            "vfmadd231ps        6 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm8  \n\t"
            "vfmadd231ps        7 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm13 \n\t"
            "vfmadd231ps        8 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm18 \n\t"
            "vfmadd231ps        9 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm23 \n\t"
            "                                                           \n\t"
            "vmovaps            13 * 64(%%rbx), %%zmm28                 \n\t"
            "vfmadd231ps        5 * 4(%%rax)%{1to16%}, %%zmm29, %%zmm4  \n\t"
            "vfmadd231ps        6 * 4(%%rax)%{1to16%}, %%zmm29, %%zmm9  \n\t"
            "vfmadd231ps        7 * 4(%%rax)%{1to16%}, %%zmm29, %%zmm14 \n\t"
            "vfmadd231ps        8 * 4(%%rax)%{1to16%}, %%zmm29, %%zmm19 \n\t"
            "vfmadd231ps        9 * 4(%%rax)%{1to16%}, %%zmm29, %%zmm24 \n\t"
            "                                                   \n\t"
            "add                $40, %%rax                      \n\t"
            "add                $640, %%rbx                     \n\t"
            "jmp                .KLOOP_5x5PF_EBCAST             \n\t"
            "                                                   \n\t"
            ".KLOOP_EXIT_5x5PF_EBCAST:                          \n\t"
            "                                                   \n\t"
            "vaddps               0(%%rcx), %%zmm0, %%zmm0      \n\t"
            "vaddps              64(%%rcx), %%zmm1, %%zmm1      \n\t"
            "vaddps             128(%%rcx), %%zmm2, %%zmm2      \n\t"
            "vaddps             192(%%rcx), %%zmm3, %%zmm3      \n\t"
            "vaddps             256(%%rcx), %%zmm4, %%zmm4      \n\t"
            "vaddps               0(%%rcx,%%rdi,4), %%zmm5, %%zmm5      \n\t"
            "vaddps              64(%%rcx,%%rdi,4), %%zmm6, %%zmm6      \n\t"
            "vaddps             128(%%rcx,%%rdi,4), %%zmm7, %%zmm7      \n\t"
            "vaddps             192(%%rcx,%%rdi,4), %%zmm8, %%zmm8      \n\t"
            "vaddps             256(%%rcx,%%rdi,4), %%zmm9, %%zmm9      \n\t"
            "vaddps               0(%%rcx,%%r10,4), %%zmm10, %%zmm10    \n\t"
            "vaddps              64(%%rcx,%%r10,4), %%zmm11, %%zmm11    \n\t"
            "vaddps             128(%%rcx,%%r10,4), %%zmm12, %%zmm12    \n\t"
            "vaddps             192(%%rcx,%%r10,4), %%zmm13, %%zmm13    \n\t"
            "vaddps             256(%%rcx,%%r10,4), %%zmm14, %%zmm14    \n\t"
            "vaddps               0(%%rcx,%%r11,4), %%zmm15, %%zmm15    \n\t"
            "vaddps              64(%%rcx,%%r11,4), %%zmm16, %%zmm16    \n\t"
            "vaddps             128(%%rcx,%%r11,4), %%zmm17, %%zmm17    \n\t"
            "vaddps             192(%%rcx,%%r11,4), %%zmm18, %%zmm18    \n\t"
            "vaddps             256(%%rcx,%%r11,4), %%zmm19, %%zmm19    \n\t"
            "vaddps               0(%%rcx,%%r12,4), %%zmm20, %%zmm20    \n\t"
            "vaddps              64(%%rcx,%%r12,4), %%zmm21, %%zmm21    \n\t"
            "vaddps             128(%%rcx,%%r12,4), %%zmm22, %%zmm22    \n\t"
            "vaddps             192(%%rcx,%%r12,4), %%zmm23, %%zmm23    \n\t"
            "vaddps             256(%%rcx,%%r12,4), %%zmm24, %%zmm24    \n\t"
            "                                                   \n\t"
            "vmovaps            %%zmm0,    0(%%rcx)             \n\t"
            "vmovaps            %%zmm1,   64(%%rcx)             \n\t"
            "vmovaps            %%zmm2,  128(%%rcx)             \n\t"
            "vmovaps            %%zmm3,  192(%%rcx)             \n\t"
            "vmovaps            %%zmm4,  256(%%rcx)             \n\t"
            "vmovaps            %%zmm5,    0(%%rcx,%%rdi,4)     \n\t"
            "vmovaps            %%zmm6,   64(%%rcx,%%rdi,4)     \n\t"
            "vmovaps            %%zmm7,  128(%%rcx,%%rdi,4)     \n\t"
            "vmovaps            %%zmm8,  192(%%rcx,%%rdi,4)     \n\t"
            "vmovaps            %%zmm9,  256(%%rcx,%%rdi,4)     \n\t"
            "vmovaps            %%zmm10,   0(%%rcx,%%r10,4)     \n\t"
            "vmovaps            %%zmm11,  64(%%rcx,%%r10,4)     \n\t"
            "vmovaps            %%zmm12, 128(%%rcx,%%r10,4)     \n\t"
            "vmovaps            %%zmm13, 192(%%rcx,%%r10,4)     \n\t"
            "vmovaps            %%zmm14, 256(%%rcx,%%r10,4)     \n\t"
            "vmovaps            %%zmm15,   0(%%rcx,%%r11,4)     \n\t"
            "vmovaps            %%zmm16,  64(%%rcx,%%r11,4)     \n\t"
            "vmovaps            %%zmm17, 128(%%rcx,%%r11,4)     \n\t"
            "vmovaps            %%zmm18, 192(%%rcx,%%r11,4)     \n\t"
            "vmovaps            %%zmm19, 256(%%rcx,%%r11,4)     \n\t"
            "vmovaps            %%zmm20,   0(%%rcx,%%r12,4)     \n\t"
            "vmovaps            %%zmm21,  64(%%rcx,%%r12,4)     \n\t"
            "vmovaps            %%zmm22, 128(%%rcx,%%r12,4)     \n\t"
            "vmovaps            %%zmm23, 192(%%rcx,%%r12,4)     \n\t"
            "vmovaps            %%zmm24, 256(%%rcx,%%r12,4)     \n\t"
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
            , "r8", "r9", "r10", "r11", "r12" //, "r13", "r14", "r15"
            , "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"
            , "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14"
            , "xmm15", "xmm16", "xmm17", "xmm18", "xmm19", "xmm20", "xmm21"
            , "xmm22", "xmm23", "xmm24", "xmm25", "xmm26", "xmm27", "xmm28"
            , "xmm29", "xmm30", "xmm31"
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

struct register_avx512_28x1asmpf_ebcast { 
    enum : int {
        BLOCK_M = 28,
        BLOCK_N = 16 * 1,
    };

    // 28x16 matrix multiplication
    static void matmul_register(
        int M, int N, int K, float *A, int lda, float *B, int ldb,
        float *C, int ldc)
    {
#if 0
        register_avx512_28x1::matmul_register(
            M, N, K, A, lda, B, ldb, C, ldc);
#else
        assert(0);
#endif
    }

    static NOINLINE void matmul_register_packed(
        int M, int N, int K, float *A, int lda, float *B, int ldb,
        float *C, int ldc)
    {
        assert(M % BLOCK_M == 0);
        assert(N % BLOCK_N == 0);
        assert(lda == BLOCK_M);
        assert(ldb == BLOCK_N);

#ifdef _MSC_VER
        register_avx512_28x1::matmul_register_packed(
            M, N, K, A, lda, B, ldb, C, ldc);
#else
        int64_t K64 = K;
        int64_t ldc64 = ldc;

        __asm__ __volatile__ (
            "                                                   \n\t"
            "vzeroall                                           \n\t"
            "vpxord             %%zmm16, %%zmm16, %%zmm16       \n\t"
            "vpxord             %%zmm17, %%zmm17, %%zmm17       \n\t"
            "vpxord             %%zmm18, %%zmm18, %%zmm18       \n\t"
            "vpxord             %%zmm19, %%zmm19, %%zmm19       \n\t"
            "vpxord             %%zmm20, %%zmm20, %%zmm20       \n\t"
            "vpxord             %%zmm21, %%zmm21, %%zmm21       \n\t"
            "vpxord             %%zmm22, %%zmm22, %%zmm22       \n\t"
            "vpxord             %%zmm23, %%zmm23, %%zmm23       \n\t"
            "vpxord             %%zmm24, %%zmm24, %%zmm24       \n\t"
            "vpxord             %%zmm25, %%zmm25, %%zmm25       \n\t"
            "vpxord             %%zmm26, %%zmm26, %%zmm26       \n\t"
            "vpxord             %%zmm27, %%zmm27, %%zmm27       \n\t"
            "vpxord             %%zmm28, %%zmm28, %%zmm28       \n\t"
            "vpxord             %%zmm29, %%zmm29, %%zmm29       \n\t"
            "vpxord             %%zmm30, %%zmm30, %%zmm30       \n\t"
            "vpxord             %%zmm31, %%zmm31, %%zmm31       \n\t"
            "                                                   \n\t"
            "mov                %0, %%rax                       \n\t" // A
            "mov                %1, %%rbx                       \n\t" // B
            "mov                %2, %%rcx                       \n\t" // C
            "mov                %3, %%rsi                       \n\t" // K
            "mov                %4, %%rdi                       \n\t" // ldc
            "                                            \n\t"
            "lea                (,%%rdi,4), %%r10        \n\t" // ldc*4*1
            "lea                (,%%r10,2), %%r11        \n\t" // ldc*4*2
            "lea                (%%r10,%%r11), %%r12     \n\t" // ldc*4*3
            "                                            \n\t"
            "prefetcht0         (%%rcx)                  \n\t" // C
            "prefetcht0         (%%rcx,%%r10)            \n\t"
            "prefetcht0         (%%rcx,%%r11)            \n\t"
            "prefetcht0         (%%rcx,%%r12)            \n\t"
            "lea                (%%rcx,%%r10,4), %%r13   \n\t" // C + ldc*4*4
            "prefetcht0         (%%r13)                  \n\t"
            "prefetcht0         (%%r13,%%r10)            \n\t"
            "prefetcht0         (%%r13,%%r11)            \n\t"
            "prefetcht0         (%%r13,%%r12)            \n\t"
            "lea                (%%r13,%%r10,4), %%r13   \n\t" // C + ldc*4*8
            "prefetcht0         (%%r13)                  \n\t"
            "prefetcht0         (%%r13,%%r10)            \n\t"
            "prefetcht0         (%%r13,%%r11)            \n\t"
            "prefetcht0         (%%r13,%%r12)            \n\t"
            "lea                (%%r13,%%r10,4), %%r13   \n\t" // C + ldc*4*12
            "prefetcht0         (%%r13)                  \n\t"
            "prefetcht0         (%%r13,%%r10)            \n\t"
            "prefetcht0         (%%r13,%%r11)            \n\t"
            "prefetcht0         (%%r13,%%r12)            \n\t"
            "lea                (%%r13,%%r10,4), %%r13   \n\t" // C + ldc*4*16
            "prefetcht0         (%%r13)                  \n\t"
            "prefetcht0         (%%r13,%%r10)            \n\t"
            "prefetcht0         (%%r13,%%r11)            \n\t"
            "prefetcht0         (%%r13,%%r12)            \n\t"
            "lea                (%%r13,%%r10,4), %%r13   \n\t" // C + ldc*4*20
            "prefetcht0         (%%r13)                  \n\t"
            "prefetcht0         (%%r13,%%r10)            \n\t"
            "prefetcht0         (%%r13,%%r11)            \n\t"
            "prefetcht0         (%%r13,%%r12)            \n\t"
            "lea                (%%r13,%%r10,4), %%r13   \n\t" // C + ldc*4*24
            "prefetcht0         (%%r13)                  \n\t"
            "prefetcht0         (%%r13,%%r10)            \n\t"
            "prefetcht0         (%%r13,%%r11)            \n\t"
            "prefetcht0         (%%r13,%%r12)            \n\t"
            "                                            \n\t"
            "lea                (,%%rsi,8), %%r8         \n\t" // 8*K
            "lea                (%%rbx,%%r8,8), %%r9     \n\t" // %rbx + 64*K
            "                                                            \n\t"
            ".KLOOP_28x1PF_EBCAST:                                       \n\t"
            "                                                            \n\t"
            "cmp                %%rbx, %%r9                              \n\t"
            "je                 .KLOOP_EXIT_28x1PF_EBCAST                \n\t"
            "                                                            \n\t"
            "vmovaps            0 * 64(%%rbx), %%zmm28                   \n\t"
            "                                                            \n\t"
            "vfmadd231ps         0 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm0  \n\t"
            "vfmadd231ps         1 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm1  \n\t"
            "vfmadd231ps         2 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm2  \n\t"
            "vfmadd231ps         3 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm3  \n\t"
            "vfmadd231ps         4 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm4  \n\t"
            "vfmadd231ps         5 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm5  \n\t"
            "vfmadd231ps         6 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm6  \n\t"
            "vfmadd231ps         7 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm7  \n\t"
            "vfmadd231ps         8 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm8  \n\t"
            "vfmadd231ps         9 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm9  \n\t"
            "vfmadd231ps        10 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm10 \n\t"
            "vfmadd231ps        11 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm11 \n\t"
            "vfmadd231ps        12 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm12 \n\t"
            "vfmadd231ps        13 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm13 \n\t"
            "vfmadd231ps        14 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm14 \n\t"
            "vfmadd231ps        15 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm15 \n\t"
            "vfmadd231ps        16 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm16 \n\t"
            "vfmadd231ps        17 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm17 \n\t"
            "vfmadd231ps        18 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm18 \n\t"
            "vfmadd231ps        19 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm19 \n\t"
            "vfmadd231ps        20 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm20 \n\t"
            "vfmadd231ps        21 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm21 \n\t"
            "vfmadd231ps        22 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm22 \n\t"
            "vfmadd231ps        23 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm23 \n\t"
            "vfmadd231ps        24 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm24 \n\t"
            "vfmadd231ps        25 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm25 \n\t"
            "vfmadd231ps        26 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm26 \n\t"
            "vfmadd231ps        27 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm27 \n\t"
            "                                                            \n\t"
            "add                $112, %%rax                              \n\t"
            "add                $64, %%rbx                               \n\t"
            "jmp                .KLOOP_28x1PF_EBCAST                     \n\t"
            "                                                           \n\t"
            ".KLOOP_EXIT_28x1PF_EBCAST:                                 \n\t"
            "                                                           \n\t"
            "vaddps             (%%rcx),         %%zmm0,  %%zmm0        \n\t"
            "vmovaps            %%zmm0,  (%%rcx)                        \n\t"
            "vaddps             (%%rcx,%%r10),   %%zmm1,  %%zmm1        \n\t"
            "vmovaps            %%zmm1,  (%%rcx,%%r10)                  \n\t"
            "vaddps             (%%rcx,%%r11),   %%zmm2,  %%zmm2        \n\t"
            "vmovaps            %%zmm2,  (%%rcx,%%r11)                  \n\t"
            "vaddps             (%%rcx,%%r12),   %%zmm3,  %%zmm3        \n\t"
            "vmovaps            %%zmm3,  (%%rcx,%%r12)                  \n\t"
            "                                                           \n\t"
            "lea                (%%rcx,%%r10,4), %%r13                  \n\t"
            "vaddps             (%%r13),         %%zmm4,  %%zmm4        \n\t"
            "vmovaps            %%zmm4,  (%%r13)                        \n\t"
            "vaddps             (%%r13,%%r10),   %%zmm5,  %%zmm5        \n\t"
            "vmovaps            %%zmm5,  (%%r13,%%r10)                  \n\t"
            "vaddps             (%%r13,%%r11),   %%zmm6,  %%zmm6        \n\t"
            "vmovaps            %%zmm6,  (%%r13,%%r11)                  \n\t"
            "vaddps             (%%r13,%%r12),   %%zmm7,  %%zmm7        \n\t"
            "vmovaps            %%zmm7,  (%%r13,%%r12)                  \n\t"
            "                                                           \n\t"
            "lea                (%%r13,%%r10,4), %%r13                  \n\t"
            "vaddps             (%%r13),         %%zmm8,  %%zmm8        \n\t"
            "vmovaps            %%zmm8,  (%%r13)                        \n\t"
            "vaddps             (%%r13,%%r10),   %%zmm9,  %%zmm9        \n\t"
            "vmovaps            %%zmm9,  (%%r13,%%r10)                  \n\t"
            "vaddps             (%%r13,%%r11),   %%zmm10, %%zmm10       \n\t"
            "vmovaps            %%zmm10, (%%r13,%%r11)                  \n\t"
            "vaddps             (%%r13,%%r12),   %%zmm11, %%zmm11       \n\t"
            "vmovaps            %%zmm11, (%%r13,%%r12)                  \n\t"
            "                                                           \n\t"
            "lea                (%%r13,%%r10,4), %%r13                  \n\t"
            "vaddps             (%%r13),         %%zmm12, %%zmm12       \n\t"
            "vmovaps            %%zmm12, (%%r13)                        \n\t"
            "vaddps             (%%r13,%%r10),   %%zmm13, %%zmm13       \n\t"
            "vmovaps            %%zmm13, (%%r13,%%r10)                  \n\t"
            "vaddps             (%%r13,%%r11),   %%zmm14, %%zmm14       \n\t"
            "vmovaps            %%zmm14, (%%r13,%%r11)                  \n\t"
            "vaddps             (%%r13,%%r12),   %%zmm15, %%zmm15       \n\t"
            "vmovaps            %%zmm15, (%%r13,%%r12)                  \n\t"
            "                                                           \n\t"
            "lea                (%%r13,%%r10,4), %%r13                  \n\t"
            "vaddps             (%%r13),         %%zmm16, %%zmm16       \n\t"
            "vmovaps            %%zmm16, (%%r13)                        \n\t"
            "vaddps             (%%r13,%%r10),   %%zmm17, %%zmm17       \n\t"
            "vmovaps            %%zmm17, (%%r13,%%r10)                  \n\t"
            "vaddps             (%%r13,%%r11),   %%zmm18, %%zmm18       \n\t"
            "vmovaps            %%zmm18, (%%r13,%%r11)                  \n\t"
            "vaddps             (%%r13,%%r12),   %%zmm19, %%zmm19       \n\t"
            "vmovaps            %%zmm19, (%%r13,%%r12)                  \n\t"
            "                                                           \n\t"
            "lea                (%%r13,%%r10,4), %%r13                  \n\t"
            "vaddps             (%%r13),         %%zmm20, %%zmm20       \n\t"
            "vmovaps            %%zmm20, (%%r13)                        \n\t"
            "vaddps             (%%r13,%%r10),   %%zmm21, %%zmm21       \n\t"
            "vmovaps            %%zmm21, (%%r13,%%r10)                  \n\t"
            "vaddps             (%%r13,%%r11),   %%zmm22, %%zmm22       \n\t"
            "vmovaps            %%zmm22, (%%r13,%%r11)                  \n\t"
            "vaddps             (%%r13,%%r12),   %%zmm23, %%zmm23       \n\t"
            "vmovaps            %%zmm23, (%%r13,%%r12)                  \n\t"
            "                                                           \n\t"
            "lea                (%%r13,%%r10,4), %%r13                  \n\t"
            "vaddps             (%%r13),         %%zmm24, %%zmm24       \n\t"
            "vmovaps            %%zmm24, (%%r13)                        \n\t"
            "vaddps             (%%r13,%%r10),   %%zmm25, %%zmm25       \n\t"
            "vmovaps            %%zmm25, (%%r13,%%r10)                  \n\t"
            "vaddps             (%%r13,%%r11),   %%zmm26, %%zmm26       \n\t"
            "vmovaps            %%zmm26, (%%r13,%%r11)                  \n\t"
            "vaddps             (%%r13,%%r12),   %%zmm27, %%zmm27       \n\t"
            "vmovaps            %%zmm27, (%%r13,%%r12)                  \n\t"
            "                                                           \n\t"
            : // output operands
            : // input operands
              "m" (A)
            , "m" (B)
            , "m" (C)
            , "m" (K64)
            , "m" (ldc64)
            : // clobbered registers
              "rax", "rbx", "rcx", "rdx", "rsi", "rdi"
            , "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"
            , "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"
            , "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14"
            , "xmm15", "xmm16", "xmm17", "xmm18", "xmm19", "xmm20", "xmm21"
            , "xmm22", "xmm23", "xmm24", "xmm25", "xmm26", "xmm27", "xmm28"
            , "xmm29", "xmm30", "xmm31"
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
struct register_avx512_28x1asmpf_ebcast_unr { 
    enum : int {
        BLOCK_M = 28,
        BLOCK_N = 16 * 1,
    };

    // 28x16 matrix multiplication
    static void matmul_register(
        int M, int N, int K, float *A, int lda, float *B, int ldb,
        float *C, int ldc)
    {
#if 0
        register_avx512_28x1::matmul_register(
            M, N, K, A, lda, B, ldb, C, ldc);
#else
        assert(0);
#endif
    }

    static NOINLINE void matmul_register_packed(
        int M, int N, int K, float *A, int lda, float *B, int ldb,
        float *C, int ldc)
    {
        assert(M % BLOCK_M == 0);
        assert(N % BLOCK_N == 0);
        assert(K % 2 == 0);
        assert(lda == BLOCK_M);
        assert(ldb == BLOCK_N);

#ifdef _MSC_VER
        register_avx512_28x1::matmul_register_packed(
            M, N, K, A, lda, B, ldb, C, ldc);
#else
        int64_t K64 = K;
        int64_t ldc64 = ldc;

        __asm__ __volatile__ (
            "                                                   \n\t"
            "vzeroall                                           \n\t"
            "vpxord             %%zmm16, %%zmm16, %%zmm16       \n\t"
            "vpxord             %%zmm17, %%zmm17, %%zmm17       \n\t"
            "vpxord             %%zmm18, %%zmm18, %%zmm18       \n\t"
            "vpxord             %%zmm19, %%zmm19, %%zmm19       \n\t"
            "vpxord             %%zmm20, %%zmm20, %%zmm20       \n\t"
            "vpxord             %%zmm21, %%zmm21, %%zmm21       \n\t"
            "vpxord             %%zmm22, %%zmm22, %%zmm22       \n\t"
            "vpxord             %%zmm23, %%zmm23, %%zmm23       \n\t"
            "vpxord             %%zmm24, %%zmm24, %%zmm24       \n\t"
            "vpxord             %%zmm25, %%zmm25, %%zmm25       \n\t"
            "vpxord             %%zmm26, %%zmm26, %%zmm26       \n\t"
            "vpxord             %%zmm27, %%zmm27, %%zmm27       \n\t"
            "vpxord             %%zmm28, %%zmm28, %%zmm28       \n\t"
            "vpxord             %%zmm29, %%zmm29, %%zmm29       \n\t"
            "vpxord             %%zmm30, %%zmm30, %%zmm30       \n\t"
            "vpxord             %%zmm31, %%zmm31, %%zmm31       \n\t"
            "                                                   \n\t"
            "mov                %0, %%rax                       \n\t" // A
            "mov                %1, %%rbx                       \n\t" // B
            "mov                %2, %%rcx                       \n\t" // C
            "mov                %3, %%rsi                       \n\t" // K
            "mov                %4, %%rdi                       \n\t" // ldc
            "                                            \n\t"
            "lea                (,%%rdi,4), %%r10        \n\t" // ldc*4*1
            "lea                (,%%r10,2), %%r11        \n\t" // ldc*4*2
            "lea                (%%r10,%%r11), %%r12     \n\t" // ldc*4*3
            "                                            \n\t"
            "prefetcht0         (%%rcx)                  \n\t" // C
            "prefetcht0         (%%rcx,%%r10)            \n\t"
            "prefetcht0         (%%rcx,%%r11)            \n\t"
            "prefetcht0         (%%rcx,%%r12)            \n\t"
            "lea                (%%rcx,%%r10,4), %%r13   \n\t" // C + ldc*4*4
            "prefetcht0         (%%r13)                  \n\t"
            "prefetcht0         (%%r13,%%r10)            \n\t"
            "prefetcht0         (%%r13,%%r11)            \n\t"
            "prefetcht0         (%%r13,%%r12)            \n\t"
            "lea                (%%r13,%%r10,4), %%r13   \n\t" // C + ldc*4*8
            "prefetcht0         (%%r13)                  \n\t"
            "prefetcht0         (%%r13,%%r10)            \n\t"
            "prefetcht0         (%%r13,%%r11)            \n\t"
            "prefetcht0         (%%r13,%%r12)            \n\t"
            "lea                (%%r13,%%r10,4), %%r13   \n\t" // C + ldc*4*12
            "prefetcht0         (%%r13)                  \n\t"
            "prefetcht0         (%%r13,%%r10)            \n\t"
            "prefetcht0         (%%r13,%%r11)            \n\t"
            "prefetcht0         (%%r13,%%r12)            \n\t"
            "lea                (%%r13,%%r10,4), %%r13   \n\t" // C + ldc*4*16
            "prefetcht0         (%%r13)                  \n\t"
            "prefetcht0         (%%r13,%%r10)            \n\t"
            "prefetcht0         (%%r13,%%r11)            \n\t"
            "prefetcht0         (%%r13,%%r12)            \n\t"
            "lea                (%%r13,%%r10,4), %%r13   \n\t" // C + ldc*4*20
            "prefetcht0         (%%r13)                  \n\t"
            "prefetcht0         (%%r13,%%r10)            \n\t"
            "prefetcht0         (%%r13,%%r11)            \n\t"
            "prefetcht0         (%%r13,%%r12)            \n\t"
            "lea                (%%r13,%%r10,4), %%r13   \n\t" // C + ldc*4*24
            "prefetcht0         (%%r13)                  \n\t"
            "prefetcht0         (%%r13,%%r10)            \n\t"
            "prefetcht0         (%%r13,%%r11)            \n\t"
            "prefetcht0         (%%r13,%%r12)            \n\t"
            "                                            \n\t"
            "vmovaps            0 * 64(%%rbx), %%zmm28   \n\t"
            "vmovaps            1 * 64(%%rbx), %%zmm29   \n\t"
            "vmovaps            2 * 64(%%rbx), %%zmm30   \n\t"
            "vmovaps            3 * 64(%%rbx), %%zmm31   \n\t"
            "                                            \n\t"
            "prefetcht0          0 * 64(%%rax)           \n\t"
            "prefetcht0          1 * 64(%%rax)           \n\t"
            "prefetcht0          2 * 64(%%rax)           \n\t"
            "prefetcht0          3 * 64(%%rax)           \n\t"
            "prefetcht0          4 * 64(%%rax)           \n\t"
            "prefetcht0          5 * 64(%%rax)           \n\t"
            "prefetcht0          6 * 64(%%rax)           \n\t"
            "prefetcht0          7 * 64(%%rax)           \n\t"
            "prefetcht0          8 * 64(%%rax)           \n\t"
            "prefetcht0          9 * 64(%%rax)           \n\t"
            "prefetcht0         10 * 64(%%rax)           \n\t"
            "prefetcht0         11 * 64(%%rax)           \n\t"
            "prefetcht0         12 * 64(%%rax)           \n\t"
            "prefetcht0         13 * 64(%%rax)           \n\t"
            "prefetcht0         14 * 64(%%rax)           \n\t"
            "prefetcht0         15 * 64(%%rax)           \n\t"
            "                                            \n\t"
            "lea                (,%%rsi,8), %%r8         \n\t" // 8*K
            "lea                (%%rbx,%%r8,8), %%r9     \n\t" // %rbx + 64*K
            "                                                            \n\t"
            ".KLOOP_28x1PF_EBCAST_UNR:                                   \n\t"
            "                                                            \n\t"
            "cmp                %%rbx, %%r9                              \n\t"
            "je                 .KLOOP_EXIT_28x1PF_EBCAST_UNR            \n\t"
            "                                                            \n\t"
            "prefetcht0         16 * 64(%%rax)                           \n\t"
            "vfmadd231ps         0 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm0  \n\t"
            "vfmadd231ps         1 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm1  \n\t"
            "vfmadd231ps         2 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm2  \n\t"
            "vfmadd231ps         3 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm3  \n\t"
            "vfmadd231ps         4 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm4  \n\t"
            "vfmadd231ps         5 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm5  \n\t"
            "vfmadd231ps         6 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm6  \n\t"
            "vfmadd231ps         7 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm7  \n\t"
            "vfmadd231ps         8 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm8  \n\t"
            "vfmadd231ps         9 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm9  \n\t"
            "vfmadd231ps        10 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm10 \n\t"
            "vfmadd231ps        11 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm11 \n\t"
            "vfmadd231ps        12 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm12 \n\t"
            "vfmadd231ps        13 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm13 \n\t"
            "prefetcht0         16 * 64(%%rbx)                           \n\t"
            "vfmadd231ps        14 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm14 \n\t"
            "vfmadd231ps        15 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm15 \n\t"
            "prefetcht0         17 * 64(%%rax)                           \n\t"
            "vfmadd231ps        16 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm16 \n\t"
            "vfmadd231ps        17 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm17 \n\t"
            "vfmadd231ps        18 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm18 \n\t"
            "vfmadd231ps        19 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm19 \n\t"
            "vfmadd231ps        20 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm20 \n\t"
            "vfmadd231ps        21 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm21 \n\t"
            "vfmadd231ps        22 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm22 \n\t"
            "vfmadd231ps        23 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm23 \n\t"
            "vfmadd231ps        24 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm24 \n\t"
            "vfmadd231ps        25 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm25 \n\t"
            "vfmadd231ps        26 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm26 \n\t"
            "vfmadd231ps        27 * 4(%%rax)%{1to16%}, %%zmm28, %%zmm27 \n\t"
            "vmovaps            4 * 64(%%rbx), %%zmm28                   \n\t"
            "                                                            \n\t"
            "vfmadd231ps        28 * 4(%%rax)%{1to16%}, %%zmm29, %%zmm0  \n\t"
            "vfmadd231ps        29 * 4(%%rax)%{1to16%}, %%zmm29, %%zmm1  \n\t"
            "vfmadd231ps        30 * 4(%%rax)%{1to16%}, %%zmm29, %%zmm2  \n\t"
            "vfmadd231ps        31 * 4(%%rax)%{1to16%}, %%zmm29, %%zmm3  \n\t"
            "prefetcht0         18 * 64(%%rax)                           \n\t"
            "vfmadd231ps        32 * 4(%%rax)%{1to16%}, %%zmm29, %%zmm4  \n\t"
            "vfmadd231ps        33 * 4(%%rax)%{1to16%}, %%zmm29, %%zmm5  \n\t"
            "vfmadd231ps        34 * 4(%%rax)%{1to16%}, %%zmm29, %%zmm6  \n\t"
            "vfmadd231ps        35 * 4(%%rax)%{1to16%}, %%zmm29, %%zmm7  \n\t"
            "vfmadd231ps        36 * 4(%%rax)%{1to16%}, %%zmm29, %%zmm8  \n\t"
            "vfmadd231ps        37 * 4(%%rax)%{1to16%}, %%zmm29, %%zmm9  \n\t"
            "vfmadd231ps        38 * 4(%%rax)%{1to16%}, %%zmm29, %%zmm10 \n\t"
            "vfmadd231ps        39 * 4(%%rax)%{1to16%}, %%zmm29, %%zmm11 \n\t"
            "vfmadd231ps        40 * 4(%%rax)%{1to16%}, %%zmm29, %%zmm12 \n\t"
            "vfmadd231ps        41 * 4(%%rax)%{1to16%}, %%zmm29, %%zmm13 \n\t"
            "prefetcht0         17 * 64(%%rbx)                           \n\t"
            "vfmadd231ps        42 * 4(%%rax)%{1to16%}, %%zmm29, %%zmm14 \n\t"
            "vfmadd231ps        43 * 4(%%rax)%{1to16%}, %%zmm29, %%zmm15 \n\t"
            "vfmadd231ps        44 * 4(%%rax)%{1to16%}, %%zmm29, %%zmm16 \n\t"
            "vfmadd231ps        45 * 4(%%rax)%{1to16%}, %%zmm29, %%zmm17 \n\t"
            "vfmadd231ps        46 * 4(%%rax)%{1to16%}, %%zmm29, %%zmm18 \n\t"
            "vfmadd231ps        47 * 4(%%rax)%{1to16%}, %%zmm29, %%zmm19 \n\t"
            "prefetcht0         19 * 64(%%rax)                           \n\t"
            "vfmadd231ps        48 * 4(%%rax)%{1to16%}, %%zmm29, %%zmm20 \n\t"
            "vfmadd231ps        49 * 4(%%rax)%{1to16%}, %%zmm29, %%zmm21 \n\t"
            "vfmadd231ps        50 * 4(%%rax)%{1to16%}, %%zmm29, %%zmm22 \n\t"
            "vfmadd231ps        51 * 4(%%rax)%{1to16%}, %%zmm29, %%zmm23 \n\t"
            "vfmadd231ps        52 * 4(%%rax)%{1to16%}, %%zmm29, %%zmm24 \n\t"
            "vfmadd231ps        53 * 4(%%rax)%{1to16%}, %%zmm29, %%zmm25 \n\t"
            "vfmadd231ps        54 * 4(%%rax)%{1to16%}, %%zmm29, %%zmm26 \n\t"
            "vfmadd231ps        55 * 4(%%rax)%{1to16%}, %%zmm29, %%zmm27 \n\t"
            "vmovaps            5 * 64(%%rbx), %%zmm29                   \n\t"
            "                                                            \n\t"
            "vfmadd231ps        56 * 4(%%rax)%{1to16%}, %%zmm30, %%zmm0  \n\t"
            "vfmadd231ps        57 * 4(%%rax)%{1to16%}, %%zmm30, %%zmm1  \n\t"
            "vfmadd231ps        58 * 4(%%rax)%{1to16%}, %%zmm30, %%zmm2  \n\t"
            "vfmadd231ps        59 * 4(%%rax)%{1to16%}, %%zmm30, %%zmm3  \n\t"
            "vfmadd231ps        60 * 4(%%rax)%{1to16%}, %%zmm30, %%zmm4  \n\t"
            "vfmadd231ps        61 * 4(%%rax)%{1to16%}, %%zmm30, %%zmm5  \n\t"
            "vfmadd231ps        62 * 4(%%rax)%{1to16%}, %%zmm30, %%zmm6  \n\t"
            "vfmadd231ps        63 * 4(%%rax)%{1to16%}, %%zmm30, %%zmm7  \n\t"
            "prefetcht0         20 * 64(%%rax)                           \n\t"
            "vfmadd231ps        64 * 4(%%rax)%{1to16%}, %%zmm30, %%zmm8  \n\t"
            "vfmadd231ps        65 * 4(%%rax)%{1to16%}, %%zmm30, %%zmm9  \n\t"
            "vfmadd231ps        66 * 4(%%rax)%{1to16%}, %%zmm30, %%zmm10 \n\t"
            "vfmadd231ps        67 * 4(%%rax)%{1to16%}, %%zmm30, %%zmm11 \n\t"
            "vfmadd231ps        68 * 4(%%rax)%{1to16%}, %%zmm30, %%zmm12 \n\t"
            "vfmadd231ps        69 * 4(%%rax)%{1to16%}, %%zmm30, %%zmm13 \n\t"
            "prefetcht0         18 * 64(%%rbx)                           \n\t"
            "vfmadd231ps        70 * 4(%%rax)%{1to16%}, %%zmm30, %%zmm14 \n\t"
            "vfmadd231ps        71 * 4(%%rax)%{1to16%}, %%zmm30, %%zmm15 \n\t"
            "vfmadd231ps        72 * 4(%%rax)%{1to16%}, %%zmm30, %%zmm16 \n\t"
            "vfmadd231ps        73 * 4(%%rax)%{1to16%}, %%zmm30, %%zmm17 \n\t"
            "vfmadd231ps        74 * 4(%%rax)%{1to16%}, %%zmm30, %%zmm18 \n\t"
            "vfmadd231ps        75 * 4(%%rax)%{1to16%}, %%zmm30, %%zmm19 \n\t"
            "vfmadd231ps        76 * 4(%%rax)%{1to16%}, %%zmm30, %%zmm20 \n\t"
            "vfmadd231ps        77 * 4(%%rax)%{1to16%}, %%zmm30, %%zmm21 \n\t"
            "vfmadd231ps        78 * 4(%%rax)%{1to16%}, %%zmm30, %%zmm22 \n\t"
            "vfmadd231ps        79 * 4(%%rax)%{1to16%}, %%zmm30, %%zmm23 \n\t"
            "prefetcht0         21 * 64(%%rax)                           \n\t"
            "vfmadd231ps        80 * 4(%%rax)%{1to16%}, %%zmm30, %%zmm24 \n\t"
            "vfmadd231ps        81 * 4(%%rax)%{1to16%}, %%zmm30, %%zmm25 \n\t"
            "vfmadd231ps        82 * 4(%%rax)%{1to16%}, %%zmm30, %%zmm26 \n\t"
            "vfmadd231ps        83 * 4(%%rax)%{1to16%}, %%zmm30, %%zmm27 \n\t"
            "vmovaps            6 * 64(%%rbx), %%zmm30                   \n\t"
            "                                                            \n\t"
            "vfmadd231ps        84 * 4(%%rax)%{1to16%}, %%zmm31, %%zmm0  \n\t"
            "vfmadd231ps        85 * 4(%%rax)%{1to16%}, %%zmm31, %%zmm1  \n\t"
            "vfmadd231ps        86 * 4(%%rax)%{1to16%}, %%zmm31, %%zmm2  \n\t"
            "vfmadd231ps        87 * 4(%%rax)%{1to16%}, %%zmm31, %%zmm3  \n\t"
            "vfmadd231ps        88 * 4(%%rax)%{1to16%}, %%zmm31, %%zmm4  \n\t"
            "vfmadd231ps        89 * 4(%%rax)%{1to16%}, %%zmm31, %%zmm5  \n\t"
            "vfmadd231ps        90 * 4(%%rax)%{1to16%}, %%zmm31, %%zmm6  \n\t"
            "vfmadd231ps        91 * 4(%%rax)%{1to16%}, %%zmm31, %%zmm7  \n\t"
            "vfmadd231ps        92 * 4(%%rax)%{1to16%}, %%zmm31, %%zmm8  \n\t"
            "vfmadd231ps        93 * 4(%%rax)%{1to16%}, %%zmm31, %%zmm9  \n\t"
            "vfmadd231ps        94 * 4(%%rax)%{1to16%}, %%zmm31, %%zmm10 \n\t"
            "vfmadd231ps        95 * 4(%%rax)%{1to16%}, %%zmm31, %%zmm11 \n\t"
            "prefetcht0         21 * 64(%%rax)                           \n\t"
            "vfmadd231ps        96 * 4(%%rax)%{1to16%}, %%zmm31, %%zmm12 \n\t"
            "vfmadd231ps        97 * 4(%%rax)%{1to16%}, %%zmm31, %%zmm13 \n\t"
            "prefetcht0         19 * 64(%%rbx)                           \n\t"
            "vfmadd231ps        98 * 4(%%rax)%{1to16%}, %%zmm31, %%zmm14 \n\t"
            "vfmadd231ps        99 * 4(%%rax)%{1to16%}, %%zmm31, %%zmm15 \n\t"
            "vfmadd231ps       100 * 4(%%rax)%{1to16%}, %%zmm31, %%zmm16 \n\t"
            "vfmadd231ps       101 * 4(%%rax)%{1to16%}, %%zmm31, %%zmm17 \n\t"
            "vfmadd231ps       102 * 4(%%rax)%{1to16%}, %%zmm31, %%zmm18 \n\t"
            "vfmadd231ps       103 * 4(%%rax)%{1to16%}, %%zmm31, %%zmm19 \n\t"
            "vfmadd231ps       104 * 4(%%rax)%{1to16%}, %%zmm31, %%zmm20 \n\t"
            "vfmadd231ps       105 * 4(%%rax)%{1to16%}, %%zmm31, %%zmm21 \n\t"
            "vfmadd231ps       106 * 4(%%rax)%{1to16%}, %%zmm31, %%zmm22 \n\t"
            "vfmadd231ps       107 * 4(%%rax)%{1to16%}, %%zmm31, %%zmm23 \n\t"
            "vfmadd231ps       108 * 4(%%rax)%{1to16%}, %%zmm31, %%zmm24 \n\t"
            "vfmadd231ps       109 * 4(%%rax)%{1to16%}, %%zmm31, %%zmm25 \n\t"
            "vfmadd231ps       110 * 4(%%rax)%{1to16%}, %%zmm31, %%zmm26 \n\t"
            "vfmadd231ps       111 * 4(%%rax)%{1to16%}, %%zmm31, %%zmm27 \n\t"
            "vmovaps            7 * 64(%%rbx), %%zmm31                   \n\t"
            "                                                            \n\t"
            "add                $448, %%rax                              \n\t"
            "add                $256, %%rbx                              \n\t"
            "jmp                .KLOOP_28x1PF_EBCAST_UNR                \n\t"
            "                                                           \n\t"
            ".KLOOP_EXIT_28x1PF_EBCAST_UNR:                             \n\t"
            "                                                           \n\t"

            "vaddps             (%%rcx),         %%zmm0,  %%zmm0        \n\t"
            "vmovaps            %%zmm0,  (%%rcx)                        \n\t"
            "vaddps             (%%rcx,%%r10),   %%zmm1,  %%zmm1        \n\t"
            "vmovaps            %%zmm1,  (%%rcx,%%r10)                  \n\t"
            "vaddps             (%%rcx,%%r11),   %%zmm2,  %%zmm2        \n\t"
            "vmovaps            %%zmm2,  (%%rcx,%%r11)                  \n\t"
            "vaddps             (%%rcx,%%r12),   %%zmm3,  %%zmm3        \n\t"
            "vmovaps            %%zmm3,  (%%rcx,%%r12)                  \n\t"
            "                                                           \n\t"
            "lea                (%%rcx,%%r10,4), %%r13                  \n\t"
            "vaddps             (%%r13),         %%zmm4,  %%zmm4        \n\t"
            "vmovaps            %%zmm4,  (%%r13)                        \n\t"
            "vaddps             (%%r13,%%r10),   %%zmm5,  %%zmm5        \n\t"
            "vmovaps            %%zmm5,  (%%r13,%%r10)                  \n\t"
            "vaddps             (%%r13,%%r11),   %%zmm6,  %%zmm6        \n\t"
            "vmovaps            %%zmm6,  (%%r13,%%r11)                  \n\t"
            "vaddps             (%%r13,%%r12),   %%zmm7,  %%zmm7        \n\t"
            "vmovaps            %%zmm7,  (%%r13,%%r12)                  \n\t"
            "                                                           \n\t"
            "lea                (%%r13,%%r10,4), %%r13                  \n\t"
            "vaddps             (%%r13),         %%zmm8,  %%zmm8        \n\t"
            "vmovaps            %%zmm8,  (%%r13)                        \n\t"
            "vaddps             (%%r13,%%r10),   %%zmm9,  %%zmm9        \n\t"
            "vmovaps            %%zmm9,  (%%r13,%%r10)                  \n\t"
            "vaddps             (%%r13,%%r11),   %%zmm10, %%zmm10       \n\t"
            "vmovaps            %%zmm10, (%%r13,%%r11)                  \n\t"
            "vaddps             (%%r13,%%r12),   %%zmm11, %%zmm11       \n\t"
            "vmovaps            %%zmm11, (%%r13,%%r12)                  \n\t"
            "                                                           \n\t"
            "lea                (%%r13,%%r10,4), %%r13                  \n\t"
            "vaddps             (%%r13),         %%zmm12, %%zmm12       \n\t"
            "vmovaps            %%zmm12, (%%r13)                        \n\t"
            "vaddps             (%%r13,%%r10),   %%zmm13, %%zmm13       \n\t"
            "vmovaps            %%zmm13, (%%r13,%%r10)                  \n\t"
            "vaddps             (%%r13,%%r11),   %%zmm14, %%zmm14       \n\t"
            "vmovaps            %%zmm14, (%%r13,%%r11)                  \n\t"
            "vaddps             (%%r13,%%r12),   %%zmm15, %%zmm15       \n\t"
            "vmovaps            %%zmm15, (%%r13,%%r12)                  \n\t"
            "                                                           \n\t"
            "lea                (%%r13,%%r10,4), %%r13                  \n\t"
            "vaddps             (%%r13),         %%zmm16, %%zmm16       \n\t"
            "vmovaps            %%zmm16, (%%r13)                        \n\t"
            "vaddps             (%%r13,%%r10),   %%zmm17, %%zmm17       \n\t"
            "vmovaps            %%zmm17, (%%r13,%%r10)                  \n\t"
            "vaddps             (%%r13,%%r11),   %%zmm18, %%zmm18       \n\t"
            "vmovaps            %%zmm18, (%%r13,%%r11)                  \n\t"
            "vaddps             (%%r13,%%r12),   %%zmm19, %%zmm19       \n\t"
            "vmovaps            %%zmm19, (%%r13,%%r12)                  \n\t"
            "                                                           \n\t"
            "lea                (%%r13,%%r10,4), %%r13                  \n\t"
            "vaddps             (%%r13),         %%zmm20, %%zmm20       \n\t"
            "vmovaps            %%zmm20, (%%r13)                        \n\t"
            "vaddps             (%%r13,%%r10),   %%zmm21, %%zmm21       \n\t"
            "vmovaps            %%zmm21, (%%r13,%%r10)                  \n\t"
            "vaddps             (%%r13,%%r11),   %%zmm22, %%zmm22       \n\t"
            "vmovaps            %%zmm22, (%%r13,%%r11)                  \n\t"
            "vaddps             (%%r13,%%r12),   %%zmm23, %%zmm23       \n\t"
            "vmovaps            %%zmm23, (%%r13,%%r12)                  \n\t"
            "                                                           \n\t"
            "lea                (%%r13,%%r10,4), %%r13                  \n\t"
            "vaddps             (%%r13),         %%zmm24, %%zmm24       \n\t"
            "vmovaps            %%zmm24, (%%r13)                        \n\t"
            "vaddps             (%%r13,%%r10),   %%zmm25, %%zmm25       \n\t"
            "vmovaps            %%zmm25, (%%r13,%%r10)                  \n\t"
            "vaddps             (%%r13,%%r11),   %%zmm26, %%zmm26       \n\t"
            "vmovaps            %%zmm26, (%%r13,%%r11)                  \n\t"
            "vaddps             (%%r13,%%r12),   %%zmm27, %%zmm27       \n\t"
            "vmovaps            %%zmm27, (%%r13,%%r12)                  \n\t"
            "                                                           \n\t"
            : // output operands
            : // input operands
              "m" (A)
            , "m" (B)
            , "m" (C)
            , "m" (K64)
            , "m" (ldc64)
            : // clobbered registers
              "rax", "rbx", "rcx", "rdx", "rsi", "rdi"
            , "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"
            , "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"
            , "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14"
            , "xmm15", "xmm16", "xmm17", "xmm18", "xmm19", "xmm20", "xmm21"
            , "xmm22", "xmm23", "xmm24", "xmm25", "xmm26", "xmm27", "xmm28"
            , "xmm29", "xmm30", "xmm31"
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

#endif
