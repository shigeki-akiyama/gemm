#pragma once

#include "00naive.h"
#include "util.h"

template <bool PACK> struct blis;

template <class MM, bool PACK>
struct blis_L1_base {
    enum : int {
        BLOCK_M = MM::BLOCK_M,
        BLOCK_N = MM::BLOCK_N,

        LDB_R = BLOCK_N, // + 64 / sizeof(float),
    };

    static float Br_buf[blis<PACK>::BLOCK_K * LDB_R];

    static void matmul(
        int M, int N, int K, float *A, int lda,
        float *B, int ldb, float *C, int ldc)
    {
        for (int j = 0; j < N - 1; j += BLOCK_N) {
            auto Br = B + ldb * 0 + j;

            if (PACK) {
                std::copy_n(Br, blis<PACK>::BLOCK_K * LDB_R, Br_buf);
                //for (int l = 0; l < blis<PACK>::BLOCK_K; l++)
                //    std::copy_n(Br + ldb * l, BLOCK_N, Br_buf + LDB_R * l);
                Br = Br_buf;
                ldb = LDB_R;
            }

            for (int i = 0; i < M - 1; i += BLOCK_M) {
                auto Ar = A + lda * i + 0;
                auto Cr = C + ldc * i + j;
                MM::matmul_register(K, Ar, lda, Br, ldb, Cr, ldc);
            }
        }
    }
};
template <class MM, bool PACK> float blis_L1_base<MM, PACK>::Br_buf[];

template <bool PACK>
struct blis {
    using blis_L1 = blis_L1_base<register_avx_3_6x2, PACK>;

    enum : int {
        BLOCK_M = 96,
        BLOCK_N = 3072,
        BLOCK_K = 256,

        LDA_C = BLOCK_K, // + 64 / sizeof(float),
    };

    alignas(32) static float Ac_buf[BLOCK_M * LDA_C];

    static void matmul(
        int M, int N, int K, float *A, int lda,
        float *B, int ldb, float *C, int ldc)
    {
        for (int j = 0; j < N - 1; j += BLOCK_N) {
            for (int k = 0; k < K - 1; k += BLOCK_K) {
                for (int i = 0; i < M - 1; i += BLOCK_M) {
                    auto Ac = A + lda * i + k;
                    auto Bc = B + ldb * k + j;
                    auto Cc = C + ldc * i + j;
                    auto Mc = std::min<int>(M - i, BLOCK_M);
                    auto Nc = std::min<int>(N - j, BLOCK_N);
                    auto Kc = std::min<int>(K - k, BLOCK_K);

                    if (PACK) {
                        // Pack A
                        std::copy_n(Ac, BLOCK_M * LDA_C, Ac_buf);
                        //for (int l = 0; l < BLOCK_M; l++)
                        //    std::copy_n(Ac + lda * l, BLOCK_K, Ac_buf + LDA_C * l);
                        Ac = Ac_buf;
                        lda = LDA_C;
                    }

                    blis_L1::matmul(Mc, Nc, Kc, Ac, lda, Bc, ldb, Cc, ldc);
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
template <bool PACK> float blis<PACK>::Ac_buf[];
