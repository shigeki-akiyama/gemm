#pragma once

#include "00naive.h"
#include "util.h"
#include <cassert>
#undef NODEBUG


template <class Opt> struct blis_copy;

template <class MM, class Opt>
struct make_blis_copy_L1 {
    using BLIS = blis_copy<Opt>;

    enum : int {
        BLOCK_M = MM::BLOCK_M,
        BLOCK_N = MM::BLOCK_N,
    };

    alignas(32) static float Br_buf[BLIS::BLOCK_K * BLOCK_N];

    static void matmul(
        int M, int N, int K, float *A, int lda,
        float *B, int ldb, float *C, int ldc)
    {
        assert(M % BLOCK_M == 0);
        assert(N % BLOCK_N == 0);

        for (int j = 0; j < N; j += BLOCK_N) {
            auto Br = B + ldb * 0 + j;

            int Kc = std::min<int>(K, BLIS::BLOCK_K);
            if (Opt::PACK >= 1) {
                // Copy B to L1 (256x16: 16KB) cache buffer
                copy2d(B, ldb, Kc, BLOCK_N, Br_buf, BLOCK_N);
                Br = Br_buf;
                ldb = BLOCK_N;
            }

            for (int i = 0; i < M; i += BLOCK_M) {
                auto Ar = A + lda * i + 0;
                auto Cr = C + ldc * i + j;
                MM::matmul_register(
                    BLOCK_M, BLOCK_N, K, Ar, lda, Br, ldb, Cr, ldc);
            }
        }
    }
};
template <class MM, class Opt> 
alignas(32) float make_blis_copy_L1<MM, Opt>::Br_buf[];

struct blis_opt {
    struct nopack {
        enum : int { PACK = 0 };
    };
    struct packL1 {
        enum : int { PACK = 1 };
    };
    struct packL2 { 
        enum : int { PACK = 2 };
    };
    struct packL3 {
        enum : int { PACK = 3 };
    };
};

template <class Opt>
struct blis_copy {
    struct L1 : make_blis_copy_L1<register_avx_3_6x2, Opt> {};

    enum : int {
        BLOCK_M = 144,
        BLOCK_N = 3072,
        BLOCK_K = 256,
    };

    alignas(32) static float Ac_buf[BLOCK_M * BLOCK_K];
    alignas(32) static float Bc_buf[BLOCK_K * BLOCK_N];

    static void matmul(
        int M, int N, int K, float *A, int lda,
        float *B, int ldb, float *C, int ldc)
    {
        for (int j = 0; j < N - 1; j += BLOCK_N) {
            for (int k = 0; k < K - 1; k += BLOCK_K) {
                auto Bc = B + ldb * k + j;
                auto Nc = std::min<int>(N - j, BLOCK_N);
                auto Kc = std::min<int>(K - k, BLOCK_K);

                if (Opt::PACK >= 3) {
                    // Copy B (256x3072) to L3 cache buffer
                    copy2d(Bc, ldb, Kc, Nc, Bc_buf, Nc);
                    Bc = Bc_buf;
                    ldb = Nc;
                }

                for (int i = 0; i < M - 1; i += BLOCK_M) {
                    auto Ac = A + lda * i + k;
                    auto Cc = C + ldc * i + j;
                    auto Mc = std::min<int>(M - i, BLOCK_M);

                    if (Opt::PACK >= 2) {
                        // Copy A (144x256) to L2 cache buffer
                        copy2d(Ac, lda, Mc, Kc, Ac_buf, Kc);
                        Ac = Ac_buf;
                        lda = Kc;
                    }

                    L1::matmul(Mc, Nc, Kc, Ac, lda, Bc, ldb, Cc, ldc);
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
template <class Opt>
alignas(32) float blis_copy<Opt>::Ac_buf[];
template <class Opt>
alignas(32) float blis_copy<Opt>::Bc_buf[];

