#pragma once

#include "00naive.h"
#include "util.h"
#include <cassert>
#undef NODEBUG

template <class Opt> struct blis;

template <class MM, class Opt>
struct blis_L1_base {
    using BLIS = blis<Opt>;

    enum : int {
        BLOCK_M = MM::BLOCK_M,
        BLOCK_N = MM::BLOCK_N,

        LDB_R = BLOCK_N, // + 64 / sizeof(float),
    };

    alignas(32) static float Br_buf[BLIS::BLOCK_K * LDB_R];

    static void matmul(
        int M, int N, int K, float *A, int lda,
        float *B, int ldb, float *C, int ldc)
    {
        assert(M % BLOCK_M == 0);
        assert(N % BLOCK_N == 0);

        for (int j = 0; j < N; j += BLOCK_N) {
            auto Br = B + ldb * 0 + j;

            int Kc = std::min<int>(K, BLIS::BLOCK_K);
            if (Opt::PACK >= 1 && K * N > Kc * 96) {
                // Pack B to L1 (256x16) cache buffer
                for (int l = 0; l < Kc; l++) {
                    std::copy_n(Br + ldb * l, BLOCK_N, Br_buf + LDB_R * l);
                }
                Br = Br_buf;
                ldb = LDB_R;
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
alignas(32) float blis_L1_base<MM, Opt>::Br_buf[];

struct blis_opt {
    struct naive {
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
struct blis {
    struct L1 : blis_L1_base<register_avx_3_6x2, Opt> {};

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

                if (Opt::PACK >= 3 && K * N > BLOCK_K * BLOCK_N) {
                    // Pack B (256x3072) to L3 cache buffer
                    for (int l = 0; l < Kc; l++)
                        std::copy_n(Bc + ldb * l, Nc, Bc_buf + Nc * l);
                    Bc = Bc_buf;
                    ldb = Nc;
                }

                for (int i = 0; i < M - 1; i += BLOCK_M) {
                    auto Ac = A + lda * i + k;
                    auto Cc = C + ldc * i + j;
                    auto Mc = std::min<int>(M - i, BLOCK_M);

                    if (Opt::PACK >= 2 && M * K > BLOCK_M * BLOCK_K) {
                        // Pack A (96x256) to L2 cache buffer
                        for (int l = 0; l < Mc; l++)
                            std::copy_n(Ac + lda * l, Kc, Ac_buf + Kc * l);
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
alignas(32) float blis<Opt>::Ac_buf[];
template <class Opt>
alignas(32) float blis<Opt>::Bc_buf[];

