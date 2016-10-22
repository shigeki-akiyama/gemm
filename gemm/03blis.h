#pragma once

#include "00naive.h"
#include "util.h"

template <class MM>
struct blis_L1_base {
    enum : int {
        BLOCK_M = MM::BLOCK_M,
        BLOCK_N = MM::BLOCK_N,
    };

    static void matmul(
        int M, int N, int K, float *A, int lda,
        float *B, int ldb, float *C, int ldc)
    {
        for (int j = 0; j < N - 1; j += BLOCK_N) {
            for (int i = 0; i < M - 1; i += BLOCK_M) {
                auto Ar = A + lda * i + 0;
                auto Br = B + ldb * 0 + j;
                auto Cr = C + ldc * i + j;
                auto Mr = std::min<int>(M - i, BLOCK_M);
                auto Nr = std::min<int>(N - j, BLOCK_N);
                MM::matmul_register(K, Ar, lda, Br, ldb, Cr, ldc);
            }
        }
    }
};

using blis_L1 = blis_L1_base<register_avx_3_6x2>;

struct blis_naive {
    enum : int {
        BLOCK_M = 96,
        BLOCK_N = 3072,
        BLOCK_K = 256,
    };

    static void matmul(
        int M, int N, int K, float *A, int lda,
        float *B, int ldb, float *C, int ldc)
    {
        for (int j = 0; j < N - 1; j += BLOCK_N) {
            for (int k = 0; k < K - 1; k += BLOCK_K) {
                for (int i = 0; i < M - 1; i += BLOCK_M) {
                    auto Ar = A + lda * i + k;
                    auto Br = B + ldb * k + j;
                    auto Cr = C + ldc * i + j;
                    auto Mr = std::min<int>(M - i, BLOCK_M);
                    auto Nr = std::min<int>(N - j, BLOCK_N);
                    auto Kr = std::min<int>(K - k, BLOCK_K);
                    blis_L1::matmul(Mr, Nr, Kr, Ar, lda, Br, ldb, Cr, ldc);
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
