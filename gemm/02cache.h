#pragma once

#include "01register.h"
#include <algorithm>

namespace cache_oblivious {

    static void matmul(
        int M, int N, int K, float *A, int lda,
        float *B, int ldb, float *C, int ldc)
    {
        using kernel = register_avx_3_4x3;

        enum : int {
            TILE_M = kernel::BLOCK_M,
            TILE_N = kernel::BLOCK_N,
            TILE_K = 32, //kernel::BLOCK_K,
        };

        if (M > std::max(N, K) && M >= 2 * TILE_M) {
            int M_half = M / 2 / sizeof(__m256) * sizeof(__m256);
            auto A_half = A + lda * M_half;
            auto C_half = C + ldc * M_half;
            matmul(M_half    , N, K, A,      lda, B, ldb, C     , ldc);
            matmul(M - M_half, N, K, A_half, lda, B, ldb, C_half, ldc);
        } else if (N > K && N >= 2 * TILE_N) {
            int N_half = N / 2 / TILE_N * TILE_N;
            auto B_half = B + N_half;
            auto C_half = C + N_half;
            matmul(M, N_half    , K, A, lda, B     , ldb, C     , ldc);
            matmul(M, N - N_half, K, A, lda, B_half, ldb, C_half, ldc);
        } else if (K >= 2 * TILE_K) {
            int K_half = K / 2 / TILE_K * TILE_K;
            auto A_half = A + K_half;
            auto B_half = B + ldb * K_half;
            matmul(M, N, K_half    , A     , lda, B,      ldb, C, ldc);
            matmul(M, N, K - K_half, A_half, lda, B_half, ldb, C, ldc);
        } else {
            kernel::matmul(M, N, K, A, lda, B, ldb, C, ldc);
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

}

template <int BLOCK_M_, int BLOCK_N_, int BLOCK_K_, class MM>
struct blocking_base {
    enum : int {
        BLOCK_M = BLOCK_M_,
        BLOCK_N = BLOCK_N_,
        BLOCK_K = BLOCK_K_,
    };

    static void matmul(
        int M, int N, int K, float *A, int lda,
        float *B, int ldb, float *C, int ldc)
    {
        for (int i = 0; i < M; i += BLOCK_M) {
            for (int j = 0; j < N; j += BLOCK_N) {
                for (int k = 0; k < K; k += BLOCK_K) {
                    auto Ab = A + lda * i + k;
                    auto Bb = B + ldb * k + j;
                    auto Cb = C + ldc * i + j;
                    auto Mr = std::min<int>(M - i, BLOCK_M);
                    auto Nr = std::min<int>(N - j, BLOCK_N);
                    auto Kr = std::min<int>(K - k, BLOCK_K);
                    MM::matmul(Mr, Nr, Kr, Ab, lda, Bb, ldb, Cb, ldc);
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

#if 0
// L1 blocking: (48x32 + 48x32 + 32x32) * 4bytes = 16KB (< 32KB)
struct cache_blocking_L1 : blocking_base<48, 32, 32, register_avx_3_6x2> {};

// L2 blocking: 96KB (< 256KB)
struct cache_blocking_L2 : blocking_base<96, 96, 96, cache_blocking_L1> {};

// L3 blocking: 432KB (< 1MB/core)
struct cache_blocking_L3 : blocking_base<192, 192, 192, cache_blocking_L2> {};
#else
struct cache_blocking_L1 : blocking_base<32, 48, 32, register_avx_3_4x3> {};
struct cache_blocking_L2 : blocking_base<64, 96, 128, cache_blocking_L1> {};
struct cache_blocking_L3 : blocking_base<128, 192, 256, cache_blocking_L2> {};
#endif
