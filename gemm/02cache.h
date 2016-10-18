#pragma once

#include "01register.h"
#include <algorithm>

namespace cache_oblivious {

    static void matmul(
        int M, int N, int K, float *A, int lda,
        float *B, int ldb, float *C, int ldc)
    {
        constexpr int TILE_M = register_avx_2::TILE_M;
        constexpr int TILE_N = register_avx_2::TILE_N;
        constexpr int TILE_K = register_avx_2::TILE_K;

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
            register_avx_2::matmul(M, N, K, A, lda, B, ldb, C, ldc);
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

namespace cache_blocking_L1 {

    static void matmul(
        int M, int N, int K, float *A, int lda,
        float *B, int ldb, float *C, int ldc)
    {
        // 32KB block (<= 32KB L1 cache)
        constexpr int L1_M = 8;
        constexpr int L1_N = 32;
        constexpr int L1_K = 32;

        for (int i1 = 0; i1 < M; i1 += L1_M) {
            for (int j1 = 0; j1 < N; j1 += L1_N) {
                for (int k1 = 0; k1 < K; k1 += L1_K) {
                    auto A1 = A + lda * i1 + k1;
                    auto B1 = B + ldb * k1 + j1;
                    auto C1 = C + ldc * i1 + j1;
                    register_avx_2::matmul(
                        L1_M, L1_N, L1_K, A1, lda, B1, ldb,
                        C1, ldc);
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

}

namespace cache_blocking_L2 {

    static void matmul(
        int M, int N, int K, float *A, int lda,
        float *B, int ldb, float *C, int ldc)
    {
        // 256KB block (<= 256KB L2 cache)
        constexpr int L2_M = 64;
        constexpr int L2_N = 32;
        constexpr int L2_K = 32;

        for (int i2 = 0; i2 < M; i2 += L2_M) {
            for (int j2 = 0; j2 < N; j2 += L2_N) {
                for (int k2 = 0; k2 < K; k2 += L2_K) {
                    auto A2 = A + lda * i2 + k2;
                    auto B2 = B + ldb * k2 + j2;
                    auto C2 = C + ldc * i2 + j2;
                    cache_blocking_L1::matmul(
                        L2_M, L2_N, L2_K, A2, lda, B2, ldb, 
                        C2, ldc);
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

}

namespace cache_blocking_L3 {

    static void matmul(
        int M, int N, int K, float *A, int lda,
        float *B, int ldb, float *C, int ldc)
    {
        // 8MB block (<= 8MB L2 cache)
        constexpr int L3_M = 256;
        constexpr int L3_N = 128;
        constexpr int L3_K = 128;
        
        for (int i3 = 0; i3 < M; i3 += L3_M) {
            for (int j3 = 0; j3 < N; j3 += L3_N) {
                for (int k3 = 0; k3 < K; k3 += L3_K) {
                    auto A2 = A + lda * i3 + k3;
                    auto B2 = B + ldb * k3 + j3;
                    auto C2 = C + ldc * i3 + j3;
                    cache_blocking_L2::matmul(
                        L3_M, L3_N, L3_K, A2, lda, B2, ldb,
                        C2, ldc);
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

}
