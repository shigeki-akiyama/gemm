#pragma once

#include "config.h"
#include "01register.h"
#include "util.h"
#include <cassert>
#undef NODEBUG

template <class Kernel, class Opt>
struct blis {
    enum : int {
        BLOCK_M = M_CACHE,
        BLOCK_N = N_CACHE,
        BLOCK_K = K_CACHE,
    };

    struct alignas(LINE_SIZE) blis_buffer {
        float Br[BLOCK_K * Kernel::BLOCK_N];    // L1: 256x24
        float Ac[BLOCK_M * BLOCK_K];            // L2: 144x256
        float Bc[BLOCK_K * BLOCK_N];            // L3: 256x3072
    };

    static blis_buffer * s_buf;

    static void matmul_cache(
        int M, int N, int K, float *A, int lda,
        float *B, int ldb, float *C, int ldc)
    {
        assert(M % Kernel::BLOCK_M == 0);
        assert(N % Kernel::BLOCK_N == 0);
        assert(K <= BLOCK_K);

        for (int j = 0; j < N; j += Kernel::BLOCK_N) {
            auto Br = B + K * j;

            // Pack B to L1 (256x16: 16KB) cache buffer
            copy2d(Br, ldb, K, Kernel::BLOCK_N, s_buf->Br, Kernel::BLOCK_N);
            Br = s_buf->Br;
            int ldb_r = Kernel::BLOCK_N;

            for (int i = 0; i < M; i += Kernel::BLOCK_M) {
                auto Ar = A + K * i;
                auto Cr = C + ldc * i + j;

#if 0
                printf("## j = %3d, i = %3d, Mr = %3d, Nr = %3d, Kr = %3d, Ar[0]= %f, Br[0] = %f\n",
                    j, i, Kernel::BLOCK_M, Kernel::BLOCK_N, K, Ar[0], Br[0]);
#endif

                Kernel::matmul_register_packed(
                    Kernel::BLOCK_M, Kernel::BLOCK_N, K, Ar, lda, Br, ldb_r, Cr, ldc);
            }
        }
    }

    static void matmul(
        int M, int N, int K, float *A, int lda,
        float *B, int ldb, float *C, int ldc)
    {
        for (int j = 0; j < N; j += BLOCK_N) {
            for (int k = 0; k < K; k += BLOCK_K) {
                auto Bc = B + ldb * k + j;
                auto Nc = std::min<int>(N - j, BLOCK_N);
                auto Kc = std::min<int>(K - k, BLOCK_K);

                // Copy B (256x3072) to L3 cache buffer
                pack2d<float, 0, Kernel::BLOCK_N>(Bc, ldb, Kc, Nc, s_buf->Bc, Nc);
                Bc = s_buf->Bc;
                auto ldb_c = Kernel::BLOCK_N;

                for (int i = 0; i < M; i += BLOCK_M) {
                    auto Ac = A + lda * i + k;
                    auto Cc = C + ldc * i + j;
                    auto Mc = std::min<int>(M - i, BLOCK_M);

                    // Copy A (144x256) to L2 cache buffer
                    pack2d<float, Kernel::BLOCK_M, 0>(Ac, lda, Mc, Kc, s_buf->Ac, Kc);
                    Ac = s_buf->Ac;
                    auto lda_c = Kernel::BLOCK_M;
#if 0
                    printf("#  j = %3d, k = %3d, i  = %3d, Mc = %3d, Nc = %3d, Kc = %3d\n",
                        j, k, i, Mc, Nc, Kc);
#endif

                    matmul_cache(Mc, Nc, Kc, Ac, lda_c, Bc, ldb_c, Cc, ldc);
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

    static void intiialize()
    {
        if (s_buf == nullptr) {
            s_buf = (blis_buffer *)_mm_malloc(sizeof(blis_buffer), PAGE_SIZE);
            memset(s_buf, 0, sizeof(blis_buffer));
        }
    }
};
template <class Kernel, class Opt>
typename blis<Kernel, Opt>::blis_buffer * blis<Kernel, Opt>::s_buf = nullptr;
