#pragma once

#include "01register.h"
#include "util.h"
#include <cassert>
#undef NODEBUG

template <class Kernel, class Opt> struct blis;

template <class Kernel, class Opt>
struct make_blis_L1_ {
    using BLIS = blis<Kernel, Opt>;

    enum : int {
        BLOCK_M = Kernel::BLOCK_M,
        BLOCK_N = Kernel::BLOCK_N,
    };

    struct alignas(LINE_SIZE) blis_buffer {
        float Br[BLIS::BLOCK_K * BLOCK_N];          // L1: 256x24
        float Ac[BLIS::BLOCK_M * BLIS::BLOCK_K];    // L2: 144x256
        float Bc[BLIS::BLOCK_K * BLIS::BLOCK_N];    // L3: 256x3072
    };

    static blis_buffer *buf;

    static void matmul(
        int M, int N, int K, float *A, int lda,
        float *B, int ldb, float *C, int ldc)
    {
        assert(M % BLOCK_M == 0);
        assert(N % BLOCK_N == 0);
        assert(K <= BLIS::BLOCK_K);

        for (int j = 0; j < N; j += BLOCK_N) {
            auto Br = B + K * j;

            // Pack B to L1 (256x16: 16KB) cache buffer
            copy2d(Br, ldb, K, BLOCK_N, buf->Br, BLOCK_N);
            Br = buf->Br;
            int ldb_r = BLOCK_N;

            for (int i = 0; i < M; i += BLOCK_M) {
                auto Ar = A + K * i;
                auto Cr = C + ldc * i + j;

#if 0
                printf("## j = %3d, i = %3d, Mr = %3d, Nr = %3d, Kr = %3d, Ar[0]= %f, Br[0] = %f\n",
                    j, i, BLOCK_M, BLOCK_N, K, Ar[0], Br[0]);
#endif

                Kernel::matmul_register_packed(BLOCK_M, BLOCK_N, K, Ar, lda, Br, ldb_r, Cr, ldc);
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

    static void initialize()
    {
        if (buf == nullptr) {
            buf = (blis_buffer *)_mm_malloc(sizeof(blis_buffer), PAGE_SIZE);
            memset(buf, 0, sizeof(blis_buffer));
        }
    }
};
template <class Kernel, class Opt>
typename make_blis_L1_<Kernel, Opt>::blis_buffer * make_blis_L1_<Kernel, Opt>::buf = nullptr;

template <class Kernel, class Opt>
struct blis_ {
    struct L1 : make_blis_L1_<Kernel, Opt> {};

    enum : int {
        BLOCK_M = 144,
        BLOCK_N = 3072,
        BLOCK_K = 256,

        LDA_C = BLOCK_K + 8,
        LDB_C = BLOCK_N + 8,
    };

    alignas(LINE_SIZE) static float Ac_buf[BLOCK_M * LDA_C];
    alignas(LINE_SIZE) static float Bc_buf[BLOCK_K * LDB_C];

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
                pack2d<float, 0, L1::BLOCK_N>(Bc, ldb, Kc, Nc, L1::buf->Bc, Nc);
                Bc = L1::buf->Bc;
                auto ldb_c = L1::BLOCK_N;

                for (int i = 0; i < M; i += BLOCK_M) {
                    auto Ac = A + lda * i + k;
                    auto Cc = C + ldc * i + j;
                    auto Mc = std::min<int>(M - i, BLOCK_M);

                    // Copy A (144x256) to L2 cache buffer
                    pack2d<float, L1::BLOCK_M, 0>(Ac, lda, Mc, Kc, L1::buf->Ac, Kc);
                    Ac = L1::buf->Ac;
                    auto lda_c = L1::BLOCK_M;
#if 0
                    printf("#  j = %3d, k = %3d, i  = %3d, Mc = %3d, Nc = %3d, Kc = %3d\n",
                        j, k, i, Mc, Nc, Kc);
#endif

                    L1::matmul(Mc, Nc, Kc, Ac, lda_c, Bc, ldb_c, Cc, ldc);
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
        L1::initialize();
    }
};
