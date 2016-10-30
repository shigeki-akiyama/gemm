#pragma once

#include "01register.h"
#include "util.h"
#include <cassert>
#undef NODEBUG

template <class Kernel, class Opt> struct blis_copy;

template <class Kernel, class Opt>
struct make_blis_copy_L1 {
    using BLIS = blis_copy<Kernel, Opt>;

    enum : int {
        BLOCK_M = Kernel::BLOCK_M,
        BLOCK_N = Kernel::BLOCK_N,
    };

    struct alignas(LINE_SIZE) blis_buffer {
        float Br[BLIS::BLOCK_K * BLOCK_N];          // L1: 256x24
        float Ac[BLIS::BLOCK_M * (BLIS::BLOCK_K /*+ 8*/)];    // L2: 144x256
        float Bc[BLIS::BLOCK_K * (BLIS::BLOCK_N /*+ 8*/)];    // L3: 256x3072
    };

    static blis_buffer *buf;

    alignas(LINE_SIZE) static float Br_buf[BLIS::BLOCK_K * BLOCK_N];

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
                copy2d(B, ldb, Kc, BLOCK_N, buf->Br, BLOCK_N);
                Br = buf->Br;
                ldb = BLOCK_N;
            }

            for (int i = 0; i < M; i += BLOCK_M) {
                auto Ar = A + lda * i + 0;
                auto Cr = C + ldc * i + j;
                lda = BLOCK_M;
                ldb = BLOCK_N;
                Kernel::matmul_register(BLOCK_M, BLOCK_N, K, Ar, lda, Br, ldb, Cr, ldc);
            }
        }
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
typename make_blis_copy_L1<Kernel, Opt>::blis_buffer * make_blis_copy_L1<Kernel, Opt>::buf = nullptr;
template <class Kernel, class Opt> 
alignas(LINE_SIZE) float make_blis_copy_L1<Kernel, Opt>::Br_buf[];

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

template <class Kernel, class Opt>
struct blis_copy {
    struct L1 : make_blis_copy_L1<Kernel, Opt> {};

    enum : int {
        BLOCK_M = 144,
        BLOCK_N = 3072,
        BLOCK_K = 256,

        LDA_C   = BLOCK_K + 8,
        LDB_C   = BLOCK_N + 8,
    };

    alignas(LINE_SIZE) static float Ac_buf[BLOCK_M * LDA_C];
    alignas(LINE_SIZE) static float Bc_buf[BLOCK_K * LDB_C];

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
                    copy2d(Bc, ldb, Kc, Nc, L1::buf->Bc, Nc);
                    Bc = L1::buf->Bc;
                    ldb = Nc;
                }

                for (int i = 0; i < M - 1; i += BLOCK_M) {
                    auto Ac = A + lda * i + k;
                    auto Cc = C + ldc * i + j;
                    auto Mc = std::min<int>(M - i, BLOCK_M);

                    if (Opt::PACK >= 2) {
                        // Copy A (144x256) to L2 cache buffer
                        copy2d(Ac, lda, Mc, Kc, L1::buf->Ac, L1::BLOCK_M + 8);
                        Ac = L1::buf->Ac;
                        lda = L1::BLOCK_M + 8;
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

    static void intiialize()
    {
        L1::initialize();
    }
};
template <class Kernel, class Opt>
alignas(LINE_SIZE) float blis_copy<Kernel, Opt>::Ac_buf[];
template <class Kernel, class Opt>
alignas(LINE_SIZE) float blis_copy<Kernel, Opt>::Bc_buf[];

