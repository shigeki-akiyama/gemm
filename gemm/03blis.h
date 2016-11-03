#pragma once

#include "01register.h"
#include "util.h"
#include <cassert>
#undef NODEBUG


template <class T, int RS, int CS>
static void pack2d(const T *A, int lda, int M, int N, T *B, int ldb)
{
    if (RS > 0) {           // for matrix A
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                int k = RS * N * (i / RS) + RS * j + i % RS;
                B[k] = A[lda * i + j];
            }
        }
    } else if (CS > 0) {    // for matrix B
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                int k = CS * M * (j / CS) + CS * i + j % CS;
                B[k] = A[lda * i + j];
            }
        }
    }
}

template <class T>
struct blis_opt {
    template <int LEVEL, class Packer>
    struct make_option {
        static constexpr int pack_level() { return LEVEL; }

        static constexpr bool do_pack() { return Packer::do_pack(); }

        template <int CURR_LEVEL>
        static void pack(const T *A, int lda, int M, int N, T *B, int ldb)
        {
            Packer::template pack<CURR_LEVEL>(A, lda, M, N, B, ldb);
        }
    };

    struct matcopy {
        static constexpr bool do_pack() { return false; }

        template <int CURR_LEVEL>
        static void pack(const T *A, int lda, int M, int N, T *B, int ldb)
        {
            copy2d(A, lda, M, N, B, ldb);
        }
    };

    template <int LEVEL, int RS, int CS>   // raw-major and column-major stride
    struct matpack {
        static constexpr bool do_pack() { return true; }

        template <int CURR_LEVEL>
        static void pack(const T *A, int lda, int M, int N, T *B, int ldb)
        {
            if (CURR_LEVEL == LEVEL && 2 <= LEVEL && LEVEL <= 3) {
                pack2d<T, RS, CS>(A, lda, M, N, B, ldb);
            } else if (CURR_LEVEL == LEVEL) {
                copy2d(A, lda, M, N, B, ldb);
            } else {
                copy2d(A, lda, M, N, B, ldb);
            }
        }
    };

    struct naive : make_option<0, matcopy> {};

    struct copyL1 : make_option<1, matcopy> {};
    struct copyL2 : make_option<2, matcopy> {};
    struct copyL3 : make_option<3, matcopy> {};

    struct packL1 : make_option<1, matpack<1, 0, 0>> {};
    struct packL2 : make_option<2, matpack<2, 6, 0>> {};
    struct packL3 : make_option<3, matpack<3, 0, 16>> {};
};


template <class Kernel, class Opt> struct blis;

template <class Kernel, class Opt>
struct make_blis_L1 {
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

        int Kc = std::min<int>(K, BLIS::BLOCK_K);

        for (int j = 0; j < N; j += BLOCK_N) {
            auto Br = B + ldb * 0 + j;
            int ldb_r = ldb;

            if (Opt::pack_level() >= 1) {
                // Copy B to L1 (256x16: 16KB) cache buffer
                Opt::template pack<1>(Br, ldb, Kc, BLOCK_N, buf->Br, BLOCK_N);
                Br = buf->Br;
                ldb_r = BLOCK_N;
            }

            for (int i = 0; i < M; i += BLOCK_M) {
                auto Ar = A + lda * i + 0;
                auto Cr = C + ldc * i + j;

                //printf("# j = %d, i = %d, Mr = %d, Nr = %d, Kr = %d\n", j, i, BLOCK_M, BLOCK_N, Kc);
                if (Opt::do_pack())
                    Kernel::matmul_register_packed(BLOCK_M, BLOCK_N, K, Ar, lda, Br, ldb_r, Cr, ldc);
                else
                    Kernel::matmul_register(BLOCK_M, BLOCK_N, Kc, Ar, lda, Br, ldb_r, Cr, ldc);
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
typename make_blis_L1<Kernel, Opt>::blis_buffer * make_blis_L1<Kernel, Opt>::buf = nullptr;

template <class Kernel, class Opt>
struct blis {
    struct L1 : make_blis_L1<Kernel, Opt> {};

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
        for (int j = 0; j < N; j += BLOCK_N) {
            for (int k = 0; k < K; k += BLOCK_K) {
                auto Bc = B + ldb * k + j;
                auto Nc = std::min<int>(N - j, BLOCK_N);
                auto Kc = std::min<int>(K - k, BLOCK_K);
                auto ldb_c = ldb;

                if (Opt::pack_level() >= 3) {
                    // Copy B (256x3072) to L3 cache buffer
                    Opt::template pack<3>(Bc, ldb, Kc, Nc, L1::buf->Bc, Nc);
                    Bc = L1::buf->Bc;
                    ldb_c = Nc;
                }

                for (int i = 0; i < M; i += BLOCK_M) {
                    auto Ac = A + lda * i + k;
                    auto Cc = C + ldc * i + j;
                    auto Mc = std::min<int>(M - i, BLOCK_M);
                    
                    auto lda_c = lda;

                    if (Opt::pack_level() >= 2) {
                        // Copy A (144x256) to L2 cache buffer
                        Opt::template pack<2>(Ac, lda, Mc, Kc, L1::buf->Ac, Kc);
                        Ac = L1::buf->Ac;
                        lda_c = Opt::do_pack() ? L1::BLOCK_M : Kc;
                    }

                    //printf("# j = %d, k = %d, i = %d, Mc = %d, Nc = %d, Kc = %d\n",
                    //    j, k, i, Mc, Nc, Kc);
#if 1
                    L1::matmul(Mc, Nc, Kc, Ac, lda_c, Bc, ldb_c, Cc, ldc);
#else
                    naive::matmul(Mc, Nc, Kc, Ac, lda_c, Bc, ldb_c, Cc, ldc);
#endif
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
