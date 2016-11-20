#pragma once

#include "config.h"
#include "01register.h"
#include "util.h"
#include <cassert>
#undef NODEBUG

struct blis_opt {
    template <int LEVEL, class Packer>
    struct make_option {
        static constexpr int pack_level() { return LEVEL; }

        static constexpr bool do_pack() { return Packer::do_pack(); }

        template <class T, int CURR_LEVEL>
        static void pack(const T *A, int lda, int M, int N, T *B, int ldb)
        {
            Packer::template pack<T, CURR_LEVEL>(A, lda, M, N, B, ldb);
        }
    };

    struct matcopy {
        static constexpr bool do_pack() { return false; }

        template <class T, int CURR_LEVEL>
        static void pack(const T *A, int lda, int M, int N, T *B, int ldb)
        {
            copy2d(A, lda, M, N, B, ldb);
        }
    };

    template <int LEVEL, int RS, int CS>   // raw-major and column-major stride
    struct matpack {
        static constexpr bool do_pack() { return true; }

        template <class T, int CURR_LEVEL>
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


template <class Arch, class Kernel, class Opt> struct blis_copy;

template <class Arch, class Kernel, class Opt>
struct make_blis_copy_L1 {
    using BLIS = blis_copy<Arch, Kernel, Opt>;

    enum : int {
        BLOCK_M = Kernel::BLOCK_M,
        BLOCK_N = Kernel::BLOCK_N,
    };
    static_assert(BLIS::BLOCK_M % BLOCK_M == 0, "BLIS::BLOCK_M is invalid.");
    static_assert(BLIS::BLOCK_N % BLOCK_N == 0, "BLIS::BLOCK_N is invalid.");

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
                Opt::template pack<float, 1>(
                    Br, ldb, Kc, BLOCK_N, buf->Br, BLOCK_N);
                Br = buf->Br;
                ldb_r = BLOCK_N;
            }

            for (int i = 0; i < M; i += BLOCK_M) {
                auto Ar = A + lda * i + 0;
                auto Cr = C + ldc * i + j;

                //printf("# j = %d, i = %d, Mr = %d, Nr = %d, Kr = %d\n",
                //       j, i_M_N, Kc);
                if (Opt::do_pack())
                    Kernel::matmul_register_packed(
                        BLOCK_M, BLOCK_N, K, Ar, lda, Br, ldb_r, Cr, ldc);
                else
                    Kernel::matmul_register(
                        BLOCK_M, BLOCK_N, Kc, Ar, lda, Br, ldb_r, Cr, ldc);
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
template <class Arch, class Kernel, class Opt>
typename make_blis_copy_L1<Arch, Kernel, Opt>::blis_buffer *
make_blis_copy_L1<Arch, Kernel, Opt>::buf = nullptr;

template <class Arch, class Kernel, class Opt>
struct blis_copy {
    struct L1 : make_blis_copy_L1<Arch, Kernel, Opt> {};

    enum : int {
        BLOCK_M = Arch::M_CACHE,
        BLOCK_N = Arch::N_CACHE,
        BLOCK_K = Arch::K_CACHE,

        LDA_C   = BLOCK_K + 8,
        LDB_C   = BLOCK_N + 8,
    };

    alignas(LINE_SIZE) static float Ac_buf[BLOCK_M * LDA_C];
    alignas(LINE_SIZE) static float Bc_buf[BLOCK_K * LDB_C];

    static void matmul(
        int M, int N, int K, float *A, int lda,
        float *B, int ldb, float *C, int ldc)
    {
        size_t total_time  = 0;
        size_t packL3_time = 0;
        size_t packL3_size = 0;
        size_t packL2_time = 0;
        size_t packL2_size = 0;

        auto t = rdtsc();

        for (int j = 0; j < N; j += BLOCK_N) {
            for (int k = 0; k < K; k += BLOCK_K) {
                auto Bc = B + ldb * k + j;
                auto Nc = std::min<int>(N - j, BLOCK_N);
                auto Kc = std::min<int>(K - k, BLOCK_K);
                auto ldb_c = ldb;

                if (Opt::pack_level() >= 3) {

                    auto t = rdtsc();

                    // Copy B (256x3072) to L3 cache buffer
                    Opt::template pack<float, 3>(
                        Bc, ldb, Kc, Nc, L1::buf->Bc, Nc);
                    Bc = L1::buf->Bc;
                    ldb_c = Nc;

                    packL3_time += rdtsc() - t;
                    packL3_size += Kc * Nc;
                }

                for (int i = 0; i < M; i += BLOCK_M) {
                    auto Ac = A + lda * i + k;
                    auto Cc = C + ldc * i + j;
                    auto Mc = std::min<int>(M - i, BLOCK_M);
                    
                    auto lda_c = lda;

                    if (Opt::pack_level() >= 2) {

                        auto t = rdtsc();

                        // Copy A (144x256) to L2 cache buffer
                        Opt::template pack<float, 2>(
                            Ac, lda, Mc, Kc, L1::buf->Ac, Kc);
                        Ac = L1::buf->Ac;
                        lda_c = Opt::do_pack() ? L1::BLOCK_M : Kc;

                        packL2_time += rdtsc() - t;
                        packL2_size += Mc * Kc;
                    }

                    //printf("# j = %d, k = %d, i = %d, "
                    //       "Mc = %d, Nc = %d, Kc = %d\n",
                    //       j, k, i, Mc, Nc, Kc);

                    L1::matmul(Mc, Nc, Kc, Ac, lda_c, Bc, ldb_c, Cc, ldc);
                }
            }
        }
#ifdef PACK_CYCLES
        total_time = rdtsc() - t;
        auto packL3_bw = double (4 * packL3_size) / double(packL3_time);
        auto packL2_bw = double (4 * packL2_size) / double(packL2_time);
        printf("total_time  = %11zu\n", total_time);
        printf("packL3_time = %11zu\n", packL3_time);
        printf("packL2_time = %11zu\n", packL2_time);
        printf("packL3_size = %11zu\n", packL3_size);
        printf("packL2_size = %11zu\n", packL3_size);
        printf("packL3_bw   = %11.6f\n", packL3_bw);
        printf("packL2_bw   = %11.6f\n", packL2_bw);
#endif
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
        L1::initialize();
    }
};
