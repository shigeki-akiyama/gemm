#pragma once

#include "config.h"
#include "01register.h"
#include "util.h"
#include <cassert>
#undef NODEBUG

#ifdef _OPENMP
#include <omp.h>
#endif

template <class Arch, class Kernel>
struct blis_omp {
    enum : int {
        BLOCK_M = Arch::M_CACHE,
        BLOCK_N = Arch::N_CACHE,
        BLOCK_K = Arch::K_CACHE,

        MAX_THREADS = 64,
    };

    struct blis_buffer {
        float Ac[MAX_THREADS][BLOCK_M * BLOCK_K];   // L2: 144x256
        float Bc[BLOCK_K * BLOCK_N];                // L3: 256x3072
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

            // for SMT
            //#pragma omp parallel for
            for (int i = 0; i < M; i += Kernel::BLOCK_M) {
                auto Ar = A + K * i;
                auto Cr = C + ldc * i + j;

#if 0
                printf("## j = %3d, i = %3d, Mr = %3d, Nr = %3d, Kr = %3d, "
                    "Ar[0]= %f, Br[0] = %f\n",
                    j, i, Kernel::BLOCK_M, Kernel::BLOCK_N, K,
                    Ar[0], Br[0]);
#endif

                Kernel::matmul_register_packed(
                    Kernel::BLOCK_M, Kernel::BLOCK_N, K,
                    Ar, lda, Br, ldb, Cr, ldc);
            }
        }
    }

    static void matmul(
        int M, int N, int K, float *A, int lda,
        float *B, int ldb, float *C, int ldc)
    {
        size_t total_time = 0;
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

                auto t = rdtsc();

                // Copy B (256x3072) to L3 cache buffer
                pack2d<float, 0, Kernel::BLOCK_N>(Bc, ldb, Kc, Nc, s_buf->Bc);
                Bc = s_buf->Bc;
                auto ldb_c = Kernel::BLOCK_N;

                packL3_time += rdtsc() - t;
                packL3_size += Kc * Nc;

                // for physical cores
                #pragma omp parallel for
                for (int i = 0; i < M; i += BLOCK_M) {
                    auto Ac = A + lda * i + k;
                    auto Cc = C + ldc * i + j;
                    auto Mc = std::min<int>(M - i, BLOCK_M);
#ifdef _OPENMP
                    auto tid = omp_get_thread_num();
                    assert(tid < MAX_THREADS);
#else
                    auto tid = 0;
#endif
                    auto t = rdtsc();

                    // Copy A (144x256) to L2 cache buffer
                    pack2d<float, Kernel::BLOCK_M, 0>(
                        Ac, lda, Mc, Kc, s_buf->Ac[tid]);
                    Ac = s_buf->Ac[tid];
                    auto lda_c = Kernel::BLOCK_M;

                    packL2_time += rdtsc() - t;
                    packL2_size += Mc * Kc;

#if 0
                    printf("#  j = %3d, k = %3d, i  = %3d, "
                        "Mc = %3d, Nc = %3d, Kc = %3d\n",
                        j, k, i, Mc, Nc, Kc);
#endif

                    matmul_cache(Mc, Nc, Kc, Ac, lda_c, Bc, ldb_c, Cc, ldc);
                }
            }
        }
#if PACK_CYCLES
        total_time = rdtsc() - t;
        auto packL3_bw = double(4 * packL3_size) / double(packL3_time);
        auto packL2_bw = double(4 * packL2_size) / double(packL2_time);
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

    static void intiialize()
    {
        if (s_buf == nullptr) {
            s_buf = (blis_buffer *)_mm_malloc(sizeof(blis_buffer), PAGE_SIZE);
            memset(s_buf, 0, sizeof(blis_buffer));
        }
    }
};
template <class Arch, class Kernel>
typename blis_omp<Arch, Kernel>::blis_buffer * 
blis_omp<Arch, Kernel>::s_buf = nullptr;
