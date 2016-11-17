#pragma once

#include "config.h"
#include "01register.h"
#include "util.h"
#include <thread>
#include <cassert>
#undef NODEBUG

#include <omp.h>
#include <hwloc.h>

static void set_affinity()
{
    hwloc_topology_t topo;

    hwloc_topology_init(&topo);
    hwloc_topology_load(topo);

    auto depth = hwloc_topology_get_depth(topo);

    auto last = depth - 1;
    auto n_objs = hwloc_get_nbobjs_by_depth(topo, last);

    #pragma omp parallel
    {
#if THREAD_BIND
        auto me = omp_get_thread_num();

        auto obj = hwloc_get_obj_by_depth(topo, last, me);
        assert(obj != nullptr);

        auto cpuset = hwloc_bitmap_dup(obj->cpuset);
        hwloc_bitmap_singlify(cpuset);

        auto flags = HWLOC_CPUBIND_THREAD | HWLOC_CPUBIND_STRICT;
        auto r = hwloc_set_cpubind(topo, cpuset, flags);
        assert(r == 0);

        hwloc_bitmap_free(cpuset);
#endif
    }

    hwloc_topology_destroy(topo);
}

template <class Arch, class Kernel>
struct blis_th {
    enum : int {
        BLOCK_M = Arch::M_CACHE,
        BLOCK_N = Arch::N_CACHE,
        BLOCK_K = Arch::K_CACHE,

        MAX_THREADS = 256,
    };

    struct blis_buffer {
        float Ac[MAX_THREADS][BLOCK_M * BLOCK_K];       // L2: 144x256
        float Bc[BLOCK_K * BLOCK_N];                    // L3: 256x3072
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

            for (int i = 0; i < M; i += Kernel::BLOCK_M) {
                auto Ar = A + K * i;
                auto Cr = C + ldc * i + j;

                Kernel::matmul_register_packed(
                    Kernel::BLOCK_M, Kernel::BLOCK_N, K,
                    Ar, lda, Br, ldb, Cr, ldc);
            }
        }
    }

    static void matmul_spmd(
        int M, int N, int K, float *A, int lda,
        float *B, int ldb, float *C, int ldc)
    {
        auto me = omp_get_thread_num();
        auto n_workers = omp_get_num_threads();

        int M_begin, M_end;
        partition(me, n_workers, M, 1, &M_begin, &M_end);

        if (0) {
            std::printf("%2d: M_begin = %10d, M_end = %10d\n",
                        me, M_begin, M_end);
        }

        for (int j = 0; j < N; j += BLOCK_N) {
            for (int k = 0; k < K; k += BLOCK_K) {
                auto Bc = B + ldb * k + j;
                auto Nc = std::min<int>(N - j, BLOCK_N);
                auto Kc = std::min<int>(K - k, BLOCK_K);

#if 1
                // Parallel packing
                {
                    int Nc_begin, Nc_end;
                    int align = Kernel::BLOCK_N;
                    partition(me, n_workers, Nc, align, &Nc_begin, &Nc_end);

                    if (0) {
                        printf("%d: Nc_begin = %5d, Nc_end = %5d\n",
                               me, Nc_begin, Nc_end);
                    }

                    // Copy B (256x3072) to L3 cache buffer
                    pack2d_col_spmd<float, Kernel::BLOCK_N>(
                        Bc, ldb, Kc, Nc_begin, Nc_end, s_buf->Bc);
                }
#else
                // Sequential packing
                if (me == 0) {
                    // Copy B (256x3072) to L3 cache buffer
                    pack2d<float,0, Kernel::BLOCK_N>(
                        Bc, ldb, Kc, Nc, s_buf->Bc);
                }
#endif
                Bc = s_buf->Bc;
                auto ldb_c = Kernel::BLOCK_N;

                #pragma omp barrier
               
                for (int i = M_begin; i < M_end; i += BLOCK_M) {
                    auto Ac = A + lda * i + k;
                    auto Cc = C + ldc * i + j;
                    auto Mc = std::min<int>(M_end - i, BLOCK_M);

                    if (0) printf("Mc = %d, M_end = %d\n", Mc, M_end);

                    // Copy A (144x256) to L2 cache buffer
                    pack2d<float, Kernel::BLOCK_M, 0>(
                        Ac, lda, Mc, Kc, s_buf->Ac[me]);
                    Ac = s_buf->Ac[me];
                    auto lda_c = Kernel::BLOCK_M;

                    matmul_cache(Mc, Nc, Kc, Ac, lda_c, Bc, ldb_c, Cc, ldc);
                }

                #pragma omp barrier
            }
        }
    }

    static void gemm(
        int M, int N, int K, float alpha, float *A, int lda,
        float *B, int ldb, float beta, float *C, int ldc)
    {
        scale_matrix(A, lda, M, K, alpha);
        scale_matrix(C, ldc, M, N, beta);

        #pragma omp parallel
        {
            matmul_spmd(M, N, K, A, lda, B, ldb, C, ldc);
        }
    }

    static void intiialize()
    {
        if (s_buf == nullptr) {
            s_buf = (blis_buffer *)_mm_malloc(sizeof(blis_buffer), PAGE_SIZE);
            memset(s_buf, 0, sizeof(blis_buffer));
        }
    }

private:

    static void partition(size_t me, size_t n_workers, int N, int align,
                          int * N_begin, int * N_end)
    {
        size_t per_worker = N / n_workers;
        size_t remain = N % n_workers;
        
        int begin, end;
        if (me < remain) {
            begin = (per_worker + 1) * me;
            end   = begin + (per_worker + 1);
        } else {
            begin = (per_worker + 1) * remain + per_worker * (me - remain);
            end   = begin + per_worker;
        }

        begin = begin / align * align;
        end   = end / align * align;

        if (me == n_workers - 1)
            end = N;

        *N_begin = begin;
        *N_end   = end;
    }

    template <class T, int CS>
    static void pack2d_col_spmd(
        const T *A, int lda, int M, int N_begin, int N_end, T *B)
    {
        assert((N_end - N_begin) % CS == 0);

        for (int i = N_begin; i < N_end; i += CS) {
            auto Ab = A + i;
            auto Bb = B + M * i;

            copy2d(Ab, lda, M, CS, Bb, CS);
        }
    }

};
template <class Arch, class Kernel>
typename blis_th<Arch, Kernel>::blis_buffer *
blis_th<Arch, Kernel>::s_buf = nullptr;

