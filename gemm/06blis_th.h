#pragma once

#include "config.h"
#include "01register.h"
#include "util.h"
#include <thread>
#include <cassert>
#undef NODEBUG

#include <pthread.h>


class spmd_master {
    size_t n_workers_;
    pthread_barrier_t barrier_;

public:
    spmd_master(size_t n_workers) : n_workers_(n_workers)
    {
        pthread_barrier_init(&barrier_, nullptr, n_workers_);
    }

    ~spmd_master()
    {
        pthread_barrier_destroy(&barrier_);
    }

    size_t n_workers() const { return n_workers_; }

    void barrier()
    {
        pthread_barrier_wait(&barrier_);
    }
};

class spmd_ops {
    spmd_master * master_;
    size_t id_;

public:
    spmd_ops(spmd_master& master, size_t id)
        : master_(&master)
        , id_(id)
    {
    }

    spmd_ops() : master_(nullptr), id_(0) {}
    spmd_ops(const spmd_ops& ops) = default;
    spmd_ops& operator=(const spmd_ops& ops) = default;

    size_t id() { return id_; }

    size_t n_workers() { return master_->n_workers(); }

    void barrier()
    {
        master_->barrier();
    }
};

class spmd_workers {
    spmd_master master_;
    std::vector<spmd_ops> spmd_ops_buf_;
    std::vector<std::thread> threads_;

public:
    template <class F, class... Args>
    spmd_workers(size_t n_workers, F start, Args... args)
        : master_(n_workers)
        , spmd_ops_buf_(n_workers)
        , threads_(n_workers)
    {
        for (size_t i = 0; i < n_workers; i++) {
            spmd_ops ops(master_, i);
            threads_[i] = std::thread([=]{
                start(ops, args...);
            });
        }

        for (auto& th : threads_) {
            th.join();
        }
    }

    ~spmd_workers() = default;
};

template <class Arch, class Kernel>
struct blis_th {
    enum : int {
        BLOCK_M = Arch::M_CACHE,
        BLOCK_N = Arch::N_CACHE,
        BLOCK_K = Arch::K_CACHE,

        MAX_THREADS = 256,
    };

    struct blis_buffer {
        float Ac[MAX_THREADS][BLOCK_M * BLOCK_K];    // L2: 144x256
        float Bc[BLOCK_K * BLOCK_N];    // L3: 256x3072
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
        spmd_ops spmd,
        int M, int N, int K, float *A, int lda,
        float *B, int ldb, float *C, int ldc)
    {
        auto me = spmd.id();
        auto n_workers = spmd.n_workers();
        auto n_blocks = (M + BLOCK_M - 1) / BLOCK_M;

        auto n_blocks_per_worker = n_blocks / n_workers;
        auto remain = n_blocks % n_workers;

        int M_begin, M_end;
        {
            int offset = 0;
            for (size_t i = 0; i < n_workers; i++) {
                M_begin = offset;
                offset += n_blocks_per_worker;
                if (i < remain) offset += 1;
                M_end = offset;

                if (i == me) break;
            }
           
            M_begin *= BLOCK_M;
            M_end *= BLOCK_M;

            if (me == n_workers - 1)
                M_end = M;
        }
#if 0
        std::printf("%zu/%zu: M_begin = %10d, M_end = %10d\n",
                    me, n_workers, M_begin, M_end);
#endif

        for (int j = 0; j < N; j += BLOCK_N) {
            for (int k = 0; k < K; k += BLOCK_K) {
                auto Bc = B + ldb * k + j;
                auto Nc = std::min<int>(N - j, BLOCK_N);
                auto Kc = std::min<int>(K - k, BLOCK_K);

                // Copy B (256x3072) to L3 cache buffer
                pack2d<float, 0, Kernel::BLOCK_N>(Bc, ldb, Kc, Nc, s_buf->Bc);
                Bc = s_buf->Bc;
                auto ldb_c = Kernel::BLOCK_N;
                
                for (int i = M_begin; i < M_end; i += BLOCK_M) {
                    auto Ac = A + lda * i + k;
                    auto Cc = C + ldc * i + j;
                    auto Mc = std::min<int>(M - i, BLOCK_M);

                    // Copy A (144x256) to L2 cache buffer
                    pack2d<float, Kernel::BLOCK_M, 0>(
                        Ac, lda, Mc, Kc, s_buf->Ac[me]);
                    Ac = s_buf->Ac[me];
                    auto lda_c = Kernel::BLOCK_M;

                    matmul_cache(Mc, Nc, Kc, Ac, lda_c, Bc, ldb_c, Cc, ldc);
                }

                spmd.barrier();
            }
        }
    }

    static void gemm(
        int M, int N, int K, float alpha, float *A, int lda,
        float *B, int ldb, float beta, float *C, int ldc)
    {
        scale_matrix(A, lda, M, K, alpha);
        scale_matrix(C, ldc, M, N, beta);

        size_t n_workers = 4;
        spmd_workers workers(n_workers, [=](spmd_ops spmd){
            matmul_spmd(spmd, M, N, K, A, lda, B, ldb, C, ldc);
        });
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
typename blis_th<Arch, Kernel>::blis_buffer *
blis_th<Arch, Kernel>::s_buf = nullptr;

