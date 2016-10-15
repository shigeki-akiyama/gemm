#pragma once

#include "util.h"

namespace naive {

    template <class T>
    static void gemm(
        int M, int N, int K, T alpha, T *A, int lda,
        T *B, int ldb, T beta, T *C, int ldc)
    {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                T ab = 0;
                for (int k = 0; k < K; k++) {
                    ab += A[lda * i + k] * B[ldb * k + j];
                }

                T& c = C[ldc * i + j];
                c = alpha * ab + beta * c;
            }
        }
    }

}

#ifdef USE_AVX
#include <immintrin.h>

namespace naive_avx {

    //template <class T>
    static void gemm(
        int M, int N, int K, float alpha, float *A, int lda,
        float *B, int ldb, float beta, float *C, int ldc)
        //int M, int N, int K, T alpha, T *A, int lda,
        //T *B, int ldb, T beta, T *C, int ldc)
    {
        auto valpha = _mm256_set1_ps(alpha);
        auto vbeta  = _mm256_set1_ps(beta);
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j+=8) {
                auto vab = _mm256_setzero_ps();
                for (int k = 0; k < K; k++) {
                    auto va = _mm256_set1_ps(A[lda * i + k]);
                    auto vb = _mm256_load_ps(&B[ldb * k + j]);
                    
                    vab = _mm256_fmadd_ps(va, vb, vab);
                }

                auto vc = _mm256_load_ps(&C[ldc * i + j]);

                vab = _mm256_mul_ps(valpha, vab);
                vc = _mm256_fmadd_ps(vbeta, vc, vab);

                _mm256_store_ps(&C[ldc * i + j], vc);
            }
        }
    }

}

#endif