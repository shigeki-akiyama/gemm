#pragma once

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