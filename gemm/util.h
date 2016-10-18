#pragma once

#include <algorithm>
#include <limits>

#ifdef __INTELLISENSE__
#undef __AVX__
#define __AVX__ 1
#endif

#ifdef __AVX__
#define USE_AVX
#include <immintrin.h>
#endif


#if 0 && defined(_WIN32)
#define NOMINMAX
#include <windows.h>

class elapsed_time {
    double *elapsed_;
    LARGE_INTEGER start_;
public:
    elapsed_time(double *elapsed)
        : elapsed_(elapsed)
    {
        QueryPerformanceCounter(&start_);
    }

    ~elapsed_time()
    {
        LARGE_INTEGER stop;
        QueryPerformanceCounter(&stop);

        LARGE_INTEGER freq;
        QueryPerformanceFrequency(&freq);

        *elapsed_ =
            static_cast<double>(stop.QuadPart - start_.QuadPart) /
            static_cast<double>(freq.QuadPart);
    }
};
#else
#include <chrono>

namespace chrono = std::chrono;
class elapsed_time {
    using clock = chrono::high_resolution_clock;

    double& elapsed_;
    chrono::time_point<clock> start_;
public:
    elapsed_time(double& elapsed)
        : elapsed_(elapsed)
        , start_(clock::now())
    {
    }

    ~elapsed_time()
    {
        auto stop = clock::now();

        auto seconds = chrono::duration_cast<chrono::duration<double>>(stop - start_);
        elapsed_ = seconds.count();
    }
};
#endif

struct measure_results {
    double avg_time;
    double min_time;
    double max_time;

    measure_results(double avg, double min, double max)
        : avg_time(avg)
        , min_time(min)
        , max_time(max)
    {
    }
};

template <class F>
static double measure_seconds(F f)
{
    double t;
    {
        elapsed_time etime(t);
        f();
    }

    return t;
}

template <class F1, class F2>
static measure_results measure_ntimes(
    size_t n_times, F1 f, F2 preprocess)
{
    auto sum = 0.0;
    auto min = std::numeric_limits<double>::max();
    auto max = std::numeric_limits<double>::min();
    for (size_t i = 0; i < n_times; i++) {
        preprocess();

        auto t = measure_seconds(f);
        sum += t;
        min = std::min(min, t);
        max = std::max(max, t);
    }

    auto avg = sum / double(n_times);
    return measure_results(avg, min, max);
}

static void scale_matrix(
    float *A, int lda, int M, int N, float coeff)
{
    if (coeff == 1.0f)
        return;

    __m256 vcoeff = _mm256_set1_ps(coeff);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j+=8) {
            auto *pa = A + lda * i + j;
            auto va = _mm256_load_ps(pa);
            va = _mm256_mul_ps(vcoeff, va);
            _mm256_store_ps(pa, va);
        }
    }
}