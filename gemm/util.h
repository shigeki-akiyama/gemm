#pragma once

#include <cstddef>
#include <cstdint>

static size_t rdtsc()
{
#if defined(_MSC_VER)
    return __rdtsc();
#else
    uint32_t hi, lo;
    asm volatile("rdtsc" : "=a"(lo), "=d"(hi));
    //asm volatile("lfence\nrdtsc" : "=a"(lo), "=d"(hi));
    return uint64_t(hi) << 32 | lo;
#endif
}

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>

class elapsed_time {
    double& elapsed_;
    LARGE_INTEGER start_;
public:
    elapsed_time(double& elapsed)
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

        elapsed_ = double(stop.QuadPart - start_.QuadPart) / double(freq.QuadPart);
    }
};

#else
#include <chrono>

#if 0
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
#else
class elapsed_time {
    double& elapsed_;
    size_t start_;
public:
    elapsed_time(double& elapsed)
        : elapsed_(elapsed)
        , start_(rdtsc())
    {
    }

    ~elapsed_time()
    {
        auto stop = rdtsc();
        elapsed_ = double(stop - start_) / 3.1e9;
    }
};
#endif

#endif
