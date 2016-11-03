#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <random>

static constexpr int LINE_SIZE = 64;
static constexpr int PAGE_SIZE = 4096;

#if defined(_MSC_VER)
#define NOINLINE    __declspec(noinline)
#else
#define NOINLINE    __attribute__((noinline))
#endif

static int alignup(int size, int align)
{
    return (size + align - 1) / align * align;
}

template <class T>
static void fill_random(T *arr, size_t size, int seed)
{
    std::mt19937 engine(seed);
    //std::uniform_real_distribution<T> dist(0.5, 2.0);
    std::uniform_int_distribution<> dist(0, 3);

    for (size_t i = 0; i < size; i++) {
#if 1
        switch (dist(engine)) {
        case 0: arr[i] = T(0.25); break;
        case 1: arr[i] = T(0.50); break;
        case 2: arr[i] = T(1.00); break;
        case 3: arr[i] = T(2.00); break;
        }
#else
        arr[i] = T(dist(engine));
#endif
    }
}

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

#if 0 //defined(_WIN32)
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

static double calc_cpu_freq()
{
	namespace C = std::chrono;
	using clock = C::high_resolution_clock;

	constexpr double time = 0.01; // sec

	auto t0 = clock::now();
	auto cycle0 = rdtsc();

	for (;;) {
		auto d = C::duration_cast<C::duration<double>>(clock::now() - t0);
		if (d.count() >= time) {
			break;
		}
	}
	auto cycle1 = rdtsc();

	return double(cycle1 - cycle0) / time * 1e-9;
}

static double g_cpu_freq = calc_cpu_freq();

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
        elapsed_ = double(stop - start_) / g_cpu_freq * 1e-9;
    }
};
#endif

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
