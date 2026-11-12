// backend/accel_proxy.hpp
// Accelerator proxy thread for gem5 SE runs (mesh config: gem5_accel/2x4_accel.py).
//
// Computes nothing: it reproduces the systolic-array accelerator's memory
// traffic on the NoC. In each period it streams through buffers larger than
// the 16 MiB LLC for duty_pct percent of the time (about 3 cache-line reads per
// write, as in the SCALE-Sim OS / 2 MB results), then spins in cache for the
// rest of the period. It runs until the main thread sets `stop`, then fills in
// the results and sets `finished`.
//
// Like main_mc.cpp's barrier, it uses only plain atomic loads and stores (no
// read-modify-write, which never completes under gem5 SE + Ruby), the two flags
// each sit on their own 64-byte cache line, and it is never pthread_join'ed
// (thread-exit wake-ups are unreliable in gem5 SE mode).
//
// Start it with pthread_create AFTER all other worker threads, so that gem5
// places it on the last CPU (cpu4 = router 3 with 2x4_accel.py).
//
// One in-order core issuing loads is much slower than a DMA engine, so this
// traffic is a LOWER BOUND on the real accelerator's load.
#pragma once
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>

struct alignas(64) AccelProxyFlag {
    std::atomic<unsigned> v{0};
    char pad[64 - sizeof(std::atomic<unsigned>)];
};

struct AccelProxyArgs {
    double period_us  = 1000.0;   // one busy + idle cycle
    double duty_pct   = 0.0;      // percent of each period spent streaming
    size_t buffer_mib = 64;       // streamed input buffer (output buffer = 1/4 of this)
    AccelProxyFlag stop;          // written only by the main thread
    AccelProxyFlag finished;      // written only by the proxy, after the results
    // Results, valid once finished.v != 0.
    uint64_t periods = 0, lines_read = 0, lines_written = 0;
    double   elapsed_s = 0.0;
    int64_t  checksum = 0;
};

inline uint64_t accel_proxy_now_ns()
{
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);   // simulated time in gem5 SE mode
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

// pthread entry point: pthread_create(&tid, nullptr, accel_proxy_thread, &args);
inline void *accel_proxy_thread(void *p)
{
    AccelProxyArgs &a = *static_cast<AccelProxyArgs *>(p);
    const size_t LINE_WORDS = 8;                       // 64-byte line = 8 x int64
    const size_t in_lines   = a.buffer_mib * 1024 * 1024 / 64;
    const size_t out_lines  = in_lines / 4;
    volatile int64_t *in  = static_cast<volatile int64_t *>(malloc(in_lines * 64));
    volatile int64_t *out = static_cast<volatile int64_t *>(malloc(out_lines * 64));
    if (!in || !out) {
        std::fprintf(stderr, "[proxy] malloc failed\n");
        a.finished.v.store(1, std::memory_order_release);
        return nullptr;
    }

    const uint64_t period = (uint64_t)(a.period_us * 1e3);
    const uint64_t busy   = (uint64_t)(period * a.duty_pct / 100.0);
    const uint64_t t0     = accel_proxy_now_ns();
    std::printf("[proxy] start: period %.1f us, duty %.0f%%, buffers %zu + %zu MiB\n",
                a.period_us, a.duty_pct, a.buffer_mib, a.buffer_mib / 4);

    size_t   ri = 0, wi = 0;
    int64_t  sum = 0;
    uint64_t nread = 0, nwritten = 0, nperiods = 0;
    auto stopped = [&]() { return a.stop.v.load(std::memory_order_acquire) != 0; };

    while (!stopped()) {
        const uint64_t p_start = accel_proxy_now_ns();

        // Busy: stream reads, one write per 4 reads; check the clock every 64 lines.
        while (accel_proxy_now_ns() - p_start < busy && !stopped()) {
            for (int b = 0; b < 64; b++) {
                sum += in[ri * LINE_WORDS];
                if (++ri == in_lines) ri = 0;
                nread++;
                if ((nread & 3) == 0) {
                    out[wi * LINE_WORDS] = sum;
                    if (++wi == out_lines) wi = 0;
                    nwritten++;
                }
            }
        }

        // Idle: spin until the period ends (no memory traffic).
        while (accel_proxy_now_ns() - p_start < period && !stopped())
            ;
        nperiods++;
    }

    a.periods = nperiods;
    a.lines_read = nread;
    a.lines_written = nwritten;
    a.elapsed_s = (accel_proxy_now_ns() - t0) / 1e9;
    a.checksum  = sum;
    a.finished.v.store(1, std::memory_order_release);  // publish the results
    return nullptr;                                     // buffers are left to process exit
}

// Call once finished.v != 0.
inline void accel_proxy_report(const AccelProxyArgs &a)
{
    const double mbytes = (a.lines_read + a.lines_written) * 64.0 / 1e6;
    std::printf("[proxy] done: %llu periods, %llu lines read, %llu written, %.1f MB in %.3f ms "
                "(%.2f GB/s avg), checksum %lld\n",
                (unsigned long long)a.periods, (unsigned long long)a.lines_read,
                (unsigned long long)a.lines_written, mbytes, a.elapsed_s * 1e3,
                a.elapsed_s > 0 ? mbytes / 1e3 / a.elapsed_s : 0.0, (long long)a.checksum);
}
