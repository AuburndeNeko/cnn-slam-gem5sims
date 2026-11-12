/*
 * accel_proxy.c - stand-in for the systolic-array accelerator in gem5 SE mode.
 *
 * Computes nothing; it only reproduces the accelerator's memory traffic on
 * the NoC. In each period it streams through buffers larger than the 16 MiB
 * LLC for `duty` percent of the time (accelerator busy: ~3 cache-line reads
 * for every write, matching the SCALE-Sim OS/2 MB results), then spins in
 * cache for the rest of the period (accelerator idle). After run_ms of
 * simulated time it prints a summary and exits.
 *
 * Usage: accel_proxy <run_ms> <period_us> <duty_pct> [buffer_mib]
 *   run_ms      simulated time to run before exiting
 *   period_us   length of one busy+idle cycle
 *   duty_pct    percent of each period spent streaming (0-100)
 *   buffer_mib  size of the streamed input buffer (default 64; output = 1/4)
 *
 * Timing uses clock_gettime, which gem5 SE mode answers with simulated time.
 * One core issuing loads is far slower than a DMA engine, so the traffic is
 * a LOWER BOUND on the real accelerator's load.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define LINE_WORDS 8          /* 64-byte cache line = 8 x int64 */

static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: %s <run_ms> <period_us> <duty_pct> [buffer_mib]\n", argv[0]);
        return 1;
    }
    double run_ms    = atof(argv[1]);
    double period_us = atof(argv[2]);
    double duty      = atof(argv[3]) / 100.0;
    size_t mib       = (argc > 4) ? strtoul(argv[4], NULL, 10) : 64;

    size_t in_lines  = mib * 1024 * 1024 / 64;
    size_t out_lines = in_lines / 4;
    volatile int64_t *in  = malloc(in_lines  * 64);
    volatile int64_t *out = malloc(out_lines * 64);
    if (!in || !out) {
        fprintf(stderr, "[proxy] malloc failed\n");
        return 1;
    }

    uint64_t period = (uint64_t)(period_us * 1e3);
    uint64_t busy   = (uint64_t)(period * duty);
    uint64_t t0     = now_ns();
    uint64_t t_end  = t0 + (uint64_t)(run_ms * 1e6);

    printf("[proxy] start: run %.3f ms, period %.1f us, duty %.0f%%, buffers %zu + %zu MiB\n",
           run_ms, period_us, duty * 100.0, mib, mib / 4);
    fflush(stdout);

    size_t   ri = 0, wi = 0;
    int64_t  sum = 0;
    uint64_t lines_read = 0, lines_written = 0, periods = 0;

    for (;;) {
        uint64_t p_start = now_ns();
        if (p_start >= t_end)
            break;

        /* Busy: stream reads, one write per 4 reads. Check the clock every 64 lines. */
        while (now_ns() - p_start < busy) {
            for (int b = 0; b < 64; b++) {
                sum += in[ri * LINE_WORDS];
                if (++ri == in_lines) ri = 0;
                lines_read++;
                if ((lines_read & 3) == 0) {
                    out[wi * LINE_WORDS] = sum;
                    if (++wi == out_lines) wi = 0;
                    lines_written++;
                }
            }
        }

        /* Idle: spin until the period ends (no memory traffic). */
        while (now_ns() - p_start < period)
            ;
        periods++;
    }

    double elapsed_s = (now_ns() - t0) / 1e9;
    double mbytes    = (lines_read + lines_written) * 64.0 / 1e6;
    printf("[proxy] done: %llu periods, %llu lines read, %llu written, %.1f MB in %.3f ms "
           "(%.2f GB/s avg), checksum %lld\n",
           (unsigned long long)periods, (unsigned long long)lines_read,
           (unsigned long long)lines_written, mbytes, elapsed_s * 1e3,
           mbytes / 1e3 / elapsed_s, (long long)sum);
    return 0;
}
