/*
 * bench/nonmatmul_bench.c - cost of the ViT's non-matrix operations on one ARM core.
 *
 * Runs softmax, LayerNorm and GELU in plain C (as a CPU fallback would) on
 * slices with the wearable operating point's shapes, measures simulated time,
 * and extrapolates to a whole frame at both operating points.
 * Meant for gem5: one ArmMinorCPU at 2 GHz, the same CPU model as the SLAM runs.
 * Plain scalar C with libm, so this is a pessimistic (upper-bound) cost;
 * hand-tuned NEON code would be several times faster.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CPU_GHZ 2.0

static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static float frand(uint32_t *s)                    /* deterministic values in [-2, 2) */
{
    *s = *s * 1664525u + 1013904223u;
    return ((*s >> 8) & 0xFFFF) / 65536.0f * 4.0f - 2.0f;
}

/* Softmax over each row: exp(x - max) / sum. */
static void softmax_rows(float *x, int rows, int len)
{
    for (int r = 0; r < rows; r++) {
        float *row = x + (size_t)r * len;
        float mx = row[0];
        for (int i = 1; i < len; i++) if (row[i] > mx) mx = row[i];
        float sum = 0.0f;
        for (int i = 0; i < len; i++) { row[i] = expf(row[i] - mx); sum += row[i]; }
        float inv = 1.0f / sum;
        for (int i = 0; i < len; i++) row[i] *= inv;
    }
}

/* LayerNorm over each token's dim values, with scale g and shift b. */
static void layernorm(float *x, const float *g, const float *b, int tokens, int dim)
{
    for (int t = 0; t < tokens; t++) {
        float *v = x + (size_t)t * dim;
        float mean = 0.0f, var = 0.0f;
        for (int i = 0; i < dim; i++) mean += v[i];
        mean /= dim;
        for (int i = 0; i < dim; i++) { float d = v[i] - mean; var += d * d; }
        float inv = 1.0f / sqrtf(var / dim + 1e-6f);
        for (int i = 0; i < dim; i++) v[i] = (v[i] - mean) * inv * g[i] + b[i];
    }
}

/* Exact GELU, x * Phi(x), as used by ViT / DINOv2. */
static void gelu(float *x, int n)
{
    for (int i = 0; i < n; i++) x[i] = 0.5f * x[i] * (1.0f + erff(x[i] * 0.70710678f));
}

static double checksum(const float *x, int n)
{
    double s = 0.0;
    for (int i = 0; i < n; i++) s += x[i];
    return s;
}

static void report(const char *name, int n, uint64_t ns, double frame_wear, double frame_full)
{
    double ns_el = (double)ns / n;
    printf("[bench] %-9s %8d elements  %7.2f ns/el  %7.1f cycles/el  "
           "-> per frame: wearable %8.1f ms (%.1fM el), full-res %9.1f ms (%.0fM el)\n",
           name, n, ns_el, ns_el * CPU_GHZ,
           ns_el * frame_wear / 1e6, frame_wear / 1e6,
           ns_el * frame_full / 1e6, frame_full / 1e6);
}

int main(void)
{
    /* Slices of the real shapes (kept small so gem5 finishes quickly). */
    const int SM_ROWS = 64,  SM_LEN = 1160;     /* 64 rows of one head's 1160 x 1160 scores */
    const int LN_TOK  = 128, LN_DIM = 384;      /* 128 tokens of the 384-wide embedding     */
    const int GE_N    = 64 * 1536;              /* 64 tokens of the 1536-wide MLP output    */

    /* Elements per frame (12 blocks): wearable 1160 tokens, full-res 4515 tokens. */
    const double SM_WEAR = 1160.0 * 1160.0 * 6 * 12, SM_FULL = 4515.0 * 4515.0 * 6 * 12;
    const double LN_WEAR = 2.0 * 1160 * 384 * 12,    LN_FULL = 2.0 * 4515 * 384 * 12;
    const double GE_WEAR = 1160.0 * 1536 * 12,       GE_FULL = 4515.0 * 1536 * 12;

    uint32_t seed = 12345;
    int n_sm = SM_ROWS * SM_LEN, n_ln = LN_TOK * LN_DIM;
    float *sm = malloc(sizeof(float) * n_sm);
    float *ln = malloc(sizeof(float) * n_ln);
    float *ge = malloc(sizeof(float) * GE_N);
    float *g  = malloc(sizeof(float) * LN_DIM);
    float *b  = malloc(sizeof(float) * LN_DIM);
    if (!sm || !ln || !ge || !g || !b) { fprintf(stderr, "malloc failed\n"); return 1; }
    for (int i = 0; i < n_sm; i++)   sm[i] = frand(&seed);
    for (int i = 0; i < n_ln; i++)   ln[i] = frand(&seed);
    for (int i = 0; i < GE_N; i++)   ge[i] = frand(&seed);
    for (int i = 0; i < LN_DIM; i++) { g[i] = 1.0f + 0.1f * frand(&seed); b[i] = 0.1f * frand(&seed); }

    printf("[bench] non-matmul ops, plain C, CPU at %.1f GHz\n", CPU_GHZ);

    uint64_t t = now_ns(); softmax_rows(sm, SM_ROWS, SM_LEN);        uint64_t t_sm = now_ns() - t;
    t = now_ns();          layernorm(ln, g, b, LN_TOK, LN_DIM);      uint64_t t_ln = now_ns() - t;
    t = now_ns();          gelu(ge, GE_N);                           uint64_t t_ge = now_ns() - t;

    report("softmax",   n_sm, t_sm, SM_WEAR, SM_FULL);
    report("layernorm", n_ln, t_ln, LN_WEAR, LN_FULL);
    report("gelu",      GE_N, t_ge, GE_WEAR, GE_FULL);
    printf("[bench] checksums %.4f %.4f %.4f\n", checksum(sm, n_sm), checksum(ln, n_ln), checksum(ge, GE_N));
    return 0;
}
