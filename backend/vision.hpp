// On-chip vision stage for the multi-core workload.
//
// Per frame, on real KITTI data:
//   1. Harris corner detection on the grayscale image (feature front-end)
//   2. Obstacle detection on the CNN depth map: three sectors (left/centre/
//      right) in the lower part of the image, counting pixels closer than a
//      hazard distance (what a wearable would warn about)
//
// Work is split into horizontal strips, one per worker thread. Each strip
// recomputes a small halo of rows, so results are bit-identical for any number
// of threads (integer arithmetic only) - a built-in correctness check.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

struct ImageSeq {
    uint32_t n = 0, w = 0, h = 0;
    std::vector<uint8_t> gray;      // n * h * w
    std::vector<uint16_t> depth;    // n * h * w, cm

    const uint8_t *g(int k) const { return &gray[(size_t)k * w * h]; }
    const uint16_t *d(int k) const { return &depth[(size_t)k * w * h]; }

    // ds = 1 (full 1242x375) or 2 (621x187, 4x less work - for long gem5 sweeps)
    bool load(const char *path, int max_frames, int ds, char *err, size_t errlen) {
        FILE *f = std::fopen(path, "rb");
        if (!f) { std::snprintf(err, errlen, "cannot open %s", path); return false; }
        char magic[4]; uint32_t ver, N, W, H;
        bool ok = std::fread(magic, 1, 4, f) == 4 && !std::memcmp(magic, "KIMG", 4) &&
                  std::fread(&ver, 4, 1, f) == 1 && std::fread(&N, 4, 1, f) == 1 &&
                  std::fread(&W, 4, 1, f) == 1 && std::fread(&H, 4, 1, f) == 1 && ver == 1;
        if (!ok) { std::snprintf(err, errlen, "bad KIMG header"); std::fclose(f); return false; }
        if (max_frames > 0 && (uint32_t)max_frames < N) N = (uint32_t)max_frames;
        if (ds < 1) ds = 1;
        n = N; w = W / ds; h = H / ds;
        gray.resize((size_t)n * w * h);
        depth.resize((size_t)n * w * h);
        std::vector<uint8_t> g0((size_t)W * H);
        std::vector<uint16_t> d0((size_t)W * H);
        for (uint32_t k = 0; k < n && ok; ++k) {
            ok = std::fread(g0.data(), 1, g0.size(), f) == g0.size() &&
                 std::fread(d0.data(), 2, d0.size(), f) == d0.size();
            uint8_t *G = &gray[(size_t)k * w * h];
            uint16_t *D = &depth[(size_t)k * w * h];
            for (uint32_t y = 0; y < h; ++y)
                for (uint32_t x = 0; x < w; ++x) {
                    if (ds == 1) { G[y * w + x] = g0[y * W + x]; D[y * w + x] = d0[y * W + x]; continue; }
                    int s = 0; uint16_t dm = 0;     // box-average gray, nearest valid depth
                    for (int dy = 0; dy < ds; ++dy)
                        for (int dx = 0; dx < ds; ++dx) {
                            size_t i = (size_t)(y * ds + dy) * W + x * ds + dx;
                            s += g0[i];
                            if (d0[i] && (!dm || d0[i] < dm)) dm = d0[i];
                        }
                    G[y * w + x] = (uint8_t)(s / (ds * ds));
                    D[y * w + x] = dm;
                }
        }
        std::fclose(f);
        if (!ok) { std::snprintf(err, errlen, "truncated KIMG file"); return false; }
        return true;
    }
};

struct VisionParams {
    int64_t harris_thresh = 20000000000LL;  // on integer Harris response
    int hazard_cm = 1000;                   // warn about obstacles within 10 m
    int min_valid_cm = 50;
    float roi_top = 0.45f;                  // obstacle ROI: lower 55% of image
    // Flat-ground model: a pixel is an obstacle only if it is clearly closer than
    // the road surface would be at that image row (i.e. it sticks up out of the
    // ground). Ground depth at row v: Z = fy * cam_height / (v - cy).
    float fy = 721.5f, cy = 172.85f;        // set from calibration (scaled by ds)
    float cam_height_m = 1.65f;             // KITTI camera height above road
    int ground_ratio_pct = 80;              // obstacle if depth < 80% of ground depth
};

struct VisionOut {
    int corners = 0;
    uint32_t checksum = 0;
    int hazard_px[3] = {0, 0, 0};           // per sector: left, centre, right
    int roi_px[3] = {0, 0, 0};
    void merge(const VisionOut &o) {
        corners += o.corners; checksum += o.checksum;
        for (int s = 0; s < 3; ++s) { hazard_px[s] += o.hazard_px[s]; roi_px[s] += o.roi_px[s]; }
    }
};

// Per-thread scratch buffers, allocated once (outside the timed loop).
struct VisionScratch {
    std::vector<int32_t> ixx, iyy, ixy;     // gradient products
    std::vector<int64_t> R;                 // Harris response
    void reserve(int rows, int w) {
        size_t n = (size_t)rows * w;
        ixx.assign(n, 0); iyy.assign(n, 0); ixy.assign(n, 0); R.assign(n, 0);
    }
};

// Process image rows [y0, y1). Halo rows are recomputed locally.
inline void vision_strip(const uint8_t *G, const uint16_t *D, int w, int h, int y0, int y1,
                         const VisionParams &P, VisionScratch &S, VisionOut &out) {
    out = VisionOut();
    // row ranges: products need rows [a0,a1), response [b0,b1), NMS [y0c,y1c)
    const int a0 = std::max(1, y0 - 2), a1 = std::min(h - 1, y1 + 2);
    const int b0 = std::max(2, y0 - 1), b1 = std::min(h - 2, y1 + 1);
    const int y0c = std::max(3, y0), y1c = std::min(h - 3, y1);
    const int base = a0;                      // scratch row 0 == image row a0
    auto idx = [&](int y, int x) { return (size_t)(y - base) * w + x; };

    // 1) Sobel gradients -> products
    for (int y = a0; y < a1; ++y) {
        const uint8_t *r0 = G + (size_t)(y - 1) * w, *r1 = G + (size_t)y * w, *r2 = G + (size_t)(y + 1) * w;
        for (int x = 1; x < w - 1; ++x) {
            int gx = (r0[x + 1] + 2 * r1[x + 1] + r2[x + 1]) - (r0[x - 1] + 2 * r1[x - 1] + r2[x - 1]);
            int gy = (r2[x - 1] + 2 * r2[x] + r2[x + 1]) - (r0[x - 1] + 2 * r0[x] + r0[x + 1]);
            size_t i = idx(y, x);
            S.ixx[i] = gx * gx; S.iyy[i] = gy * gy; S.ixy[i] = gx * gy;
        }
    }
    // 2) 3x3 window sums -> Harris response R = det - trace^2 / 25  (k = 0.04)
    for (int y = b0; y < b1; ++y)
        for (int x = 2; x < w - 2; ++x) {
            int64_t sxx = 0, syy = 0, sxy = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                size_t i = idx(y + dy, x);
                sxx += (int64_t)S.ixx[i - 1] + S.ixx[i] + S.ixx[i + 1];
                syy += (int64_t)S.iyy[i - 1] + S.iyy[i] + S.iyy[i + 1];
                sxy += (int64_t)S.ixy[i - 1] + S.ixy[i] + S.ixy[i + 1];
            }
            int64_t tr = sxx + syy;
            S.R[idx(y, x)] = sxx * syy - sxy * sxy - (tr * tr) / 25;
        }
    // 3) threshold + 3x3 non-maximum suppression
    for (int y = y0c; y < y1c; ++y)
        for (int x = 3; x < w - 3; ++x) {
            int64_t r = S.R[idx(y, x)];
            if (r <= P.harris_thresh) continue;
            bool mx = true;
            for (int dy = -1; dy <= 1 && mx; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    if (!dy && !dx) continue;
                    int64_t q = S.R[idx(y + dy, x + dx)];
                    // strict on one side, non-strict on the other: unique max on plateaus
                    if ((dy < 0 || (dy == 0 && dx < 0)) ? q >= r : q > r) { mx = false; break; }
                }
            if (mx) { out.corners++; out.checksum += (uint32_t)(x * 2654435761u) ^ (uint32_t)(y * 40503u); }
        }
    // 4) obstacle sectors on CNN depth (ground-plane test, integer per pixel)
    const int ytop = (int)(P.roi_top * h);
    for (int y = std::max(y0, ytop); y < y1; ++y) {
        const uint16_t *dr = D + (size_t)y * w;
        float dv = (float)y - P.cy;
        // obstacle limit for this row, in cm: min(hazard, ratio * ground depth)
        int lim = P.hazard_cm;
        if (dv > 0.5f) {
            float zg_cm = 100.0f * P.fy * P.cam_height_m / dv;
            int g = (int)(zg_cm * P.ground_ratio_pct / 100);
            if (g < lim) lim = g;
        }
        for (int x = 0; x < w; ++x) {
            int s = (3 * x) / w;
            int dcm = dr[x];
            if (dcm < P.min_valid_cm) continue;
            out.roi_px[s]++;
            if (dcm < lim) out.hazard_px[s]++;
        }
    }
}
