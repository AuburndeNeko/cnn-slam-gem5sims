// Reader for the KSLM binary produced by frontend/build_dataset.py.
// Field-by-field reads -> independent of struct padding (x86-64 host and aarch64 gem5).
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

struct Obs { uint32_t id; float u, v, d; };

struct Frame {
    double t;
    double gt[12];              // T_wc, 3x4 row-major (evaluation only)
    uint32_t vo_valid, vo_inliers;
    double vo_R[9], vo_t[3];    // T_{k-1,k} from CNN-depth PnP
    double vel[3], omega[3];    // OXTS body motion in camera frame
    std::vector<Obs> obs;
};

struct Dataset {
    uint32_t version = 0, n = 0, width = 0, height = 0;
    double fx = 0, fy = 0, cx = 0, cy = 0;
    std::vector<Frame> frames;
    size_t total_obs = 0;

    bool load(const char *path, int max_frames, char *err, size_t errlen) {
        FILE *f = std::fopen(path, "rb");
        if (!f) { std::snprintf(err, errlen, "cannot open %s", path); return false; }
        char magic[4]; uint32_t reserved;
        bool ok = std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "KSLM", 4) == 0;
        ok = ok && rd(f, version) && rd(f, n) && rd(f, width) && rd(f, height) && rd(f, reserved);
        ok = ok && rd(f, fx) && rd(f, fy) && rd(f, cx) && rd(f, cy);
        if (!ok || version != 1) {
            std::snprintf(err, errlen, "bad header (magic/version)"); std::fclose(f); return false;
        }
        uint32_t m = n;
        if (max_frames > 0 && (uint32_t)max_frames < m) m = (uint32_t)max_frames;
        frames.resize(m);
        for (uint32_t k = 0; k < m && ok; ++k) {
            Frame &F = frames[k];
            uint32_t nobs, pad;
            ok = rd(f, F.t) && rdn(f, F.gt, 12) && rd(f, F.vo_valid) && rd(f, F.vo_inliers) &&
                 rdn(f, F.vo_R, 9) && rdn(f, F.vo_t, 3) && rdn(f, F.vel, 3) && rdn(f, F.omega, 3) &&
                 rd(f, nobs) && rd(f, pad);
            if (!ok) break;
            F.obs.resize(nobs);
            for (uint32_t j = 0; j < nobs && ok; ++j)
                ok = rd(f, F.obs[j].id) && rd(f, F.obs[j].u) && rd(f, F.obs[j].v) && rd(f, F.obs[j].d);
            total_obs += nobs;
        }
        std::fclose(f);
        if (!ok) { std::snprintf(err, errlen, "truncated file"); return false; }
        n = m;
        return true;
    }

  private:
    template <class T> static bool rd(FILE *f, T &x) { return std::fread(&x, sizeof(T), 1, f) == 1; }
    template <class T> static bool rdn(FILE *f, T *x, size_t k) { return std::fread(x, sizeof(T), k, f) == k; }
};