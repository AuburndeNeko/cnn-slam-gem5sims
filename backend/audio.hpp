// audio.hpp: 2-mic direction of arrival (GCC-PHAT) for the CNN-SLAM wearable workload.
// Header-only, FP32, no allocation after construction (suits gem5 SE runs).
#pragma once
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace audio {

using cf = std::complex<float>;
constexpr float kSpeedOfSound = 343.0f;
constexpr float kPi = 3.14159265358979f;

// ---------- KAUD v1 loader ----------
struct Clip {
    uint32_t fs = 0, n_mics = 0, n = 0, hop = 0, win = 0;
    float spacing = 0.f;
    std::vector<int16_t> pcm;  // interleaved [n][n_mics]
    uint32_t n_hops() const { return n < win ? 0 : (n - win) / hop + 1; }
    const int16_t* hop_ptr(uint32_t h) const { return pcm.data() + size_t(h) * hop * n_mics; }
};

inline bool load_kaud(const char* path, Clip& c) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "audio: cannot open %s\n", path); return false; }
    char magic[4]; uint32_t hdr[6];
    bool ok = std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "KAUD", 4) == 0 &&
              std::fread(hdr, 4, 6, f) == 6 && hdr[0] == 1 &&
              std::fread(&c.spacing, 4, 1, f) == 1;
    if (ok) {
        c.fs = hdr[1]; c.n_mics = hdr[2]; c.n = hdr[3]; c.hop = hdr[4]; c.win = hdr[5];
        c.pcm.resize(size_t(c.n) * c.n_mics);
        ok = c.n_mics == 2 &&
             std::fread(c.pcm.data(), 2, c.pcm.size(), f) == c.pcm.size();
    }
    std::fclose(f);
    if (!ok) std::fprintf(stderr, "audio: bad KAUD file %s\n", path);
    return ok;
}

// ---------- in-place iterative radix-2 FFT (n = power of 2) ----------
class FFT {
public:
    explicit FFT(int n) : n_(n), rev_(n), tw_(n / 2) {
        int bits = 0; while ((1 << bits) < n) ++bits;
        for (int i = 0; i < n; ++i) {
            int r = 0; for (int b = 0; b < bits; ++b) r |= ((i >> b) & 1) << (bits - 1 - b);
            rev_[i] = r;
        }
        for (int k = 0; k < n / 2; ++k)
            tw_[k] = cf(std::cos(-2 * kPi * k / n), std::sin(-2 * kPi * k / n));
    }
    // inverse=true computes the unnormalised inverse (caller divides by n if needed)
    void run(cf* x, bool inverse) const {
        for (int i = 0; i < n_; ++i) if (i < rev_[i]) std::swap(x[i], x[rev_[i]]);
        for (int len = 2; len <= n_; len <<= 1) {
            int half = len / 2, step = n_ / len;
            for (int i = 0; i < n_; i += len)
                for (int j = 0; j < half; ++j) {
                    cf w = tw_[j * step];
                    if (inverse) w = std::conj(w);
                    cf u = x[i + j], v = x[i + j + half] * w;
                    x[i + j] = u + v;
                    x[i + j + half] = u - v;
                }
        }
    }
    int size() const { return n_; }
private:
    int n_;
    std::vector<int> rev_;
    std::vector<cf> tw_;
};

// ---------- GCC-PHAT direction of arrival ----------
struct Doa {
    int active;    // 1 if a coherent source was detected in this window
    float lag;     // TDOA in samples (left lags right when > 0)
    float az_deg;  // azimuth, 0 = ahead, + = right; NaN-free (0 when inactive)
    float peak;    // GCC-PHAT peak height (coherence, about 0..1)
};

class GccPhat {
public:
    GccPhat(int win, int fs, float spacing, float peak_thresh = 0.25f)
        : win_(win), fs_(fs), spacing_(spacing), thresh_(peak_thresh), fft_(win),
          w_(win), a_(win), b_(win) {
        for (int i = 0; i < win; ++i) w_[i] = 0.5f - 0.5f * std::cos(2 * kPi * i / (win - 1));
        max_lag_ = int(std::ceil(spacing / kSpeedOfSound * fs));
    }
    // x: interleaved stereo int16, at least win frames (L = channel 0, R = channel 1)
    Doa process(const int16_t* x) {
        for (int i = 0; i < win_; ++i) {
            a_[i] = cf(w_[i] * x[2 * i], 0.f);
            b_[i] = cf(w_[i] * x[2 * i + 1], 0.f);
        }
        fft_.run(a_.data(), false);
        fft_.run(b_.data(), false);
        for (int k = 0; k < win_; ++k) {           // cross-spectrum, phase transform
            cf c = a_[k] * std::conj(b_[k]);
            float m = std::abs(c);
            a_[k] = m > 1e-12f ? c / m : cf(0.f, 0.f);
        }
        fft_.run(a_.data(), true);                 // a_ = correlation * win
        const float inv = 1.0f / win_;
        auto r = [&](int lag) { return a_[(lag + win_) % win_].real() * inv; };

        int best = -max_lag_; float vb = r(best);
        for (int l = -max_lag_ + 1; l <= max_lag_; ++l) { float v = r(l); if (v > vb) { vb = v; best = l; } }
        float off = 0.f;
        if (best > -max_lag_ && best < max_lag_) {  // parabolic sub-sample refinement
            float vm = r(best - 1), vp = r(best + 1), den = vm - 2 * vb + vp;
            if (den != 0.f) off = 0.5f * (vm - vp) / den;
        }
        Doa d;
        d.peak = vb;
        d.lag = best + off;
        d.active = vb > thresh_;
        float s = d.lag * kSpeedOfSound / (fs_ * spacing_);
        s = s > 1.f ? 1.f : (s < -1.f ? -1.f : s);
        d.az_deg = d.active ? std::asin(s) * 180.f / kPi : 0.f;
        return d;
    }
    int max_lag() const { return max_lag_; }
private:
    int win_, fs_, max_lag_;
    float spacing_, thresh_;
    FFT fft_;
    std::vector<float> w_;
    std::vector<cf> a_, b_;
};

// L / C / R sector, same idea as the vision hazard sectors
inline char sector(float az_deg, float centre_half_width = 15.f) {
    return az_deg < -centre_half_width ? 'L' : (az_deg > centre_half_width ? 'R' : 'C');
}

}  // namespace audio
