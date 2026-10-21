// audio_test.cpp: standalone check of audio.hpp against the make_audio.py ground truth.
// usage: ./audio_test --audio data/audio_0005.kaud [--gt data/audio_0005_gt.txt] [--out data/doa.txt]
//                     [--thresh 0.25] [--frames N]
#include "audio.hpp"
#include <cstdlib>
#include <ctime>
#include <string>

static double now_ms() {
    timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

int main(int argc, char** argv) {
    std::string apath = "data/audio_0005.kaud", gpath, opath;
    float thresh = 0.25f; int frames = -1;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string k = argv[i];
        if (k == "--audio") apath = argv[i + 1];
        else if (k == "--gt") gpath = argv[i + 1];
        else if (k == "--out") opath = argv[i + 1];
        else if (k == "--thresh") thresh = std::atof(argv[i + 1]);
        else if (k == "--frames") frames = std::atoi(argv[i + 1]);
        else { std::fprintf(stderr, "unknown arg %s\n", k.c_str()); return 1; }
    }
    audio::Clip clip;
    if (!audio::load_kaud(apath.c_str(), clip)) return 1;
    uint32_t n_hops = clip.n_hops();
    if (frames > 0 && uint32_t(frames) * 10 < n_hops) n_hops = frames * 10;
    std::printf("audio: %u Hz, %u samples, hop %u, win %u, spacing %.3f m, %u hops\n",
                clip.fs, clip.n, clip.hop, clip.win, clip.spacing, n_hops);

    audio::GccPhat doa(clip.win, clip.fs, clip.spacing, thresh);
    std::vector<audio::Doa> res(n_hops);
    double t0 = now_ms();
    for (uint32_t h = 0; h < n_hops; ++h) res[h] = doa.process(clip.hop_ptr(h));
    double t1 = now_ms();

    uint32_t n_act = 0; double sum_lag = 0;
    for (auto& d : res) { n_act += d.active; sum_lag += d.lag; }
    std::printf("doa: %u/%u hops active, max lag %d, lag sum %.4f\n", n_act, n_hops, doa.max_lag(), sum_lag);
    std::printf("doa time: %.3f ms total, %.4f ms/hop, %.3f ms per 100 ms video frame\n",
                t1 - t0, (t1 - t0) / n_hops, (t1 - t0) / n_hops * 10);

    if (!opath.empty()) {
        FILE* f = std::fopen(opath.c_str(), "w");
        std::fprintf(f, "# hop active lag az_deg peak sector\n");
        for (uint32_t h = 0; h < n_hops; ++h)
            std::fprintf(f, "%u %d %.4f %.3f %.4f %c\n", h, res[h].active, res[h].lag, res[h].az_deg,
                         res[h].peak, res[h].active ? audio::sector(res[h].az_deg) : '-');
        std::fclose(f);
    }

    if (!gpath.empty()) {   // accuracy vs ground truth
        FILE* f = std::fopen(gpath.c_str(), "r");
        if (!f) { std::fprintf(stderr, "cannot open %s\n", gpath.c_str()); return 1; }
        char line[256]; uint32_t h; int act; float tc, az, tau;
        uint32_t det_ok = 0, both = 0, within10 = 0, sec_ok = 0, n = 0;
        double sum_err = 0, max_err = 0;
        while (std::fgets(line, sizeof line, f)) {
            if (line[0] == '#' || std::sscanf(line, "%u %f %d %f %f", &h, &tc, &act, &az, &tau) != 5) continue;
            if (h >= n_hops) break;
            ++n;
            det_ok += (act == res[h].active);
            if (act && res[h].active) {
                double e = std::fabs(res[h].az_deg - az);
                ++both; sum_err += e; if (e > max_err) max_err = e;
                within10 += e < 10; sec_ok += audio::sector(az) == audio::sector(res[h].az_deg);
            }
        }
        std::fclose(f);
        std::printf("vs gt: detection %u/%u correct; on %u hops active in both: MAE %.2f deg, max %.2f, "
                    "within 10 deg %.0f%%, sector L/C/R %.0f%%\n", det_ok, n, both,
                    both ? sum_err / both : 0, max_err, both ? 100.0 * within10 / both : 0,
                    both ? 100.0 * sec_ok / both : 0);
    }
    return 0;
}
