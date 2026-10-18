// Multi-core CNN-SLAM workload (wearable-navigation style), for gem5 Ruby/Garnet.
//
//   slam_mc --data seq.bin --images seq_img.bin [--threads W] [--frames N]
//           [--filter ekf|ukf] [--ds 1|2] [--verbose]
//
// Threads (needs W+1 simulated CPUs in gem5 SE mode):
//   main thread      : EKF/UKF back-end on frame k-1
//   W worker threads : vision on frame k (Harris corners + obstacle sectors),
//                      image split into W horizontal strips
// Each "step" runs both stages concurrently and ends at a barrier, so a frame
// enters vision at step k and leaves the filter at step k+1 (2-stage pipeline).
// --threads 0 runs everything sequentially on one core (baseline, no pipeline).
//
// Timing uses clock_gettime, which gem5 SE answers with *simulated* time,
// so the reported latencies are on the simulated chip.
#include <pthread.h>
#include <time.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>
#include <atomic>
#include "dataset.hpp"
#include "ekf.hpp"
#include "ukf.hpp"
#include "vision.hpp"

#ifdef USE_M5OPS
#include <gem5/m5ops.h>
#define ROI_BEGIN() m5_reset_stats(0, 0)
#define ROI_END() m5_dump_stats(0, 0)
#else
#define ROI_BEGIN()
#define ROI_END()
#endif

static double now_ms() {
    timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
}

// ------------------------------------------------------------ filter stage
struct FilterStage {
    const Dataset &D; Params prm; Camera cam; SlamState st;
    std::unique_ptr<Filter> F; Motion last; bool have_last = false;
    long n_upd = 0, n_rej = 0;
    FilterStage(const Dataset &d, const char *filt) : D(d) {
        cam = {Real(D.fx), Real(D.fy), Real(D.cx), Real(D.cy)};
        if (!std::strcmp(filt, "ukf")) F.reset(new UkfSlam(prm)); else F.reset(new EkfSlam());
        for (int i = 0; i < NPOSE; ++i) st.P(i, i) = Real(1e-6);
    }
    Motion motion(int k) {
        Motion m; const Frame &Fr = D.frames[k];
        if (Fr.vo_valid) {
            for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) m.dR.m[i][j] = Real(Fr.vo_R[3 * i + j]);
            orthonormalize(m.dR);
            m.dt = v3(Real(Fr.vo_t[0]), Real(Fr.vo_t[1]), Real(Fr.vo_t[2]));
            Real s_t = prm.vo_t_abs + prm.vo_t_rel * norm(m.dt);
            Real s_r = prm.vo_r_abs + prm.vo_r_rel * norm(so3_log(m.dR));
            for (int a = 0; a < 3; ++a) { m.qt[a] = s_t * s_t; m.qr[a] = s_r * s_r; }
            last = m; have_last = true; return m;
        }
        if (have_last) m = last; else { m.dR = eye3(); m.dt = v3(0, 0, 0); for (int a = 0; a < 3; ++a) m.qt[a] = m.qr[a] = Real(1e-2); }
        Real f2 = prm.vo_fail_inflate * prm.vo_fail_inflate;
        for (int a = 0; a < 3; ++a) { m.qt[a] *= f2; m.qr[a] *= f2; }
        return m;
    }
    void step(int k) {
        const Frame &Fr = D.frames[k];
        if (k > 0) F->predict(st, motion(k));
        bool seen[MAX_LM] = {};
        for (const Obs &o : Fr.obs) {
            int s = st.find(o.id); if (s < 0) continue;
            seen[s] = true;
            Real z[3] = {Real(o.u), Real(o.v), Real(o.d)}, Rn[3], nis = 0;
            meas_noise(prm, z, Rn);
            if (F->update(st, cam, s, z, Rn, prm.gate_chi2, &nis)) { ++n_upd; st.slot[s].rejects = 0; }
            else { ++n_rej; if (++st.slot[s].rejects >= prm.max_rejects) st.release(s); }
        }
        for (int s = 0; s < MAX_LM; ++s) if (st.slot[s].id >= 0 && !seen[s]) st.release(s);
        for (const Obs &o : Fr.obs) {
            if (st.find(o.id) >= 0) continue;
            int s = st.free_slot(); if (s < 0) continue;
            Real z[3] = {Real(o.u), Real(o.v), Real(o.d)}, Rn[3];
            meas_noise(prm, z, Rn);
            F->init_landmark(st, cam, s, z, Rn);
            st.slot[s].id = o.id; st.slot[s].rejects = 0;
        }
    }
};

// Master/worker flag barrier: plain ordered loads/stores only, no atomic
// read-modify-write (which never completes for the worker under gem5 SE + Ruby).
// Every flag has exactly one writer and sits on its own 64-byte cache line.
static const int MAXW = 64;
struct alignas(64) Flag {
    std::atomic<unsigned> v{0};
    char pad[64 - sizeof(std::atomic<unsigned>)];
};
struct FlagBarrier {
    int W = 0;
    Flag arrived[MAXW];     // arrived[t] written only by worker t
    Flag release;           // written only by the main thread
    void init(int nworkers) { W = nworkers; }
    void worker_wait(int t, unsigned phase) {
        arrived[t].v.store(phase, std::memory_order_release);
        while (release.v.load(std::memory_order_acquire) < phase) { /* spin */ }
    }
    void master_wait(unsigned phase) {
        for (int t = 0; t < W; ++t)
            while (arrived[t].v.load(std::memory_order_acquire) < phase) { /* spin */ }
        release.v.store(phase, std::memory_order_release);
    }
};

// ------------------------------------------------------------ worker pool
struct Pool {
    int W = 0;
    const ImageSeq *I = nullptr; VisionParams vp;
    std::vector<VisionScratch> scratch;
    std::vector<VisionOut> out;
    std::vector<pthread_t> tid;
    FlagBarrier start, done;
    volatile int frame = -1; volatile bool stop = false;
    int y0(int t) const { return (int)((long)I->h * t / W); }
};
struct WorkerArg { Pool *P; int t; };

static void *worker(void *a_) {
    WorkerArg *a = (WorkerArg *)a_; Pool &P = *a->P; int t = a->t;
    unsigned ph = 0;
    for (;;) {
        ++ph;
        P.start.worker_wait(t, ph);
        if (P.stop) break;
        int k = P.frame;
        std::printf("[w%d] released, frame %d\n", t, k);
        if (k >= 0)
            vision_strip(P.I->g(k), P.I->d(k), P.I->w, P.I->h, P.y0(t), P.y0(t + 1), P.vp,
                         P.scratch[t], P.out[t]);
        std::printf("[w%d] vision done\n", t);
        P.done.worker_wait(t, ph);
    }
    return nullptr;
}

static void usage(const char *p) {
    std::printf("usage: %s --data seq.bin --images seq_img.bin [--threads W] [--frames N]\n"
                "          [--filter ekf|ukf] [--ds 1|2] [--period-ms 100] [--verbose]\n", p);
}

int main(int argc, char **argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);   // print immediately (no buffering), so progress shows in gem5
    const char *data = nullptr, *images = nullptr, *filt = "ekf";
    int W = 1, frames = 0, ds = 1; bool verbose = false; double period = 100.0;
    for (int i = 1; i < argc; ++i) {
        auto nxt = [&]() -> const char * { if (i + 1 >= argc) { usage(argv[0]); std::exit(1); } return argv[++i]; };
        if (!std::strcmp(argv[i], "--data")) data = nxt();
        else if (!std::strcmp(argv[i], "--images")) images = nxt();
        else if (!std::strcmp(argv[i], "--threads")) W = std::atoi(nxt());
        else if (!std::strcmp(argv[i], "--frames")) frames = std::atoi(nxt());
        else if (!std::strcmp(argv[i], "--filter")) filt = nxt();
        else if (!std::strcmp(argv[i], "--ds")) ds = std::atoi(nxt());
        else if (!std::strcmp(argv[i], "--period-ms")) period = std::atof(nxt());
        else if (!std::strcmp(argv[i], "--verbose")) verbose = true;
        else { usage(argv[0]); return 1; }
    }
    if (!data || !images || W < 0) { usage(argv[0]); return 1; }

    Dataset D; ImageSeq I; char err[256];
    if (!D.load(data, frames, err, sizeof err)) { std::fprintf(stderr, "data: %s\n", err); return 1; }
    if (!I.load(images, frames, ds, err, sizeof err)) { std::fprintf(stderr, "images: %s\n", err); return 1; }
    const int n = (int)std::min<uint32_t>(D.n, I.n);

    std::printf("--- CNN-SLAM multi-core workload (%s, %s) ---\n", filt, REAL_NAME);
    std::printf("frames=%d  image=%ux%u (ds=%d)  vision_threads=%d  filter_thread=main  cpus_needed=%d\n",
                n, I.w, I.h, ds, W, W + 1);

    FilterStage FS(D, filt);
    Pool P; P.W = W > 0 ? W : 1; P.I = &I;
    P.vp.fy = (float)(D.fy / ds); P.vp.cy = (float)(D.cy / ds);   // calibration for ground model
    P.scratch.resize(P.W); P.out.resize(P.W);
    for (int t = 0; t < P.W; ++t) P.scratch[t].reserve(P.y0(t + 1) - P.y0(t) + 6, I.w);
    std::vector<WorkerArg> args(W);
    if (W > 0) {
        P.start.init(W);
        P.done.init(W);
        P.tid.resize(W);
        for (int t = 0; t < W; ++t) { args[t] = {&P, t}; pthread_create(&P.tid[t], nullptr, worker, &args[t]); }
    }

    std::vector<VisionOut> vres(n);
    std::vector<double> vis_start(n, 0), lat(n, 0), step_ms;

    ROI_BEGIN();
    double t_all = now_ms();
    if (W == 0) {                                   // sequential baseline
        for (int k = 0; k < n; ++k) {
            double t0 = now_ms();
            vision_strip(I.g(k), I.d(k), I.w, I.h, 0, I.h, P.vp, P.scratch[0], vres[k]);
            FS.step(k);
            double t1 = now_ms();
            step_ms.push_back(t1 - t0); lat[k] = t1 - t0;
        }
    } else {                                        // 2-stage pipeline
    unsigned ph = 0;
    for (int s = 0; s <= n; ++s) {
        double t0 = now_ms();
        P.frame = (s < n) ? s : -1;
        if (s < n) vis_start[s] = t0;
        ++ph;
        std::printf("[m] step %d: at start barrier\n", s);
        P.start.master_wait(ph);                // release workers on frame s
        if (s >= 1) FS.step(s - 1);             // filter frame s-1 concurrently
        std::printf("[m] step %d: at done barrier\n", s);
        P.done.master_wait(ph);
        std::printf("[m] step %d: complete\n", s);
        double t1 = now_ms();
        if (s < n) { vres[s] = VisionOut(); for (int t = 0; t < W; ++t) vres[s].merge(P.out[t]); }
        if (s >= 1) lat[s - 1] = t1 - vis_start[s - 1];
        step_ms.push_back(t1 - t0);
    }
        P.stop = true;
        P.start.master_wait(++ph);                  // let workers see stop and return
        std::printf("[debug] pipeline finished\n");
        // No pthread_join: thread-exit wake-ups are unreliable in gem5 SE mode.
        // Workers return on their own; process exit cleans up.
    }
    double t_total = now_ms() - t_all;
    ROI_END();

    // ---------------------------------------------------------- report
    long corners = 0; uint32_t cks = 0; int haz_frames = 0;
    for (int k = 0; k < n; ++k) {
        const VisionOut &v = vres[k];
        corners += v.corners; cks = cks * 31u + v.checksum + (uint32_t)v.corners;
        bool haz = false; char flags[4] = "---";
        for (int s = 0; s < 3; ++s)
            if (v.roi_px[s] && v.hazard_px[s] * 50 > v.roi_px[s]) { haz = true; flags[s] = "LCR"[s]; }  // >2% of sector
        haz_frames += haz;
        if (verbose)
            std::printf("frame %3d  corners=%5d  hazard=%s (%d/%d/%d px)  latency=%.3f ms\n", k, v.corners, flags,
                        v.hazard_px[0], v.hazard_px[1], v.hazard_px[2], lat[k]);
    }
    double smax = 0, ssum = 0, lmax = 0, lsum = 0;
    for (double x : step_ms) { ssum += x; if (x > smax) smax = x; }
    for (int k = 0; k < n; ++k) { lsum += lat[k]; if (lat[k] > lmax) lmax = lat[k]; }
    const Frame &L = D.frames[n - 1];
    std::printf("vision: corners=%ld (%.0f/frame)  hazard_frames=%d/%d  checksum=%08x\n",
                corners, (double)corners / n, haz_frames, n, cks);
    std::printf("filter: updates=%ld rejected=%ld  final=(%.3f, %.3f, %.3f)  GT=(%.3f, %.3f, %.3f)\n",
                FS.n_upd, FS.n_rej, (double)FS.st.p.x, (double)FS.st.p.y, (double)FS.st.p.z, L.gt[3], L.gt[7], L.gt[11]);
    std::printf("timing (simulated in gem5): total=%.3f ms  step mean=%.3f max=%.3f ms  "
                "frame latency mean=%.3f max=%.3f ms\n", t_total, ssum / step_ms.size(), smax, lsum / n, lmax);
    std::printf("throughput=%.1f fps  deadline %.0f ms: %s (worst latency %.1f%% of budget)\n",
                1000.0 * n / t_total, period, lmax <= period ? "MET" : "MISSED", 100.0 * lmax / period);
    return 0;
}
