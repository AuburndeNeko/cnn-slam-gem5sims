// CNN-SLAM back-end: EKF / UKF over KITTI front-end measurements.
//   slam --data seq.bin [--filter ekf|ukf|vo] [--motion vo|oxts] [--frames N]
//        [--out traj.txt] [--verbose] ...
// -DUSE_M5OPS (+ link libm5.a) brackets the filter loop with m5_reset/dump_stats.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include "dataset.hpp"
#include "ekf.hpp"
#include "ukf.hpp"

#ifdef USE_M5OPS
#include <gem5/m5ops.h>
#define ROI_BEGIN() m5_reset_stats(0, 0)
#define ROI_END() m5_dump_stats(0, 0)
#else
#define ROI_BEGIN()
#define ROI_END()
#endif

static void usage(const char *p) {
    std::printf("usage: %s --data seq.bin [--filter ekf|ukf|vo] [--motion vo|oxts] [--frames N]\n"
                "          [--out traj.txt] [--verbose] [--sigma-px f] [--depth-rel f] [--depth-abs f]\n"
                "          [--vo-t-rel f] [--vo-r-rel f] [--gate f]\n", p);
}

static Motion make_motion(const Dataset &D, int k, const Params &prm, bool use_oxts,
                          Motion &last, bool &have_last) {
    Motion m;
    const Frame &F = D.frames[k];
    if (use_oxts) {
        const Frame &Fp = D.frames[k - 1];
        Real dt = Real(F.t - Fp.t);
        m.dR = so3_exp(v3(Real(Fp.omega[0] * dt), Real(Fp.omega[1] * dt), Real(Fp.omega[2] * dt)));
        m.dt = v3(Real(Fp.vel[0] * dt), Real(Fp.vel[1] * dt), Real(Fp.vel[2] * dt));
        Real qv = prm.oxts_v_sigma * prm.oxts_v_sigma * dt, qw = prm.oxts_w_sigma * prm.oxts_w_sigma * dt;
        for (int a = 0; a < 3; ++a) { m.qt[a] = qv; m.qr[a] = qw; }
        return m;
    }
    if (F.vo_valid) {
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) m.dR.m[i][j] = Real(F.vo_R[3 * i + j]);
        orthonormalize(m.dR);
        m.dt = v3(Real(F.vo_t[0]), Real(F.vo_t[1]), Real(F.vo_t[2]));
        Real st = prm.vo_t_abs + prm.vo_t_rel * norm(m.dt);
        Real sr = prm.vo_r_abs + prm.vo_r_rel * norm(so3_log(m.dR));
        for (int a = 0; a < 3; ++a) { m.qt[a] = st * st; m.qr[a] = sr * sr; }
        last = m; have_last = true;
        return m;
    }
    // VO failure: constant-velocity fallback with inflated noise
    if (have_last) m = last; else { m.dR = eye3(); m.dt = v3(0, 0, 0); for (int a = 0; a < 3; ++a) m.qt[a] = m.qr[a] = Real(1e-2); }
    Real f2 = prm.vo_fail_inflate * prm.vo_fail_inflate;
    for (int a = 0; a < 3; ++a) { m.qt[a] *= f2; m.qr[a] *= f2; }
    return m;
}

int main(int argc, char **argv) {
    const char *data = nullptr, *out = nullptr, *filt = "ekf", *motion = "vo";
    int frames = 0; bool verbose = false;
    Params prm;
    for (int i = 1; i < argc; ++i) {
        auto nxt = [&](void) -> const char * { if (i + 1 >= argc) { usage(argv[0]); std::exit(1); } return argv[++i]; };
        if (!std::strcmp(argv[i], "--data")) data = nxt();
        else if (!std::strcmp(argv[i], "--filter")) filt = nxt();
        else if (!std::strcmp(argv[i], "--motion")) motion = nxt();
        else if (!std::strcmp(argv[i], "--frames")) frames = std::atoi(nxt());
        else if (!std::strcmp(argv[i], "--out")) out = nxt();
        else if (!std::strcmp(argv[i], "--verbose")) verbose = true;
        else if (!std::strcmp(argv[i], "--sigma-px")) prm.sigma_px = Real(std::atof(nxt()));
        else if (!std::strcmp(argv[i], "--depth-rel")) prm.depth_rel = Real(std::atof(nxt()));
        else if (!std::strcmp(argv[i], "--depth-abs")) prm.depth_abs = Real(std::atof(nxt()));
        else if (!std::strcmp(argv[i], "--vo-t-rel")) prm.vo_t_rel = Real(std::atof(nxt()));
        else if (!std::strcmp(argv[i], "--vo-r-rel")) prm.vo_r_rel = Real(std::atof(nxt()));
        else if (!std::strcmp(argv[i], "--gate")) prm.gate_chi2 = Real(std::atof(nxt()));
        else { usage(argv[0]); return 1; }
    }
    if (!data) { usage(argv[0]); return 1; }

    Dataset D; char err[256];
    if (!D.load(data, frames, err, sizeof err)) { std::fprintf(stderr, "load failed: %s\n", err); return 1; }
    Camera cam = {Real(D.fx), Real(D.fy), Real(D.cx), Real(D.cy)};
    const bool use_oxts = !std::strcmp(motion, "oxts");
    const bool dead_reckon = !std::strcmp(filt, "vo");

    std::unique_ptr<Filter> F;
    if (!std::strcmp(filt, "ukf")) F.reset(new UkfSlam(prm));
    else F.reset(new EkfSlam());

    std::printf("--- gem5 CNN-SLAM back-end (%s, %s) ---\n", dead_reckon ? "vo-only" : F->name(), REAL_NAME);
    std::printf("frames=%u obs=%zu  max_lm=%d state_dim=%d  motion=%s  image=%ux%u\n",
                D.n, D.total_obs, MAX_LM, NSTATE, motion, D.width, D.height);

    SlamState st;
    for (int i = 0; i < NPOSE; ++i) st.P(i, i) = Real(1e-6);
    std::vector<Real> traj(12 * D.n);
    Motion last; bool have_last = false;
    long n_upd = 0, n_rej = 0, n_init = 0, n_noslot = 0, n_free = 0;
    double nis_sum = 0, sq_err = 0;
    bool seen[MAX_LM];

    ROI_BEGIN();
    auto t0 = std::chrono::steady_clock::now();
    for (uint32_t k = 0; k < D.n; ++k) {
        const Frame &Fr = D.frames[k];
        if (k > 0) F->predict(st, make_motion(D, k, prm, use_oxts, last, have_last));

        if (!dead_reckon) {
            for (int s = 0; s < MAX_LM; ++s) seen[s] = false;
            for (const Obs &o : Fr.obs) {               // 1) update
                int s = st.find(o.id);
                if (s < 0) continue;
                seen[s] = true;
                Real z[3] = {Real(o.u), Real(o.v), Real(o.d)}, Rn[3], nis = 0;
                meas_noise(prm, z, Rn);
                if (F->update(st, cam, s, z, Rn, prm.gate_chi2, &nis)) {
                    ++n_upd; nis_sum += nis; st.slot[s].rejects = 0; st.slot[s].age++;
                } else {
                    ++n_rej;
                    if (++st.slot[s].rejects >= prm.max_rejects) { st.release(s); ++n_free; }
                }
            }
            for (int s = 0; s < MAX_LM; ++s)            // 2) marginalise ended tracks
                if (st.slot[s].id >= 0 && !seen[s]) { st.release(s); ++n_free; }
            for (const Obs &o : Fr.obs) {               // 3) init new landmarks
                if (st.find(o.id) >= 0) continue;
                int s = st.free_slot();
                if (s < 0) { ++n_noslot; continue; }
                Real z[3] = {Real(o.u), Real(o.v), Real(o.d)}, Rn[3];
                meas_noise(prm, z, Rn);
                F->init_landmark(st, cam, s, z, Rn);
                st.slot[s].id = o.id; st.slot[s].rejects = 0; st.slot[s].age = 0;
                ++n_init;
            }
        }
        Real *T = &traj[12 * k];
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) T[4 * i + j] = st.R.m[i][j];
            T[4 * i + 3] = at(st.p, i);
        }
        double ex = st.p.x - Fr.gt[3], ey = st.p.y - Fr.gt[7], ez = st.p.z - Fr.gt[11];
        sq_err += ex * ex + ey * ey + ez * ez;
        if (verbose)
            std::printf("k=%3u  p=(%8.3f %8.3f %8.3f)  gt=(%8.3f %8.3f %8.3f)  lm=%2d  vo=%u\n", k,
                        (double)st.p.x, (double)st.p.y, (double)st.p.z, Fr.gt[3], Fr.gt[7], Fr.gt[11],
                        st.active(), Fr.vo_valid);
    }
    auto t1 = std::chrono::steady_clock::now();
    ROI_END();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const Frame &L = D.frames[D.n - 1];
    double fe = std::sqrt((st.p.x - L.gt[3]) * (st.p.x - L.gt[3]) + (st.p.y - L.gt[7]) * (st.p.y - L.gt[7]) +
                          (st.p.z - L.gt[11]) * (st.p.z - L.gt[11]));
    std::printf("updates=%ld rejected=%ld inits=%ld released=%ld no_slot=%ld  mean_NIS=%.2f (ideal 3)\n",
                n_upd, n_rej, n_init, n_free, n_noslot, n_upd ? nis_sum / n_upd : 0.0);
    std::printf("Final Estimated Position: (%.3f, %.3f, %.3f) m   GT: (%.3f, %.3f, %.3f) m\n",
                (double)st.p.x, (double)st.p.y, (double)st.p.z, L.gt[3], L.gt[7], L.gt[11]);
    std::printf("Final position error: %.3f m   ATE-RMSE (unaligned): %.3f m\n", fe, std::sqrt(sq_err / D.n));
    std::printf("Filter loop: %.3f ms total, %.4f ms/frame (host clock; use sim_ticks in gem5)\n", ms, ms / D.n);

    if (out) {
        FILE *f = std::fopen(out, "w");
        if (!f) { std::fprintf(stderr, "cannot write %s\n", out); return 1; }
        for (uint32_t k = 0; k < D.n; ++k) {
            for (int i = 0; i < 12; ++i) std::fprintf(f, i ? " %.9e" : "%.9e", (double)traj[12 * k + i]);
            std::fprintf(f, "\n");
        }
        std::fclose(f);
        std::printf("trajectory written to %s (KITTI 3x4 format)\n", out);
    }
    return 0;
}