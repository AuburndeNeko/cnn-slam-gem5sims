// Shared state, models and bookkeeping for EKF-SLAM and UKF-SLAM.
//
// nominal : p (3, world), R (SO3, cam->world), l_1..l_M (3 each, world)
// error   : dx = [dp, dtheta, dl_1 .. dl_M],  R = R_hat * Exp(dtheta)
// P       : N x N, N = 6 + 3*MAX_LM (fixed -> fixed per-frame cost for gem5)
// Motion  : p_k = p_{k-1} + R_{k-1}(dt + n_t),  R_k = R_{k-1} dR Exp(n_r)
// Meas.   : p_c = R^T(l - p),  z = (fx X/Z + cx, fy Y/Z + cy, Z)
#pragma once
#include "linalg.hpp"
#include <cstdint>

#ifndef SLAM_MAX_LM
#define SLAM_MAX_LM 24
#endif
static const int MAX_LM = SLAM_MAX_LM;
static const int NPOSE = 6;
static const int NSTATE = NPOSE + 3 * MAX_LM;

struct Params {
    Real sigma_px = Real(1.5);
    Real depth_rel = Real(0.12);         // CNN depth sigma = rel*d + abs
    Real depth_abs = Real(0.10);
    Real vo_t_abs = Real(0.02), vo_t_rel = Real(0.03);
    Real vo_r_abs = Real(0.0015), vo_r_rel = Real(0.02);
    Real vo_fail_inflate = Real(10);
    Real oxts_v_sigma = Real(0.10), oxts_w_sigma = Real(0.005);
    Real gate_chi2 = Real(11.34);        // chi2(3) 99%
    Real min_depth = Real(0.5);
    int max_rejects = 2;
    Real ukf_alpha = Real(1), ukf_beta = Real(2), ukf_kappa = Real(0);
};

struct Camera { Real fx, fy, cx, cy; };
struct Slot { int64_t id = -1; int rejects = 0; int age = 0; };

struct SlamState {
    V3 p;
    M3 R;
    Real lm[3 * MAX_LM];
    Mat P;
    Slot slot[MAX_LM];
    SlamState() : P(NSTATE, NSTATE) {
        p = v3(0, 0, 0); R = eye3();
        for (int i = 0; i < 3 * MAX_LM; ++i) lm[i] = 0;
    }
    V3 landmark(int s) const { return v3(lm[3 * s], lm[3 * s + 1], lm[3 * s + 2]); }
    int find(int64_t id) const {
        for (int s = 0; s < MAX_LM; ++s) if (slot[s].id == id) return s;
        return -1;
    }
    int free_slot() const {
        for (int s = 0; s < MAX_LM; ++s) if (slot[s].id < 0) return s;
        return -1;
    }
    int active() const { int a = 0; for (int s = 0; s < MAX_LM; ++s) a += slot[s].id >= 0; return a; }
    void release(int s) {   // marginalisation = dropping rows/cols
        int o = NPOSE + 3 * s;
        for (int a = 0; a < 3; ++a)
            for (int j = 0; j < NSTATE; ++j) { P(o + a, j) = 0; P(j, o + a) = 0; }
        slot[s] = Slot();
        lm[3 * s] = lm[3 * s + 1] = lm[3 * s + 2] = 0;
    }
    void retract(const Real *dx) {
        p = p + v3(dx[0], dx[1], dx[2]);
        R = R * so3_exp(v3(dx[3], dx[4], dx[5]));
        orthonormalize(R);
        for (int i = 0; i < 3 * MAX_LM; ++i) lm[i] += dx[NPOSE + i];
    }
};

inline bool project(const Camera &cam, V3 p, const M3 &R, V3 l, Real z[3], V3 *pc_out) {
    V3 pc = tr(R) * (l - p);
    if (pc_out) *pc_out = pc;
    if (pc.z < Real(1e-3)) return false;
    Real iz = Real(1) / pc.z;
    z[0] = cam.fx * pc.x * iz + cam.cx;
    z[1] = cam.fy * pc.y * iz + cam.cy;
    z[2] = pc.z;
    return true;
}

inline V3 backproject(const Camera &cam, const Real z[3]) {
    return v3(z[2] * (z[0] - cam.cx) / cam.fx, z[2] * (z[1] - cam.cy) / cam.fy, z[2]);
}

inline void meas_noise(const Params &P, const Real z[3], Real Rn[3]) {
    Real sd = P.depth_rel * z[2] + P.depth_abs;
    Rn[0] = Rn[1] = P.sigma_px * P.sigma_px;
    Rn[2] = sd * sd;
}

struct Motion { M3 dR; V3 dt; Real qt[3], qr[3]; };

// Shared correction: C = P H^T (or P_xz for UKF). Gate, then
// x <- x [+] C S^-1 y,  P <- P - C S^-1 C^T.
struct KalmanWork { Mat C, K; KalmanWork() : C(NSTATE, 3), K(NSTATE, 3) {} };

inline bool kalman_apply(SlamState &st, KalmanWork &w, Real S[3][3], const Real y[3],
                         Real gate_chi2, Real *nis_out) {
    Real Si[3][3];
    if (!inv3(S, Si)) return false;
    Real nis = 0;
    for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) nis += y[a] * Si[a][b] * y[b];
    if (nis_out) *nis_out = nis;
    if (!(nis < gate_chi2)) return false;

    Real dx[NSTATE];
    for (int i = 0; i < NSTATE; ++i) {
        const Real *c = w.C.row(i);
        Real *k = w.K.row(i);
        for (int b = 0; b < 3; ++b) k[b] = c[0] * Si[0][b] + c[1] * Si[1][b] + c[2] * Si[2][b];
        dx[i] = k[0] * y[0] + k[1] * y[1] + k[2] * y[2];
    }
    for (int i = 0; i < NSTATE; ++i) {
        const Real *k = w.K.row(i);
        Real *Pi = st.P.row(i);
        for (int j = i; j < NSTATE; ++j) {
            const Real *c = w.C.row(j);
            Pi[j] -= k[0] * c[0] + k[1] * c[1] + k[2] * c[2];
        }
    }
    for (int i = 0; i < NSTATE; ++i)
        for (int j = 0; j < i; ++j) st.P(i, j) = st.P(j, i);
    st.retract(dx);
    return true;
}

// Linearised landmark init (EKF).
inline void init_landmark_linear(SlamState &st, const Camera &cam, int s, const Real z[3],
                                 const Real Rn[3]) {
    V3 pc = backproject(cam, z);
    V3 l = st.R * pc + st.p;
    int o = NPOSE + 3 * s;
    st.lm[3 * s] = l.x; st.lm[3 * s + 1] = l.y; st.lm[3 * s + 2] = l.z;

    M3 Jth = st.R * skew(pc);
    Real Jx[3][6];
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) { Jx[a][b] = (a == b); Jx[a][3 + b] = -Jth.m[a][b]; }
    M3 B = {{{z[2] / cam.fx, 0, (z[0] - cam.cx) / cam.fx},
             {0, z[2] / cam.fy, (z[1] - cam.cy) / cam.fy},
             {0, 0, 1}}};
    M3 Jz = st.R * B;

    for (int j = 0; j < NSTATE; ++j) {
        if (j >= o && j < o + 3) continue;
        for (int a = 0; a < 3; ++a) {
            Real s_ = 0;
            for (int b = 0; b < 6; ++b) s_ += Jx[a][b] * st.P(b, j);
            st.P(o + a, j) = s_; st.P(j, o + a) = s_;
        }
    }
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) {
            Real s_ = 0;
            for (int i = 0; i < 6; ++i)
                for (int j = 0; j < 6; ++j) s_ += Jx[a][i] * st.P(i, j) * Jx[b][j];
            for (int i = 0; i < 3; ++i) s_ += Jz.m[a][i] * Rn[i] * Jz.m[b][i];
            st.P(o + a, o + b) = s_;
        }
}

struct Filter {
    virtual ~Filter() {}
    virtual const char *name() const = 0;
    virtual void predict(SlamState &st, const Motion &m) = 0;
    virtual bool update(SlamState &st, const Camera &cam, int s, const Real z[3], const Real Rn[3],
                        Real gate, Real *nis) = 0;
    virtual void init_landmark(SlamState &st, const Camera &cam, int s, const Real z[3],
                               const Real Rn[3]) = 0;
};