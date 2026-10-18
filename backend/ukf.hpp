// Manifold UKF-SLAM with local sigma-point sets:
//   predict: pose(6)+process noise(6) -> 25 pts
//   update : pose(6)+one landmark(3)  -> 19 pts
//   init   : pose(6)+meas noise(3)    -> 19 pts
// Other states enter via P_{xo,z} = P_{xo,s} P_ss^-1 P_{s,z}, which is exact
// for Gaussian priors -> same posterior as a full-state UT, far cheaper/stabler in FP32.
#pragma once
#include "slam_core.hpp"

class UkfSlam : public Filter {
  public:
    explicit UkfSlam(const Params &p) : prm_(p) {}
    const char *name() const override { return "ukf"; }

    void predict(SlamState &st, const Motion &m) override {
        const int na = 12, np = 2 * na + 1;
        Weights w = weights(na);
        Real L[6 * 6];
        pose_chol(st, L);
        Real qs[6] = {std::sqrt(m.qt[0]), std::sqrt(m.qt[1]), std::sqrt(m.qt[2]),
                      std::sqrt(m.qr[0]), std::sqrt(m.qr[1]), std::sqrt(m.qr[2])};

        V3 pp[np]; M3 Rp[np]; Real dx[np][6];
        for (int i = 0; i < np; ++i) {
            Real a[12] = {};
            if (i > 0) {
                int j = (i - 1) % na; Real sg = (i <= na) ? w.gamma : -w.gamma;
                if (j < 6) for (int r = 0; r < 6; ++r) a[r] = sg * L[r * 6 + j];
                else a[j] = sg * qs[j - 6];
            }
            for (int r = 0; r < 6; ++r) dx[i][r] = a[r];
            V3 pi = st.p + v3(a[0], a[1], a[2]);
            M3 Ri = st.R * so3_exp(v3(a[3], a[4], a[5]));
            pp[i] = pi + Ri * (m.dt + v3(a[6], a[7], a[8]));
            Rp[i] = Ri * m.dR * so3_exp(v3(a[9], a[10], a[11]));
        }
        V3 pm = v3(0, 0, 0), em = v3(0, 0, 0);
        M3 R0t = tr(Rp[0]);
        for (int i = 0; i < np; ++i) {
            Real wm = i ? w.wi : w.wm0;
            pm = pm + wm * pp[i];
            em = em + wm * so3_log(R0t * Rp[i]);
        }
        M3 Rm = Rp[0] * so3_exp(em);
        orthonormalize(Rm);
        Real Pn[6][6] = {}, Cx[6][6] = {};
        M3 Rmt = tr(Rm);
        for (int i = 0; i < np; ++i) {
            Real wc = i ? w.wi : w.wc0;
            V3 dp = pp[i] - pm, dth = so3_log(Rmt * Rp[i]);
            Real d[6] = {dp.x, dp.y, dp.z, dth.x, dth.y, dth.z};
            for (int a = 0; a < 6; ++a)
                for (int b = 0; b < 6; ++b) { Pn[a][b] += wc * d[a] * d[b]; Cx[a][b] += wc * d[a] * dx[i][b]; }
        }
        Real Ft[6 * 6];
        for (int a = 0; a < 6; ++a) for (int b = 0; b < 6; ++b) Ft[a * 6 + b] = Cx[b][a];
        chol_solve(L, 6, 6, Ft, 6, 6);       // F^T = P_pp^-1 Cx^T
        Mat &P = st.P;
        Real col[6];
        for (int j = NPOSE; j < NSTATE; ++j) {
            for (int a = 0; a < 6; ++a) {
                Real s = 0;
                for (int k = 0; k < 6; ++k) s += Ft[k * 6 + a] * P(k, j);
                col[a] = s;
            }
            for (int a = 0; a < 6; ++a) { P(a, j) = col[a]; P(j, a) = col[a]; }
        }
        for (int a = 0; a < 6; ++a)
            for (int b = 0; b < 6; ++b) P(a, b) = Real(0.5) * (Pn[a][b] + Pn[b][a]);
        st.p = pm; st.R = Rm;
    }

    bool update(SlamState &st, const Camera &cam, int s, const Real z[3], const Real Rn[3],
                Real gate, Real *nis) override {
        const int ns = 9, np = 2 * ns + 1;
        Weights w = weights(ns);
        int idx[9];
        sub_index(s, idx);
        Mat &P = st.P;
        Real L[9 * 9];
        for (int a = 0; a < 9; ++a) for (int b = 0; b < 9; ++b) L[a * 9 + b] = P(idx[a], idx[b]);
        if (!chol_jitter(L, 9)) return false;

        V3 l = st.landmark(s);
        Real hz[np][3], ds[np][9];
        for (int i = 0; i < np; ++i) {
            Real a[9] = {};
            if (i > 0) {
                int j = (i - 1) % ns; Real sg = (i <= ns) ? w.gamma : -w.gamma;
                for (int r = 0; r < 9; ++r) a[r] = sg * L[r * 9 + j];
            }
            for (int r = 0; r < 9; ++r) ds[i][r] = a[r];
            V3 pi = st.p + v3(a[0], a[1], a[2]);
            M3 Ri = st.R * so3_exp(v3(a[3], a[4], a[5]));
            V3 li = l + v3(a[6], a[7], a[8]), pc;
            if (!project(cam, pi, Ri, li, hz[i], &pc) || pc.z < prm_.min_depth) return false;
        }
        Real zm[3] = {0, 0, 0};
        for (int i = 0; i < np; ++i) {
            Real wm = i ? w.wi : w.wm0;
            for (int a = 0; a < 3; ++a) zm[a] += wm * hz[i][a];
        }
        Real S[3][3] = {}, Psz[9 * 3] = {};
        for (int i = 0; i < np; ++i) {
            Real wc = i ? w.wi : w.wc0, e[3];
            for (int a = 0; a < 3; ++a) e[a] = hz[i][a] - zm[a];
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b) S[a][b] += wc * e[a] * e[b];
            for (int r = 0; r < 9; ++r) for (int b = 0; b < 3; ++b) Psz[r * 3 + b] += wc * ds[i][r] * e[b];
        }
        for (int a = 0; a < 3; ++a) S[a][a] += Rn[a];
        chol_solve(L, 9, 9, Psz, 3, 3);      // A = P_ss^-1 P_sz
        Mat &C = w_.C;
        for (int i = 0; i < NSTATE; ++i) {   // C = P_{x,s} A = P_xz
            const Real *Pi = P.row(i);
            for (int b = 0; b < 3; ++b) {
                Real v = 0;
                for (int r = 0; r < 9; ++r) v += Pi[idx[r]] * Psz[r * 3 + b];
                C(i, b) = v;
            }
        }
        Real y[3] = {z[0] - zm[0], z[1] - zm[1], z[2] - zm[2]};
        return kalman_apply(st, w_, S, y, gate, nis);
    }

    void init_landmark(SlamState &st, const Camera &cam, int s, const Real z[3],
                       const Real Rn[3]) override {
        const int na = 9, np = 2 * na + 1;
        Weights w = weights(na);
        Real L[6 * 6];
        pose_chol(st, L);
        Real rs[3] = {std::sqrt(Rn[0]), std::sqrt(Rn[1]), std::sqrt(Rn[2])};
        V3 g[np]; Real dx[np][6];
        for (int i = 0; i < np; ++i) {
            Real a[9] = {};
            if (i > 0) {
                int j = (i - 1) % na; Real sg = (i <= na) ? w.gamma : -w.gamma;
                if (j < 6) for (int r = 0; r < 6; ++r) a[r] = sg * L[r * 6 + j];
                else a[j] = sg * rs[j - 6];
            }
            for (int r = 0; r < 6; ++r) dx[i][r] = a[r];
            Real zi[3] = {z[0] + a[6], z[1] + a[7], z[2] + a[8]};
            M3 Ri = st.R * so3_exp(v3(a[3], a[4], a[5]));
            g[i] = Ri * backproject(cam, zi) + st.p + v3(a[0], a[1], a[2]);
        }
        V3 lm = v3(0, 0, 0);
        for (int i = 0; i < np; ++i) lm = lm + (i ? w.wi : w.wm0) * g[i];
        Real Pll[3][3] = {}, Plx[3][6] = {};
        for (int i = 0; i < np; ++i) {
            Real wc = i ? w.wi : w.wc0;
            V3 e = g[i] - lm;
            for (int a = 0; a < 3; ++a) {
                for (int b = 0; b < 3; ++b) Pll[a][b] += wc * at(e, a) * at(e, b);
                for (int b = 0; b < 6; ++b) Plx[a][b] += wc * at(e, a) * dx[i][b];
            }
        }
        Real Jt[6 * 3];
        for (int r = 0; r < 6; ++r) for (int a = 0; a < 3; ++a) Jt[r * 3 + a] = Plx[a][r];
        chol_solve(L, 6, 6, Jt, 3, 3);       // J^T = P_pp^-1 P_xl
        Mat &P = st.P;
        const int o = NPOSE + 3 * s;
        st.lm[3 * s] = lm.x; st.lm[3 * s + 1] = lm.y; st.lm[3 * s + 2] = lm.z;
        for (int j = 0; j < NSTATE; ++j) {
            if (j >= o && j < o + 3) continue;
            for (int a = 0; a < 3; ++a) {
                Real v;
                if (j < 6) v = Plx[a][j];
                else { v = 0; for (int r = 0; r < 6; ++r) v += Jt[r * 3 + a] * P(r, j); }
                P(o + a, j) = v; P(j, o + a) = v;
            }
        }
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) P(o + a, o + b) = Real(0.5) * (Pll[a][b] + Pll[b][a]);
    }

  private:
    struct Weights { Real gamma, wm0, wc0, wi; };
    Weights weights(int n) const {
        Real a2 = prm_.ukf_alpha * prm_.ukf_alpha;
        Real lam = a2 * (Real(n) + prm_.ukf_kappa) - Real(n);
        Weights w;
        w.gamma = std::sqrt(Real(n) + lam);
        w.wm0 = lam / (Real(n) + lam);
        w.wc0 = w.wm0 + Real(1) - a2 + prm_.ukf_beta;
        w.wi = Real(1) / (Real(2) * (Real(n) + lam));
        return w;
    }
    static void sub_index(int s, int idx[9]) {
        for (int i = 0; i < 6; ++i) idx[i] = i;
        for (int i = 0; i < 3; ++i) idx[6 + i] = NPOSE + 3 * s + i;
    }
    static bool chol_jitter(Real *A, int n) {
        Real backup[9 * 9];
        for (int i = 0; i < n * n; ++i) backup[i] = A[i];
        Real jit = 0;
        for (int t = 0; t < 6; ++t) {
            if (cholesky(A, n, n)) return true;
            jit = jit == 0 ? Real(1e-9) : jit * 100;
            for (int i = 0; i < n * n; ++i) A[i] = backup[i];
            for (int i = 0; i < n; ++i) A[i * n + i] += jit;
        }
        return false;
    }
    static void pose_chol(const SlamState &st, Real L[36]) {
        for (int a = 0; a < 6; ++a) for (int b = 0; b < 6; ++b) L[a * 6 + b] = st.P(a, b);
        if (!chol_jitter(L, 6)) {
            for (int i = 0; i < 36; ++i) L[i] = 0;
            for (int i = 0; i < 6; ++i) L[i * 6 + i] = Real(1e-4);
        }
    }
    Params prm_;
    KalmanWork w_;
};