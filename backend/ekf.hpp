// Error-state EKF-SLAM.
// Predict: F = [[I, -R[dt]x],[0, dR^T]], G = diag(R, I); landmarks static -> O(N).
// Update : H has 9 non-zero columns -> P H^T is O(9N), rank-3 downdate O(N^2).
#pragma once
#include "slam_core.hpp"

class EkfSlam : public Filter {
  public:
    const char *name() const override { return "ekf"; }

    void predict(SlamState &st, const Motion &m) override {
        Mat &P = st.P;
        M3 A = st.R * skew(m.dt);
        M3 dRt = tr(m.dR);
        Real F[6][6] = {};
        for (int i = 0; i < 3; ++i) {
            F[i][i] = 1;
            for (int j = 0; j < 3; ++j) { F[i][3 + j] = -A.m[i][j]; F[3 + i][3 + j] = dRt.m[i][j]; }
        }
        Real tmp[6];
        for (int j = 0; j < NSTATE; ++j) {          // P_p* <- F P_p*
            for (int i = 0; i < 6; ++i) {
                Real s = 0;
                for (int k = 0; k < 6; ++k) s += F[i][k] * P(k, j);
                tmp[i] = s;
            }
            for (int i = 0; i < 6; ++i) P(i, j) = tmp[i];
        }
        for (int i = 0; i < NSTATE; ++i) {          // P_*p <- P_*p F^T
            Real *Pi = P.row(i);
            for (int c = 0; c < 6; ++c) {
                Real s = 0;
                for (int k = 0; k < 6; ++k) s += Pi[k] * F[c][k];
                tmp[c] = s;
            }
            for (int c = 0; c < 6; ++c) Pi[c] = tmp[c];
        }
        for (int a = 0; a < 3; ++a)                 // + G Q G^T
            for (int b = 0; b < 3; ++b) {
                Real s = 0;
                for (int k = 0; k < 3; ++k) s += st.R.m[a][k] * m.qt[k] * st.R.m[b][k];
                P(a, b) += s;
            }
        for (int a = 0; a < 3; ++a) P(3 + a, 3 + a) += m.qr[a];

        st.p = st.p + st.R * m.dt;
        st.R = st.R * m.dR;
        orthonormalize(st.R);
    }

    bool update(SlamState &st, const Camera &cam, int s, const Real z[3], const Real Rn[3],
                Real gate, Real *nis) override {
        V3 l = st.landmark(s), pc;
        Real h[3];
        if (!project(cam, st.p, st.R, l, h, &pc)) return false;
        Real iz = Real(1) / pc.z;
        Real J[3][3] = {{cam.fx * iz, 0, -cam.fx * pc.x * iz * iz},
                        {0, cam.fy * iz, -cam.fy * pc.y * iz * iz},
                        {0, 0, 1}};
        M3 Rt = tr(st.R), Spc = skew(pc);
        Real Hp[3][3], Ht[3][3], Hl[3][3];
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) {
                Real sl = 0, st_ = 0;
                for (int k = 0; k < 3; ++k) { sl += J[a][k] * Rt.m[k][b]; st_ += J[a][k] * Spc.m[k][b]; }
                Hl[a][b] = sl; Hp[a][b] = -sl; Ht[a][b] = st_;
            }
        const int o = NPOSE + 3 * s;
        Mat &P = st.P, &C = w_.C;
        for (int i = 0; i < NSTATE; ++i) {
            const Real *Pi = P.row(i);
            for (int a = 0; a < 3; ++a) {
                Real v = 0;
                for (int k = 0; k < 3; ++k)
                    v += Pi[k] * Hp[a][k] + Pi[3 + k] * Ht[a][k] + Pi[o + k] * Hl[a][k];
                C(i, a) = v;
            }
        }
        Real S[3][3];
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) {
                Real v = 0;
                for (int k = 0; k < 3; ++k)
                    v += Hp[a][k] * C(k, b) + Ht[a][k] * C(3 + k, b) + Hl[a][k] * C(o + k, b);
                S[a][b] = v + (a == b ? Rn[a] : Real(0));
            }
        Real y[3] = {z[0] - h[0], z[1] - h[1], z[2] - h[2]};
        return kalman_apply(st, w_, S, y, gate, nis);
    }

    void init_landmark(SlamState &st, const Camera &cam, int s, const Real z[3],
                       const Real Rn[3]) override {
        init_landmark_linear(st, cam, s, z, Rn);
    }

  private:
    KalmanWork w_;
};