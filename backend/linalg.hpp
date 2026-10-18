// Minimal, dependency-free linear algebra for the gem5 back-end.
#pragma once
#include <cmath>
#include <cstddef>
#include <vector>
#include <algorithm>

#ifdef SLAM_USE_DOUBLE
typedef double Real;
#define REAL_NAME "fp64"
#else
typedef float Real;
#define REAL_NAME "fp32"
#endif

struct Mat {
    int r = 0, c = 0;
    std::vector<Real> d;
    Mat() {}
    Mat(int R, int C) : r(R), c(C), d((size_t)R * C, Real(0)) {}
    void resize(int R, int C) { r = R; c = C; d.assign((size_t)R * C, Real(0)); }
    inline Real &operator()(int i, int j) { return d[(size_t)i * c + j]; }
    inline Real operator()(int i, int j) const { return d[(size_t)i * c + j]; }
    inline Real *row(int i) { return &d[(size_t)i * c]; }
    void zero() { std::fill(d.begin(), d.end(), Real(0)); }
    void symmetrize() {
        for (int i = 0; i < r; ++i)
            for (int j = i + 1; j < c; ++j) {
                Real m = Real(0.5) * ((*this)(i, j) + (*this)(j, i));
                (*this)(i, j) = m; (*this)(j, i) = m;
            }
    }
};

// In-place lower Cholesky. Returns false if not positive definite.
inline bool cholesky(Real *A, int n, int lda) {
    for (int j = 0; j < n; ++j) {
        Real s = A[j * lda + j];
        for (int k = 0; k < j; ++k) s -= A[j * lda + k] * A[j * lda + k];
        if (!(s > Real(0))) return false;
        Real ljj = std::sqrt(s);
        A[j * lda + j] = ljj;
        for (int i = j + 1; i < n; ++i) {
            Real t = A[i * lda + j];
            for (int k = 0; k < j; ++k) t -= A[i * lda + k] * A[j * lda + k];
            A[i * lda + j] = t / ljj;
        }
        for (int k = j + 1; k < n; ++k) A[j * lda + k] = Real(0);
    }
    return true;
}

// Solve (L L^T) X = B in place, B is n x m.
inline void chol_solve(const Real *L, int n, int lda, Real *B, int m, int ldb) {
    for (int col = 0; col < m; ++col) {
        for (int i = 0; i < n; ++i) {
            Real s = B[i * ldb + col];
            for (int k = 0; k < i; ++k) s -= L[i * lda + k] * B[k * ldb + col];
            B[i * ldb + col] = s / L[i * lda + i];
        }
        for (int i = n - 1; i >= 0; --i) {
            Real s = B[i * ldb + col];
            for (int k = i + 1; k < n; ++k) s -= L[k * lda + i] * B[k * ldb + col];
            B[i * ldb + col] = s / L[i * lda + i];
        }
    }
}

struct V3 { Real x, y, z; };
struct M3 { Real m[3][3]; };

inline V3 v3(Real x, Real y, Real z) { V3 v = {x, y, z}; return v; }
inline V3 operator+(V3 a, V3 b) { return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
inline V3 operator-(V3 a, V3 b) { return v3(a.x - b.x, a.y - b.y, a.z - b.z); }
inline V3 operator*(Real s, V3 a) { return v3(s * a.x, s * a.y, s * a.z); }
inline Real dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Real norm(V3 a) { return std::sqrt(dot(a, a)); }
inline Real &at(V3 &v, int i) { return i == 0 ? v.x : (i == 1 ? v.y : v.z); }
inline Real at(const V3 &v, int i) { return i == 0 ? v.x : (i == 1 ? v.y : v.z); }

inline M3 eye3() { M3 I = {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}}; return I; }
inline M3 zero3() { M3 Z = {{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}}; return Z; }
inline V3 operator*(const M3 &A, V3 v) {
    return v3(A.m[0][0] * v.x + A.m[0][1] * v.y + A.m[0][2] * v.z,
              A.m[1][0] * v.x + A.m[1][1] * v.y + A.m[1][2] * v.z,
              A.m[2][0] * v.x + A.m[2][1] * v.y + A.m[2][2] * v.z);
}
inline M3 operator*(const M3 &A, const M3 &B) {
    M3 C;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            C.m[i][j] = A.m[i][0] * B.m[0][j] + A.m[i][1] * B.m[1][j] + A.m[i][2] * B.m[2][j];
    return C;
}
inline M3 tr(const M3 &A) {
    M3 T;
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) T.m[i][j] = A.m[j][i];
    return T;
}
inline M3 skew(V3 w) {
    M3 S = {{{0, -w.z, w.y}, {w.z, 0, -w.x}, {-w.y, w.x, 0}}};
    return S;
}
inline M3 so3_exp(V3 w) {
    Real th = norm(w);
    M3 K = skew(w), K2 = K * K, R = eye3();
    Real a, b;
    if (th < Real(1e-5)) { a = Real(1) - th * th / 6; b = Real(0.5) - th * th / 24; }
    else { a = std::sin(th) / th; b = (Real(1) - std::cos(th)) / (th * th); }
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) R.m[i][j] += a * K.m[i][j] + b * K2.m[i][j];
    return R;
}
inline V3 so3_log(const M3 &R) {
    Real c = (R.m[0][0] + R.m[1][1] + R.m[2][2] - Real(1)) * Real(0.5);
    c = std::max(Real(-1), std::min(Real(1), c));
    Real th = std::acos(c);
    V3 v = v3(R.m[2][1] - R.m[1][2], R.m[0][2] - R.m[2][0], R.m[1][0] - R.m[0][1]);
    Real s = (th < Real(1e-4)) ? Real(0.5) : th / (Real(2) * std::sin(th));
    return s * v;
}
inline void orthonormalize(M3 &R) {
    V3 r0 = v3(R.m[0][0], R.m[0][1], R.m[0][2]);
    V3 r1 = v3(R.m[1][0], R.m[1][1], R.m[1][2]);
    r0 = (Real(1) / norm(r0)) * r0;
    r1 = r1 - dot(r0, r1) * r0;
    r1 = (Real(1) / norm(r1)) * r1;
    V3 r2 = v3(r0.y * r1.z - r0.z * r1.y, r0.z * r1.x - r0.x * r1.z, r0.x * r1.y - r0.y * r1.x);
    Real rows[3][3] = {{r0.x, r0.y, r0.z}, {r1.x, r1.y, r1.z}, {r2.x, r2.y, r2.z}};
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) R.m[i][j] = rows[i][j];
}
inline bool inv3(const Real S[3][3], Real Si[3][3]) {
    Real a = S[0][0], b = S[0][1], c = S[0][2], d = S[1][0], e = S[1][1], f = S[1][2],
         g = S[2][0], h = S[2][1], i = S[2][2];
    Real A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
    Real det = a * A + b * B + c * C;
    if (std::fabs(det) < Real(1e-20)) return false;
    Real id = Real(1) / det;
    Si[0][0] = A * id; Si[0][1] = -(b * i - c * h) * id; Si[0][2] = (b * f - c * e) * id;
    Si[1][0] = B * id; Si[1][1] = (a * i - c * g) * id;  Si[1][2] = -(a * f - c * d) * id;
    Si[2][0] = C * id; Si[2][1] = -(a * h - b * g) * id; Si[2][2] = (a * e - b * d) * id;
    return true;
}