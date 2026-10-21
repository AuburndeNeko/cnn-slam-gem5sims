#!/usr/bin/env python3
"""
Synthetic 2-mic recording aligned with KITTI drive 0005 (10 Hz video frames).

Scenario: one broadband sound source (e.g. a passing vehicle) sweeps from
--az-start to --az-end degrees over the clip; it is silent during --gap-frame.
Two mics on a glasses frame, --spacing metres apart, left/right of the camera.
Azimuth convention: 0 = straight ahead, + = right (camera +x), - = left.

Outputs:
  <out>.kaud   KAUD v1: "KAUD", u32 version, u32 fs, u32 n_mics, u32 n_samples,
               u32 hop, u32 win, f32 spacing_m, then int16 samples interleaved [n][2]
  <out>_gt.txt per hop: hop  t_center  active  az_true_deg  tdoa_samples
Hop i uses samples [i*hop, i*hop + win); frame f = hops 10f .. 10f+9.
"""
import argparse, struct
import numpy as np

C = 343.0  # speed of sound, m/s

def delayed(x, delay, taps=32):
    """y[n] = x(n - delay[n]) with windowed-sinc interpolation (delay in samples, per sample)."""
    n = np.arange(len(x))
    pos = n - delay                                   # fractional read position
    base = np.floor(pos).astype(int)
    k = np.arange(-taps // 2 + 1, taps // 2 + 1)      # kernel offsets
    idx = base[:, None] + k[None, :]
    frac = pos[:, None] - idx                         # distance to each tap
    w = np.sinc(frac) * np.hanning(taps + 2)[1:-1][None, :]
    valid = (idx >= 0) & (idx < len(x))
    return np.sum(np.where(valid, x[np.clip(idx, 0, len(x) - 1)], 0.0) * w, axis=1)

def gcc_phat_ref(a, b, win, max_lag):
    """Python reference: lag (samples) of b relative to a, parabolic-interpolated."""
    W = np.hanning(win)
    X = np.fft.rfft(a * W) * np.conj(np.fft.rfft(b * W))
    r = np.fft.irfft(X / (np.abs(X) + 1e-12), win)
    lags = np.arange(-max_lag, max_lag + 1)
    v = r[lags % win]
    i = int(np.argmax(v))
    if 0 < i < len(v) - 1:
        den = v[i - 1] - 2 * v[i] + v[i + 1]
        off = 0.5 * (v[i - 1] - v[i + 1]) / den if den != 0 else 0.0
    else:
        off = 0.0
    return lags[i] + off

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/audio_0005")
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--fs", type=int, default=16000)
    ap.add_argument("--hop", type=int, default=160)       # 10 ms
    ap.add_argument("--win", type=int, default=512)       # 32 ms FFT window
    ap.add_argument("--spacing", type=float, default=0.15)
    ap.add_argument("--az-start", type=float, default=-60.0)
    ap.add_argument("--az-end", type=float, default=60.0)
    ap.add_argument("--gap-frame", type=int, default=4)   # -1 = no silent gap
    ap.add_argument("--snr-db", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    n_hops = a.frames * 10
    n = (n_hops - 1) * a.hop + a.win                  # exactly 10 hops per video frame
    t = np.arange(n) / a.fs
    dur = n_hops * a.hop / a.fs

    # source azimuth over time (linear sweep) and on/off envelope with 10 ms ramps
    az = np.deg2rad(a.az_start + (a.az_end - a.az_start) * np.clip(t / dur, 0, 1))
    env = np.ones(n)
    if a.gap_frame >= 0:
        g0, g1 = a.gap_frame * 0.1, (a.gap_frame + 1) * 0.1
        ramp = 0.01
        env = np.clip(np.minimum(np.abs(t - g0), np.abs(t - g1)) / ramp, 0, 1)
        env[(t > g0) & (t < g1)] = 0.0

    src = rng.standard_normal(n) * env                # broadband source
    tau = a.spacing * np.sin(az) / C * a.fs           # TDOA in samples (L lags R if > 0)
    left = delayed(src, +tau / 2)
    right = delayed(src, -tau / 2)

    sig_pow = np.mean(src[env > 0] ** 2) if np.any(env > 0) else 1.0
    nstd = np.sqrt(sig_pow / 10 ** (a.snr_db / 10))
    left += nstd * rng.standard_normal(n)
    right += nstd * rng.standard_normal(n)

    peak = max(np.abs(left).max(), np.abs(right).max())
    pcm = np.round(np.stack([left, right], 1) / peak * 0.9 * 32767).astype("<i2")

    with open(a.out + ".kaud", "wb") as f:
        f.write(b"KAUD")
        f.write(struct.pack("<6I", 1, a.fs, 2, n, a.hop, a.win))
        f.write(struct.pack("<f", a.spacing))
        f.write(pcm.tobytes())

    max_lag = int(np.ceil(a.spacing / C * a.fs))
    errs = []
    with open(a.out + "_gt.txt", "w") as f:
        f.write("# hop t_center active az_true_deg tdoa_samples\n")
        for h in range(n_hops):
            s0, c = h * a.hop, h * a.hop + a.win // 2
            active = int(env[s0:s0 + a.win].mean() > 0.5)
            f.write(f"{h} {t[c]:.4f} {active} {np.rad2deg(az[c]):.3f} {tau[c]:.4f}\n")
            if active:
                x = pcm[s0:s0 + a.win].astype(float)
                lag = gcc_phat_ref(x[:, 0], x[:, 1], a.win, max_lag)
                est = np.rad2deg(np.arcsin(np.clip(lag * C / (a.fs * a.spacing), -1, 1)))
                errs.append(abs(est - np.rad2deg(az[c])))

    errs = np.array(errs)
    print(f"wrote {a.out}.kaud: {n} samples x 2 mics, {n_hops} hops, {a.frames} frames, "
          f"spacing {a.spacing} m, max lag {max_lag} samples, SNR {a.snr_db} dB")
    print(f"python GCC-PHAT reference on {len(errs)} active hops: "
          f"MAE {errs.mean():.2f} deg, median {np.median(errs):.2f}, max {errs.max():.2f}, "
          f"within 10 deg {100 * np.mean(errs < 10):.0f}%")

if __name__ == "__main__":
    main()
