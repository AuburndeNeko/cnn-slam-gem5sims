#!/usr/bin/env python3
"""ATE (SE(3) Umeyama, no scale) / RPE for KITTI-format trajectories.
Usage: evaluate.py --gt gt.txt --est name=traj.txt [...] [--plot out.png] [--json out.json]"""
import argparse, json
import numpy as np


def load(path):
    M = np.loadtxt(path).reshape(-1, 3, 4)
    T = np.tile(np.eye(4), (len(M), 1, 1))
    T[:, :3, :4] = M
    return T


def umeyama_se3(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    C = (dst - mu_d).T @ (src - mu_s) / len(src)
    U, _, Vt = np.linalg.svd(C)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    return R, mu_d - R @ mu_s


def metrics(gt, est, seg=10.0):
    n = min(len(gt), len(est))
    gt, est = gt[:n], est[:n]
    pg, pe = gt[:, :3, 3], est[:, :3, 3]
    R, t = umeyama_se3(pe, pg)
    pa = (R @ pe.T).T + t
    ate = float(np.sqrt(np.mean(np.sum((pa - pg) ** 2, 1))))
    ate_raw = float(np.sqrt(np.mean(np.sum((pe - pg) ** 2, 1))))
    dist = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pg, axis=0), axis=1))]
    errs = []
    for i in range(n):
        j = np.searchsorted(dist, dist[i] + seg)
        if j >= n:
            break
        E = np.linalg.inv(np.linalg.inv(gt[i]) @ gt[j]) @ (np.linalg.inv(est[i]) @ est[j])
        errs.append(np.linalg.norm(E[:3, 3]) / (dist[j] - dist[i]))
    rot = []
    for i in range(n - 1):
        E = np.linalg.inv(np.linalg.inv(gt[i]) @ gt[i + 1]) @ (np.linalg.inv(est[i]) @ est[i + 1])
        rot.append(np.degrees(np.arccos(np.clip((np.trace(E[:3, :3]) - 1) / 2, -1, 1))))
    path = float(dist[-1])
    final = float(np.linalg.norm(pe[-1] - pg[-1]))
    return {'ate_rmse_m': ate, 'ate_unaligned_m': ate_raw,
            'final_err_m': final, 'final_err_pct_path': 100 * final / max(path, 1e-9),
            f'drift_{int(seg)}m_pct': 100 * float(np.mean(errs)) if errs else None,
            'rpe_rot_deg_per_frame': float(np.mean(rot)), 'path_m': path, 'frames': n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', required=True)
    ap.add_argument('--est', nargs='+', required=True)
    ap.add_argument('--plot')
    ap.add_argument('--json')
    a = ap.parse_args()
    gt = load(a.gt)
    res, trajs = {}, {}
    for spec in a.est:
        name, path = spec.split('=', 1)
        trajs[name] = load(path)
        res[name] = metrics(gt, trajs[name])
    hdr = f"{'method':<14}{'ATE[m]':>9}{'raw[m]':>9}{'final[m]':>10}{'final%':>8}{'drift10m%':>11}{'rot[deg/f]':>12}"
    print(hdr); print('-' * len(hdr))
    for k, r in res.items():
        print(f"{k:<14}{r['ate_rmse_m']:>9.3f}{r['ate_unaligned_m']:>9.3f}{r['final_err_m']:>10.3f}"
              f"{r['final_err_pct_path']:>8.2f}{(r['drift_10m_pct'] or 0):>11.2f}{r['rpe_rot_deg_per_frame']:>12.4f}")
    print(f"(GT path length {next(iter(res.values()))['path_m']:.1f} m)")
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(res, f, indent=2)
    if a.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
        ax[0].plot(gt[:, 0, 3], gt[:, 2, 3], 'k-', lw=2.5, label='GT (OXTS)')
        for k, T in trajs.items():
            ax[0].plot(T[:, 0, 3], T[:, 2, 3], lw=1.5, label=f"{k} (ATE {res[k]['ate_rmse_m']:.2f} m)")
        ax[0].set_xlabel('x [m]'); ax[0].set_ylabel('z [m] (forward)')
        ax[0].set_title('Top-down trajectory'); ax[0].axis('equal'); ax[0].grid(alpha=.3); ax[0].legend()
        for k, T in trajs.items():
            n = min(len(T), len(gt))
            ax[1].plot(np.linalg.norm(T[:n, :3, 3] - gt[:n, :3, 3], axis=1), label=k)
        ax[1].set_xlabel('frame'); ax[1].set_ylabel('position error [m]')
        ax[1].set_title('Error vs. time'); ax[1].grid(alpha=.3); ax[1].legend()
        fig.tight_layout(); fig.savefig(a.plot, dpi=130)
        print('plot ->', a.plot)


if __name__ == '__main__':
    main()