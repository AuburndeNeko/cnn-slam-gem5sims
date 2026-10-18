#!/usr/bin/env python3
"""
Host-side CNN-SLAM front-end for KITTI raw. Per frame k:
  1. CNN metric depth map D_k
  2. KLT feature tracks with persistent integer IDs        (data association)
  3. VO dT_{k-1,k} from PnP-RANSAC (CNN depth at k-1 -> 3-D, pixels at k)
  4. Landmark measurements z = (u, v, d) for up to --max-obs tracks
  5. Ground-truth camera pose from OXTS (evaluation only)

Binary layout (little-endian, see backend/dataset.hpp):
  header : char[4] "KSLM", u32 version, u32 n_frames, u32 width, u32 height,
           u32 reserved, f64 fx, fy, cx, cy
  frame  : f64 t, f64 gt_T_wc[12], u32 vo_valid, u32 vo_inliers,
           f64 vo_R[9], f64 vo_t[3], f64 vel_cam[3], f64 omega_cam[3],
           u32 n_obs, u32 pad, n_obs * {u32 id, f32 u, f32 v, f32 d}
"""
import argparse, json, os, struct, sys, time
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kitti_raw import KittiRaw                      # noqa: E402
from cnn_depth import CnnDepth, depth_metrics       # noqa: E402

MAGIC = b'KSLM'
VERSION = 1


def bilinear(D, pts):
    x, y = pts[:, 0].astype(np.float32), pts[:, 1].astype(np.float32)
    return cv2.remap(D, x.reshape(-1, 1), y.reshape(-1, 1), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0).ravel()


class Tracker:
    """Pyramidal KLT with forward-backward check and replenishment."""

    def __init__(self, target=250, min_dist=15, fb_thresh=1.0):
        self.target, self.min_dist, self.fb = target, min_dist, fb_thresh
        self.next_id = 0
        self.ids = np.zeros(0, np.int64)
        self.pts = np.zeros((0, 2), np.float32)
        self.age = np.zeros(0, np.int64)
        self.prev = None
        self.lk = dict(winSize=(21, 21), maxLevel=3,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    def step(self, gray):
        prev_ids, prev_pts = self.ids.copy(), self.pts.copy()
        if self.prev is not None and len(self.pts):
            p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, gray, self.pts, None, **self.lk)
            p0r, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev, p1, None, **self.lk)
            fb = np.linalg.norm(p0r - self.pts, axis=1)
            h, w = gray.shape
            ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < self.fb) & \
                 (p1[:, 0] >= 0) & (p1[:, 0] < w - 1) & (p1[:, 1] >= 0) & (p1[:, 1] < h - 1)
            self.ids, self.pts, self.age = self.ids[ok], p1[ok], self.age[ok] + 1
        if len(self.pts) < self.target:
            mask = np.full(gray.shape, 255, np.uint8)
            for p in self.pts:
                cv2.circle(mask, (int(p[0]), int(p[1])), self.min_dist, 0, -1)
            new = cv2.goodFeaturesToTrack(gray, self.target - len(self.pts), 0.01,
                                          self.min_dist, mask=mask)
            if new is not None:
                new = new.reshape(-1, 2).astype(np.float32)
                n = len(new)
                self.ids = np.concatenate([self.ids, np.arange(self.next_id, self.next_id + n)])
                self.pts = np.concatenate([self.pts, new])
                self.age = np.concatenate([self.age, np.zeros(n, np.int64)])
                self.next_id += n
        self.prev = gray
        return prev_ids, prev_pts


def pnp_vo(K, ids0, pts0, D0, ids1, pts1, dmin, dmax):
    """Relative camera motion T_{k-1,k} from 3-D(k-1, CNN depth) <-> 2-D(k)."""
    common, i0, i1 = np.intersect1d(ids0, ids1, return_indices=True)
    if len(common) < 12:
        return None, np.zeros(0, np.int64)
    p0, p1 = pts0[i0], pts1[i1]
    d0 = bilinear(D0, p0)
    ok = (d0 > dmin) & (d0 < dmax)
    if ok.sum() < 12:
        return None, np.zeros(0, np.int64)
    p0, p1, d0, cid = p0[ok], p1[ok], d0[ok], common[ok]
    Kinv = np.linalg.inv(K)
    X0 = (Kinv @ np.c_[p0, np.ones(len(p0))].T).T * d0[:, None]
    good, rvec, tvec, inl = cv2.solvePnPRansac(
        X0.astype(np.float64), p1.astype(np.float64), K, None,
        iterationsCount=300, reprojectionError=2.0, confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP)
    if not good or inl is None or len(inl) < 10:
        return None, np.zeros(0, np.int64)
    inl = inl.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(X0[inl], p1[inl].astype(np.float64), K, None, rvec, tvec)
    R10, _ = cv2.Rodrigues(rvec)                    # X_k = R10 X_{k-1} + t10
    dR = R10.T
    dt = (-R10.T @ tvec).ravel()
    if np.linalg.norm(dt) > 5.0:                    # >50 m/s at 10 Hz: reject
        return None, np.zeros(0, np.int64)
    return (dR, dt, len(inl)), cid[inl]


def select_obs(ids, pts, age, D, allowed, max_obs, grid, w, h, dmin, dmax):
    """Pick well-spread, long-lived tracks with valid CNN depth."""
    d = bilinear(D, pts)
    ok = (age >= 1) & (d > dmin) & (d < dmax) & np.isin(ids, allowed)
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return []
    gx, gy = grid
    cell = (np.minimum((pts[idx, 0] / w * gx).astype(int), gx - 1) +
            gx * np.minimum((pts[idx, 1] / h * gy).astype(int), gy - 1))
    buckets = {}
    for r in np.argsort(-age[idx], kind='stable'):
        buckets.setdefault(int(cell[r]), []).append(idx[r])
    out, rnd = [], 0
    while len(out) < max_obs:
        added = False
        for c in sorted(buckets):
            if rnd < len(buckets[c]):
                out.append(buckets[c][rnd])
                added = True
                if len(out) >= max_obs:
                    break
        if not added:
            break
        rnd += 1
    out.sort(key=lambda j: -age[j])   # back-end fills free slots oldest-first
    return [(int(ids[j]), float(pts[j, 0]), float(pts[j, 1]), float(d[j])) for j in out]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base', required=True, help='dir containing <date>/ (KITTI raw)')
    ap.add_argument('--date', default='2011_09_26')
    ap.add_argument('--drive', default='0005')
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-frames', type=int, default=0)
    ap.add_argument('--max-obs', type=int, default=48)
    ap.add_argument('--dmin', type=float, default=1.5)
    ap.add_argument('--dmax', type=float, default=40.0)
    ap.add_argument('--depth-source', choices=['cnn', 'lidar'], default='cnn')
    ap.add_argument('--cnn-input', default='518x1708', help='HxW, multiples of 14')
    ap.add_argument('--eval-depth-every', type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    seq = KittiRaw(args.base, args.date, args.drive)
    n = seq.n if args.max_frames <= 0 else min(seq.n, args.max_frames)
    tag = f'{args.date}_drive_{args.drive}'
    print(f'[frontend] {tag}: {n} frames, {seq.width}x{seq.height}, '
          f'fx={seq.fx:.2f} cx={seq.cx:.2f} cy={seq.cy:.2f}')

    H, W = [int(x) for x in args.cnn_input.split('x')]
    cnn = CnnDepth(input_hw=(H, W), cache_dir=os.path.join(args.out, f'depth_cache_{H}x{W}'))
    trk = Tracker()

    bin_path = os.path.join(args.out, f'{tag}.bin')
    f = open(bin_path, 'wb')
    f.write(MAGIC + struct.pack('<5I4d', VERSION, n, seq.width, seq.height, 0,
                                seq.fx, seq.fy, seq.cx, seq.cy))

    D_prev = None
    vo_T = np.eye(4)
    vo_traj, depth_stats, n_obs_hist, inl_hist = [], [], [], []
    n_vo_fail = 0
    t_start = time.time()
    for k in range(n):
        img = seq.image(k)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        D = cnn(img, key=f'{k:010d}') if args.depth_source == 'cnn' else seq.lidar_depth(k)

        prev_ids, prev_pts = trk.step(gray)

        vo, inlier_ids = (None, np.zeros(0, np.int64))
        if k > 0:
            vo, inlier_ids = pnp_vo(seq.K, prev_ids, prev_pts, D_prev, trk.ids, trk.pts,
                                    args.dmin, args.dmax)
        if vo is None:
            dR, dt, ninl, valid = np.eye(3), np.zeros(3), 0, 0
            if k > 0:
                n_vo_fail += 1
            allowed = trk.ids
        else:
            dR, dt, ninl = vo
            valid = 1
            allowed = inlier_ids
        dT = np.eye(4); dT[:3, :3] = dR; dT[:3, 3] = dt
        vo_T = vo_T @ dT
        vo_traj.append(vo_T[:3, :4].ravel())
        inl_hist.append(ninl)

        obs = select_obs(trk.ids, trk.pts, trk.age, D, allowed, args.max_obs, (8, 3),
                         seq.width, seq.height, args.dmin, args.dmax)
        n_obs_hist.append(len(obs))

        f.write(struct.pack('<d', seq.timestamps[k]))
        f.write(struct.pack('<12d', *seq.gt_T_wc[k][:3, :4].ravel()))
        f.write(struct.pack('<2I', valid, ninl))
        f.write(struct.pack('<9d', *dR.ravel()))
        f.write(struct.pack('<3d', *dt))
        f.write(struct.pack('<3d', *seq.vel_cam[k]))
        f.write(struct.pack('<3d', *seq.omega_cam[k]))
        f.write(struct.pack('<2I', len(obs), 0))
        for (i, u, v, d) in obs:
            f.write(struct.pack('<I3f', i & 0xFFFFFFFF, u, v, d))

        if args.depth_source == 'cnn' and args.eval_depth_every > 0 and k % args.eval_depth_every == 0:
            L = seq.lidar_depth(k)
            if L is not None:
                depth_stats.append(depth_metrics(D, L, args.dmax))

        D_prev = D
        if k % 10 == 0 or k == n - 1:
            print(f'[frontend] frame {k:4d}/{n}  tracks={len(trk.ids):3d}  '
                  f'vo_inliers={ninl:3d}  obs={len(obs):2d}  ({time.time()-t_start:.0f}s)')
    f.close()

    np.savetxt(os.path.join(args.out, f'{tag}_gt.txt'),
               seq.gt_T_wc[:n, :3, :4].reshape(n, 12), fmt='%.9f')
    np.savetxt(os.path.join(args.out, f'{tag}_vo_py.txt'), np.array(vo_traj), fmt='%.9f')

    summary = {
        'sequence': tag, 'frames': n, 'bin': os.path.basename(bin_path),
        'bin_bytes': os.path.getsize(bin_path),
        'depth_source': args.depth_source, 'cnn_model': cnn.model_id,
        'cnn_input_hw': [H, W],
        'cnn_mean_infer_s': float(np.mean(cnn.infer_times)) if cnn.infer_times else None,
        'vo_failures': n_vo_fail,
        'vo_inliers_mean': float(np.mean(inl_hist[1:])) if n > 1 else 0,
        'obs_per_frame_mean': float(np.mean(n_obs_hist)),
        'gt_path_length_m': float(np.sum(np.linalg.norm(
            np.diff(seq.gt_T_wc[:n, :3, 3], axis=0), axis=1))),
    }
    if depth_stats:
        summary['cnn_depth_vs_lidar'] = {k: float(np.mean([d[k] for d in depth_stats]))
                                         for k in depth_stats[0]}
    with open(os.path.join(args.out, f'{tag}_frontend.json'), 'w') as jf:
        json.dump(summary, jf, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()