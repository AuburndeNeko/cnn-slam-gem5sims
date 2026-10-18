#!/usr/bin/env python3
"""Synthetic KSLM sequence (same format as KITTI export) for back-end tests."""
import argparse, struct
import numpy as np
import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='synthetic.bin')
    ap.add_argument('--frames', type=int, default=150)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--scale-bias', type=float, default=1.03)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    fx = fy = 720.0; cx, cy, W, H = 620.0, 187.0, 1242, 375
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    T = [np.eye(4)]
    for k in range(1, a.frames):
        d = np.eye(4)
        d[:3, :3] = cv2.Rodrigues(np.array([0.0, 0.01 * np.sin(k / 15), 0.0]))[0]
        d[:3, 3] = [0.02 * np.sin(k / 10), 0.0, 1.0]
        T.append(T[-1] @ d)
    T = np.array(T)
    pts = []
    for c in T[::2, :3, 3]:
        for _ in range(40):
            pts.append(c + [rng.uniform(-15, 15), rng.uniform(-3, 2), rng.uniform(3, 30)])
    pts = np.array(pts)

    f = open(a.out, 'wb')
    f.write(b'KSLM' + struct.pack('<5I4d', 1, a.frames, W, H, 0, fx, fy, cx, cy))
    for k in range(a.frames):
        if k == 0:
            dR, dt = np.eye(3), np.zeros(3)
        else:
            d = np.linalg.inv(T[k - 1]) @ T[k]
            dR = d[:3, :3] @ cv2.Rodrigues(rng.normal(0, 0.002, 3))[0]
            dt = d[:3, 3] * a.scale_bias + rng.normal(0, 0.02, 3)
        Tcw = np.linalg.inv(T[k])
        pc = (Tcw[:3, :3] @ pts.T).T + Tcw[:3, 3]
        uvw = (K @ pc.T).T
        vis = np.where((pc[:, 2] > 2) & (pc[:, 2] < 40))[0]
        uv = uvw[vis, :2] / uvw[vis, 2:3]
        inimg = (uv[:, 0] > 0) & (uv[:, 0] < W) & (uv[:, 1] > 0) & (uv[:, 1] < H)
        vis, uv = vis[inimg][:48], uv[inimg][:48]
        f.write(struct.pack('<d', 0.1 * k))
        f.write(struct.pack('<12d', *T[k][:3, :4].ravel()))
        f.write(struct.pack('<2I', int(k > 0), 100))
        f.write(struct.pack('<9d', *dR.ravel()))
        f.write(struct.pack('<3d', *dt))
        f.write(struct.pack('<6d', 0, 0, 10, 0, 0, 0))
        f.write(struct.pack('<2I', len(vis), 0))
        for i, (u, v) in zip(vis, uv):
            d = pc[i, 2]
            f.write(struct.pack('<I3f', int(i), u + rng.normal(0, 1.0), v + rng.normal(0, 1.0),
                                d * (1 + rng.normal(0, 0.05))))
    f.close()
    np.savetxt(a.out.replace('.bin', '_gt.txt'), T[:, :3, :4].reshape(-1, 12), fmt='%.9f')
    print(f'wrote {a.out}: {a.frames} frames, {len(pts)} landmarks')


if __name__ == '__main__':
    main()