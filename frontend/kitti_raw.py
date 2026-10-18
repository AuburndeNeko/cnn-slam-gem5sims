"""
KITTI *raw* (synced+rectified) loader for the CNN-SLAM front-end.

Why raw and not the odometry benchmark? Raw drives ship OXTS (GPS/IMU) packets
and Velodyne scans alongside the images, so we get:
  * ground-truth 6-DoF poses (OXTS -> metric poses, per the KITTI devkit),
  * optional body velocities / angular rates as an alternative motion input,
  * LiDAR to sanity-check the CNN depth.

Conventions
-----------
All poses are T_wc (camera-to-world) for the left colour camera (cam 2,
rectified). "World" = camera frame of the first image, so GT starts at identity.
"""
import os
import glob
import numpy as np


def _read_calib_file(path):
    out = {}
    with open(path) as f:
        for line in f:
            if ':' not in line:
                continue
            k, v = line.split(':', 1)
            try:
                out[k.strip()] = np.array([float(x) for x in v.split()])
            except ValueError:
                pass  # e.g. calib_time
    return out


def _rt(R, t):
    T = np.eye(4)
    T[:3, :3] = np.asarray(R).reshape(3, 3)
    T[:3, 3] = np.asarray(t).reshape(3)
    return T


def _read_timestamps(path):
    from datetime import datetime
    ts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            date, time = line.split(' ')
            hms, frac = time.split('.')
            dt = datetime.strptime(date + ' ' + hms, '%Y-%m-%d %H:%M:%S')
            ts.append(dt.timestamp() + float('0.' + frac))
    ts = np.array(ts)
    return ts - ts[0]


class KittiRaw:
    def __init__(self, base_dir, date, drive, cam=2):
        self.date, self.drive, self.cam = date, drive, cam
        self.calib_dir = os.path.join(base_dir, date)
        self.seq_dir = os.path.join(base_dir, date, f'{date}_drive_{drive}_sync')
        if not os.path.isdir(self.seq_dir):
            raise FileNotFoundError(self.seq_dir)

        self.image_files = sorted(glob.glob(
            os.path.join(self.seq_dir, f'image_{cam:02d}', 'data', '*.png')))
        self.velo_files = sorted(glob.glob(
            os.path.join(self.seq_dir, 'velodyne_points', 'data', '*.bin')))
        self.oxts_files = sorted(glob.glob(
            os.path.join(self.seq_dir, 'oxts', 'data', '*.txt')))
        assert len(self.image_files) == len(self.oxts_files), \
            'synced drive should have one OXTS packet per image'
        self.n = len(self.image_files)
        self.timestamps = _read_timestamps(
            os.path.join(self.seq_dir, f'image_{cam:02d}', 'timestamps.txt'))

        self._load_calib()
        self._load_oxts()

    def _load_calib(self):
        c2c = _read_calib_file(os.path.join(self.calib_dir, 'calib_cam_to_cam.txt'))
        v2c = _read_calib_file(os.path.join(self.calib_dir, 'calib_velo_to_cam.txt'))
        i2v = _read_calib_file(os.path.join(self.calib_dir, 'calib_imu_to_velo.txt'))

        P = c2c[f'P_rect_{self.cam:02d}'].reshape(3, 4)
        self.K = P[:, :3].copy()
        self.fx, self.fy = P[0, 0], P[1, 1]
        self.cx, self.cy = P[0, 2], P[1, 2]
        sz = c2c[f'S_rect_{self.cam:02d}']
        self.width, self.height = int(sz[0]), int(sz[1])

        R_rect = np.eye(4)
        R_rect[:3, :3] = c2c['R_rect_00'].reshape(3, 3)
        T_cam_rect0 = np.eye(4)
        T_cam_rect0[0, 3] = P[0, 3] / P[0, 0]

        T_cam0_velo = _rt(v2c['R'], v2c['T'])
        T_velo_imu = _rt(i2v['R'], i2v['T'])

        self.T_cam_velo = T_cam_rect0 @ R_rect @ T_cam0_velo
        self.T_cam_imu = self.T_cam_velo @ T_velo_imu
        self.T_imu_cam = np.linalg.inv(self.T_cam_imu)

    def _load_oxts(self):
        packets = np.array([np.loadtxt(f) for f in self.oxts_files])
        self.oxts = packets
        er = 6378137.0
        lat0 = packets[0, 0]
        scale = np.cos(lat0 * np.pi / 180.0)

        T_w_imu = []
        for p in packets:
            lat, lon, alt, roll, pitch, yaw = p[:6]
            tx = scale * lon * np.pi * er / 180.0
            ty = scale * er * np.log(np.tan((90.0 + lat) * np.pi / 360.0))
            tz = alt
            cr, sr = np.cos(roll), np.sin(roll)
            cp, sp = np.cos(pitch), np.sin(pitch)
            cyw, syw = np.cos(yaw), np.sin(yaw)
            Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
            Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            Rz = np.array([[cyw, -syw, 0], [syw, cyw, 0], [0, 0, 1]])
            T_w_imu.append(_rt(Rz @ Ry @ Rx, [tx, ty, tz]))
        T_w_imu = np.array(T_w_imu)

        T_wc = np.array([T @ self.T_imu_cam for T in T_w_imu])
        self.gt_T_wc = np.array([np.linalg.inv(T_wc[0]) @ T for T in T_wc])

        # body-frame velocity / angular rate -> camera frame (with lever arm)
        R_ci = self.T_cam_imu[:3, :3]
        r_cam_in_imu = self.T_imu_cam[:3, 3]
        v_i = packets[:, 8:11]          # vf, vl, vu
        w_i = packets[:, 20:23]         # wf, wl, wu
        v_cam_i = v_i + np.cross(w_i, r_cam_in_imu)
        self.vel_cam = v_cam_i @ R_ci.T
        self.omega_cam = w_i @ R_ci.T

    def image(self, k):
        import cv2
        return cv2.imread(self.image_files[k], cv2.IMREAD_COLOR)

    def lidar_depth(self, k):
        """Sparse LiDAR depth image (0 = no return) for CNN depth validation."""
        if k >= len(self.velo_files):
            return None
        pts = np.fromfile(self.velo_files[k], dtype=np.float32).reshape(-1, 4)[:, :3]
        pc = (self.T_cam_velo[:3, :3] @ pts.T).T + self.T_cam_velo[:3, 3]
        pc = pc[pc[:, 2] > 0.5]
        uv = (self.K @ pc.T).T
        u = np.round(uv[:, 0] / uv[:, 2]).astype(int)
        v = np.round(uv[:, 1] / uv[:, 2]).astype(int)
        ok = (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        D = np.zeros((self.height, self.width), np.float32)
        order = np.argsort(-pc[ok, 2])          # nearest point wins
        D[v[ok][order], u[ok][order]] = pc[ok, 2][order]
        return D