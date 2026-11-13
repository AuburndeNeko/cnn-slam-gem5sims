#!/usr/bin/env python3
"""Where is the CNN depth biased?

Prints the median of (CNN depth / LiDAR depth) by LiDAR range, for all LiDAR
pixels and for LiDAR pixels close to the corners the tracker would pick, and
separately for the first and second part of the drive.
Ratio > 1 means the CNN OVERestimates depth (-> VO translation too long).
(Note: 'median_ratio' in *_frontend.json is the inverse, LiDAR / CNN.)

Place in CNN_SLAM/tools/. Uses the depth cache, so pass the same --out, --model
and --cnn-input as the build_dataset.py run you want to inspect:

  python tools/depth_bias.py --base kitti --drive 0005 --out data_md2 --model monodepth2
  python tools/depth_bias.py --base kitti --drive 0005 --out data
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'frontend'))
from kitti_raw import KittiRaw      # noqa: E402
from cnn_depth import CnnDepth      # noqa: E402

BINS = [(1.5, 10), (10, 20), (20, 40), (40, 80)]

ap = argparse.ArgumentParser()
ap.add_argument('--base', required=True)
ap.add_argument('--date', default='2011_09_26')
ap.add_argument('--drive', default='0005')
ap.add_argument('--out', required=True, help='same --out as the build_dataset.py run')
ap.add_argument('--model', choices=['dav2', 'monodepth2'], default='dav2')
ap.add_argument('--cnn-input', default='518x1708')
ap.add_argument('--md2-repo', default='monodepth2')
ap.add_argument('--md2-weights', default='monodepth2/models/mono+stereo_640x192')
ap.add_argument('--every', type=int, default=5, help='use every Nth frame')
ap.add_argument('--radius', type=int, default=6, help='"near a corner" = within this many pixels')
ap.add_argument('--split', type=int, default=50, help='frame index separating the two parts')
a = ap.parse_args()

seq = KittiRaw(a.base, a.date, a.drive)
H, W = [int(x) for x in a.cnn_input.split('x')]
if a.model == 'monodepth2':
    W, H = [int(x) for x in os.path.basename(a.md2_weights.rstrip('/')).split('_')[-1].split('x')]
    cnn = CnnDepth(backend='monodepth2', md2_repo=a.md2_repo, md2_weights=a.md2_weights,
                   input_hw=(H, W), cache_dir=os.path.join(a.out, f'depth_cache_md2_{H}x{W}'))
else:
    cnn = CnnDepth(input_hw=(H, W), cache_dir=os.path.join(a.out, f'depth_cache_{H}x{W}'))

lidar, ratio, near, part = [], [], [], []
for k in range(0, seq.n, a.every):
    L = seq.lidar_depth(k)
    if L is None:
        continue
    img = seq.image(k)
    D = cnn(img, key=f'{k:010d}')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 250, 0.01, 15)   # same detector settings as Tracker
    mask = np.zeros(gray.shape, np.uint8)
    if corners is not None:
        for x, y in corners.reshape(-1, 2):
            cv2.circle(mask, (int(x), int(y)), a.radius, 255, -1)
    ok = (L > 1e-3) & (D > 1e-3)
    lidar.append(L[ok])
    ratio.append(D[ok] / L[ok])
    near.append(mask[ok] > 0)
    part.append(np.full(int(ok.sum()), 0 if k < a.split else 1))

lidar, ratio, near, part = map(np.concatenate, (lidar, ratio, near, part))


def cell(sel):
    return f'{np.median(ratio[sel]):6.3f} ({int(sel.sum()):7d})' if sel.sum() >= 50 else '   n/a           '


print(f'{a.model}, drive {a.drive}: median CNN/LiDAR ratio (number of LiDAR pixels)')
for p, label in ((0, f'frames < {a.split}'), (1, f'frames >= {a.split}')):
    print(f'\n{label}')
    print(f'{"range [m]":>10s}  {"all pixels":>16s}  {"near corners":>16s}')
    for lo, hi in BINS:
        r = (lidar >= lo) & (lidar < hi) & (part == p)
        print(f'{lo:4.1f}-{hi:<4.0f}   {cell(r)}  {cell(r & near)}')
    r = (lidar >= 1.5) & (lidar < 40) & (part == p)
    print(f'{"1.5-40":>10s}  {cell(r)}  {cell(r & near)}')
