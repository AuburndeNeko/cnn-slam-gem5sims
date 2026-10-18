#!/usr/bin/env python3
"""
Export real KITTI grayscale frames + CNN depth for the on-chip vision stage.

Uses the depth maps already cached by build_dataset.py (no CNN rerun).

Binary layout (little-endian, read by backend/vision.hpp):
  header : char[4] "KIMG", u32 version=1, u32 n_frames, u32 width, u32 height
  frame  : u8  gray[height*width]            (row-major)
           u16 depth_cm[height*width]        (CNN depth in cm, 0 = invalid)
"""
import argparse
import os
import struct
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kitti_raw import KittiRaw  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True, help='dir containing <date>/ (KITTI raw)')
    ap.add_argument('--date', default='2011_09_26')
    ap.add_argument('--drive', default='0005')
    ap.add_argument('--out', required=True, help='data directory used by build_dataset.py')
    ap.add_argument('--frames', type=int, default=10)
    ap.add_argument('--cache', default='depth_cache_518x1708')
    a = ap.parse_args()

    seq = KittiRaw(a.base, a.date, a.drive)
    n = min(a.frames, seq.n)
    cache = os.path.join(a.out, a.cache)
    tag = f'{a.date}_drive_{a.drive}'
    path = os.path.join(a.out, f'{tag}_img{n}.bin')

    with open(path, 'wb') as f:
        f.write(b'KIMG' + struct.pack('<4I', 1, n, seq.width, seq.height))
        for k in range(n):
            dpath = os.path.join(cache, f'{k:010d}.npy')
            if not os.path.exists(dpath):
                sys.exit(f'missing {dpath}: run build_dataset.py for at least {n} frames first')
            gray = cv2.cvtColor(seq.image(k), cv2.COLOR_BGR2GRAY)
            depth = np.load(dpath).astype(np.float32)
            assert gray.shape == depth.shape == (seq.height, seq.width)
            dcm = np.clip(np.round(depth * 100.0), 0, 65535).astype('<u2')
            f.write(np.ascontiguousarray(gray, dtype=np.uint8).tobytes())
            f.write(np.ascontiguousarray(dcm).tobytes())
    mb = os.path.getsize(path) / 1e6
    print(f'wrote {path}: {n} frames, {seq.width}x{seq.height}, {mb:.1f} MB')


if __name__ == '__main__':
    main()
