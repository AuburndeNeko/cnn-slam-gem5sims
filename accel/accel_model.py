#!/usr/bin/env python3
"""Analytical cycle model for an output-stationary systolic array.

Reads a SCALE-Sim topology CSV (conv or GEMM), predicts cycles per layer for
any array shape, optionally checks the prediction against a SCALE-Sim
COMPUTE_REPORT.csv, and prints frame time at several clock periods.

Each layer is treated as a GEMM: M = output rows (pixels or tokens),
K = inner dimension (fh*fw*cin), N = output columns (filters).
  folds = ceil(M / R) * ceil(N / C)            for an R x C array
  scalesim model: cycles = folds * (K + R + C - 2)          (fitted on one layer so far)
  rtl model:      cycles = folds * max(K, R + C - 1) + (R + C + 6)   (handoff formula, checked at 4x4)

Place in CNN_SLAM/accel/. Examples (from CNN_SLAM):
  # validate against the simulated run, then sweep array shapes
  python accel/accel_model.py accel/monodepth2_192x640.csv \
      --report accel/results/monodepth2_192x640_32x32_os_2048/COMPUTE_REPORT.csv
  python accel/accel_model.py accel/monodepth2_192x640.csv \
      --arrays 8x8,16x16,32x32,64x16,16x64,64x64 --clocks 1.0,2.22,2.77,8.43
  # ViT encoder block (GEMM CSV, 12 blocks per frame)
  python accel/accel_model.py accel/vits_block_266x854.csv --gemm --repeat 12
"""
import argparse
import csv
import math

ap = argparse.ArgumentParser()
ap.add_argument("topology")
ap.add_argument("--gemm", action="store_true", help="CSV columns are Layer, M, N, K")
ap.add_argument("--repeat", type=int, default=1, help="times the CSV runs per frame (12 for a ViT block)")
ap.add_argument("--arrays", default="32x32", help="comma list of RxC array shapes")
ap.add_argument("--clocks", default="1.0,2.22,2.77,8.43", help="comma list of clock periods in ns")
ap.add_argument("--model", choices=["scalesim", "rtl"], default="scalesim")
ap.add_argument("--report", default=None, help="SCALE-Sim COMPUTE_REPORT.csv to validate against")
ap.add_argument("--pe-area", type=float, default=1522.0, help="um^2 per PE (Nangate45, final PE)")
args = ap.parse_args()


def load(path, gemm):
    layers = []
    with open(path) as f:
        rows = [r for r in csv.reader(f) if r and r[0].strip()]
    for r in rows[1:]:
        v = [x.strip() for x in r]
        if gemm:
            M, N, K = int(v[1]), int(v[2]), int(v[3])
        else:
            ih, iw, kh, kw, c, nf, s = (int(x) for x in v[1:8])
            M = ((ih - kh) // s + 1) * ((iw - kw) // s + 1)
            K, N = kh * kw * c, nf
        layers.append((v[0], M, K, N))
    return layers


def cycles(M, K, N, R, C, model):
    folds = math.ceil(M / R) * math.ceil(N / C)
    if model == "scalesim":
        return folds * (K + R + C - 2)
    return folds * max(K, R + C - 1) + (R + C + 6)


layers = load(args.topology, args.gemm)
shapes = [tuple(int(x) for x in s.lower().split("x")) for s in args.arrays.split(",")]
clocks = [float(x) for x in args.clocks.split(",")]
macs = sum(M * K * N for _, M, K, N in layers)
print(f"{len(layers)} layers, {macs * args.repeat / 1e9:.2f} GMAC per frame "
      f"(x{args.repeat}), cycle model: {args.model}")

if args.report:
    R, C = shapes[0]
    sim = {}
    with open(args.report) as f:
        for r in list(csv.reader(f))[1:]:
            if r and r[0].strip():
                sim[int(r[0])] = int(float(r[2]))       # column 2 = "Total Cycles" (no prefetch)
    print(f"\nValidation against {args.report} on {R}x{C}")
    print(f"{'layer':52s} {'simulated':>10s} {'model':>10s} {'error':>7s}")
    ts = tm = 0
    for i, (name, M, K, N) in enumerate(layers):
        if i not in sim:
            continue
        m = cycles(M, K, N, R, C, args.model)
        ts += sim[i]
        tm += m
        print(f"{name[:52]:52s} {sim[i]:10d} {m:10d} {100 * (m - sim[i]) / sim[i]:6.1f}%")
    if ts:
        print(f"{'TOTAL':52s} {ts:10d} {tm:10d} {100 * (tm - ts) / ts:6.1f}%")

print(f"\n{'array':>8s} {'area mm2':>9s} {'Mcycles':>9s} {'util':>6s}  "
      + "  ".join(f"{c:g} ns" .rjust(9) for c in clocks) + "   (ms per frame, matmuls only)")
for R, C in shapes:
    cyc = sum(cycles(M, K, N, R, C, args.model) for _, M, K, N in layers) * args.repeat
    util = macs * args.repeat / (cyc * R * C)
    area = R * C * args.pe_area / 1e6
    print(f"{R:>4d}x{C:<3d} {area:9.2f} {cyc / 1e6:9.2f} {100 * util:5.1f}%  "
          + "  ".join(f"{cyc * c / 1e6:9.1f}" for c in clocks))
