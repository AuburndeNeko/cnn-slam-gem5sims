#!/usr/bin/env python3
"""Analytical model of a vector / special-function unit next to the systolic array.

Inputs
  - an op trace from accel/trace_ops.py (element counts of every non-matmul op), and
  - the topology CSVs, for the array cycles (same formula as accel/accel_model.py:
    folds * (K + R + C - 2), validated against SCALE-Sim).

Vector unit: L lanes, each lane does one elementwise step per cycle (add, multiply,
compare, LUT lookup with linear interpolation). An op category costs `cost` lane-cycles
per element, so cycles = elements * cost / L. Costs are assumptions, set on the command line.

Two placements:
  standalone  every non-matmul op is a separate pass through the vector unit.
  readout     elementwise ops that directly follow a conv/matmul (BatchNorm/bias/scale folds,
              ReLU/ELU/sigmoid/GELU, requantisation) are applied while the array's results drain,
              so they cost no extra cycles. This needs L >= C (the array drains up to one
              C-wide row per cycle). Residual adds join them with --fuse-residual (needs the
              skip tensor streamed in beside the drain). Softmax, LayerNorm, pooling and
              bilinear upsampling are row/window operations and stay as separate passes.

With --hide-softmax (fused attention, see attention_model.py) softmax is treated like the
readout-fused ops: its cycles overlap the array's work on the next query tile.

Frame time is given two ways:
  serial   = array + vector cycles (no overlap; conservative)
  overlap  = max(array, vector)   (perfect double buffering; optimistic)

Place in CNN_SLAM/accel/. Examples (from CNN_SLAM):
  python accel/vector_model.py accel/ops_md2_192x640.json --conv-csv accel/monodepth2_192x640.csv
  python accel/vector_model.py accel/ops_dav2_266x854.json --conv-csv accel/vits_dpt_266x854.csv \\
      --gemm-csv accel/vits_block_266x854.csv --repeat 12 --heads 6
  python accel/vector_model.py accel/ops_dav2_518x1708.json --conv-csv accel/vits_dpt_518x1708.csv \\
      --gemm-csv accel/vits_block_518x1708.csv --repeat 12 --heads 6 --array 64x64
"""
import argparse
import csv
import json
import math

# Default cost per element in lane-cycles (assumptions; see --cost)
DEFAULT_COST = {
    "fold": 1,               # BatchNorm / bias / scale / requantise: one multiply-add
    "act": 1,                # ReLU, ELU, sigmoid, GELU: one LUT + interpolation step (pipelined)
    "residual": 1,           # one add
    "softmax": 3,            # pass 1 max; pass 2 exp + running sum; pass 3 multiply by 1/sum
    "layernorm": 3,          # pass 1 sum and sum of squares; pass 2 subtract mean; multiply-add
    "pool": None,            # window - 1 compares per output (taken from the trace)
    "upsample_bilinear": 3,  # three linear interpolations per output
    "move": 0,               # concat / pad / nearest upsample / copies: address generation only
    "const": 0,              # position-embedding resize: precomputed offline per resolution
}
FUSABLE = {"fold", "act"}    # elementwise ops that can ride on the array's readout path

ap = argparse.ArgumentParser()
ap.add_argument("trace", help="JSON from trace_ops.py")
ap.add_argument("--conv-csv", help="SCALE-Sim conv topology (layer per row)")
ap.add_argument("--gemm-csv", help="SCALE-Sim GEMM topology (Layer, M, N, K)")
ap.add_argument("--repeat", type=int, default=1, help="times the GEMM CSV runs per frame (12 blocks)")
ap.add_argument("--heads", type=int, default=1,
                help="attention heads: rows whose name starts with 'attn' are multiplied by this "
                     "(the block CSVs list attention for ONE head)")
ap.add_argument("--array", default="32x32")
ap.add_argument("--lanes", default="8,16,32,64,128")
ap.add_argument("--clock-ns", default="2.77", help="comma list of clock periods (2.77 = lop at 360 MHz)")
ap.add_argument("--fuse-residual", action="store_true")
ap.add_argument("--hide-softmax", action="store_true",
                help="fused attention: softmax runs on chip while the array computes the next tile, "
                     "so its cycles are hidden (check pacing with attention_model.py)")
ap.add_argument("--cost", action="append", default=[], help="override, e.g. --cost softmax=4")
args = ap.parse_args()

cost = dict(DEFAULT_COST)
for c in args.cost:
    k, v = c.split("=")
    cost[k] = float(v)
R, C = (int(x) for x in args.array.lower().split("x"))
lanes = [int(x) for x in args.lanes.split(",")]
clocks = [float(x) for x in args.clock_ns.split(",")]


def cycles(M, K, N):
    return math.ceil(M / R) * math.ceil(N / C) * (K + R + C - 2)


def array_cycles():
    total, macs, detail = 0, 0, []
    if args.conv_csv:
        rows = [r for r in csv.reader(open(args.conv_csv)) if r and r[0].strip()][1:]
        cc = cm = 0
        for v in rows:
            ih, iw, kh, kw, ch, nf, s = (int(x) for x in v[1:8])
            M = ((ih - kh) // s + 1) * ((iw - kw) // s + 1)
            cc += cycles(M, kh * kw * ch, nf)
            cm += M * kh * kw * ch * nf
        detail.append(("convs", cc, cm))
        total += cc
        macs += cm
    if args.gemm_csv:
        rows = [r for r in csv.reader(open(args.gemm_csv)) if r and r[0].strip()][1:]
        gc = gm = 0
        for v in rows:
            M, N, K = int(v[1]), int(v[2]), int(v[3])
            h = args.heads if v[0].strip().startswith("attn") else 1
            gc += h * cycles(M, K, N)
            gm += h * M * K * N
        detail.append((f"GEMM block x{args.repeat}", gc * args.repeat, gm * args.repeat))
        total += gc * args.repeat
        macs += gm * args.repeat
    return total, macs, detail


tr = json.load(open(args.trace))
ops = tr["ops"]
traced_macs = sum(r.get("macs", 0) for r in ops)

# lane-cycles per category, split into fusable-at-readout and not
work = {}
for r in ops:
    cat = r["cat"]
    if cat == "matmul":
        continue
    c = (r["window"] - 1) if cat == "pool" else cost[cat]
    w = work.setdefault(cat, dict(elems=0, total=0.0, fusable=0.0))
    w["elems"] += r["elems"]
    w["total"] += r["elems"] * c
    if (r["from_mm"] and (cat in FUSABLE or (cat == "residual" and args.fuse_residual))) or \
            (cat == "softmax" and args.hide_softmax):
        w["fusable"] += r["elems"] * c

arr, amacs, detail = array_cycles()
print(f"{tr['desc']}  {tr['width']}x{tr['height']}  array {R}x{C}")
for name, cc, cm in detail:
    print(f"  array: {name:22s} {cc / 1e6:8.2f} M cycles  {cm / 1e9:7.2f} GMAC")
print(f"  array total {arr / 1e6:.2f} M cycles, {amacs / 1e9:.2f} GMAC "
      f"(trace: {traced_macs / 1e9:.2f} GMAC{' OK' if abs(amacs - traced_macs) < 0.005 * traced_macs else ' MISMATCH'})")

print(f"\n{'category':18s} {'M elem':>9s} {'cost':>5s} {'M lane-cyc':>11s} {'fusable':>9s}")
tot = fus = 0.0
for cat, w in sorted(work.items(), key=lambda kv: -kv[1]["total"]):
    cs = "win-1" if cat == "pool" else f"{cost[cat]:g}"
    print(f"{cat:18s} {w['elems'] / 1e6:9.2f} {cs:>5s} {w['total'] / 1e6:11.2f} {w['fusable'] / 1e6:9.2f}")
    tot += w["total"]
    fus += w["fusable"]
print(f"{'TOTAL':18s} {'':9s} {'':5s} {tot / 1e6:11.2f} {fus / 1e6:9.2f}")

print(f"\nVector cycles per frame and frame time (array {arr / 1e6:.2f} M cycles)")
hdr = f"{'lanes':>5s} {'standalone':>11s} {'readout':>9s} {'+% vs array':>12s}"
for ck in clocks:
    hdr += f"  {'serial ms':>9s} {'overlap ms':>10s} @{ck:g}ns"
print(hdr + "    (serial/overlap use the readout placement)")
for L in lanes:
    sa = tot / L
    ro = (tot - fus) / L
    note = "" if L >= C else "  (L < C: readout cannot keep up, use standalone)"
    v = ro if L >= C else sa
    line = f"{L:5d} {sa / 1e6:9.2f} M {ro / 1e6:7.2f} M {100 * v / arr:11.1f}%"
    for ck in clocks:
        line += f"  {(arr + v) * ck / 1e6:9.1f} {max(arr, v) * ck / 1e6:10.1f}         "
    print(line + note)
