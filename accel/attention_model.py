#!/usr/bin/env python3
"""DRAM traffic and tile pacing of ViT attention, unfused vs fused, on an output-stationary array.

Per head and per block, with N tokens and head dimension d (words = INT8 bytes):

  unfused, SCALE-Sim view (softmax not modelled, as in our SCALE-Sim runs):
      Q + K                (attn_scores inputs)
    + N*N  written          (score matrix out of attn_scores)
    + N*N  read             (score matrix into attn_values)
    + V * v_passes          (V re-read per query tile if it does not fit the weight buffer)
    + O                     (attn_values output)
  unfused, with a real softmax pass: + 2*N*N (read scores, write probabilities)

  fused (scores -> softmax -> x V for one tile of R query rows, all on chip):
      Q + O + (K + V) * kv_passes
    The R x N score tile lives in the output buffer and never goes to DRAM. kv_passes = 1 when
    K and V of the head stay resident on chip, else ceil(N / R) (re-streamed per query tile).
    Softmax is exact (all N keys of the tile are on chip), so no FlashAttention rescaling is needed.

Pacing: per query tile the array computes scores (R x d)(d x N) and then P(R x N)(N x d); the
vector unit's softmax for that tile (R*N elements * softmax_cost / lanes) can run while the array
works on the next tile. If softmax time <= array time per tile, the softmax is hidden.

Place in CNN_SLAM/accel/. Examples (from CNN_SLAM):
  # wearable: 32x32, 1160 tokens
  python accel/attention_model.py --tokens 1160 --array 32x32 --lanes 32 --block-words 54.5e6
  # full resolution: 64x64, 4515 tokens, SCALE-Sim block traffic 515 M words
  python accel/attention_model.py --tokens 4515 --array 64x64 --lanes 64 --block-words 515e6 \\
      --scalesim-attn 373.4e6
"""
import argparse
import math

ap = argparse.ArgumentParser()
ap.add_argument("--tokens", type=int, required=True, help="N (1160 wearable, 4515 full-res)")
ap.add_argument("--heads", type=int, default=6)
ap.add_argument("--head-dim", type=int, default=64)
ap.add_argument("--blocks", type=int, default=12)
ap.add_argument("--array", default="32x32")
ap.add_argument("--rows", type=int, default=0, help="query rows per tile (default: array rows R)")
ap.add_argument("--lanes", type=int, default=32, help="vector lanes")
ap.add_argument("--softmax-cost", type=float, default=3.0, help="lane-cycles per softmax element")
ap.add_argument("--weight-buf", type=int, default=512 * 1024, help="weight buffer words (V operand)")
ap.add_argument("--out-buf", type=int, default=512 * 1024, help="output buffer words (score tile)")
ap.add_argument("--in-buf", type=int, default=2048 * 1024, help="input buffer words")
ap.add_argument("--double-buffer", action="store_true", default=True,
                help="streaming operands get half of each buffer (as in SCALE-Sim)")
ap.add_argument("--block-words", type=float, default=None,
                help="SCALE-Sim DRAM words per block (all six GEMMs), to compute the frame total")
ap.add_argument("--scalesim-attn", type=float, default=None,
                help="SCALE-Sim attention words per block (attn_scores + attn_values, all heads), "
                     "if known; otherwise the analytical unfused value is used")
ap.add_argument("--dpt-words", type=float, default=None, help="DPT + patch embedding words per frame")
args = ap.parse_args()

N, h, d, nb = args.tokens, args.heads, args.head_dim, args.blocks
R, C = (int(x) for x in args.array.lower().split("x"))
Rq = args.rows or R
eff = 0.5 if args.double_buffer else 1.0
Q = K = V = O = N * d

# ---------------- unfused (per head)
v_fits = V <= args.weight_buf * eff
v_passes = 1 if v_fits else math.ceil(N / R)
unf_ss = Q + K + 2 * N * N + V * v_passes + O          # what SCALE-Sim counts
unf_real = unf_ss + 2 * N * N                           # plus a separate softmax pass over DRAM

# ---------------- fused (per head)
score_tile = Rq * N
tile_fits = score_tile <= args.out_buf                  # single tile, reused: no double buffering
kv_resident = (K + V) <= args.in_buf * eff              # K and V of one head kept in the input buffer
kv_passes = 1 if kv_resident else math.ceil(N / Rq)
fused = Q + O + (K + V) * kv_passes

# ---------------- pacing per query tile (array formula: folds * (K + R + C - 2))
def cyc(M, Kd, Nn):
    return math.ceil(M / R) * math.ceil(Nn / C) * (Kd + R + C - 2)

arr_tile = cyc(Rq, d, N) + cyc(Rq, N, d)                # scores tile, then P x V
smx_tile = Rq * N * args.softmax_cost / args.lanes
tiles = math.ceil(N / Rq)

print(f"N = {N} tokens, {h} heads x d = {d}, {nb} blocks, array {R}x{C}, query tile {Rq} rows, "
      f"{args.lanes} vector lanes")
print(f"\nPer head (M words)")
print(f"  score matrix N*N                  {N * N / 1e6:8.2f}")
print(f"  unfused, SCALE-Sim view           {unf_ss / 1e6:8.2f}   (V {'fits, read once' if v_fits else f're-read {v_passes}x'})")
print(f"  unfused, with softmax pass        {unf_real / 1e6:8.2f}")
print(f"  fused                             {fused / 1e6:8.2f}   (K+V {'resident' if kv_resident else f're-read {kv_passes}x'};"
      f" score tile {score_tile / 1e3:.0f} k words {'fits' if tile_fits else 'DOES NOT FIT'} the output buffer)")

blk = lambda x: x * h                                   # noqa: E731
print(f"\nPer block, all heads (M words)")
attn_unf = args.scalesim_attn if args.scalesim_attn else blk(unf_ss)
src = "SCALE-Sim" if args.scalesim_attn else "model"
print(f"  attention unfused ({src})        {attn_unf / 1e6:8.1f}")
if args.scalesim_attn:
    print(f"  attention unfused (model)         {blk(unf_ss) / 1e6:8.1f}   (check against SCALE-Sim)")
print(f"  attention unfused + softmax pass  {(attn_unf + blk(2 * N * N)) / 1e6:8.1f}")
print(f"  attention fused                   {blk(fused) / 1e6:8.1f}")

if args.block_words:
    rest = args.block_words - attn_unf
    print(f"\n  block total, SCALE-Sim            {args.block_words / 1e6:8.1f}   (non-attention part {rest / 1e6:.1f})")
    print(f"  block total, fused                {(rest + blk(fused)) / 1e6:8.1f}   "
          f"({args.block_words / (rest + blk(fused)):.2f}x less)")
    enc_unf = nb * args.block_words
    enc_fus = nb * (rest + blk(fused))
    dpt = args.dpt_words or 0
    print(f"\nPer frame (encoder{' + DPT' if dpt else ''}), 1 word = 1 byte")
    print(f"  unfused (SCALE-Sim)               {(enc_unf + dpt) / 1e6:8.0f} MB   {(enc_unf + dpt) * 10 / 1e9:6.2f} GB/s at 10 fps")
    print(f"  unfused + softmax pass            {(enc_unf + dpt + nb * blk(2 * N * N)) / 1e6:8.0f} MB   "
          f"{(enc_unf + dpt + nb * blk(2 * N * N)) * 10 / 1e9:6.2f} GB/s")
    print(f"  fused                             {(enc_fus + dpt) / 1e6:8.0f} MB   {(enc_fus + dpt) * 10 / 1e9:6.2f} GB/s")

print(f"\nPacing per query tile: array {arr_tile:,} cycles, softmax {smx_tile:,.0f} cycles "
      f"-> softmax {'HIDDEN behind the array' if smx_tile <= arr_tile else 'is the bottleneck'}")
lanes_needed = math.ceil(Rq * N * args.softmax_cost / arr_tile)
print(f"  lanes needed to hide softmax: {lanes_needed}")
stall = max(0.0, smx_tile - arr_tile) * tiles * h * nb
print(f"  frame: {tiles} tiles x {h} heads x {nb} blocks; softmax stall {stall / 1e6:.2f} M cycles "
      f"(0 if hidden); all softmax {Rq * N * args.softmax_cost / args.lanes * tiles * h * nb / 1e6:.2f} M cycles if not overlapped")
two = 2 * score_tile
spare_in = args.in_buf - (K + V)          # K and V resident, rest of the input buffer is free
where = ("output buffer" if two <= args.out_buf else
         "input buffer (beside K and V)" if two <= spare_in else "NOWHERE: overlap not possible")
print(f"  overlap needs 2 score tiles on chip ({two / 1e3:.0f} k words): fit in the {where}")
print(f"\nOn-chip need (fused, per head): K+V {2 * N * d / 1e3:.0f} k words in the input buffer "
      f"(limit {args.in_buf * eff / 1e3:.0f} k), score tile {score_tile / 1e3:.0f} k in the output buffer "
      f"(limit {args.out_buf / 1e3:.0f} k)")
