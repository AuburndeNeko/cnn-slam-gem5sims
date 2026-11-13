"""Write a SCALE-Sim *conv* topology CSV for Monodepth2 (ResNet-18 + depth decoder).

Layer shapes are read from the real PyTorch model with forward hooks, so nothing
is typed in by hand. No weights are needed (shapes only).

Place in CNN_SLAM/accel/. Run from CNN_SLAM with the CNN venv active:
  python accel/make_cnn_topology.py --repo monodepth2 --height 192 --width 640 \
      --out accel/monodepth2_192x640.csv

Notes
- IFMAP size is written as (out-1)*stride + filter, i.e. already padded and trimmed
  so SCALE-Sim's output size equals PyTorch's exactly (SCALE-Sim has no padding).
- Only convolutions are listed. Max-pool, BatchNorm (folds into the conv at
  inference), ReLU/ELU, upsampling and the final sigmoid are not matmuls.
- The decoder is built with scales=[0]: only the full-resolution disparity head,
  which is all inference needs.
"""
import argparse
import os
import sys

import torch
import torch.nn as nn

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True, help="path to the monodepth2 clone")
ap.add_argument("--height", type=int, default=192)   # must be a multiple of 32
ap.add_argument("--width", type=int, default=640)    # must be a multiple of 32
ap.add_argument("--out", required=True)
ap.add_argument("--array", type=int, default=32, help="systolic array size N (for the K >= 2N-1 check)")
args = ap.parse_args()

sys.path.insert(0, os.path.abspath(args.repo))
import networks  # noqa: E402

enc = networks.ResnetEncoder(18, False).eval()
dec = networks.DepthDecoder(num_ch_enc=enc.num_ch_enc, scales=[0]).eval()

rows = []  # filled in execution order


def make_hook(prefix):
    def hook(m, inp, out):
        cin = inp[0].shape[1]
        kh, kw = m.kernel_size
        sh, sw = m.stride
        ho, wo = out.shape[2], out.shape[3]
        assert m.groups == 1 and sh == sw, "grouped / anisotropic conv not handled"
        name = f"{prefix}{len(rows):02d}_{kh}x{kw}_{cin}to{m.out_channels}"
        rows.append(dict(name=name,
                         ih=(ho - 1) * sh + kh, iw=(wo - 1) * sw + kw,
                         kh=kh, kw=kw, c=cin, nf=m.out_channels, s=sh,
                         M=ho * wo, K=kh * kw * cin))
    return hook


for prefix, net in (("enc", enc), ("dec", dec)):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.register_forward_hook(make_hook(prefix))

with torch.no_grad():
    dec(enc(torch.zeros(1, 3, args.height, args.width)))

with open(args.out, "w") as f:
    f.write("Layer name, IFMAP Height, IFMAP Width, Filter Height, Filter Width, "
            "Channels, Num Filter, Strides,\n")
    for r in rows:
        # trailing comma is intentional (SCALE-Sim drops the last field)
        f.write(f"{r['name']}, {r['ih']}, {r['iw']}, {r['kh']}, {r['kw']}, "
                f"{r['c']}, {r['nf']}, {r['s']},\n")

# Summary: each conv seen as a GEMM (M = output pixels, K = fh*fw*cin, N = filters)
kmin = 2 * args.array - 1
total = 0
print(f"{'layer':28s} {'M (out px)':>10s} {'K':>6s} {'N (filt)':>8s} {'MACs (M)':>9s}  note")
for r in rows:
    macs = r["M"] * r["K"] * r["nf"]
    total += macs
    note = []
    if r["K"] < kmin:
        note.append(f"K<{kmin}: RTL pads tiles")
    if r["nf"] < args.array:
        note.append(f"only {r['nf']} of {args.array} columns used")
    print(f"{r['name']:28s} {r['M']:10d} {r['K']:6d} {r['nf']:8d} {macs / 1e6:9.1f}  {'; '.join(note)}")
print(f"\n{len(rows)} conv layers, {total / 1e9:.2f} GMAC per frame "
      f"at {args.width}x{args.height}  ->  {args.out}")
