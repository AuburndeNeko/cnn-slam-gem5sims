"""Write a SCALE-Sim *conv* topology CSV for the convolutional parts of
Depth Anything V2 (ViT-S): the patch embedding and the DPT neck + head.

Shapes come from forward hooks on the real Hugging Face model, so nothing is
typed in by hand. The transformer blocks (nn.Linear) are NOT listed here; they
are already modelled as GEMMs in accel/vits_block_*.csv.

Place in CNN_SLAM/accel/. Run from CNN_SLAM with the CNN venv active:
  python accel/make_dpt_topology.py --height 266 --width 854  --array 32 \
      --out accel/vits_dpt_266x854.csv
  python accel/make_dpt_topology.py --height 518 --width 1708 --array 64 \
      --out accel/vits_dpt_518x1708.csv

Notes
- IFMAP size is written as (out-1)*stride + filter (padded and trimmed), so
  SCALE-Sim's output size equals PyTorch's.
- Transposed convolutions with kernel == stride (the DPT "reassemble" upsamplers)
  do not overlap, so each one is exactly a GEMM with K = Cin and Cout*k*k outputs
  per input pixel. They are written as a 1x1 conv with Cout*k*k filters.
- Not matmuls, so not listed: bilinear upsampling, ReLU, the final sigmoid.
"""
import argparse

import torch
import torch.nn as nn
from transformers import AutoModelForDepthEstimation

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf")
ap.add_argument("--height", type=int, default=518)   # multiples of 14
ap.add_argument("--width", type=int, default=1708)
ap.add_argument("--out", required=True)
ap.add_argument("--array", type=int, default=32, help="systolic array size N (for the notes)")
args = ap.parse_args()

model = AutoModelForDepthEstimation.from_pretrained(args.model).eval()
rows = []  # filled in execution order


def make_hook(qual):
    def hook(m, inp, out):
        cin = inp[0].shape[1]
        hi, wi = inp[0].shape[2], inp[0].shape[3]
        ho, wo = out.shape[2], out.shape[3]
        kh, kw = m.kernel_size
        sh, sw = m.stride
        tag = qual.replace(".", "_")
        if isinstance(m, nn.ConvTranspose2d):
            if (kh, kw) != (sh, sw) or tuple(m.padding) != (0, 0):
                print(f"WARNING: {qual}: transposed conv with kernel != stride, skipped")
                return
            rows.append(dict(name=f"{len(rows):02d}_tconv_{tag}", ih=hi, iw=wi, kh=1, kw=1,
                             c=cin, nf=m.out_channels * kh * kw, s=1, M=hi * wi, K=cin))
            return
        assert m.groups == 1 and sh == sw, f"{qual}: grouped / anisotropic conv not handled"
        rows.append(dict(name=f"{len(rows):02d}_{tag}",
                         ih=(ho - 1) * sh + kh, iw=(wo - 1) * sw + kw, kh=kh, kw=kw,
                         c=cin, nf=m.out_channels, s=sh, M=ho * wo, K=kh * kw * cin))
    return hook


for qual, m in model.named_modules():
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        m.register_forward_hook(make_hook(qual))

with torch.no_grad():
    model(pixel_values=torch.zeros(1, 3, args.height, args.width))

with open(args.out, "w") as f:
    f.write("Layer name, IFMAP Height, IFMAP Width, Filter Height, Filter Width, "
            "Channels, Num Filter, Strides,\n")
    for r in rows:
        # trailing comma is intentional (SCALE-Sim drops the last field)
        f.write(f"{r['name']}, {r['ih']}, {r['iw']}, {r['kh']}, {r['kw']}, "
                f"{r['c']}, {r['nf']}, {r['s']},\n")

kmin = 2 * args.array - 1
total = 0
print(f"{'layer':60s} {'M (out px)':>10s} {'K':>6s} {'N (filt)':>8s} {'MACs (M)':>9s}  note")
for r in rows:
    macs = r["M"] * r["K"] * r["nf"]
    total += macs
    note = []
    if r["K"] < kmin:
        note.append(f"K<{kmin}: RTL pads tiles")
    if r["nf"] < args.array:
        note.append(f"only {r['nf']} of {args.array} columns used")
    print(f"{r['name']:60s} {r['M']:10d} {r['K']:6d} {r['nf']:8d} {macs / 1e6:9.1f}  {'; '.join(note)}")
print(f"\n{len(rows)} conv layers, {total / 1e9:.2f} GMAC per frame "
      f"at {args.width}x{args.height}  ->  {args.out}")
