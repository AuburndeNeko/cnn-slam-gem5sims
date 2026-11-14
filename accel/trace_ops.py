#!/usr/bin/env python3
"""Count every operation a depth network executes, split into matmul and vector work.

Runs the real PyTorch model once under a TorchDispatchMode, so every aten op is
seen (modules AND functional calls such as F.interpolate, torch.cat, softmax).
For each op it records the element count and whether its input came straight
from a conv/matmul (through BatchNorm/activations/scaling), which decides whether
the op could be fused into the systolic array's output readout path.

Writes a JSON file read by accel/vector_model.py. No weights are needed (shapes only).

Place in CNN_SLAM/accel/. Run from CNN_SLAM with the CNN venv active:
  python accel/trace_ops.py md2  --repo monodepth2 --height 192 --width 640  --out accel/ops_md2_192x640.json
  python accel/trace_ops.py dav2 --height 266 --width 854  --out accel/ops_dav2_266x854.json
  python accel/trace_ops.py dav2 --height 518 --width 1708 --out accel/ops_dav2_518x1708.json

Categories (elements = output elements unless noted):
  matmul      conv / mm / bmm / addmm. MACs are counted (validation against the topology CSVs).
  fold        BatchNorm (eval), per-channel / scalar mul, add, sub, div: folds into the conv weights
              or the requantisation step at no extra cost when it follows a matmul.
  act         ReLU / ELU / sigmoid / GELU / tanh / exp (elementwise, LUT-type)
  residual    add/sub/mul of two full-size tensors (needs a second operand stream)
  softmax     row softmax (row reduction: max, exp + sum, scale)
  layernorm   row LayerNorm (row reduction: mean, variance, normalise + affine)
  pool        max pooling (elements = outputs; window size recorded)
  upsample_bilinear   bilinear interpolation (elements = outputs)
  move        nearest upsample, concat, padding, copies: data movement only, no arithmetic
  layout      views / permutes / reshapes: free
  const       position-embedding resize (bicubic): depends only on the input size, precomputed offline
Any op not in the tables is reported as UNKNOWN so nothing is silently dropped.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import torch
from torch.utils._python_dispatch import TorchDispatchMode

MATMUL = {"convolution", "mm", "bmm", "addmm", "baddbmm"}
LAYOUT = {"view", "_unsafe_view", "permute", "transpose", "t", "expand", "reshape", "unsqueeze",
          "squeeze", "select", "slice", "split", "split_with_sizes", "alias", "detach", "unbind",
          "_reshape_alias", "as_strided", "empty", "zeros", "ones", "new_empty", "lift_fresh",
          "_to_copy", "unflatten", "flatten", "chunk", "narrow", "lift_fresh_copy", "full", "arange",
          "scalar_tensor", "_local_scalar_dense", "sym_size", "sym_stride", "is_same_size"}
MOVE = {"cat", "reflection_pad2d", "constant_pad_nd", "upsample_nearest2d", "clone", "copy_",
        "contiguous", "_unsafe_index", "pad", "fill_", "zero_"}
FOLD_UNARY_PARAM = {"_native_batch_norm_legit_no_training", "native_batch_norm", "batch_norm",
                    "_native_batch_norm_legit"}
ACT = {"relu", "relu_", "elu", "elu_", "sigmoid", "sigmoid_", "gelu", "tanh", "exp", "hardtanh",
       "clamp", "clamp_min"}
BINARY = {"add", "add_", "sub", "sub_", "mul", "mul_", "div", "div_", "rsub"}
SOFTMAX = {"_softmax", "softmax"}
LAYERNORM = {"native_layer_norm"}
POOL = {"max_pool2d_with_indices", "max_pool2d", "avg_pool2d"}
BILINEAR = {"upsample_bilinear2d", "_upsample_bilinear2d_aa"}
CONST = {"upsample_bicubic2d"}     # DINOv2 position-embedding resize: depends only on resolution


def numel(x):
    return x.numel() if isinstance(x, torch.Tensor) else 0


class Tracer(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.rows = []                      # one dict per counted op
        self.tag = {}                       # data_ptr -> True if produced by a matmul chain
        self.unknown = defaultdict(int)

    def _from_mm(self, t):
        return isinstance(t, torch.Tensor) and t.numel() > 0 and self.tag.get(t.data_ptr(), False)

    def _set(self, out, from_mm):
        outs = out if isinstance(out, (tuple, list)) else [out]
        for o in outs:
            if isinstance(o, torch.Tensor) and o.numel() > 0:
                self.tag[o.data_ptr()] = from_mm

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = func.overloadpacket.__name__
        tensors_in = [a for a in args if isinstance(a, torch.Tensor)]
        first = tensors_in[0] if tensors_in else None
        out = func(*args, **kwargs)
        o0 = out[0] if isinstance(out, (tuple, list)) else out
        n_out = numel(o0)
        row = None

        if name in MATMUL:
            if name == "convolution":
                w, transposed = args[1], bool(args[6]) if len(args) > 6 else False
                if transposed:      # weight is (Cin, Cout/groups, kh, kw); kernel == stride here,
                    macs = n_out * w.shape[0]                    # so each output sees Cin inputs once
                else:               # weight is (Cout, Cin/groups, kh, kw)
                    macs = n_out * (w.shape[1] * w.shape[2] * w.shape[3])
            elif name in ("mm", "addmm"):
                a, b = (args[1], args[2]) if name == "addmm" else (args[0], args[1])
                macs = a.shape[0] * a.shape[1] * b.shape[1]
            else:                                                       # bmm / baddbmm
                a, b = (args[1], args[2]) if name == "baddbmm" else (args[0], args[1])
                macs = a.shape[0] * a.shape[1] * a.shape[2] * b.shape[2]
            row = dict(op=name, cat="matmul", elems=n_out, macs=int(macs), from_mm=False)
            self._set(out, True)
        elif name in LAYOUT:
            # views keep the producer tag of their input (same data)
            self._set(out, self._from_mm(first))
        elif name in MOVE:
            row = dict(op=name, cat="move", elems=n_out, from_mm=self._from_mm(first))
            self._set(out, False)
        elif name in FOLD_UNARY_PARAM:
            f = self._from_mm(first)
            row = dict(op=name, cat="fold", elems=n_out, from_mm=f)
            self._set(out, f)
        elif name in ACT:
            f = self._from_mm(first)
            row = dict(op=name, cat="act", elems=n_out, from_mm=f)
            self._set(out, f)
        elif name in BINARY:
            sizes = sorted((numel(a) if isinstance(a, torch.Tensor) else 1) for a in args[:2])
            full_pair = len(tensors_in) == 2 and sizes[0] == sizes[1] == n_out
            if full_pair:
                # residual add (or elementwise product) of two full tensors
                f = any(self._from_mm(a) for a in tensors_in)
                row = dict(op=name, cat="residual", elems=n_out, from_mm=f)
                self._set(out, False)
            else:
                # one side is a scalar or a broadcast parameter (bias, BN, layer scale, normalisation)
                big = max(tensors_in, key=numel) if tensors_in else None
                f = self._from_mm(big)
                row = dict(op=name, cat="fold", elems=n_out, from_mm=f)
                self._set(out, f)
        elif name in SOFTMAX:
            row = dict(op=name, cat="softmax", elems=n_out, row_len=int(o0.shape[-1]),
                       from_mm=self._from_mm(first))
            self._set(out, False)
        elif name in LAYERNORM:
            row = dict(op=name, cat="layernorm", elems=n_out, row_len=int(o0.shape[-1]),
                       from_mm=self._from_mm(first))
            self._set(out, False)
        elif name in POOL:
            k = args[1] if len(args) > 1 else kwargs.get("kernel_size")
            row = dict(op=name, cat="pool", elems=n_out, window=int(k[0] * k[1]),
                       from_mm=self._from_mm(first))
            self._set(out, False)
        elif name in BILINEAR:
            row = dict(op=name, cat="upsample_bilinear", elems=n_out, from_mm=self._from_mm(first))
            self._set(out, False)
        elif name in CONST:
            row = dict(op=name, cat="const", elems=n_out, from_mm=False)
            self._set(out, False)
        else:
            self.unknown[name] += n_out
            self._set(out, False)

        if row is not None:
            row["shape"] = list(o0.shape) if isinstance(o0, torch.Tensor) else []
            self.rows.append(row)
        return out


def build_md2(args):
    sys.path.insert(0, os.path.abspath(args.repo))
    import networks  # noqa: E402
    enc = networks.ResnetEncoder(18, False).eval()
    dec = networks.DepthDecoder(num_ch_enc=enc.num_ch_enc, scales=[0]).eval()  # inference: scale 0 only
    x = torch.zeros(1, 3, args.height, args.width)
    return (lambda: dec(enc(x))), "Monodepth2 ResNet-18 (scales=[0])"


def build_dav2(args):
    from transformers import AutoModelForDepthEstimation
    model = AutoModelForDepthEstimation.from_pretrained(args.model, attn_implementation="eager").eval()
    x = torch.zeros(1, 3, args.height, args.width)
    return (lambda: model(pixel_values=x)), args.model


ap = argparse.ArgumentParser()
ap.add_argument("net", choices=["md2", "dav2"])
ap.add_argument("--repo", default="monodepth2", help="monodepth2 clone (md2 only)")
ap.add_argument("--model", default="depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf")
ap.add_argument("--height", type=int, required=True)
ap.add_argument("--width", type=int, required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

run, desc = build_md2(args) if args.net == "md2" else build_dav2(args)
tr = Tracer()
with torch.no_grad(), tr:
    run()

# Summary by category (and by op inside each category)
summ = defaultdict(lambda: dict(elems=0, elems_from_mm=0, macs=0, n=0))
byop = defaultdict(lambda: dict(elems=0, n=0))
for r in tr.rows:
    s = summ[r["cat"]]
    s["elems"] += r["elems"]
    s["elems_from_mm"] += r["elems"] if r["from_mm"] else 0
    s["macs"] += r.get("macs", 0)
    s["n"] += 1
    b = byop[(r["cat"], r["op"])]
    b["elems"] += r["elems"]
    b["n"] += 1

print(f"{desc}  input {args.width}x{args.height}")
print(f"{'category':20s} {'ops':>5s} {'M elements':>11s} {'of which after matmul':>22s} {'GMAC':>7s}")
for cat in ("matmul", "fold", "act", "residual", "softmax", "layernorm", "pool",
            "upsample_bilinear", "move", "const"):
    if cat in summ:
        s = summ[cat]
        print(f"{cat:20s} {s['n']:5d} {s['elems'] / 1e6:11.2f} {s['elems_from_mm'] / 1e6:22.2f} "
              f"{s['macs'] / 1e9:7.2f}")
print("\nper op:")
for (cat, op), b in sorted(byop.items()):
    print(f"  {cat:18s} {op:40s} {b['n']:5d} ops {b['elems'] / 1e6:10.2f} M el")
if tr.unknown:
    print("\nUNKNOWN ops (not classified, check!):")
    for k, v in tr.unknown.items():
        print(f"  {k:40s} {v / 1e6:10.2f} M el")

json.dump(dict(net=args.net, desc=desc, height=args.height, width=args.width,
               summary=summ, ops=tr.rows, unknown=tr.unknown), open(args.out, "w"), indent=1)
print(f"\n-> {args.out}")
