#!/usr/bin/env python3
"""Compare FP32 vs dynamic-INT8 Depth Anything V2 (Linear layers quantized) on a few KITTI frames.

Run from CNN_SLAM with the venv active:
    python tools/int8_depth_check.py
    python tools/int8_depth_check.py --frames 0 38 76 114 152 --out data/int8_check.txt

Only nn.Linear layers (the ViT backbone, most of the compute) are quantized to INT8.
The convolutional depth head stays FP32. Comparison is INT8 vs FP32 output (not vs LiDAR).
"""
import argparse
import glob
import io
import os
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"


def model_size_mb(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 1e6


def infer(model, inputs):
    t0 = time.perf_counter()
    with torch.no_grad():
        depth = model(**inputs).predicted_depth[0].numpy()
    return depth, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images",
                    default="kitti/2011_09_26/2011_09_26_drive_0005_sync/image_02/data")
    ap.add_argument("--frames", type=int, nargs="+", default=[0, 38, 76, 114, 152])
    ap.add_argument("--out", default="data/int8_check.txt")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.images, "*.png")))
    if not files:
        raise SystemExit(f"No PNGs found in {args.images}")

    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    fp32 = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).eval()
    # quantize_dynamic returns a quantized copy; fp32 is left unchanged
    int8 = torch.ao.quantization.quantize_dynamic(fp32, {torch.nn.Linear}, dtype=torch.qint8)

    lines = [f"model size (MB): fp32 {model_size_mb(fp32):.1f}  int8 {model_size_mb(int8):.1f}"]
    print(lines[0])

    rel_all, t32_all, t8_all = [], [], []
    for i in args.frames:
        img = Image.open(files[i]).convert("RGB")
        inputs = proc(images=img, return_tensors="pt")
        if i == args.frames[0]:
            print("model input shape:", tuple(inputs["pixel_values"].shape))

        d32, t32 = infer(fp32, inputs)
        d8, t8 = infer(int8, inputs)

        valid = (d32 > 0.1) & (d8 > 0.1)
        a, b = d8[valid], d32[valid]
        rel = np.abs(a - b) / b
        ratio = np.maximum(a / b, b / a)
        delta1 = float(np.mean(ratio < 1.25))

        line = (f"frame {i:3d}: mean rel diff {rel.mean():.4f}  median {np.median(rel):.4f}  "
                f"delta1 agreement {delta1:.4f}  time fp32 {t32:.2f}s int8 {t8:.2f}s")
        print(line)
        lines.append(line)
        rel_all.append(rel.mean())
        t32_all.append(t32)
        t8_all.append(t8)

    summary = (f"SUMMARY over {len(args.frames)} frames: mean rel diff {np.mean(rel_all):.4f}  "
               f"avg time fp32 {np.mean(t32_all):.2f}s  int8 {np.mean(t8_all):.2f}s  "
               f"speedup {np.mean(t32_all) / np.mean(t8_all):.2f}x")
    print(summary)
    lines.append(summary)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
