"""Monodepth2 depth wrapper (ResNet-18 encoder + depth decoder).

Returns METRIC depth in metres for stereo-trained weights (stereo or mono+stereo),
using the fixed 5.4 scale factor from the authors' evaluate_depth.py.

Place in CNN_SLAM/frontend/. Smoke test (from CNN_SLAM, venv active):
  python frontend/monodepth2_depth.py \
      --repo monodepth2 --weights monodepth2/models/mono+stereo_640x192 \
      --image kitti/2011_09_26/2011_09_26_drive_0005_sync/image_02/data/0000000000.png
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

STEREO_SCALE_FACTOR = 5.4          # monodepth2: network units -> metres (stereo baseline)
MIN_DEPTH, MAX_DEPTH = 0.1, 100.0  # monodepth2 disp_to_depth() defaults


class Monodepth2Depth:
    def __init__(self, repo, weights, device=None):
        sys.path.insert(0, os.path.abspath(repo))
        import networks  # from the monodepth2 clone

        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        enc_sd = torch.load(os.path.join(weights, "encoder.pth"), map_location="cpu")
        self.h, self.w = int(enc_sd["height"]), int(enc_sd["width"])  # e.g. 192, 640

        self.encoder = networks.ResnetEncoder(18, False)
        own = self.encoder.state_dict()
        self.encoder.load_state_dict({k: v for k, v in enc_sd.items() if k in own})

        self.decoder = networks.DepthDecoder(num_ch_enc=self.encoder.num_ch_enc, scales=range(4))
        self.decoder.load_state_dict(
            torch.load(os.path.join(weights, "depth.pth"), map_location="cpu"))

        self.encoder.to(self.dev).eval()
        self.decoder.to(self.dev).eval()

    @torch.no_grad()
    def predict(self, rgb):
        """rgb: HxWx3 uint8 (RGB order). Returns HxW float32 depth in metres."""
        H, W = rgb.shape[:2]
        x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div(255.0)
        x = x.unsqueeze(0).to(self.dev)
        # ImageNet-style normalisation happens inside ResnetEncoder, not here.
        x = F.interpolate(x, (self.h, self.w), mode="bilinear",
                          align_corners=False, antialias=True)

        disp = self.decoder(self.encoder(x))[("disp", 0)]            # sigmoid output, 0..1
        disp = F.interpolate(disp, (H, W), mode="bilinear", align_corners=False)

        scaled_disp = 1.0 / MAX_DEPTH + (1.0 / MIN_DEPTH - 1.0 / MAX_DEPTH) * disp
        depth = STEREO_SCALE_FACTOR / scaled_disp
        return depth[0, 0].cpu().numpy().astype(np.float32)


if __name__ == "__main__":
    from PIL import Image

    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="path to the monodepth2 clone")
    ap.add_argument("--weights", required=True, help="folder with encoder.pth and depth.pth")
    ap.add_argument("--image", required=True)
    ap.add_argument("--save", default=None, help="optional .npy output path")
    a = ap.parse_args()

    net = Monodepth2Depth(a.repo, a.weights)
    rgb = np.array(Image.open(a.image).convert("RGB"))
    d = net.predict(rgb)
    print(f"network input {net.w}x{net.h}, image {rgb.shape[1]}x{rgb.shape[0]}, device {net.dev}")
    print(f"depth [m]: min {d.min():.2f}  median {np.median(d):.2f}  max {d.max():.2f}")
    if a.save:
        np.save(a.save, d)
