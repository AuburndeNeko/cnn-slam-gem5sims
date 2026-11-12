"""
CNN monocular depth front-end.

Default network: Depth Anything V2 (ViT-S) fine-tuned for *metric* outdoor
depth. Metric output makes monocular SLAM scale-observable: every tracked pixel
gets a depth in metres, feeding both PnP VO and the (u, v, d) landmark
measurements of the EKF/UKF. Depth maps are cached as float16 .npy so the CNN
runs once per sequence; the gem5 back-end never runs the network.
"""
import os
import time
import numpy as np

DEFAULT_MODEL = 'depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf'


class CnnDepth:
    def __init__(self, model_id=DEFAULT_MODEL, input_hw=(518, 1708), cache_dir=None,
                 threads=None):
        self.model_id = model_id
        self.input_hw = input_hw          # must be multiples of 14 for ViT patches
        self.cache_dir = cache_dir
        self._model = None
        self.threads = threads
        self.infer_times = []
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def _lazy_init(self):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        if self.threads:
            torch.set_num_threads(self.threads)
        self._torch = torch
        self._proc = AutoImageProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForDepthEstimation.from_pretrained(self.model_id).eval()

    def __call__(self, bgr, key=None):
        if self.cache_dir and key is not None:
            path = os.path.join(self.cache_dir, f'{key}.npy')
            if os.path.exists(path):
                return np.load(path).astype(np.float32)
        if self._model is None:
            self._lazy_init()
        torch = self._torch
        H, W = bgr.shape[:2]
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        inputs = self._proc(images=rgb, return_tensors='pt',
                            size={'height': self.input_hw[0], 'width': self.input_hw[1]},
                            keep_aspect_ratio=False)
        t0 = time.time()
        with torch.no_grad():
            pred = self._model(**inputs).predicted_depth
        self.infer_times.append(time.time() - t0)
        depth = torch.nn.functional.interpolate(
            pred[:, None], size=(H, W), mode='bilinear', align_corners=False)[0, 0]
        depth = depth.numpy().astype(np.float32)
        if self.cache_dir and key is not None:
            np.save(path, depth.astype(np.float16))
        return depth


def depth_metrics(pred, gt, max_depth=80.0):
    m = (gt > 1e-3) & (gt < max_depth) & (pred > 1e-3)
    p, g = pred[m], gt[m]
    thresh = np.maximum(p / g, g / p)
    return {
        'abs_rel': float(np.mean(np.abs(p - g) / g)),
        'rmse': float(np.sqrt(np.mean((p - g) ** 2))),
        'delta1': float(np.mean(thresh < 1.25)),
        'median_ratio': float(np.median(g / p)),   # >1 => CNN underestimates
        'n': int(m.sum()),
    }