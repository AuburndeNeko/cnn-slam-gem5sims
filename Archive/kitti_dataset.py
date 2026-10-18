# kitti_dataset.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pandas as pd


class AbhishekKITTIDataset(Dataset):
    def __init__(self, sequence="00", transform=None, dataset_root="dataset"):
        self.transform = transform

        # Primary image search path: image_2 -> fallback: image_0
        self.seq_dir = os.path.join(dataset_root, "sequences", sequence, "image_2")
        if not os.path.exists(self.seq_dir):
            self.seq_dir = os.path.join(dataset_root, "sequences", sequence, "image_0")

        self.pose_file = os.path.join(dataset_root, "poses", f"{sequence}.txt")

        # Load sorted frame paths
        if os.path.exists(self.seq_dir):
            self.image_files = sorted([
                os.path.join(self.seq_dir, f) for f in os.listdir(self.seq_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
        else:
            self.image_files = []

        if len(self.image_files) < 2:
            raise ValueError(
                f"Found {len(self.image_files)} images in '{self.seq_dir}'. "
                f"At least 2 frames are required for optical flow training."
            )

        # Load ground truth poses
        if not os.path.exists(self.pose_file):
            raise FileNotFoundError(f"Pose file not found at: {self.pose_file}")

        self.gt_poses = pd.read_csv(self.pose_file, delimiter=' ', header=None).values

    def __len__(self):
        # Bound valid index range by images and poses count
        valid_samples = min(len(self.image_files), len(self.gt_poses))
        return max(0, valid_samples - 1)

    def __getitem__(self, idx):
        # Read consecutive image frames (t and t+1)
        img1 = Image.open(self.image_files[idx]).convert('L')
        img2 = Image.open(self.image_files[idx + 1]).convert('L')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        # Concatenate into 2-channel tensor [2, H, W]
        x_input = torch.cat([img1, img2], dim=0)

        # Extract pose matrices and calculate ground truth delta Z (translation)
        p1 = self.gt_poses[idx].reshape(3, 4)
        p2 = self.gt_poses[idx + 1].reshape(3, 4)

        delta_z = np.sqrt(np.sum((p2[:, 3] - p1[:, 3]) ** 2))
        gt_delta = torch.tensor([delta_z], dtype=torch.float32)

        return x_input, gt_delta