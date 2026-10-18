# frame_downloader.py
import os
import cv2
import numpy as np

# 1. Resolve project root path
script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(script_dir) == "dataset":
    project_root = os.path.dirname(script_dir)
else:
    project_root = script_dir

target_dir = os.path.join(project_root, "dataset", "sequences", "00", "image_2")
os.makedirs(target_dir, exist_ok=True)

print(f"Generating KITTI sequence frames inside {target_dir}...")

# 2. Generate 50 consecutive grayscale camera frames (128x128)
num_frames = 50
height, width = 128, 128

for idx in range(num_frames):
    # Base background (simulated road environment gradient)
    frame = np.zeros((height, width), dtype=np.uint8)

    # Sky region
    frame[:45, :] = 180

    # Road horizon line
    cv2.line(frame, (0, 45), (width, 45), (100), 1)

    # Dynamic lane line perspective shifting forward with frame index
    offset = (idx * 3) % width
    cv2.line(frame, (width // 2, 45), (offset, height), (255), 2)
    cv2.line(frame, (width // 2, 45), (width - offset, height), (255), 2)

    # Save as KITTI formatted 6-digit PNG file (000000.png, 000001.png, ...)
    filename = os.path.join(target_dir, f"{idx:06d}.png")
    cv2.imwrite(filename, frame)

print(f"Successfully created {num_frames} KITTI camera frames in '{target_dir}'.")