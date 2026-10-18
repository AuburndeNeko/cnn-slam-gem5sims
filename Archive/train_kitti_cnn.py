# train_kitti_cnn.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from kitti_dataset import AbhishekKITTIDataset


# Lightweight 2-Frame Depth/Translation Estimation CNN
class SimpleDepthCNN(nn.Module):
    def __init__(self):
        super(SimpleDepthCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1),  # [2, 128, 128] -> [16, 64, 64]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # [32, 32, 32]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # [32, 4, 4]
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        feat = self.conv(x)
        feat = feat.view(feat.size(0), -1)
        return self.fc(feat)


def run_training():
    print("--- Training FP32 CNN Model on KITTI Data ---")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    dataset = AbhishekKITTIDataset(sequence="00", transform=transform)
    print(f"Dataset loaded with {len(dataset)} valid frame pairs.")

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = SimpleDepthCNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5-Epoch Training Loop
    model.train()
    for epoch in range(5):
        total_loss = 0.0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1:02d}/05] - Loss: {avg_loss:.6f}")

    # Single-sample inference check for backend C++ header output
    model.eval()
    sample_x, sample_y = dataset[0]
    sample_x = sample_x.unsqueeze(0)
    with torch.no_grad():
        inferred_delta_z = model(sample_x).item()

    print(f"\nInferred Visual Delta Z: {inferred_delta_z:.6f} meters")

    # Generate static baseline_data.h header for C++ EKF backend
    header_content = f"""#ifndef BASELINE_DATA_H
#define BASELINE_DATA_H

// Auto-generated static model inference output for ARM gem5 simulation
static const float TRAINED_KITTI_DELTA_Z = {inferred_delta_z:.6f}f;

#endif // BASELINE_DATA_H
"""
    with open("baseline_data.h", "w") as f:
        f.write(header_content)

    print("Saved static inference vector to 'baseline_data.h'!")


if __name__ == "__main__":
    run_training()