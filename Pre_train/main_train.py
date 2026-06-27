import os
import glob
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import IonUnlabeledDataset
from model_denoise import DWNetV2DenoiseUNet


DATA_DIR = r"F:\Slice_5000frame\Data2000"
SAVE_DIR = r"F:\AAA_Project_300ions_CNNDetect\Pre_train\Run_pretrain"

EPOCHS = 100
BATCH_SIZE = 16
LR = 1e-3
NUM_WORKERS = 0
RESIZE_W = 456
RESIZE_H = 88

MAX_MASK_FRACTION = 0.12


L1_WEIGHT = 0.8
SSIM_WEIGHT = 0.2


GAUSS_SIGMA = 0.055
OFFSET_B = 0.045
OFFSET_FRACTION = 0.10
BLUR_SIGMA = 0.8


POISSON_PHOTON_SCALE = float(os.environ.get("POISSON_PHOTON_SCALE", 40.0))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {DEVICE}")


def gaussian_window(
    window_size=11, sigma=1.5, channels=1, device="cpu", dtype=torch.float32
):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    gauss = torch.exp(-(coords**2) / (2 * sigma**2))
    gauss = gauss / gauss.sum()
    kernel_2d = torch.outer(gauss, gauss)
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d.view(1, 1, window_size, window_size).repeat(channels, 1, 1, 1)


def ssim_loss(pred, target, window_size=11, sigma=1.5, c1=0.01**2, c2=0.03**2):
    channels = pred.size(1)
    window = gaussian_window(window_size, sigma, channels, pred.device, pred.dtype)

    mu_x = F.conv2d(pred, window, padding=window_size // 2, groups=channels)
    mu_y = F.conv2d(target, window, padding=window_size // 2, groups=channels)

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = (
        F.conv2d(pred * pred, window, padding=window_size // 2, groups=channels)
        - mu_x_sq
    )
    sigma_y_sq = (
        F.conv2d(target * target, window, padding=window_size // 2, groups=channels)
        - mu_y_sq
    )
    sigma_xy = (
        F.conv2d(pred * target, window, padding=window_size // 2, groups=channels)
        - mu_xy
    )

    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    ssim_map = numerator / (denominator + 1e-8)
    return 1.0 - ssim_map.mean()


def estimate_photon_scale(
    image_paths, n_frames=1000, bright_percentile=99.0, gain=None
):
    if gain is None:
        return POISSON_PHOTON_SCALE

    n = min(n_frames, len(image_paths))
    acc = None
    count = 0
    for p in image_paths[:n]:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        f = img.astype(np.float64) / 255.0
        acc = f if acc is None else acc + f
        count += 1

    if acc is None or count == 0:
        return POISSON_PHOTON_SCALE

    avg = acc / count
    bright_level = float(np.percentile(avg, bright_percentile))
    return bright_level * float(gain)


def add_noise(x):

    noisy = x + torch.randn_like(x) * GAUSS_SIGMA

    noisy = (
        torch.poisson(torch.clamp(noisy, 0.0, 1.0) * POISSON_PHOTON_SCALE)
        / POISSON_PHOTON_SCALE
    )

    offset = torch.empty_like(noisy).uniform_(0.0, OFFSET_B)
    offset_mask = (torch.rand_like(noisy) < OFFSET_FRACTION).to(noisy.dtype)
    noisy = noisy + offset * offset_mask

    noisy_np = noisy.detach().cpu().numpy()
    blurred = []
    for idx in range(noisy_np.shape[0]):
        img = cv2.GaussianBlur(noisy_np[idx, 0], (3, 3), sigmaX=BLUR_SIGMA)
        blurred.append(img[None, ...])
    noisy = torch.from_numpy(np.stack(blurred, axis=0)).to(
        device=x.device, dtype=x.dtype
    )

    batch_size, _, height, width = noisy.shape
    for idx in range(batch_size):
        mask_h = random.randint(
            max(1, int(height * 0.03)), max(1, int(height * MAX_MASK_FRACTION))
        )
        mask_w = random.randint(
            max(1, int(width * 0.03)), max(1, int(width * MAX_MASK_FRACTION))
        )
        top = random.randint(0, max(0, height - mask_h))
        left = random.randint(0, max(0, width - mask_w))
        noisy[idx, :, top : top + mask_h, left : left + mask_w] = 0.0

    noisy = torch.clamp(noisy, 0.0, 1.0)
    return noisy


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.*")))
    assert len(image_paths) > 0, " DATA_DIR 里没有图片！"

    print(f"Found {len(image_paths)} images")
    print(f"Resize target: {RESIZE_W}x{RESIZE_H}")

    dataset = IonUnlabeledDataset(
        image_paths,
        resize=(RESIZE_W, RESIZE_H),
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    model = DWNetV2DenoiseUNet(in_channels=1, out_channels=1).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    criterion_l1 = nn.L1Loss()

    best_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        for batch in loader:
            clean = batch["image"].to(DEVICE)
            noisy = add_noise(clean)
            pred = model(noisy)

            loss_l1 = criterion_l1(pred, clean)
            loss_ssim = ssim_loss(pred, clean)
            loss = L1_WEIGHT * loss_l1 + SSIM_WEIGHT * loss_ssim

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.6f}")

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
            },
            os.path.join(SAVE_DIR, "last.pth"),
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                },
                os.path.join(SAVE_DIR, "best.pth"),
            )
            print("Saved best.pth")

    print("Pretraining finished")


if __name__ == "__main__":
    main()
