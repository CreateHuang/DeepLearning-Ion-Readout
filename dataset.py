import os
from glob import glob
import random

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from config import IMG_DIR


MAX_IONS = 300
MASK_THRESHOLD = 127
MIN_COMPONENT_AREA = 1


def _mask_to_img(mask_file):
    mask_dir, mask_filename = os.path.split(mask_file)
    img_dir = mask_dir.replace("masks", "images")
    img_file = os.path.splitext(mask_filename)[0] + ".png"
    return os.path.join(img_dir, img_file)


def _img_to_mask(img_file):
    img_dir, img_filename = os.path.split(img_file)
    mask_dir = img_dir.replace("images", "masks")
    mask_file = os.path.splitext(img_filename)[0] + ".png"
    return os.path.join(mask_dir, mask_file)


def get_img_files():
    mask_files = sorted(glob(os.path.join(IMG_DIR, "masks", "*.png")))
    img_files = [_mask_to_img(f) for f in mask_files]

    valid_img_files = []
    for img_file in img_files:
        mask_file = _img_to_mask(img_file)
        if os.path.exists(img_file) and os.path.exists(mask_file):
            valid_img_files.append(img_file)

    return np.array(valid_img_files)


def extract_centers_from_mask(mask_np, max_ions=MAX_IONS, min_area=MIN_COMPONENT_AREA):
    binary = (mask_np > MASK_THRESHOLD).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    centers_list = []
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        cx, cy = centroids[label_id]
        centers_list.append((float(cx), float(cy)))

    centers_list = sorted(centers_list, key=lambda p: p[0])

    if len(centers_list) > max_ions:
        centers_list = centers_list[:max_ions]

    centers = np.zeros((max_ions, 2), dtype=np.float32)
    valid = np.zeros((max_ions,), dtype=np.float32)

    for i, (cx, cy) in enumerate(centers_list):
        centers[i, 0] = cx
        centers[i, 1] = cy
        valid[i] = 1.0

    return centers, valid


def compute_curriculum_difficulty(img_files, max_ions=MAX_IONS):
    difficulties = []
    for img_file in img_files:
        mask = cv2.imread(_img_to_mask(img_file), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            difficulties.append(0.0)
            continue
        binary = (mask > MASK_THRESHOLD).astype(np.uint8)
        num_labels, _ = cv2.connectedComponents(binary, connectivity=8)
        n_bright = max(0, num_labels - 1)
        n_dark = max(0, max_ions - n_bright)
        difficulties.append(float(n_bright * n_dark) / float(max_ions))
    return np.asarray(difficulties, dtype=np.float32)


def apply_physics_noise(
    img, alpha, rng, sigma=0.055, b=0.045, frac=0.10, blur_sigma=0.8
):
    if alpha <= 0.0:
        return img
    out = img + rng.normal(0.0, sigma * alpha, img.shape).astype(np.float32)
    offset_mask = (rng.random(img.shape) < frac).astype(np.float32)
    out = out + rng.uniform(0.0, b * alpha, img.shape).astype(np.float32) * offset_mask
    if blur_sigma * alpha > 1e-3:
        out = cv2.GaussianBlur(out, (3, 3), sigmaX=float(blur_sigma * alpha))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


class MaskDataset(Dataset):
    def __init__(
        self,
        img_files,
        transform=None,
        mask_transform=None,
        max_ions=MAX_IONS,
        noise_strength=0.0,
        noise_seed=1,
    ):
        self.img_files = list(img_files)
        self.mask_files = [_img_to_mask(f) for f in self.img_files]
        self.transform = transform if transform is not None else ToTensor()
        self.mask_transform = (
            mask_transform if mask_transform is not None else ToTensor()
        )
        self.max_ions = max_ions

        self.noise_strength = noise_strength
        self._rng = np.random.default_rng(noise_seed)

    def set_noise_strength(self, alpha):
        self.noise_strength = float(alpha)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        mask_path = self.mask_files[idx]

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError(f"Failed to read image: {img_path}")
        if mask is None:
            raise ValueError(f"Failed to read mask: {mask_path}")

        centers, centers_valid = extract_centers_from_mask(mask, max_ions=self.max_ions)

        img_f = img.astype(np.float32) / 255.0
        if self.noise_strength > 0.0:
            img_f = apply_physics_noise(img_f, self.noise_strength, self._rng)
        img_tensor = torch.from_numpy(img_f[None, ...]).float()

        mask_f = mask.astype(np.float32) / 255.0
        mask_tensor = (torch.from_numpy(mask_f[None, ...]) > 0.5).float()

        sample = {
            "image": img_tensor,
            "mask": mask_tensor,
            "centers_gt": torch.from_numpy(centers),
            "centers_valid": torch.from_numpy(centers_valid),
            "img_path": img_path,
        }
        return sample


if __name__ == "__main__":
    pass
