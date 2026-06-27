import glob
import os
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


VALID_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.npy")


def collect_image_paths(data_dir: str) -> List[str]:
    paths: List[str] = []
    for ext in VALID_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(data_dir, ext)))
    return sorted(paths)


class IonUnlabeledDataset(Dataset):

    def __init__(
        self, image_paths: Sequence[str], resize: Optional[Tuple[int, int]] = None
    ):
        self.image_paths = list(image_paths)
        self.resize = resize

        if len(self.image_paths) == 0:
            raise ValueError("No images found. Please check --data_dir.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_image(self, path: str) -> np.ndarray:
        if path.lower().endswith(".npy"):
            img = np.load(path)
            if img.ndim == 3:
                img = img.squeeze()
            img = img.astype(np.float32)
        else:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Failed to read image: {path}")
            img = img.astype(np.float32)

        img_min = float(img.min())
        img_max = float(img.max())
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img, dtype=np.float32)

        if self.resize is not None:
            target_w, target_h = self.resize
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

        img = np.expand_dims(img, axis=0)
        return img.astype(np.float32)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        img = self._load_image(path)
        return {
            "image": torch.from_numpy(img),
            "path": path,
        }
