from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy.io import loadmat


SUPPORTED_SUFFIXES = {".npy", ".mat"}
MAT_KEY_PRIORITY = ("bright", "dark", "data", "image", "images", "img")


def normalize_to_uint8(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame, dtype=np.float32)
    min_val = float(np.min(frame))
    max_val = float(np.max(frame))

    if max_val == min_val:
        if min_val == 0:
            return np.zeros(frame.shape, dtype=np.uint8)
        return np.full(frame.shape, 255, dtype=np.uint8)

    normalized = (frame - min_val) / (max_val - min_val)
    return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)


def iter_candidate_arrays_from_mat(file_path: Path) -> Iterator[Tuple[str, np.ndarray]]:
    mat_data = loadmat(file_path)

    yielded = set()
    for key in MAT_KEY_PRIORITY:
        if key in mat_data and isinstance(mat_data[key], np.ndarray):
            yielded.add(key)
            yield key, mat_data[key]

    numeric_candidates: List[Tuple[str, np.ndarray]] = []
    for key, value in mat_data.items():
        if key.startswith("__") or key in yielded:
            continue
        if (
            isinstance(value, np.ndarray)
            and value.ndim >= 2
            and np.issubdtype(value.dtype, np.number)
        ):
            numeric_candidates.append((key, value))

    numeric_candidates.sort(key=lambda item: item[1].size, reverse=True)
    for item in numeric_candidates:
        yield item


def choose_primary_mat_array(file_path: Path) -> Tuple[str, np.ndarray]:
    for key, value in iter_candidate_arrays_from_mat(file_path):
        return key, value
    raise ValueError(f"No numeric image-like array found in MAT file: {file_path}")


def load_payloads(file_path: Path) -> List[Tuple[str, np.ndarray]]:
    suffix = file_path.suffix.lower()
    if suffix == ".npy":
        array = np.load(file_path, allow_pickle=False)
        return [("npy", array)]
    if suffix == ".mat":
        key, array = choose_primary_mat_array(file_path)
        return [(key, array)]
    raise ValueError(f"Unsupported file type: {file_path}")


def iter_frames(array: np.ndarray) -> Iterator[Tuple[int, np.ndarray]]:
    array = np.asarray(array)

    if array.ndim == 2:
        yield 0, array
        return

    if array.ndim == 3:
        for idx in range(array.shape[0]):
            yield idx, array[idx, :, :]
        return

    raise ValueError(f"Only 2D or 3D arrays are supported, got shape {array.shape}")


def collect_source_files(input_dir: Path) -> List[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def convert_dataset(input_dir: Path, output_dir: Path, start_index: int = 0) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"

    source_files = collect_source_files(input_dir)
    if not source_files:
        raise FileNotFoundError(f"No .npy or .mat files found under {input_dir}")

    next_index = start_index
    total_saved = 0

    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.writer(manifest_file)
        writer.writerow(
            ["png_name", "source_file", "payload_key", "frame_index", "shape"]
        )

        for source_path in source_files:
            payloads = load_payloads(source_path)
            for payload_key, array in payloads:
                for frame_index, frame in iter_frames(array):
                    png_name = f"{next_index:06d}.png"
                    png_path = output_dir / png_name

                    image = Image.fromarray(normalize_to_uint8(frame))
                    image.save(png_path)

                    writer.writerow(
                        [
                            png_name,
                            str(source_path),
                            payload_key,
                            frame_index,
                            "x".join(str(dim) for dim in frame.shape),
                        ]
                    )

                    next_index += 1
                    total_saved += 1

            print(f"Converted: {source_path}")

    print(f"Finished. Saved {total_saved} PNG files to {output_dir}")
    print(f"Manifest written to {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert all .npy/.mat slices under a folder to 8-bit PNG images."
    )
    parser.add_argument(
        "--input_dir",
        default=r"F:\Slice_data",
        help="Source directory containing .npy/.mat files",
    )
    parser.add_argument(
        "--output_dir",
        default=r"F:\Slice_data\Pretrain_Dataset",
        help="Directory to save PNG slices",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Starting numeric index for output PNG names",
    )
    args = parser.parse_args()

    convert_dataset(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        start_index=args.start_index,
    )


if __name__ == "__main__":
    main()
