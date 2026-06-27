import logging
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

from dataset import MaskDataset
from nets.DWNetV2_unet import DWNetV2_unet
from nets.SegformerB5 import SegformerB5

np.random.seed(1)
torch.backends.cudnn.deterministic = True
torch.manual_seed(1)


N_CV = 5
RANDOM_STATE = 1
EXPERIMENT = "train_unet"
Test_OUT_DIR = "outputs/UNET_224_weights_100000_days"


RESULT_DIR = r"F:\AAA_Project_300ions_CNNDetect\Results"
os.makedirs(RESULT_DIR, exist_ok=True)
ION_COUNT_FILE = os.path.join(RESULT_DIR, "ion_counts.txt")


def get_data_loaders(image_files):
    val_transform = Compose(
        [
            ToTensor(),
        ]
    )

    val_loader = DataLoader(
        MaskDataset(image_files, val_transform),
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
    )

    return val_loader


def analyze_ions(binary_image):

    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    centers = []
    for contour in contours:

        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append((cX, cY))
        else:
            x, y, w, h = cv2.boundingRect(contour)

            cX = x + w // 2
            cY = y + h // 2
            centers.append((cX, cY))

    ion_count = len(contours)
    return ion_count, centers


def save_centers(image_name, centers):

    base_name = os.path.splitext(image_name)[0]
    center_file = os.path.join(RESULT_DIR, f"{base_name}_centers.txt")

    with open(center_file, "w") as f:
        for idx, (x, y) in enumerate(centers, 1):
            f.write(f"ion {idx}: ({x}, {y})\n")


def evaluate(image_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的设备: {device}")

    image_files = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith(".png")
    ]

    if not image_files:
        print(f"在目录 {image_dir} 中未找到PNG图片")
        return

    print(f"找到 {len(image_files)} 张PNG图片，开始处理...")

    data_loader = get_data_loaders(image_files)

    ckpt_path = r"F:\AAA_Project_300ions_CNNDetect\else\train_unet_0\0-best.pth"
    if "segformer" in ckpt_path.lower():
        model = SegformerB5(pre_trained=None, mode="eval")
    else:
        model = DWNetV2_unet(pre_trained=None, mode="eval")
    try:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    all_ion_counts = []

    with open(ION_COUNT_FILE, "w") as f:
        f.write("Image name\t Number of ions\n")

    with torch.no_grad():
        for batch in data_loader:

            image_path = batch["img_path"][0]
            image_name = os.path.basename(image_path)

            inputs = batch["image"].to(device)
            outputs = model(inputs)

            for i, o in zip(inputs, outputs):

                i = i.cpu().numpy() * 255
                i = i.astype(np.uint8).squeeze(0)

                o = o.cpu().numpy()
                o = o.astype(np.uint8).squeeze(0)

                ion_count, centers = analyze_ions(o)
                print(f"检测到的离子个数: {ion_count}")

                with open(ION_COUNT_FILE, "a") as f:
                    f.write(f"{image_name}\t{ion_count}\n")

                save_centers(image_name, centers)

                all_ion_counts.append(ion_count)

    if all_ion_counts:
        average_ions = sum(all_ion_counts) / len(all_ion_counts)
        print(f"\n所有图片的平均离子数量: {average_ions:.2f}")

    print(f"\n处理完成！结果保存在 {RESULT_DIR} 目录下")


if __name__ == "__main__":
    start_time = time.time()
    TEST_IMAGE_DIR = r"F:\Slice_5000frame\TEST_1000"
    evaluate(TEST_IMAGE_DIR)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"程序运行总时间：{total_time:.6f} 秒")
