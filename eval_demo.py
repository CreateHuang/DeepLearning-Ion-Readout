import logging
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor

from dataset import MaskDataset, get_img_files, get_img_files_eval
from nets.DWNetV2_unet import DWNetV2_unet
from nets.SegformerB5 import SegformerB5

np.random.seed(1)
torch.backends.cudnn.deterministic = True
torch.manual_seed(1)


N_CV = 5

RANDOM_STATE = 1

EXPERIMENT = "train_unet"
Test_OUT_DIR = "outputs/UNET_224_weights_100000_days"


def get_data_loaders(val_files):
    val_transform = Compose(
        [
            ToTensor(),
        ]
    )

    val_loader = DataLoader(
        MaskDataset(val_files, val_transform),
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )

    return val_loader


def analyze_ions(binary_image):

    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    ion_count = len(contours)

    centers = []
    for contour in contours:

        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append((cX, cY))

    return ion_count, centers


def evaluate():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的设备: {device}")

    val_files = [r"F:\IONS_DATASET\images\0001.png"]
    data_loader = get_data_loaders(val_files)

    model = DWNetV2_unet(pre_trained=None, mode="eval")
    try:

        model.load_state_dict(
            torch.load(
                r"F:\AAA_Project_300ions_CNNDetect\else\train_unet_1\2-best.pth",
                map_location=device,
            )
        )
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            for i, o in zip(inputs, outputs):

                i = i.cpu().numpy() * 255

                o = o.cpu().numpy()

                print("i shape:", i.shape)

                print("o shape:", o.shape)

                i = i.astype(np.uint8).squeeze(0)

                o = o.astype(np.uint8).squeeze(0)

                ion_count, centers = analyze_ions(o)
                print(f"检测到的离子个数: {ion_count}")
                print(f"检测到的离子个数1: {len(centers)}")
                print(f"离子中心点坐标: {centers}")

                o_color = cv2.cvtColor(o, cv2.COLOR_GRAY2BGR)

                for x, y in centers:
                    cv2.circle(
                        o_color, (x, y), 1, (0, 255, 0), -1, lineType=cv2.LINE_AA
                    )

                h, w = i.shape[:2]

                h, w = i.shape

                fig_w = 5
                fig_h = fig_w * (h / w) * 2

                fig, axes = plt.subplots(2, 1, figsize=(fig_w, fig_h), dpi=300)

                axes[0].imshow(i)
                axes[0].set_title("sCMOS Readout Image", fontsize=6)
                axes[0].axis("off")

                axes[1].imshow(o_color)
                axes[1].set_title(
                    f"Inference result : The number of quantum bits (ions) in the |0⟩ state is {ion_count}",
                    fontsize=6,
                )
                axes[1].axis("off")

                plt.tight_layout()

                save_path = os.path.join(Test_OUT_DIR, "result_vertical.png")
                plt.savefig(save_path, dpi=300, bbox_inches="tight")

                plt.show()


if __name__ == "__main__":
    if not os.path.exists(Test_OUT_DIR):
        os.makedirs(Test_OUT_DIR)

    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)
    if not logger.hasHandlers():
        logger.addHandler(logging.FileHandler(filename="outputs/evaluation.log"))

    evaluate()
