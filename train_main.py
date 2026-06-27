import argparse
import logging
import os
import random

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

from curriculum import CurriculumSampler, noise_alpha
from dataset import MaskDataset, compute_curriculum_difficulty, get_img_files
from loss import HybridSegmentationMultiIonLoss
from nets.DWNetV2_unet import DWNetV2_unet
from trainer import Trainer


BATCH_SIZE = 16
INITIAL_LR = 1e-4
RANDOM_STATE = 1


MAX_EPOCHS = 1000
VAL_FRACTION = 0.2

CENTROID_RADIUS = 3


USE_CURRICULUM = True
CURRICULUM_P0 = 0.2
CURRICULUM_WARMUP_FRAC = 0.5
NOISE_WARMUP_FRAC = 0.5
NOISE_ALPHA0 = 0.0

EXPERIMENT = "train_unet_multiion"
OUT_DIR = f"outputs/{EXPERIMENT}"


def set_seed(seed):
    global RANDOM_STATE
    RANDOM_STATE = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_data_loaders(train_files, val_files):
    train_dataset = MaskDataset(
        train_files, noise_strength=0.0, noise_seed=RANDOM_STATE
    )
    val_dataset = MaskDataset(val_files)

    if USE_CURRICULUM:
        difficulties = compute_curriculum_difficulty(train_files)
        train_sampler = CurriculumSampler(
            difficulties,
            max_epochs=MAX_EPOCHS,
            p0=CURRICULUM_P0,
            warmup_frac=CURRICULUM_WARMUP_FRAC,
            seed=RANDOM_STATE,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=train_sampler,
            pin_memory=True,
            num_workers=4,
        )
    else:
        train_sampler = None
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )

    return train_loader, val_loader, train_dataset, train_sampler


def save_best_model(model, df_hist):
    if df_hist["val_loss"].tail(1).iloc[0] <= df_hist["val_loss"].min():
        torch.save(model.state_dict(), f"{OUT_DIR}/best.pth")


def write_on_board(writer, df_hist):
    row = df_hist.tail(1).iloc[0]

    writer.add_scalars(
        f"{EXPERIMENT}/loss",
        {
            "train": row.train_loss,
            "val": row.val_loss,
        },
        row.epoch,
    )

    writer.add_scalars(
        f"{EXPERIMENT}/bce",
        {
            "train": row.train_bce,
            "val": row.val_bce,
        },
        row.epoch,
    )

    writer.add_scalars(
        f"{EXPERIMENT}/dice",
        {
            "train": row.train_dice,
            "val": row.val_dice,
        },
        row.epoch,
    )

    writer.add_scalars(
        f"{EXPERIMENT}/centroid",
        {
            "train": row.train_centroid,
            "val": row.val_centroid,
        },
        row.epoch,
    )


def log_hist(df_hist):
    last = df_hist.tail(1)
    best = df_hist.sort_values("val_loss").head(1)
    summary = pd.concat((last, best)).reset_index(drop=True)
    summary["name"] = ["Last", "Best"]
    logger.debug(
        summary[
            [
                "name",
                "epoch",
                "train_loss",
                "train_bce",
                "train_dice",
                "train_centroid",
                "val_loss",
                "val_bce",
                "val_dice",
                "val_centroid",
            ]
        ]
    )
    logger.debug("")


def load_pretrained_weights(model, ckpt_path, device, load_mode="full"):
    if not ckpt_path:
        print("No pretrained checkpoint provided.")
        return model

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    model_state = model.state_dict()
    loaded_keys = []

    if load_mode == "full":
        filtered = {}
        for k, v in state_dict.items():
            if k in model_state and model_state[k].shape == v.shape:
                filtered[k] = v
                loaded_keys.append(k)

        incompatible = model.load_state_dict(filtered, strict=False)
        print(f"[Pretrain-FULL] loaded={len(loaded_keys)}")
        print(f"[Pretrain-FULL] missing={len(incompatible.missing_keys)}")
        print(f"[Pretrain-FULL] unexpected={len(incompatible.unexpected_keys)}")

    elif load_mode == "backbone":
        filtered = {}
        for k, v in state_dict.items():
            candidate_keys = [k]

            if k.startswith("backbone."):
                candidate_keys.append(k[len("backbone.") :])

            for ck in candidate_keys:
                if ck in model_state and model_state[ck].shape == v.shape:
                    filtered[ck] = v
                    loaded_keys.append(ck)
                    break

                bk = f"backbone.{ck}"
                if bk in model_state and model_state[bk].shape == v.shape:
                    filtered[bk] = v
                    loaded_keys.append(bk)
                    break

        incompatible = model.load_state_dict(filtered, strict=False)
        print(f"[Pretrain-BACKBONE] loaded={len(loaded_keys)}")
        print(f"[Pretrain-BACKBONE] missing={len(incompatible.missing_keys)}")
        print(f"[Pretrain-BACKBONE] unexpected={len(incompatible.unexpected_keys)}")

    else:
        raise ValueError(f"Unsupported load_mode: {load_mode}")

    return model


def run_training(pre_trained, pretrained_ckpt="", load_mode="full", model_name="dwnet"):
    image_files = get_img_files()
    if len(image_files) == 0:
        raise RuntimeError(
            "No training images found. Please check IMG_DIR/images and IMG_DIR/masks."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
    else:
        print("Training on CPU")

    train_files, val_files = train_test_split(
        image_files,
        test_size=VAL_FRACTION,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    print(
        f"Train: {len(train_files)} images, Val: {len(val_files)} images, Max epochs: {MAX_EPOCHS}"
    )

    writer = SummaryWriter(log_dir=os.path.join(OUT_DIR, "tb"))

    def on_after_epoch(m, df_hist):
        save_best_model(m, df_hist)
        write_on_board(writer, df_hist)
        log_hist(df_hist)

    criterion = HybridSegmentationMultiIonLoss(
        bce_weight=1.0, dice_weight=1.0, centroid_weight=0.1, radius=CENTROID_RADIUS
    )

    train_loader, val_loader, train_dataset, train_sampler = get_data_loaders(
        train_files, val_files
    )
    data_loaders = (train_loader, val_loader)

    def on_before_epoch(epoch):

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if USE_CURRICULUM:
            alpha = noise_alpha(
                epoch, MAX_EPOCHS, warmup_frac=NOISE_WARMUP_FRAC, alpha0=NOISE_ALPHA0
            )
            train_dataset.set_noise_strength(alpha)

    if model_name == "dwnet":
        model = DWNetV2_unet(pre_trained)
    elif model_name == "segformer":
        from nets.SegformerB5 import SegformerB5

        model = SegformerB5(pre_trained)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    model.to(device)

    if pretrained_ckpt:
        print(f"Loading pretrained checkpoint: {pretrained_ckpt}")
        print(f"Load mode: {load_mode}")
        model = load_pretrained_weights(
            model=model, ckpt_path=pretrained_ckpt, device=device, load_mode=load_mode
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=INITIAL_LR, weight_decay=1e-5)

    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    trainer = Trainer(
        data_loaders=data_loaders,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        on_after_epoch=on_after_epoch,
        on_before_epoch=on_before_epoch,
    )

    hist = trainer.train(model, optimizer, num_epochs=MAX_EPOCHS)
    hist.to_csv(f"{OUT_DIR}/hist.csv", index=False)

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_ckpt", type=str, default="", help="Path to pretrained checkpoint"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dwnet",
        choices=["dwnet", "segformer"],
        help="Model architecture: dwnet or segformer",
    )
    parser.add_argument(
        "--load_mode", type=str, default="full", choices=["full", "backbone"]
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Maximum number of training epochs (default: 1000)",
    )
    parser.add_argument(
        "--no_curriculum",
        action="store_true",
        help="Disable curriculum learning (density staging + noise ramp)",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    if args.out_dir is not None:
        OUT_DIR = args.out_dir
    else:
        OUT_DIR = f"outputs/{EXPERIMENT}_seed{args.seed}"

    if args.epochs is not None:
        MAX_EPOCHS = args.epochs

    if args.no_curriculum:
        USE_CURRICULUM = False

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    logger.setLevel(logging.DEBUG)
    if not logger.hasHandlers():
        logger.addHandler(
            logging.FileHandler(
                filename=os.path.join(OUT_DIR, f"{EXPERIMENT}_seed{args.seed}.log")
            )
        )
        logger.addHandler(logging.StreamHandler())

    if torch.cuda.is_available():
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_accumulated_memory_stats()

    run_training(
        pre_trained=None,
        pretrained_ckpt=args.pretrained_ckpt,
        load_mode=args.load_mode,
        model_name=args.model,
    )
