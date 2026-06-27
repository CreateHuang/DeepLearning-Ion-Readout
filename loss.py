import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(pred, target, eps=1.0):
    pred = pred.contiguous().view(pred.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)

    loss = 1.0 - (2.0 * intersection + eps) / (union + eps)
    return loss.mean()


def extract_local_soft_centroid(
    prob_map, centers_gt, centers_valid, radius=3, eps=1e-6
):
    B, C, H, W = prob_map.shape
    assert C == 1

    device = prob_map.device
    dtype = prob_map.dtype
    K = centers_gt.shape[1]

    ys_full = torch.arange(H, device=device, dtype=dtype).view(1, 1, H, 1)
    xs_full = torch.arange(W, device=device, dtype=dtype).view(1, 1, 1, W)

    pred_centers = torch.zeros((B, K, 2), device=device, dtype=dtype)

    for b in range(B):
        for k in range(K):
            if centers_valid[b, k] < 0.5:
                continue

            x0 = centers_gt[b, k, 0]
            y0 = centers_gt[b, k, 1]

            x_min = max(0, int(torch.floor(x0).item()) - radius)
            x_max = min(W, int(torch.floor(x0).item()) + radius + 1)
            y_min = max(0, int(torch.floor(y0).item()) - radius)
            y_max = min(H, int(torch.floor(y0).item()) + radius + 1)

            patch = prob_map[b : b + 1, :, y_min:y_max, x_min:x_max]
            if patch.numel() == 0:
                pred_centers[b, k] = centers_gt[b, k]
                continue

            xs = xs_full[:, :, :, x_min:x_max].expand_as(patch)
            ys = ys_full[:, :, y_min:y_max, :].expand_as(patch)

            mass = patch.sum() + eps
            cx = (patch * xs).sum() / mass
            cy = (patch * ys).sum() / mass

            pred_centers[b, k, 0] = cx
            pred_centers[b, k, 1] = cy

    return pred_centers


def multi_ion_centroid_loss(pred, centers_gt, centers_valid, radius=3):
    pred_centers = extract_local_soft_centroid(
        pred, centers_gt, centers_valid, radius=radius
    )

    diff = (pred_centers - centers_gt) ** 2
    diff = diff.sum(dim=-1)

    valid = centers_valid.float()
    loss = (diff * valid).sum() / (valid.sum() + 1e-6)
    return loss


class HybridSegmentationMultiIonLoss(nn.Module):

    def __init__(self, bce_weight=1.0, dice_weight=1.0, centroid_weight=0.1, radius=3):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.centroid_weight = centroid_weight
        self.radius = radius
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target, centers_gt, centers_valid):
        pred = torch.sigmoid(logits)

        loss_bce = self.bce(logits, target)
        loss_dice = dice_loss(pred, target)
        loss_centroid = multi_ion_centroid_loss(
            pred, centers_gt, centers_valid, radius=self.radius
        )

        total_loss = (
            self.bce_weight * loss_bce
            + self.dice_weight * loss_dice
            + self.centroid_weight * loss_centroid
        )

        return {
            "loss": total_loss,
            "loss_bce": loss_bce,
            "loss_dice": loss_dice,
            "loss_centroid": loss_centroid,
        }
