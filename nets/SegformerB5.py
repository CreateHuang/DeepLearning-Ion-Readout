import torch
import torch.nn as nn
from transformers import SegformerConfig, SegformerForSemanticSegmentation


class SegformerB5(nn.Module):
    def __init__(self, pre_trained=None, mode="train"):
        super(SegformerB5, self).__init__()
        self.mode = mode

        config = SegformerConfig(
            num_channels=1,
            num_labels=1,
            depths=[3, 6, 40, 3],
            hidden_sizes=[64, 128, 320, 512],
            decoder_hidden_size=768,
            classifier_dropout_prob=0.1,
        )

        self.model = SegformerForSemanticSegmentation(config)

        if pre_trained is not None:
            print(f"Loading weights from {pre_trained}")
            self.model.load_state_dict(torch.load(pre_trained))

    def forward(self, x):
        outputs = self.model(x)
        logits = outputs.logits

        logits_resized = torch.nn.functional.interpolate(
            logits, size=x.shape[2:], mode="bilinear", align_corners=True
        )

        return torch.sigmoid(logits_resized)
