import logging
import math
import sys

import torch
import torch.nn as nn
from torch.nn import functional as F
from nets.DWNetV2 import DWNetV2, InvertedResidual


class DWNetV2_unet(nn.Module):
    def __init__(self, pre_trained, mode="train"):
        super(DWNetV2_unet, self).__init__()

        self.mode = mode
        self.backbone = DWNetV2()

        self.dconv1 = nn.ConvTranspose2d(1280, 96, 4, padding=1, stride=2)
        self.invres1 = InvertedResidual(192, 96, 1, 6)

        self.dconv2 = nn.ConvTranspose2d(96, 32, 4, padding=1, stride=2)
        self.invres2 = InvertedResidual(64, 32, 1, 6)

        self.dconv3 = nn.ConvTranspose2d(32, 24, 4, padding=1, stride=2)
        self.invres3 = InvertedResidual(48, 24, 1, 6)

        self.dconv4 = nn.ConvTranspose2d(24, 16, 4, padding=1, stride=2)
        self.invres4 = InvertedResidual(32, 16, 1, 6)

        self.conv_last = nn.Conv2d(16, 3, 1)

        self.conv_score = nn.Conv2d(3, 1, 1)

        self._init_weights()

        if pre_trained is not None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            full_state_dict = torch.load(pre_trained, map_location=device)

            backbone_state_dict = {}
            for key, value in full_state_dict.items():

                if key.startswith("backbone."):

                    new_key = key[len("backbone.") :]
                    backbone_state_dict[new_key] = value

            self.backbone.load_state_dict(backbone_state_dict)

        else:
            print("No pretrained weights — training from scratch.")

    def forward(self, x):
        for n in range(0, 2):
            x = self.backbone.features[n](x)
        x1 = x
        logging.debug((x1.shape, "x1"))

        for n in range(2, 4):
            x = self.backbone.features[n](x)
        x2 = x
        logging.debug((x2.shape, "x2"))

        for n in range(4, 7):
            x = self.backbone.features[n](x)
        x3 = x
        logging.debug((x3.shape, "x3"))

        for n in range(7, 14):
            x = self.backbone.features[n](x)
        x4 = x
        logging.debug((x4.shape, "x4"))

        for n in range(14, 19):
            x = self.backbone.features[n](x)
        x5 = x
        logging.debug((x5.shape, "x5"))

        decoder_feature0 = self.dconv1(x)

        decoder_feature0_resized = F.interpolate(
            decoder_feature0, size=x4.shape[2:], mode="bilinear", align_corners=True
        )

        up1 = torch.cat([x4, decoder_feature0_resized], dim=1)
        up1 = self.invres1(up1)
        logging.debug((up1.shape, "up1"))

        decoder_feature1 = self.dconv2(up1)

        decoder_feature1_resized = F.interpolate(
            decoder_feature1, size=x3.shape[2:], mode="bilinear", align_corners=True
        )
        up2 = torch.cat([x3, decoder_feature1_resized], dim=1)
        up2 = self.invres2(up2)
        logging.debug((up2.shape, "up2"))

        decoder_feature2 = self.dconv3(up2)
        decoder_feature2_resized = F.interpolate(
            decoder_feature2, size=x2.shape[2:], mode="bilinear", align_corners=True
        )
        up3 = torch.cat([x2, decoder_feature2_resized], dim=1)
        up3 = self.invres3(up3)
        logging.debug((up3.shape, "up3"))

        decoder_feature3 = self.dconv4(up3)
        decoder_feature3_resized = F.interpolate(
            decoder_feature3, size=x1.shape[2:], mode="bilinear", align_corners=True
        )
        up4 = torch.cat([x1, decoder_feature3_resized], dim=1)
        up4 = self.invres4(up4)
        logging.debug((up4.shape, "up4"))

        x = self.conv_last(up4)
        logging.debug((x.shape, "conv_last"))

        x = self.conv_score(x)
        logging.debug((x.shape, "conv_score"))

        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


if __name__ == "__main__":

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    net = DWNetV2_unet(pre_trained=None)
    net(torch.randn(1, 3, 224, 224))
