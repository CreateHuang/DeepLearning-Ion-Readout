import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(inp: int, oup: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


def conv_1x1_bn(inp: int, oup: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: int):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(
                    hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(
                    hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class DWNetV2Backbone(nn.Module):

    def __init__(self, in_channels: int = 1, width_mult: float = 1.0):
        super().__init__()
        block = InvertedResidual
        input_channel = int(32 * width_mult)
        last_channel = 1280

        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        self.last_channel = (
            int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        )
        features = [conv_bn(in_channels, input_channel, 1)]

        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    block(input_channel, output_channel, stride, expand_ratio=t)
                )
                input_channel = output_channel

        features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features = nn.Sequential(*features)

        self._initialize_weights()

    def forward(self, x: torch.Tensor):

        for n in range(0, 2):
            x = self.features[n](x)
        x1 = x

        for n in range(2, 4):
            x = self.features[n](x)
        x2 = x

        for n in range(4, 7):
            x = self.features[n](x)
        x3 = x

        for n in range(7, 14):
            x = self.features[n](x)
        x4 = x

        for n in range(14, 19):
            x = self.features[n](x)
        x5 = x

        return x1, x2, x3, x4, x5

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class DWNetV2DenoiseUNet(nn.Module):

    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()
        self.backbone = DWNetV2Backbone(in_channels=in_channels)

        self.dconv1 = nn.ConvTranspose2d(1280, 96, 4, padding=1, stride=2)
        self.invres1 = InvertedResidual(192, 96, 1, 6)

        self.dconv2 = nn.ConvTranspose2d(96, 32, 4, padding=1, stride=2)
        self.invres2 = InvertedResidual(64, 32, 1, 6)

        self.dconv3 = nn.ConvTranspose2d(32, 24, 4, padding=1, stride=2)
        self.invres3 = InvertedResidual(48, 24, 1, 6)

        self.dconv4 = nn.ConvTranspose2d(24, 16, 4, padding=1, stride=2)
        self.invres4 = InvertedResidual(32, 16, 1, 6)

        self.recon_head = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=1),
        )

        self._init_decoder_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]
        x1, x2, x3, x4, x5 = self.backbone(x)

        d1 = self.dconv1(x5)
        d1 = F.interpolate(d1, size=x4.shape[2:], mode="bilinear", align_corners=True)
        u1 = self.invres1(torch.cat([x4, d1], dim=1))

        d2 = self.dconv2(u1)
        d2 = F.interpolate(d2, size=x3.shape[2:], mode="bilinear", align_corners=True)
        u2 = self.invres2(torch.cat([x3, d2], dim=1))

        d3 = self.dconv3(u2)
        d3 = F.interpolate(d3, size=x2.shape[2:], mode="bilinear", align_corners=True)
        u3 = self.invres3(torch.cat([x2, d3], dim=1))

        d4 = self.dconv4(u3)
        d4 = F.interpolate(d4, size=x1.shape[2:], mode="bilinear", align_corners=True)
        u4 = self.invres4(torch.cat([x1, d4], dim=1))

        out = self.recon_head(u4)
        out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=True)
        return out

    def _init_decoder_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
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

    def load_backbone_from_classifier_ckpt(
        self, ckpt_path: str, device: str = "cpu"
    ) -> None:
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)

        converted: Dict[str, torch.Tensor] = {}
        backbone_state = self.backbone.state_dict()

        for key, value in state_dict.items():
            if key.startswith("backbone."):
                new_key = key[len("backbone.") :]
            else:
                new_key = key

            if (
                new_key in backbone_state
                and backbone_state[new_key].shape == value.shape
            ):
                converted[new_key] = value

        missing, unexpected = self.backbone.load_state_dict(converted, strict=False)
        print(
            f"[load_backbone_from_classifier_ckpt] loaded={len(converted)}, missing={len(missing)}, unexpected={len(unexpected)}"
        )

    def load_denoise_pretrain(
        self, ckpt_path: str, device: str = "cpu", strict: bool = True
    ) -> None:
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = self.load_state_dict(state_dict, strict=strict)
        print(
            f"[load_denoise_pretrain] missing={len(missing)}, unexpected={len(unexpected)}"
        )


if __name__ == "__main__":
    model = DWNetV2DenoiseUNet(in_channels=1, out_channels=1)
    x = torch.randn(2, 1, 224, 224)
    y = model(x)
    print("input :", x.shape)
    print("output:", y.shape)
