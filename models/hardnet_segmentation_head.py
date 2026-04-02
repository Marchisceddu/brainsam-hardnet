import torch
import torch.nn as nn
import torch.nn.functional as F

from .hardnet import HarDNet


class ConvUpsample(nn.Module):
    def __init__(self, in_channels=256, out_channels=1, upsample_factors=(2, 2, 2, 2, 2)):
        super().__init__()
        layers = []
        current_channels = in_channels
        for factor in upsample_factors:
            next_channels = max(current_channels // 2, out_channels)
            layers.append(nn.Conv2d(current_channels, next_channels, kernel_size=1))
            layers.append(nn.Upsample(scale_factor=factor, mode="bilinear", align_corners=False))
            current_channels = next_channels
        layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=1))
        self.upsample_net = nn.Sequential(*layers)

    def forward(self, x):
        return self.upsample_net(x)


class HardNetSegmentationHead(nn.Module):
    def __init__(
        self,
        arch=85,
        pretrained=True,
        in_channels=3,
        backbone_input_size=512,
        prompt_size=256,
        out_channels=1,
        freeze_backbone=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.backbone_input_size = backbone_input_size
        self.prompt_size = prompt_size

        self.backbone = HarDNet(arch=arch, pretrained=pretrained)
        if self.in_channels == 1:
            old_layer = self.backbone.base[0].conv
            old_weights = old_layer.weight.data
            new_layer = nn.Conv2d(
                1,
                old_layer.out_channels,
                kernel_size=old_layer.kernel_size,
                stride=old_layer.stride,
                padding=old_layer.padding,
                bias=old_layer.bias is not None,
            )
            new_layer.weight.data = old_weights.mean(dim=1, keepdim=True)
            if old_layer.bias is not None:
                new_layer.bias.data = old_layer.bias.data
            self.backbone.base[0].conv = new_layer
        self.head = ConvUpsample(
            in_channels=self.backbone.features,
            out_channels=out_channels,
            upsample_factors=(2, 2, 2, 2, 2),
        )

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def forward(self, x):
        if self.in_channels == 1 and x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)
        elif self.in_channels == 3 and x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        if x.shape[-1] != self.backbone_input_size or x.shape[-2] != self.backbone_input_size:
            x = F.interpolate(
                x,
                size=(self.backbone_input_size, self.backbone_input_size),
                mode="bilinear",
                align_corners=False,
            )

        _, _, _, _, bottleneck = self.backbone(x)
        mask_logits = self.head(bottleneck)

        if mask_logits.shape[-1] != self.prompt_size or mask_logits.shape[-2] != self.prompt_size:
            mask_logits = F.interpolate(
                mask_logits,
                size=(self.prompt_size, self.prompt_size),
                mode="bilinear",
                align_corners=False,
            )

        return mask_logits
