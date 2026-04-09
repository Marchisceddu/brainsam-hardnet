'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hardnet import HarDNet


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: gating signal from decoder, x: skip connection
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        if skip_channels > 0:
            # We use typically half the skip channels for the intermediate representation
            f_int = max(skip_channels // 2, 1)
            self.ag = AttentionGate(F_g=in_channels, F_l=skip_channels, F_int=f_int)
        else:
            self.ag = None

        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip=None):
        x_up = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            if x_up.shape[-2:] != skip.shape[-2:]:
                x_up = F.interpolate(x_up, size=skip.shape[-2:], mode="bilinear", align_corners=False)

            if self.ag is not None:
                skip = self.ag(g=x_up, x=skip)

            x_up = torch.cat([x_up, skip], dim=1)
            
        x_out = self.conv1(x_up)
        x_out = self.bn1(x_out)
        x_out = self.relu(x_out)
        x_out = self.conv2(x_out)
        x_out = self.bn2(x_out)
        x_out = self.relu(x_out)
        return x_out


class HardNetUNetHead(nn.Module):
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
        
        # Adatta il primo layer se in_channels == 1
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

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        # Canali di output per architettura 85: full_features = [96, 192, 320, 720, 1280]
        # Equivalenti a [x2, x4, x8, x16, x32]
        if not hasattr(self.backbone, 'full_features'):
            raise ValueError(f"HarDNet arch={arch} non supportata o full_features mancante.")
            
        ch_x2, ch_x4, ch_x8, ch_x16, ch_x32 = self.backbone.full_features

        # Decoder Stages
        self.dec1 = DecoderBlock(in_channels=ch_x32, skip_channels=ch_x16, out_channels=512)
        
        # Multi-scale FPN Dense Features projection
        self.proj_d1 = nn.Conv2d(512, 256, kernel_size=1)
        
        self.dec2 = DecoderBlock(in_channels=512, skip_channels=ch_x8, out_channels=256)
        self.proj_d2 = nn.Conv2d(256, 256, kernel_size=1)
        
        self.dec3 = DecoderBlock(in_channels=256, skip_channels=ch_x4, out_channels=128)
        self.proj_d3 = nn.Conv2d(128, 256, kernel_size=1)
        
        self.dec4 = DecoderBlock(in_channels=128, skip_channels=ch_x2, out_channels=64)
        
        # Layer per generare il logit finale della maschera (senza passaggi di sigmoid qui)
        self.head = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Adattamento input
        if self.in_channels == 1 and x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)
        elif self.in_channels == 3 and x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # Assicurati che l'input size coincida (ad es. 512x512)
        if x.shape[-1] != self.backbone_input_size or x.shape[-2] != self.backbone_input_size:
            x = F.interpolate(
                x,
                size=(self.backbone_input_size, self.backbone_input_size),
                mode="bilinear",
                align_corners=False,
            )

        # HardNet Forward (restituisce 5 feature map)
        features = self.backbone(x)
        if len(features) == 5:
            x2, x4, x8, x16, x32 = features
        else:
            raise RuntimeError("La backbone HarDNet non ha restituito 5 feature map come previsto.")

        # U-Net Decoding
        # 1. Da x32 (16x16) a x16 (32x32)
        d1 = self.dec1(x32, x16) 
        
        # 2. Da x16 (32x32) a x8 (64x64)
        d2 = self.dec2(d1, x8)
        
        # 3. Da x8 (64x64) a x4 (128x128)
        d3 = self.dec3(d2, x4)
        
        # Feature Pyramid fusion for dense features (allineate a d1, cioè 1/16)
        dense_d1 = self.proj_d1(d1)
        dense_d2 = F.interpolate(self.proj_d2(d2), size=dense_d1.shape[-2:], mode="bilinear", align_corners=False)
        dense_d3 = F.interpolate(self.proj_d3(d3), size=dense_d1.shape[-2:], mode="bilinear", align_corners=False)
        
        dense_features = dense_d1 + dense_d2 + dense_d3
        
        # 4. Da x4 (128x128) a x2 (256x256)
        d4 = self.dec4(d3, x2)
        
        # Final logit
        mask_logits = self.head(d4)

        # Se la maschera non è esatamente 256x256 come richiesto dal prompt di SAM, aggiustala
        if mask_logits.shape[-1] != self.prompt_size or mask_logits.shape[-2] != self.prompt_size:
            mask_logits = F.interpolate(
                mask_logits,
                size=(self.prompt_size, self.prompt_size),
                mode="bilinear",
                align_corners=False,
            )

        return mask_logits, dense_features
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hardnet import HarDNet


class ASPP(nn.Module):
    def __init__(self, in_channels, branch_channels=256):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(branch_channels)
        self.conv2 = nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=6, dilation=6, bias=False)
        self.bn2 = nn.BatchNorm2d(branch_channels)
        self.conv3 = nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=12, dilation=12, bias=False)
        self.bn3 = nn.BatchNorm2d(branch_channels)
        self.conv4 = nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=18, dilation=18, bias=False)
        self.bn4 = nn.BatchNorm2d(branch_channels)
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        self.project = nn.Sequential(
            nn.Conv2d(branch_channels * 5, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res1 = self.relu(self.bn1(self.conv1(x)))
        res2 = self.relu(self.bn2(self.conv2(x)))
        res3 = self.relu(self.bn3(self.conv3(x)))
        res4 = self.relu(self.bn4(self.conv4(x)))
        res5 = self.gap(x)
        res5 = F.interpolate(res5, size=x.shape[-2:], mode='bilinear', align_corners=False)
        out = torch.cat([res1, res2, res3, res4, res5], dim=1)
        return self.project(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return x * self.sigmoid(out)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        reduced_channels = max(in_channels // reduction_ratio, 1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.mlp(self.avg_pool(x).view(b, c))
        max_out = self.mlp(self.max_pool(x).view(b, c))
        
        out = avg_out + max_out
        out = self.sigmoid(out).view(b, c, 1, 1)
        return x * out


class DHAModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(DHAModule, self).__init__()
        self.spatial = SpatialAttention()
        self.channel = ChannelAttention(in_channels, reduction_ratio)

    def forward(self, x):
        out = self.spatial(x)
        out = self.channel(out)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        if skip_channels > 0:
            self.dha = DHAModule(in_channels=skip_channels)
        else:
            self.dha = None

        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip=None):
        x_up = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            if x_up.shape[-2:] != skip.shape[-2:]:
                x_up = F.interpolate(x_up, size=skip.shape[-2:], mode="bilinear", align_corners=False)

            if self.dha is not None:
                skip = self.dha(skip)

            x_up = torch.cat([x_up, skip], dim=1)
            
        x_out = self.conv1(x_up)
        x_out = self.bn1(x_out)
        x_out = self.relu(x_out)
        x_out = self.conv2(x_out)
        x_out = self.bn2(x_out)
        x_out = self.relu(x_out)
        return x_out


class HardNetUNetHead(nn.Module):
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
        
        # Adatta il primo layer se in_channels == 1
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

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        # Canali di output per architettura 85: full_features = [96, 192, 320, 720, 1280]
        # Equivalenti a [x2, x4, x8, x16, x32]
        if not hasattr(self.backbone, 'full_features'):
            raise ValueError(f"HarDNet arch={arch} non supportata o full_features mancante.")
            
        ch_x2, ch_x4, ch_x8, ch_x16, ch_x32 = self.backbone.full_features

        # Bottleneck: ASPP
        self.aspp = ASPP(in_channels=ch_x32, branch_channels=256)

        # Decoder Stages
        self.dec1 = DecoderBlock(in_channels=ch_x32, skip_channels=ch_x16, out_channels=512)
        
        self.dec2 = DecoderBlock(in_channels=512, skip_channels=ch_x8, out_channels=256)
        
        self.dec3 = DecoderBlock(in_channels=256, skip_channels=ch_x4, out_channels=128)
        
        self.dec4 = DecoderBlock(in_channels=128, skip_channels=ch_x2, out_channels=64)
        
        # Layer per generare il logit finale della maschera (senza passaggi di sigmoid qui)
        self.head = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Adattamento input
        if self.in_channels == 1 and x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)
        elif self.in_channels == 3 and x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # Assicurati che l'input size coincida (ad es. 512x512)
        if x.shape[-1] != self.backbone_input_size or x.shape[-2] != self.backbone_input_size:
            x = F.interpolate(
                x,
                size=(self.backbone_input_size, self.backbone_input_size),
                mode="bilinear",
                align_corners=False,
            )

        # HardNet Forward (restituisce 5 feature map)
        features = self.backbone(x)
        if len(features) == 5:
            x2, x4, x8, x16, x32 = features
        else:
            raise RuntimeError("La backbone HarDNet non ha restituito 5 feature map come previsto.")

        # U-Net Decoding
        # 0. Bottleneck
        x32 = self.aspp(x32)

        # 1. Da x32 (16x16) a x16 (32x32)
        d1 = self.dec1(x32, x16) 
        
        # 2. Da x16 (32x32) a x8 (64x64)
        d2 = self.dec2(d1, x8)
        
        # 3. Da x8 (64x64) a x4 (128x128)
        d3 = self.dec3(d2, x4)
        
        # 4. Da x4 (128x128) a x2 (256x256)
        d4 = self.dec4(d3, x2)
        
        # Final logit
        mask_logits = self.head(d4)

        # Se la maschera non è esatamente 256x256 come richiesto dal prompt di SAM, aggiustala
        if mask_logits.shape[-1] != self.prompt_size or mask_logits.shape[-2] != self.prompt_size:
            mask_logits = F.interpolate(
                mask_logits,
                size=(self.prompt_size, self.prompt_size),
                mode="bilinear",
                align_corners=False,
            )

        return mask_logits, (d1, d2, d3)
