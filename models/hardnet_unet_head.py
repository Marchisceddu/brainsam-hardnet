import torch
import torch.nn as nn
import torch.nn.functional as F

from .hardnet import HarDNet


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # Usiamo ConvTranspose2d per l'upsampling per avere parametri apprendibili, 
        # oppure si può usare interpolate + Conv. Usiamo interpolate + Conv 
        # per ridurre artefatti a scacchiera (checkerboard artifacts).
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            # Assicuriamoci che le dimensioni coincidano (nel caso di dispari)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


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
        
        # Estrazione Dense Features (es. a risoluzione 1/16 per allinearsi con SAM ViT)
        self.dense_features_proj = nn.Conv2d(512, 256, kernel_size=1)
        
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
        # 1. Da x32 (16x16) a x16 (32x32)
        d1 = self.dec1(x32, x16) 
        
        # Estrai dense_features a 1/16 risoluzione (stessa del ViT embedding)
        dense_features = self.dense_features_proj(d1)
        
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

        return mask_logits, dense_features
