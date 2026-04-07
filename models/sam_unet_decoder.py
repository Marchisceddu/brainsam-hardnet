import torch
import torch.nn as nn
import torch.nn.functional as F

from segment_anything.modeling.mask_decoder import MaskDecoder


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return self.sigmoid(y) * x


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(F.adaptive_avg_pool2d(x, 1))))
        max_out = self.fc2(self.relu(self.fc1(F.adaptive_max_pool2d(x, 1))))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class FeatureFusionBlock(nn.Module):
    """
    Fonde le features estratte dal ViT di SAM con le feature dense estratte dalla U-Net di Hardnet,
    usando un meccanismo combinato di Channel e Spatial Attention (ispirato a CBAM).
    Entrambi sono tensori di dimensione (B, 256, H/16, W/16).
    """
    def __init__(self, channels=256):
        super().__init__()
        # Normalizzazione della varianza delle feature HarDNet
        self.hardnet_norm = nn.GroupNorm(32, channels)

        # Proiezione post concatenazione
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # Attention blocks
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention(kernel_size=7)

        # Connessione finale
        self.conv_out = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, sam_features, hardnet_features):
        hardnet_features = self.hardnet_norm(hardnet_features)
        x = torch.cat([sam_features, hardnet_features], dim=1)
        x = self.conv_fusion(x)
        # Apply Channel Attention
        x = self.ca(x)
        # Apply Spatial Attention
        x = self.sa(x)
        return self.conv_out(x)


class SamUNetDecoder(nn.Module):
    """
    Eredita le funzioni del MaskDecoder ma applica prima il FeatureFusionBlock sugli image_embeddings e le features HardNet.
    Invece di ereditare, possiamo semplicemente wrapparlo.
    """
    def __init__(self, mask_decoder: MaskDecoder):
        super().__init__()
        self.fusion_block = FeatureFusionBlock(channels=mask_decoder.transformer_dim)
        self.mask_decoder = mask_decoder

    def forward(
        self,
        image_embeddings: torch.Tensor,
        hardnet_dense_features: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ):
        # 1. Fondere le feature del ViT e della U-Net
        fused_embeddings = self.fusion_block(image_embeddings, hardnet_dense_features)
        
        # 2. Passarle nel MaskDecoder originale di SAM
        masks, iou_pred = self.mask_decoder(
            image_embeddings=fused_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=multimask_output,
        )
        
        return masks, iou_pred
