import torch
import torch.nn as nn
import torch.nn.functional as F

from segment_anything.modeling.mask_decoder import MaskDecoder


class FeatureFusionBlock(nn.Module):
    """
    Fonds le features estratte dal ViT di SAM con le feature dense estratte dalla U-Net di Hardnet.
    Entrambi sono tensori di dimensione (B, 256, H/16, W/16).
    """
    def __init__(self, channels=256):
        super().__init__()
        # Concateniamo i due tensori -> 512, riportiamo a 256, e applichiamo 3x3 conv
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, sam_features, hardnet_features):
        x = torch.cat([sam_features, hardnet_features], dim=1)
        return self.conv_fusion(x)


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
