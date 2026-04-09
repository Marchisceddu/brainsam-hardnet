import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F


class LearnablePromptRefiner(nn.Module):
    """
    Raffina i logits grezzi dello stage 1 per generare un mask-prompt più pulito
    e semanticamente coerente rispetto a una semplice operazione di clamp/sigmoid.
    """
    def __init__(self, in_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # output ranges [0, 1] as required by SAM prompt encoder
        )

    def forward(self, x):
        return self.net(x)


class HardNetFeatSegUNet(nn.Module):
    def __init__(
        self,
        image_encoder,
        hardnet_unet_stage,
        prompt_encoder,
        sam_unet_decoder,
        img_size,
        iter_2stage,
    ):
        super().__init__()
        self.iter_2stage = iter_2stage
        self.img_size = img_size
        self.image_encoder = image_encoder
        self.hardnet_unet_stage = hardnet_unet_stage
        self.prompt_encoder = prompt_encoder
        self.sam_unet_decoder = sam_unet_decoder
        self.prompt_refiner = LearnablePromptRefiner(in_channels=1)

        # Multi-scale FPN Dense Features projection
        self.proj_d1 = nn.Conv2d(512, 256, kernel_size=1)
        self.proj_d2 = nn.Conv2d(256, 256, kernel_size=1)
        self.proj_d3 = nn.Conv2d(128, 256, kernel_size=1)

    @property
    def mask_decoder(self):
        return self.sam_unet_decoder.mask_decoder

    def forward(self, x, stage1_only=False):
        original_size = x.shape[-1]
        
        # 2. HardNet U-Net estrae logits e feature maps (d1, d2, d3)
        mask_logits, (d1, d2, d3) = self.hardnet_unet_stage(x)
        out1 = mask_logits
        
        # Feature Pyramid fusion for dense features (allineate a d1, cioè 1/16)
        dense_d1 = self.proj_d1(d1)
        dense_d2 = F.interpolate(self.proj_d2(d2), size=dense_d1.shape[-2:], mode="bilinear", align_corners=False)
        dense_d3 = F.interpolate(self.proj_d3(d3), size=dense_d1.shape[-2:], mode="bilinear", align_corners=False)
        
        hardnet_dense_features = dense_d1 + dense_d2 + dense_d3
        
        if out1.shape[-1] != original_size:
            out1_resized = F.interpolate(
                out1,
                (original_size, original_size),
                mode="bilinear",
                align_corners=False,
            )
        else:
            out1_resized = out1

        if stage1_only:
            return out1_resized, None, mask_logits

        # Resize per il ViT
        if x.shape[-1] != self.image_encoder.img_size:
            x_vit = F.interpolate(
                x,
                (self.image_encoder.img_size, self.image_encoder.img_size),
                mode="bilinear",
                align_corners=False,
            )
        else:
            x_vit = x

        # 1. SAM ViT estrae le image_embeddings
        image_embeddings = self.image_encoder(x_vit)
        
        # Estrarre Positional Embeddings dall'encoder del prompt
        image_pe = self.prompt_encoder.get_dense_pe()

        out_d = out1

        for _ in range(self.iter_2stage):
            # Learnable Prompt Refiner in sostituzione della semplice sigmoid
            if out_d.shape[1] > 1:
                p_in = self.prompt_refiner(out_d[:, 1:2, ...])
            else:
                p_in = self.prompt_refiner(out_d)

            # Evita interpolazioni inutili se la risoluzione e' gia' corretta.
            target_mask_size = self.prompt_encoder.mask_input_size
            if p_in.shape[-2:] != target_mask_size:
                p_in = F.interpolate(
                    p_in,
                    target_mask_size,
                    mode="bilinear",
                    align_corners=False,
                )
            
            # Generazione prompt
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=None,
                masks=p_in,
            )
            
            # Decodifica basata su MaskDecoder di SAM + Fusion Block
            masks, iou_pred = self.sam_unet_decoder(
                image_embeddings=image_embeddings,
                hardnet_dense_features=hardnet_dense_features,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            
            out_d = masks
            
        out2 = F.interpolate(
            out_d,
            (original_size, original_size),
            mode="bilinear",
            align_corners=False,
        )

        return out1_resized, out2, mask_logits
