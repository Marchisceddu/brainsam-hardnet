import torch
import torch.nn as nn
import torch.nn.functional as F


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

    @property
    def mask_decoder(self):
        return self.sam_unet_decoder.mask_decoder

    def forward(self, x):
        original_size = x.shape[-1]
        
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
        
        # 2. HardNet U-Net estrae logits e dense_features
        mask_logits, hardnet_dense_features = self.hardnet_unet_stage(x)
        out1 = mask_logits
        
        if out1.shape[-1] != original_size:
            out1_resized = F.interpolate(
                out1,
                (original_size, original_size),
                mode="bilinear",
                align_corners=False,
            )
        else:
            out1_resized = out1

        out_d = out1

        for _ in range(self.iter_2stage):
            # Sigmoid e selezione canali (nel caso num_classes > 1 prendiamo il canale foreground)
            if out_d.shape[1] > 1:
                p_in = torch.sigmoid(out_d[:, 1:2, ...])
            else:
                p_in = torch.sigmoid(out_d)

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
