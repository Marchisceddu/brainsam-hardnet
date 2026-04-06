import torch
import torch.nn as nn
import torch.nn.functional as F


class HardNetFeatSegUNetCNN(nn.Module):
    def __init__(
        self,
        image_encoder,
        hardnet_unet_stage,
        prompt_encoder_end,
        seg_decoder_end,
        img_size,
        iter_2stage,
    ):
        super().__init__()
        self.iter_2stage = iter_2stage
        self.img_size = img_size
        self.image_encoder = image_encoder
        self.hardnet_unet_stage = hardnet_unet_stage
        self.prompt_encoder_end = prompt_encoder_end
        self.seg_decoder_end = seg_decoder_end

    def forward(self, x):
        original_size = x.shape[-1]

        if x.shape[-1] != self.image_encoder.img_size:
            x_vit = F.interpolate(
                x,
                (self.image_encoder.img_size, self.image_encoder.img_size),
                mode="bilinear",
                align_corners=False,
            )
        else:
            x_vit = x

        image_embeddings = self.image_encoder(x_vit)

        # HardNet U-Net stage outputs first-stage mask logits and dense features.
        # This variant uses mask logits as prompt source and a CNN decoder for stage 2.
        mask_logits, _ = self.hardnet_unet_stage(x)
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
            if out_d.shape[1] > 1:
                p_in = torch.sigmoid(out_d[:, 1:2, ...])
            else:
                p_in = torch.sigmoid(out_d)

            target_mask_size = self.prompt_encoder_end.mask_input_size
            if p_in.shape[-2:] != target_mask_size:
                p_in = F.interpolate(
                    p_in,
                    target_mask_size,
                    mode="bilinear",
                    align_corners=False,
                )

            _, dense_prompt_embeddings = self.prompt_encoder_end(
                points=None,
                boxes=None,
                masks=p_in,
            )

            out_d = self.seg_decoder_end(image_embeddings, dense_prompt_embeddings)

        out2 = F.interpolate(
            out_d,
            (original_size, original_size),
            mode="bilinear",
            align_corners=False,
        )

        return out1_resized, out2, mask_logits
