import torch
import torch.nn as nn
import torch.nn.functional as F


class HardNetFeatSeg(nn.Module):
    def __init__(
        self,
        image_encoder,
        hardnet_first_stage,
        prompt_encoder_end,
        seg_decoder_end,
        img_size,
        iter_2stage,
    ):
        super().__init__()
        self.iter_2stage = iter_2stage
        self.img_size = img_size
        self.image_encoder = image_encoder
        self.hardnet_first_stage = hardnet_first_stage
        self.prompt_encoder_end = prompt_encoder_end
        self.seg_decoder_end = seg_decoder_end

    def forward(self, x):
        original_size = x.shape[-1]
        x = F.interpolate(
            x,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )

        image_embedding = self.image_encoder(x)
        prompt_embedding = self.hardnet_first_stage(x)
        out1 = prompt_embedding

        if out1.shape[-1] != original_size:
            out1 = F.interpolate(
                out1,
                (original_size, original_size),
                mode="bilinear",
                align_corners=False,
            )

        out1_d = out1

        for _ in range(self.iter_2stage):
            # Convert logits → probabilities before passing as mask prompt.
            # SAM's prompt encoder mask_downscaling expects bounded [0,1] masks
            # with exactly 1 channel (the foreground probability).
            if out1_d.shape[1] > 1:
                # If multi-class, we take the foreground channel (index 1)
                p2_in = torch.sigmoid(out1_d[:, 1:2, ...])
            else:
                p2_in = torch.sigmoid(out1_d)

            p2_in = F.interpolate(
                p2_in,
                self.prompt_encoder_end.mask_input_size,
                mode="bilinear",
                align_corners=False,
            )
            _, prompt_embedding2 = self.prompt_encoder_end(
                points=None,
                boxes=None,
                masks=p2_in,
            )
            out1_d = self.seg_decoder_end(image_embedding, prompt_embedding2)
            out1_d = F.interpolate(
                out1_d,
                (original_size, original_size),
                mode="bilinear",
                align_corners=False,
            )

        out2 = out1_d
        return out1, out2, prompt_embedding
