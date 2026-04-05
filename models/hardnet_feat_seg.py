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


class SegDecoderCNN(nn.Module):
    def __init__(self,
                 num_classes=2,
                 x_embed_dim=256,
                 num_depth=4,
                 top_channel=64,
                 p_channel=3,
                 promptemd_channel=256,
                 first_p=True,
                 ):
        super().__init__()
        self.first_p = first_p
        self.input_block = nn.Sequential(
            nn.Conv2d(x_embed_dim+promptemd_channel, top_channel*2**num_depth, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(top_channel * 2 ** num_depth, top_channel * 2 ** num_depth, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList()
        for i in range(num_depth):
            if num_depth > 2 > i:
                block = nn.Sequential(
                    nn.Conv2d(top_channel * 2 ** (num_depth - i), top_channel * 2 ** (num_depth - i), 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(top_channel * 2 ** (num_depth - i), top_channel * 2 ** (num_depth - i - 1), 3, padding=1),
                    nn.ReLU(inplace=True),
                )
            else:
                block = nn.Sequential(
                    nn.Conv2d(top_channel * 2 ** (num_depth - i),  top_channel * 2 ** (num_depth - i), 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(top_channel * 2 ** (num_depth - i),  top_channel * 2 ** (num_depth - i), 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(top_channel * 2 ** (num_depth - i), top_channel * 2 ** (num_depth - i - 1), 2, stride=2),
                )
            self.blocks.append(block)

        self.final = nn.Sequential(
            nn.Conv2d(top_channel+p_channel, int(top_channel), 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(top_channel, num_classes, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(num_classes, num_classes, 1),
        )

    def forward(self, x, p):
        if self.first_p:
            x = self.input_block(x)
            for blk in self.blocks:
                x = blk(x)
            y = torch.cat([x, p], 1)
            y = self.final(y)
        else:
            x = torch.cat([x, p], 1)
            x = self.input_block(x)
            for blk in self.blocks:
                x = blk(x)
            y = self.final(x)

        return y
