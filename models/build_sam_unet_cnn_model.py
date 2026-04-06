import torch

from functools import partial

from .hardnet_feat_seg import SegDecoderCNN
from .hardnet_feat_seg_unet_cnn import HardNetFeatSegUNetCNN
from .hardnet_unet_head import HardNetUNetHead
from segment_anything.modeling import ImageEncoderViT, PromptEncoder
from .build_sam_feat_seg_model import (
    _adapt_sam_patch_embed_in_channels,
    _resize_sam_pos_embed,
    _is_global_rel_pos_key,
    _resize_sam_rel_pos,
)


def load_checkpoint_safely_unet_cnn(model, checkpoint_path, encoder_global_attn_indexes):
    """Load compatible tensors for HardNetUNet + SegDecoderCNN architecture."""
    with open(checkpoint_path, "rb") as f:
        state_dict = torch.load(f, weights_only=False)

    if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]

    model_state = model.state_dict()
    filtered_state = {}
    loaded_keys = []
    resized_keys = []
    skipped_shape = []

    for raw_key, value in state_dict.items():
        key = raw_key[7:] if raw_key.startswith("module.") else raw_key

        target_key = key
        if target_key.startswith("hardnet_first_stage."):
            target_key = target_key.replace("hardnet_first_stage.", "hardnet_unet_stage.")
        if target_key.startswith("prompt_encoder."):
            target_key = target_key.replace("prompt_encoder.", "prompt_encoder_end.")

        if target_key not in model_state:
            continue

        target = model_state[target_key]

        if value.shape == target.shape:
            filtered_state[target_key] = value
            loaded_keys.append(target_key)
            continue

        if key == "image_encoder.patch_embed.proj.weight":
            if value.ndim == 4 and target.ndim == 4 and value.shape[1] == 3 and target.shape[1] == 1 and value.shape[0] == target.shape[0]:
                filtered_state[target_key] = value.mean(dim=1, keepdim=True)
                loaded_keys.append(target_key)
                continue

        if key == "image_encoder.pos_embed":
            if value.ndim == 4 and target.ndim == 4 and value.shape[0] == target.shape[0] and value.shape[-1] == target.shape[-1]:
                filtered_state[target_key] = _resize_sam_pos_embed(value, target.shape)
                loaded_keys.append(target_key)
                resized_keys.append((target_key, tuple(value.shape), tuple(target.shape)))
                continue

        if _is_global_rel_pos_key(key, encoder_global_attn_indexes):
            if value.ndim == 2 and target.ndim == 2:
                filtered_state[target_key] = _resize_sam_rel_pos(value, tuple(target.shape))
                loaded_keys.append(target_key)
                resized_keys.append((target_key, tuple(value.shape), tuple(target.shape)))
                continue

        skipped_shape.append((target_key, tuple(value.shape), tuple(target.shape)))

    model.load_state_dict(filtered_state, strict=False)
    print(
        f"Loaded UNet-CNN keys: {len(loaded_keys)}, "
        f"Resized: {len(resized_keys)}, Skipped Shape: {len(skipped_shape)}"
    )
    if skipped_shape:
        print("E.g. Skipped keys:", skipped_shape[:5])


def _checkpoint_has_hardnet_weights(checkpoint_path):
    with open(checkpoint_path, "rb") as f:
        state_dict = torch.load(f, weights_only=False)

    if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]

    if not isinstance(state_dict, dict):
        return False

    for raw_key in state_dict.keys():
        key = raw_key[7:] if raw_key.startswith("module.") else raw_key
        if key.startswith("hardnet_first_stage.") or key.startswith("hardnet_unet_stage."):
            return True
    return False


def _build_feat_seg_model_hardnet_unet_cnn(
    img_size,
    iter_2stage,
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    num_classes,
    input_channels=3,
    checkpoint=None,
):
    prompt_embed_dim = 256
    vit_patch_size = 16
    image_embedding_size = img_size // vit_patch_size

    image_encoder = ImageEncoderViT(
        depth=encoder_depth,
        embed_dim=encoder_embed_dim,
        img_size=img_size,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=encoder_num_heads,
        patch_size=vit_patch_size,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=encoder_global_attn_indexes,
        window_size=14,
        out_chans=prompt_embed_dim,
    )
    _adapt_sam_patch_embed_in_channels(image_encoder, input_channels)

    hardnet_pretrained = True
    if checkpoint is not None:
        hardnet_pretrained = not _checkpoint_has_hardnet_weights(checkpoint)

    prompt_encoder_end = PromptEncoder(
        embed_dim=prompt_embed_dim,
        image_embedding_size=(image_embedding_size, image_embedding_size),
        input_image_size=(img_size, img_size),
        mask_in_chans=16,
    )
    prompt_mask_size = prompt_encoder_end.mask_input_size[0]

    hardnet_unet_stage = HardNetUNetHead(
        arch=85,
        pretrained=hardnet_pretrained,
        in_channels=input_channels,
        backbone_input_size=img_size,
        prompt_size=prompt_mask_size,
        out_channels=1 if num_classes == 2 else num_classes,
        freeze_backbone=False,
    )

    seg_decoder_end = SegDecoderCNN(
        num_classes=num_classes,
        num_depth=4,
        p_channel=0,
        promptemd_channel=256,
        first_p=False,
    )

    model = HardNetFeatSegUNetCNN(
        iter_2stage=iter_2stage,
        img_size=img_size,
        image_encoder=image_encoder,
        hardnet_unet_stage=hardnet_unet_stage,
        prompt_encoder_end=prompt_encoder_end,
        seg_decoder_end=seg_decoder_end,
    )

    if checkpoint is not None:
        load_checkpoint_safely_unet_cnn(
            model,
            checkpoint,
            encoder_global_attn_indexes=encoder_global_attn_indexes,
        )

    return model


def build_sam_vit_l_hardnet_unet_cnn(num_classes=2, checkpoint=None, img_size=512, iter_2stage=1, input_channels=1):
    return _build_feat_seg_model_hardnet_unet_cnn(
        img_size=img_size,
        iter_2stage=iter_2stage,
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        num_classes=num_classes,
        input_channels=input_channels,
        checkpoint=checkpoint,
    )


def build_sam_vit_b_hardnet_unet_cnn(num_classes=2, checkpoint=None, img_size=512, iter_2stage=1, input_channels=1):
    return _build_feat_seg_model_hardnet_unet_cnn(
        img_size=img_size,
        iter_2stage=iter_2stage,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        num_classes=num_classes,
        input_channels=input_channels,
        checkpoint=checkpoint,
    )


def build_sam_vit_h_hardnet_unet_cnn(num_classes=2, checkpoint=None, img_size=512, iter_2stage=1, input_channels=1):
    return _build_feat_seg_model_hardnet_unet_cnn(
        img_size=img_size,
        iter_2stage=iter_2stage,
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        num_classes=num_classes,
        input_channels=input_channels,
        checkpoint=checkpoint,
    )


sam_unet_cnn_model_registry = {
    "vit_h_hardnet_unet_cnn": build_sam_vit_h_hardnet_unet_cnn,
    "vit_l_hardnet_unet_cnn": build_sam_vit_l_hardnet_unet_cnn,
    "vit_b_hardnet_unet_cnn": build_sam_vit_b_hardnet_unet_cnn,
}
