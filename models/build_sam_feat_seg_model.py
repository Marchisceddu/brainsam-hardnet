import torch
import torch.nn.functional as F

from functools import partial

from .SamFeatSeg import SamFeatSeg, SegDecoderCNN, UPromptCNN
from .hardnet_segmentation_head import HardNetSegmentationHead
from .hardnet_feat_seg import HardNetFeatSeg
# from .sam_decoder import MaskDecoder
from segment_anything.modeling import ImageEncoderViT, PromptEncoder,TwoWayTransformer


def _resize_sam_pos_embed(pos_embed, target_shape):
    """Resize SAM absolute positional embedding [1,H,W,C] to target [1,Ht,Wt,C]."""
    _, target_h, target_w, _ = target_shape
    pos_embed = pos_embed.permute(0, 3, 1, 2)
    pos_embed = F.interpolate(
        pos_embed,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    )
    return pos_embed.permute(0, 2, 3, 1)


def _resize_sam_rel_pos(rel_pos, target_shape):
    """Resize SAM relative positional embedding [L,D] to target [Lt,Dt]."""
    rel_pos = rel_pos.unsqueeze(0).unsqueeze(0)
    rel_pos = F.interpolate(
        rel_pos,
        size=target_shape,
        mode="bilinear",
        align_corners=False,
    )
    return rel_pos[0, 0, ...]


def _is_global_rel_pos_key(key, encoder_global_attn_indexes):
    if "image_encoder.blocks." not in key or ".attn.rel_pos" not in key:
        return False
    try:
        block_idx = int(key.split(".")[2])
    except (IndexError, ValueError):
        return False
    return block_idx in encoder_global_attn_indexes


def _adapt_sam_patch_embed_in_channels(image_encoder, input_channels):
    if input_channels == 3:
        return
    if input_channels != 1:
        raise ValueError(f"Unsupported input_channels={input_channels}. Use 1 or 3.")

    old_proj = image_encoder.patch_embed.proj
    new_proj = torch.nn.Conv2d(
        1,
        old_proj.out_channels,
        kernel_size=old_proj.kernel_size,
        stride=old_proj.stride,
        padding=old_proj.padding,
        bias=old_proj.bias is not None,
    )
    new_proj.weight.data = old_proj.weight.data.mean(dim=1, keepdim=True)
    if old_proj.bias is not None:
        new_proj.bias.data = old_proj.bias.data
    image_encoder.patch_embed.proj = new_proj


def _load_checkpoint_safely(model, checkpoint_path, encoder_global_attn_indexes):
    """Load only compatible checkpoint tensors; adapt SAM patch-embed 3->1 when needed."""
    with open(checkpoint_path, "rb") as f:
        state_dict = torch.load(f, weights_only=False)

    if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]

    model_state = model.state_dict()
    filtered_state = {}
    loaded_keys = []
    resized_keys = []
    skipped_shape = []

    for key, value in state_dict.items():
        if key not in model_state:
            continue

        target = model_state[key]
        if value.shape == target.shape:
            filtered_state[key] = value
            loaded_keys.append(key)
            continue

        # Special case: SAM patch embedding 3ch checkpoint -> 1ch model.
        if key == "image_encoder.patch_embed.proj.weight":
            if value.ndim == 4 and target.ndim == 4 and value.shape[1] == 3 and target.shape[1] == 1 and value.shape[0] == target.shape[0]:
                filtered_state[key] = value.mean(dim=1, keepdim=True)
                loaded_keys.append(key)
                continue

        # Special case: SAM absolute positional embedding interpolation.
        if key == "image_encoder.pos_embed":
            if value.ndim == 4 and target.ndim == 4 and value.shape[0] == target.shape[0] and value.shape[-1] == target.shape[-1]:
                filtered_state[key] = _resize_sam_pos_embed(value, target.shape)
                loaded_keys.append(key)
                resized_keys.append((key, tuple(value.shape), tuple(target.shape)))
                continue

        # Special case: SAM global-attention relative positional embeddings.
        if _is_global_rel_pos_key(key, encoder_global_attn_indexes):
            if value.ndim == 2 and target.ndim == 2:
                filtered_state[key] = _resize_sam_rel_pos(value, tuple(target.shape))
                loaded_keys.append(key)
                resized_keys.append((key, tuple(value.shape), tuple(target.shape)))
                continue

        skipped_shape.append((key, tuple(value.shape), tuple(target.shape)))

    model.load_state_dict(filtered_state, strict=False)
    print(
        f"load keys over! loaded={len(loaded_keys)} "
        f"resized={len(resized_keys)} skipped_shape={len(skipped_shape)}"
    )
    if resized_keys:
        print("Resized keys example:", resized_keys[:5])
    if skipped_shape:
        print("Skipped (shape mismatch) keys example:", skipped_shape[:5])


def _checkpoint_has_hardnet_weights(checkpoint_path):
    """Return True if the checkpoint contains HardNet-first-stage weights."""
    with open(checkpoint_path, "rb") as f:
        state_dict = torch.load(f, weights_only=False)

    if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]

    if not isinstance(state_dict, dict):
        return False

    for key in state_dict.keys():
        if key.startswith("hardnet_first_stage.backbone."):
            return True
    return False


def _build_feat_seg_model(
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

    sam_seg = SamFeatSeg(
        iter_2stage=iter_2stage,
        img_size=img_size,
        image_encoder=image_encoder,
        promptcnn_first=UPromptCNN(n_channels=input_channels),
        seg_decoder_first=SegDecoderCNN(num_classes=num_classes, num_depth=4, p_channel=3, promptemd_channel=0, first_p=True),

        prompt_encoder_end=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(img_size, img_size),
            mask_in_chans=16,
        ),
        seg_decoder_end=SegDecoderCNN(num_classes=num_classes, num_depth=4, p_channel=0, promptemd_channel=256, first_p=False),
    )

    if checkpoint is not None:
        _load_checkpoint_safely(
            sam_seg,
            checkpoint,
            encoder_global_attn_indexes=encoder_global_attn_indexes,
        )
    return sam_seg


def _build_feat_seg_model_hardnet(
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
        # If checkpoint already contains hardnet weights, avoid extra ImageNet download
        # and avoid initializing then immediately overriding those same tensors.
        hardnet_pretrained = not _checkpoint_has_hardnet_weights(checkpoint)

    sam_seg = HardNetFeatSeg(
        iter_2stage=iter_2stage,
        img_size=img_size,
        image_encoder=image_encoder,
        hardnet_first_stage=HardNetSegmentationHead(
            arch=85,
            pretrained=hardnet_pretrained,
            in_channels=input_channels,
            backbone_input_size=512,
            prompt_size=256,
            out_channels=1,
            freeze_backbone=False,
        ),
        prompt_encoder_end=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(img_size, img_size),
            mask_in_chans=16,
        ),
        seg_decoder_end=SegDecoderCNN(
            num_classes=num_classes,
            num_depth=4,
            p_channel=0,
            promptemd_channel=256,
            first_p=False,
        ),
    )

    if checkpoint is not None:
        _load_checkpoint_safely(
            sam_seg,
            checkpoint,
            encoder_global_attn_indexes=encoder_global_attn_indexes,
        )
    return sam_seg


def build_sam_vit_h_seg_cnn(num_classes=2, checkpoint=None, img_size=320, iter_2stage=1, input_channels=3):
    return _build_feat_seg_model(
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


build_sam_seg = build_sam_vit_h_seg_cnn


def build_sam_vit_l_seg_cnn(num_classes=2, checkpoint=None, img_size=320, iter_2stage=1, input_channels=3):
    return _build_feat_seg_model(
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


def build_sam_vit_l_hardnet_seg_cnn(num_classes=2, checkpoint=None, img_size=320, iter_2stage=1, input_channels=1):
    return _build_feat_seg_model_hardnet(
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


def build_sam_vit_b_hardnet_seg_cnn(num_classes=2, checkpoint=None, img_size=320, iter_2stage=1, input_channels=1):
    return _build_feat_seg_model_hardnet(
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


def build_sam_vit_h_hardnet_seg_cnn(num_classes=2, checkpoint=None, img_size=320, iter_2stage=1, input_channels=1):
    return _build_feat_seg_model_hardnet(
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


def build_sam_vit_b_seg_cnn(num_classes=2, checkpoint=None, img_size=320, iter_2stage=1, input_channels=3):
    return _build_feat_seg_model(
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


sam_feat_seg_model_registry = {
    "default": build_sam_seg,
    "vit_h": build_sam_seg,
    "vit_l": build_sam_vit_l_seg_cnn,
    "vit_h_hardnet": build_sam_vit_h_hardnet_seg_cnn,
    "vit_l_hardnet": build_sam_vit_l_hardnet_seg_cnn,
    "vit_b_hardnet": build_sam_vit_b_hardnet_seg_cnn,
    "vit_b": build_sam_vit_b_seg_cnn,
}

