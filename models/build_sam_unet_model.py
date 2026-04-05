import torch
import torch.nn.functional as F

from functools import partial

from .hardnet_feat_seg_unet import HardNetFeatSegUNet
from .hardnet_unet_head import HardNetUNetHead
from .sam_unet_decoder import SamUNetDecoder

from segment_anything.modeling import ImageEncoderViT, PromptEncoder, TwoWayTransformer, MaskDecoder
from .build_sam_feat_seg_model import _adapt_sam_patch_embed_in_channels, _resize_sam_pos_embed, _is_global_rel_pos_key, _resize_sam_rel_pos


def _load_checkpoint_safely_unet(model, checkpoint_path, encoder_global_attn_indexes):
    """Load matching tensors from SAM/HardNet checkpoints to the new UNet architecture."""
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
        # Checkpoint da DDP possono avere il prefisso "module.".
        # Lo normalizziamo per confrontarlo con i nomi dei parametri del modello.
        key = raw_key[7:] if raw_key.startswith("module.") else raw_key

        # Mappa i pesi originali di SAM al nuovo SamUNetDecoder
        target_key = key
        if target_key.startswith("mask_decoder."):
            target_key = "sam_unet_decoder." + target_key
            
        # Mappa i pesi originali della HardNetSegmentationHead alla nuova HardNetUNetHead
        if target_key.startswith("hardnet_first_stage."):
            target_key = target_key.replace("hardnet_first_stage.", "hardnet_unet_stage.")

        if target_key not in model_state:
            continue

        target = model_state[target_key]
        
        # Saltare gli eventuali pesi lineari che sono totalmente cambiati in dimensioni
        if value.shape == target.shape:
            filtered_state[target_key] = value
            loaded_keys.append(target_key)
            continue

        # Special case: SAM patch embedding 3ch -> 1ch
        if key == "image_encoder.patch_embed.proj.weight":
            if value.ndim == 4 and target.ndim == 4 and value.shape[1] == 3 and target.shape[1] == 1 and value.shape[0] == target.shape[0]:
                filtered_state[target_key] = value.mean(dim=1, keepdim=True)
                loaded_keys.append(target_key)
                continue

        # Special case: SAM absolute positional embedding
        if key == "image_encoder.pos_embed":
            if value.ndim == 4 and target.ndim == 4 and value.shape[0] == target.shape[0] and value.shape[-1] == target.shape[-1]:
                filtered_state[target_key] = _resize_sam_pos_embed(value, target.shape)
                loaded_keys.append(target_key)
                resized_keys.append((target_key, tuple(value.shape), tuple(target.shape)))
                continue

        # Special case: SAM global-attention
        if _is_global_rel_pos_key(key, encoder_global_attn_indexes):
            if value.ndim == 2 and target.ndim == 2:
                filtered_state[target_key] = _resize_sam_rel_pos(value, tuple(target.shape))
                loaded_keys.append(target_key)
                resized_keys.append((target_key, tuple(value.shape), tuple(target.shape)))
                continue

        skipped_shape.append((target_key, tuple(value.shape), tuple(target.shape)))

    model.load_state_dict(filtered_state, strict=False)
    print(f"Loaded UNet keys: {len(loaded_keys)}, Resized: {len(resized_keys)}, Skipped Shape: {len(skipped_shape)}")
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


def _build_feat_seg_model_hardnet_unet(
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

    # Image Encoder SAM
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

    # SAM Prompt Encoder
    prompt_encoder = PromptEncoder(
        embed_dim=prompt_embed_dim,
        image_embedding_size=(image_embedding_size, image_embedding_size),
        input_image_size=(img_size, img_size),
        mask_in_chans=16,
    )
    # Keep stage-1 mask logits resolution aligned with SAM mask prompt resolution.
    # For SAM: mask_input_size = 4 * image_embedding_size = img_size // 4.
    prompt_mask_size = prompt_encoder.mask_input_size[0]

    # SAM Mask Decoder (incorporato in SamUNetDecoder)
    mask_decoder = MaskDecoder(
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=prompt_embed_dim,
            mlp_dim=2048,
            num_heads=8,
        ),
        transformer_dim=prompt_embed_dim,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
    )
    sam_unet_dec = SamUNetDecoder(mask_decoder=mask_decoder)

    # HardNet U-Net Head
    hardnet_unet_stage = HardNetUNetHead(
        arch=85,
        pretrained=hardnet_pretrained,
        in_channels=input_channels,
        backbone_input_size=img_size,
        prompt_size=prompt_mask_size,
        out_channels=1 if num_classes == 2 else num_classes,
        freeze_backbone=False,
    )

    # Top Level Model
    sam_seg = HardNetFeatSegUNet(
        iter_2stage=iter_2stage,
        img_size=img_size,
        image_encoder=image_encoder,
        hardnet_unet_stage=hardnet_unet_stage,
        prompt_encoder=prompt_encoder,
        sam_unet_decoder=sam_unet_dec,
    )

    if checkpoint is not None:
        _load_checkpoint_safely_unet(
            sam_seg,
            checkpoint,
            encoder_global_attn_indexes=encoder_global_attn_indexes,
        )
    return sam_seg


def build_sam_vit_l_hardnet_unet(num_classes=2, checkpoint=None, img_size=512, iter_2stage=1, input_channels=1):
    return _build_feat_seg_model_hardnet_unet(
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


def build_sam_vit_b_hardnet_unet(num_classes=2, checkpoint=None, img_size=512, iter_2stage=1, input_channels=1):
    return _build_feat_seg_model_hardnet_unet(
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


def build_sam_vit_h_hardnet_unet(num_classes=2, checkpoint=None, img_size=512, iter_2stage=1, input_channels=1):
    return _build_feat_seg_model_hardnet_unet(
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


sam_unet_model_registry = {
    "vit_h_hardnet_unet": build_sam_vit_h_hardnet_unet,
    "vit_l_hardnet_unet": build_sam_vit_l_hardnet_unet,
    "vit_b_hardnet_unet": build_sam_vit_b_hardnet_unet,
}
