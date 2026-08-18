import math

import numpy as np
import timm
import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoModel


def extract_model_embeddings(model, batch_tensors, representation_type='cls'):
    """
    Extracts image embeddings from a TIPSv2 model (local checkpoints or Hugging Face wrappers)
    in a single optimized forward pass without redundant block execution.
    
    Args:
        model: TIPSv2 model instance (local checkpoint or Hugging Face wrapper).
        batch_tensors (torch.Tensor): Preprocessed image batch tensor of shape (B, 3, H, W).
        representation_type (str): Type of representation to extract ('cls', 'avg_patch', or 'cls_avg_patch').
        
    Returns:
        np.ndarray: Extracted feature matrix of shape (B, D) or (B, 2*D).
    """
    # Resolve the underlying vision encoder (nested under vision_encoder for HF wrappers)
    vision_encoder = model.vision_encoder if hasattr(model, 'vision_encoder') else model

    if representation_type in ['avg_patch', 'cls_avg_patch']:
        # Extract both CLS and Value Attention Patch tokens in a single optimized pass
        H, W = batch_tensors.shape[2:]
        P = vision_encoder.patch_size if hasattr(vision_encoder, 'patch_size') else 14
        new_H = math.ceil(H / P) * P
        new_W = math.ceil(W / P) * P
        if (H, W) != (new_H, new_W):
            batch_tensors_resized = F.interpolate(batch_tensors, size=(new_H, new_W), mode='bicubic', align_corners=False)
        else:
            batch_tensors_resized = batch_tensors

        x = vision_encoder.prepare_tokens_with_masks(batch_tensors_resized)
        num_register = getattr(vision_encoder, 'num_register_tokens', 1)
        
        # 1. Forward through first N-1 blocks (out of 12)
        all_blocks = list(vision_encoder.blocks)
        for blk in all_blocks[:-1]:
            x = blk(x)
        
        # 2. Standard forward for the last block to get standard output (CLS token)
        # This is mathematically identical to the default model forward path
        x_standard = all_blocks[-1](x)
        x_standard_norm = vision_encoder.norm(x_standard)
        cls_token = x_standard_norm[:, 0]
        if cls_token.ndim == 3:
            cls_token = cls_token.squeeze(1)
        if hasattr(model, 'head'):
            cls_token = model.head(cls_token)
        cls_token = cls_token.cpu().numpy()

        # 3. Value attention forward for the last block to get patch tokens (MaskCLIP values trick)
        x_normed = all_blocks[-1].norm1(x)
        b_dim, n_dim, c_dim = x_normed.shape
        qkv = (
            all_blocks[-1].attn.qkv(x_normed)
            .reshape(b_dim, n_dim, 3, all_blocks[-1].attn.num_heads, c_dim // all_blocks[-1].attn.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        v = qkv[2]
        v_out = v.transpose(1, 2).reshape(b_dim, n_dim, c_dim)
        v_out = all_blocks[-1].attn.proj(v_out)
        v_out = all_blocks[-1].ls1(v_out)
        x_val = v_out + x

        y_val = all_blocks[-1].norm2(x_val)
        y_val = all_blocks[-1].ls2(all_blocks[-1].mlp(y_val))
        x_val = x_val + y_val

        x_val_norm = vision_encoder.norm(x_val)
        patch_tokens = x_val_norm[:, 1 + num_register:, :]
        avg_patch = np.mean(patch_tokens.cpu().numpy(), axis=1)

        if representation_type == 'cls_avg_patch':
            return np.concatenate([cls_token, avg_patch], axis=1)
        else:
            return avg_patch
    else:
        # Standard fast path (only CLS is requested)
        if hasattr(model, 'encode_image'):
            out = model.encode_image(batch_tensors)
            cls_token = out.cls_token
        else:
            cls_token, _, _ = model(batch_tensors)
        if cls_token.ndim == 3:
            cls_token = cls_token.squeeze(1)
        return cls_token.cpu().numpy()


def extract_benchmark_features_single_pass(model, batch_tensors, is_local=False):
    """
    Extracts standard CLS tokens (and second CLS token if local) and the MaskCLIP 
    value attention patch tokens in a single optimized forward pass.
    
    Args:
        model: TIPSv2 model instance, local checkpoint, or standard timm/HF model.
        batch_tensors (torch.Tensor): Preprocessed image batch tensor of shape (B, 3, H, W).
        is_local (bool): True if using local check-pointed ImageEncoder, False if Hugging Face.
        
    Returns:
        tuple: (cls_out, patch_tokens_vals) where:
            - cls_out is either (first_cls, second_cls) or a single cls_token array.
            - patch_tokens_vals is the MaskCLIP value attention patch tokens array.
    """
    # Case 1: timm Model Integration
    if not is_local and hasattr(model, 'forward_features'):
        return _extract_features_timm(model, batch_tensors)

    # Case 2: TIPSv2 model (local or Hugging Face wrapper)
    return _extract_features_tipsv2(model, batch_tensors, is_local)


def _extract_features_timm(model, batch_tensors):
    """Extracts features from a timm model."""
    features = model.forward_features(batch_tensors)
    
    if features.ndim == 4:
        # CNN output: pool spatial dims for CLS, reshape for patches
        cls_out = torch.nn.functional.adaptive_avg_pool2d(features, 1).flatten(1).cpu().numpy()
        b, c, h, w = features.shape
        patch_tokens_vals = features.permute(0, 2, 3, 1).reshape(b, h*w, c).cpu().numpy()
    else:
        # Transformer output: token 0 is CLS, tokens after prefix are patches
        cls_out = features[:, 0].cpu().numpy()
        num_prefix = getattr(model, 'num_prefix_tokens', 1)
        patch_tokens_vals = features[:, num_prefix:].cpu().numpy()
    return cls_out, patch_tokens_vals


def _extract_features_tipsv2(model, batch_tensors, is_local):
    """Extracts features from a TIPSv2 model (local or HF)."""
    vision_encoder = model if is_local else (model.vision_encoder if hasattr(model, 'vision_encoder') else model)
    
    # 1. Prepare tokens
    x = vision_encoder.prepare_tokens_with_masks(batch_tensors)
    num_register = getattr(vision_encoder, 'num_register_tokens', 1)
    
    # 2. Forward through first N-1 blocks
    all_blocks = list(vision_encoder.blocks)
    for blk in all_blocks[:-1]:
        x = blk(x)
        
    # 3. Last block standard output (CLS token)
    x_standard = all_blocks[-1](x)
    x_standard_norm = vision_encoder.norm(x_standard)
    
    if is_local:
        first_cls = model.head(x_standard_norm[:, :1])
        if first_cls.ndim == 3:
            first_cls = first_cls.squeeze(1)
        first_cls = first_cls.cpu().numpy()
        
        second_cls = model.head(x_standard_norm[:, 1 : num_register + 1])
        if second_cls.ndim == 3:
            second_cls = second_cls.squeeze(1)
        second_cls = second_cls.cpu().numpy()
        
        cls_out = (first_cls, second_cls)
    else:
        cls_token = x_standard_norm[:, 0]
        if cls_token.ndim == 3:
            cls_token = cls_token.squeeze(1)
        cls_out = cls_token.cpu().numpy()
        
    # 4. Last block value attention projection (MaskCLIP values trick)
    x_normed = all_blocks[-1].norm1(x)
    b_dim, n_dim, c_dim = x_normed.shape
    qkv = (
        all_blocks[-1].attn.qkv(x_normed)
        .reshape(b_dim, n_dim, 3, all_blocks[-1].attn.num_heads, c_dim // all_blocks[-1].attn.num_heads)
        .permute(2, 0, 3, 1, 4)
    )
    v = qkv[2]
    v_out = v.transpose(1, 2).reshape(b_dim, n_dim, c_dim)
    v_out = all_blocks[-1].attn.proj(v_out)
    v_out = all_blocks[-1].ls1(v_out)
    x_val = v_out + x

    y_val = all_blocks[-1].norm2(x_val)
    y_val = all_blocks[-1].ls2(all_blocks[-1].mlp(y_val))
    x_val = x_val + y_val

    x_val_norm = vision_encoder.norm(x_val)
    patch_tokens = x_val_norm[:, 1 + num_register:, :]
    patch_tokens_vals = patch_tokens.cpu().numpy()
    
    return cls_out, patch_tokens_vals


def load_vision_model(model_name, device):
    """
    Loads a vision model (timm or Hugging Face AutoModel) and resolves its required 
    image size and transform.
    
    Args:
        model_name (str): Hugging Face identifier or timm model name.
        device (torch.device): Device to load the model on.
        
    Returns:
        tuple: (model, transform, image_size)
    """
    
    is_timm = False
    try:
        is_timm = model_name in timm.list_models() or any(
            x in model_name for x in ['vit_', 'resnet', 'efficientnet', 'convnext', 'swin']
        )
    except Exception:
        pass
        
    if is_timm:
        print(f"Loading timm model: {model_name}...")
        model = timm.create_model(model_name, pretrained=True).eval().to(device)
        
        # Resolve data config and create transform
        data_config = timm.data.resolve_model_data_config(model)
        timm_transform = timm.data.create_transform(**data_config, is_training=False)
        
        # Get required image size (timm uses a 3-element tuple (C, H, W))
        input_size = data_config.get('input_size', (3, 224, 224))
        image_size = input_size[1]
        
        return model, timm_transform, image_size
    else:
        # Load TIPSv2 (local or Hugging Face wrapper)
        print(f"Loading TIPSv2 model: {model_name}...")
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).eval().to(device)
        image_size = 448
        hf_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        return model, hf_transform, image_size

