import math

import numpy as np
import torch
import torch.nn.functional as F


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
        out = model.encode_image(batch_tensors)
        cls_token = out.cls_token
        if cls_token.ndim == 3:
            cls_token = cls_token.squeeze(1)
        return cls_token.cpu().numpy()
