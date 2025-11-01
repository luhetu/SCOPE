# -*- coding: utf-8 -*-
# CoPE for ViT: 在 Attention Logits 上加偏置（与用户代码一致）
# CoPE adds bias to attention logits (same as user-provided code)

import torch
import torch.nn as nn
import math
from einops.layers.torch import Rearrange

# ======================================================
# CoPE: 在 Attention Logits 上加偏置（与用户代码完全一致）
# ======================================================
class CoPE(nn.Module):
    """
    CoPE Implementation: Adds bias to attention logits
    Exactly matches user-provided code structure
    """
    def __init__(self, npos_max, head_dim):
        super().__init__()
        self.npos_max = npos_max
        self.head_dim = head_dim
        # pos_emb: [1, head_dim, npos_max]
        self.pos_emb = nn.Parameter(torch.zeros(1, head_dim, npos_max))
        nn.init.xavier_uniform_(self.pos_emb)

    def forward(self, query, attn_logits):
        """
        Args:
            query: [B, H, N, head_dim] - query embeddings
            attn_logits: [B, H, N, N] - attention logits (before softmax)
        Returns:
            bias: [B, H, N, N] - bias to add to attention logits
        """
        B, H, N, D = query.shape
        
        # Memory-efficient implementation: process per query position
        # Compute logits_int once (can be reused)
        pos_emb_matrix = self.pos_emb[0]  # [head_dim, npos_max]
        logits_int = torch.matmul(query, pos_emb_matrix)  # [B, H, N, npos_max]
        
        # Process each query position separately to save memory
        bias_output = []
        for i in range(N):
            # For query position i, process all key positions
            attn_logits_i = attn_logits[:, :, i, :]  # [B, H, N]
            gates_i = torch.sigmoid(attn_logits_i)  # [B, H, N]
            
            # Flip-cumsum-flip on the key dimension
            pos_i = gates_i.flip(-1).cumsum(dim=-1).flip(-1)  # [B, H, N]
            pos_i = pos_i.clamp(max=self.npos_max - 1)
            
            # Interpolate positions
            pos_ceil_i = pos_i.ceil().long().clamp(0, self.npos_max - 1)  # [B, H, N]
            pos_floor_i = pos_i.floor().long().clamp(0, self.npos_max - 1)  # [B, H, N]
            w_i = pos_i - pos_floor_i  # [B, H, N]
            
            # Gather from logits_int for this query position
            logits_i = logits_int[:, :, i, :]  # [B, H, npos_max]
            
            logits_ceil_i = torch.gather(logits_i, dim=2, index=pos_ceil_i)  # [B, H, N]
            logits_floor_i = torch.gather(logits_i, dim=2, index=pos_floor_i)  # [B, H, N]
            
            # Interpolate
            bias_i = logits_ceil_i * w_i + logits_floor_i * (1 - w_i)  # [B, H, N]
            bias_output.append(bias_i)
        
        # Stack: [B, H, N] * N -> [B, H, N, N]
        bias = torch.stack(bias_output, dim=2)  # [B, H, N, N]
        return bias

# ======================================================
# Multi-Head Self Attention with CoPE
# ======================================================
class MultiHeadSelfAttention(nn.Module):
    """Self Attention with CoPE (adds bias to attention logits)"""
    def __init__(self, dim: int, heads: int, num_patches: int, drop: float = 0.1):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.cope = CoPE(npos_max=num_patches + 1, head_dim=self.head_dim)  # +1 for CLS token
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        """
        Args:
            x: [B, N, C] - input tokens (with CLS token)
        """
        B, N, C = x.shape
        
        # Compute QKV
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv  # Each: [B, H, N, head_dim]
        
        # Compute attention logits
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, N, N]
        
        # Add CoPE bias to logits (exactly as user code: attn_logits += self.cope(query, attn_logits))
        cope_bias = self.cope(q, attn_logits)  # [B, H, N, N]
        attn_logits = attn_logits + cope_bias  # ✅ 在 logits 上加偏置
        
        # Apply softmax
        attn = attn_logits.softmax(dim=-1)  # [B, H, N, N]
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # [B, H, N, head_dim]
        out = out.transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        return self.drop(self.proj(out))

# ======================================================
# Standard MLP
# ======================================================
class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int, drop: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x

# ======================================================
# Transformer Block with CoPE
# ======================================================
class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, num_patches: int, mlp_ratio: float = 4.0, drop: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads, num_patches, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# ======================================================
# ViT 主干 (CoPE in Attention Logits + CLS)
# ======================================================
class ViTCoPE(nn.Module):
    """
    ViT + CoPE (在 Attention Logits 上加偏置)
    - CoPE adds bias to attention logits (same as user-provided code)
    - 强制使用 CLS Token
    """
    def __init__(self,
                 image_size: int = 224,
                 patch_size: int = 16,
                 num_classes: int = 1000,
                 dim: int = 192,
                 depth: int = 12,
                 heads: int = 3,
                 mlp_dim: int = 768,
                 drop: float = 0.1,
                 emb_drop: float = 0.1):
        super().__init__()
        assert image_size % patch_size == 0
        self.h = self.w = image_size // patch_size
        num_patches = self.h * self.w
        patch_dim = 3 * patch_size * patch_size

        # 1) Patch Embedding
        self.to_patch = nn.Sequential(
            Rearrange('b c (h p1)(w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim)
        )

        # 2) CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # 3) Dropout + Transformer 堆叠 (CoPE 在每个 Attention 层中)
        self.drop = nn.Dropout(emb_drop)
        self.blocks = nn.Sequential(*[
            TransformerBlock(dim, heads, num_patches, mlp_dim / dim, drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

        # 4) 分类头
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        B = img.size(0)
        
        # Patch embedding
        x = self.to_patch(img)  # [B, N, C]

        # 添加 CLS token
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, C]
        x = torch.cat([cls, x], dim=1)  # [B, 1+N, C]

        # Transformer with CoPE (CoPE 在 Attention Logits 上加偏置)
        x = self.drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        cls_feat = x[:, 0]  # [B, C] - CLS token
        return self.head(cls_feat)
