# -*- coding: utf-8 -*-
# === Embedding-level SCoPE (CoPE + HKPool 残差门控) ===
# 改 embedding 阶段，不动 attention
# HKPool 仅使用 max pooling，不包含卷积

import torch
from torch import nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# ------------------------------------------------------------------
# HKPool：MaxPool-based 门控模块
# ------------------------------------------------------------------
class HKPool(nn.Module):
    """Hybrid Kernel Pool (无卷积版)
       多层 MaxPool 生成语义门控信号 [B, N]
    """
    def __init__(self):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max4 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        # 全局对比增强
        mu = self.global_avgpool(x)
        delta = x - mu
        x = x + (1 + self.alpha) * delta

        # 层叠 MaxPool：2×2 → 2×2 → 4×4
        x = self.max4(x)
        x = self.max4(x)
        x = self.max2(x)

        # 通道平均，flatten 成 [B, N]
        x = x.mean(dim=1)          # [B, H, W]
        return x.flatten(1)        # [B, N]


# ------------------------------------------------------------------
# CoPE Embedding：动态位置偏移模块（embedding-level）
# ------------------------------------------------------------------
class CoPEEmbedding(nn.Module):
    """动态偏移位置编码: 由 token 内容生成 ΔP"""
    def __init__(self, npos_max, dim):
        super().__init__()
        self.npos_max = npos_max
        self.dim = dim
        self.pos_table = nn.Parameter(torch.randn(1, npos_max, dim) * 0.02)

    def forward(self, x):
        B, N, C = x.shape
        gate = torch.sigmoid(x.mean(dim=-1))                    # [B,N]
        pos = gate * (self.npos_max - 1)
        pos_floor = pos.floor().long()
        pos_ceil  = pos.ceil().long()
        w = (pos - pos_floor).unsqueeze(-1)
        tbl = self.pos_table[0]                                 # [N,C]
        emb_f = tbl.index_select(0, pos_floor.reshape(-1)).view(B, N, C)
        emb_c = tbl.index_select(0, pos_ceil.reshape(-1)).view(B, N, C)
        cope_offset = emb_f * (1 - w) + emb_c * w               # [B,N,C]
        return cope_offset, gate


# ------------------------------------------------------------------
# Transformer 模块（保持标准 ViT）
# ------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, dim, mlp_hidden, drop=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, mlp_hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_hidden, dim)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc2(self.drop(self.act(self.fc1(x))))
        return self.drop(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads, drop=0.1):
        super().__init__()
        self.heads = heads
        self.dk = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv
        attn = (q @ k.transpose(-2, -1)) * (self.dk ** -0.5)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.drop(self.proj(out))


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ------------------------------------------------------------------
# ViT-SCoPE Embedding-level 主体
# ------------------------------------------------------------------
class ViTSCoPE_Embed(nn.Module):
    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 num_classes=1000,
                 dim=192,
                 depth=12,
                 heads=3,
                 mlp_dim=768,
                 drop=0.1,
                 emb_drop=0.1,
                 use_cls=True):
        super().__init__()
        assert image_size % patch_size == 0
        self.h = self.w = image_size // patch_size
        num_patches = self.h * self.w
        patch_dim = 3 * patch_size * patch_size
        self.use_cls = use_cls

        # patch embedding
        self.to_patch = nn.Sequential(
            Rearrange('b c (h p1)(w p2)->b (h w)(p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim)
        )

        # 固定位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # 动态偏移与 HKPool
        self.cope_emb = CoPEEmbedding(num_patches, dim)
        self.hkpool = HKPool()
        self.alpha = nn.Parameter(torch.tensor(0.1))

        # 可选 CLS
        if use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

        # Transformer backbone
        self.drop = nn.Dropout(emb_drop)
        self.blocks = nn.Sequential(*[
            TransformerBlock(dim, heads, mlp_dim / dim, drop) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)

    def forward(self, img):
        B = img.size(0)
        x = self.to_patch(img)                   # [B,N,C]

        cope_offset, cope_gate = self.cope_emb(x)       # [B,N,C], [B,N]
        hk_gate = self.hkpool(img)                      # [B,N]

        # 残差门控融合（embedding阶段执行）
        fused_gate = cope_gate + (1 + self.alpha) * (hk_gate - cope_gate)
        fused_gate = torch.sigmoid(fused_gate).unsqueeze(-1)  # [B,N,1]

        # 最终 token embedding
        x = x + self.pos_embed + cope_offset * fused_gate

        # CLS token
        if self.use_cls:
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)

        x = self.drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        out = x[:, 0] if self.use_cls else x.mean(dim=1)
        return self.head(out)
