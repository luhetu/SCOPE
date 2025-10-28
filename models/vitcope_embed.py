# -*- coding: utf-8 -*-
# Pure Embedding-level CoPE for ViT (CLS + no fixed pos)
# 仅在 embedding 阶段使用 CoPE 动态位置编码，Attention 不改动。

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

# ======================================================
# 动态位置编码模块（CoPE）
# ======================================================
class CoPEEmbedding(nn.Module):
    """根据 token 内容生成动态位置嵌入 ΔP（不含固定索引）"""
    def __init__(self, npos_max: int, dim: int):
        super().__init__()
        self.npos_max = int(npos_max)
        self.dim = int(dim)
        # 可学习查表，用于插值生成偏移向量
        self.pos_table = nn.Parameter(torch.randn(1, self.npos_max, self.dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, C]
        输出: 动态位置嵌入 ΔP [B, N, C]
        """
        B, N, C = x.shape
        # 简单 gate：内容均值后 sigmoid
        gate = torch.sigmoid(x.mean(dim=-1))              # [B, N]
        pos = (gate * (self.npos_max - 1)).clamp(0, self.npos_max - 1)
        f = pos.floor().long()
        c = pos.ceil().long()
        w = (pos - f).unsqueeze(-1)                       # [B, N, 1]
        tbl = self.pos_table[0]                           # [npos_max, C]
        emb_f = tbl.index_select(0, f.reshape(-1)).view(B, N, C)
        emb_c = tbl.index_select(0, c.reshape(-1)).view(B, N, C)
        cope = emb_f * (1 - w) + emb_c * w
        return cope                                       # [B, N, C]


# ======================================================
# 标准 Transformer 模块（未修改）
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

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, drop: float = 0.1):
        super().__init__()
        assert dim % heads == 0
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
        out  = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.drop(self.proj(out))

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, drop: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = MultiHeadSelfAttention(dim, heads, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = MLP(dim, int(dim * mlp_ratio), drop)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ======================================================
# ViT 主干 (Pure CoPE + CLS)
# ======================================================
class ViTCoPE(nn.Module):
    """ViT + 纯动态 CoPE（无固定位置索引）+ CLS token"""
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
        patch_dim   = 3 * patch_size * patch_size

        # 1) Patch Embedding
        self.to_patch = nn.Sequential(
            Rearrange('b c (h p1)(w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim)
        )

        # 2) 动态位置编码（唯一位置建模方式）
        self.cope_emb = CoPEEmbedding(num_patches, dim)

        # 3) CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # 4) Dropout + Transformer 堆叠
        self.drop = nn.Dropout(emb_drop)
        self.blocks = nn.Sequential(*[
            TransformerBlock(dim, heads, mlp_dim / dim, drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

        # 5) 分类头
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        B = img.size(0)
        # patch embedding
        x = self.to_patch(img)                    # [B, N, C]

        # 动态位置偏移（仅作用于 patch tokens）
        cope_embed = self.cope_emb(x)             # [B, N, C]
        x = x + cope_embed                        # ✅ 纯动态位置建模

        # 添加 CLS token（不加 CoPE）
        cls = self.cls_token.expand(B, -1, -1)    # [B, 1, C]
        x = torch.cat([cls, x], dim=1)            # [B, 1+N, C]

        # Transformer + 分类头
        x = self.drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        cls_feat = x[:, 0]                        # [B, C]
        return self.head(cls_feat)


