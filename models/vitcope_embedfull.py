# -*- coding: utf-8 -*-
# Embedding-level CoPE for ViT (fixed + dynamic pos)
# 对比标准 ViT: 仅修改 embedding 阶段, Transformer 未改动

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

# ---------------- 动态位置编码 ----------------
class CoPEEmbedding(nn.Module):
    """动态偏移位置编码: 由token内容生成 ΔP"""
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
        pos_ceil = pos.ceil().long()
        w = (pos - pos_floor).unsqueeze(-1)
        tbl = self.pos_table[0]                                 # [N,C]
        emb_f = tbl.index_select(0, pos_floor.reshape(-1)).view(B, N, C)
        emb_c = tbl.index_select(0, pos_ceil.reshape(-1)).view(B, N, C)
        cope_offset = emb_f * (1 - w) + emb_c * w               # [B,N,C]
        return cope_offset


# ---------------- Transformer Block ----------------
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


# ---------------- ViT 主干 ----------------
class ViTCoPE_EmbedFull(nn.Module):
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

        # 动态位置偏移
        self.cope_emb = CoPEEmbedding(num_patches, dim)

        # 可选 CLS
        if use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

        self.drop = nn.Dropout(emb_drop)
        self.blocks = nn.Sequential(*[
            TransformerBlock(dim, heads, mlp_dim / dim, drop) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)

    def forward(self, img):
        B = img.size(0)
        x = self.to_patch(img)                        # [B,N,C]
        cope_offset = self.cope_emb(x)                # [B,N,C]
        x = x + self.pos_embed + cope_offset          # 固定+动态位置编码
        if self.use_cls:
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)
        x = self.drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        out = x[:, 0] if self.use_cls else x.mean(dim=1)
        return self.head(out)


# ---------------- 测试 ----------------
if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    net = ViTCoPE_EmbedFull()
    y = net(x)
    print("logits:", y.shape)  # [2,1000]
