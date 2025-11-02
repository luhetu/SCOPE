# -*- coding: utf-8 -*-
# ViT-Tiny + CoPE（paper-correct：对注意力 logits 加偏置，保留 CLS）
# 仅加入 Top-k 稀疏门控（不增参），不使用 CB，不做分块
# 默认：image_size=224, patch=16, dim=192, depth=12, heads=3, dim_head=64, num_classes=1000

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

# ---------------- helpers ----------------
def pair(v):
    return v if isinstance(v, tuple) else (v, v)

# ---------------- CoPE: attention-level (add-to-logits) ----------------
class CoPE(nn.Module):
    """
    输入:
      q:           [B, H, N, D]
      attn_logits: [B, H, N, N]  （此张量仅用于 gates 计算）
    输出:
      bias:        [B, H, N, N]  （直接加到最终的注意力 logits 上）
    """
    def __init__(self, npos_max: int, head_dim: int):
        super().__init__()
        self.npos_max = int(npos_max)
        self.head_dim = int(head_dim)
        # pos_emb: [1, D, L]
        self.pos_emb = nn.Parameter(torch.zeros(1, self.head_dim, self.npos_max))
        nn.init.xavier_uniform_(self.pos_emb)

    @torch.no_grad()
    def _resize_pos_len_(self, cur_len: int):
        if self.pos_emb.shape[-1] == cur_len:
            return
        pe = F.interpolate(self.pos_emb, size=cur_len, mode='linear', align_corners=False)
        self.pos_emb.data.copy_(pe)

    def forward(self, q: torch.Tensor, attn_logits: torch.Tensor) -> torch.Tensor:
        # q: [B,H,N,D], attn_logits: [B,H,N,N]
        B, H, N, D = q.shape
        self._resize_pos_len_(N)
        L = self.pos_emb.shape[-1]  # == N（含 CLS）

        # 1) 二维门控（pairwise）
        gates = torch.sigmoid(attn_logits)                 # [B,H,N,N]

        # 2) 累积得到动态位置（flip-cumsum-flip）
        pos = gates.flip(-1).cumsum(dim=-1).flip(-1)
        pos = pos.clamp(max=L - 1)

        # 3) 线性插值索引
        f = pos.floor().long()
        c = pos.ceil().long()
        w = (pos - f).to(q.dtype)                          # [B,H,N,N]

        # 4) q · pos_emb -> gather -> 线性插值
        logits_int = torch.einsum('bhnd,dl->bhnl', q, self.pos_emb[0])  # [B,H,N,L]
        logits_floor = logits_int.gather(-1, f)            # [B,H,N,N]
        logits_ceil  = logits_int.gather(-1, c)            # [B,H,N,N]
        bias = logits_ceil * w + logits_floor * (1. - w)   # [B,H,N,N]
        return bias

# ---------------- MLP ----------------
class MLP(nn.Module):
    def __init__(self, dim, mlp_hidden, p=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p)
    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x

# ---------------- Attention（paper-correct + Top-k 稀疏门控） ----------------
class Attention(nn.Module):
    """
    方法学：先算完整 qk^T/√d，再对其做 Top-k 阈值得到“稀疏 logits”喂给 CoPE 产生 bias；
           之后重算一次完整 logits 与 bias 相加，softmax 得注意力。
    仅加入 Top-k（不增参，不分块）：
      topk: 每个 query（行）仅保留最大的 k 个 key 位置参与 CoPE 的 gates（其余≈0）
    """
    def __init__(self, dim, heads=3, dim_head=64, num_patches=196, p=0.1, topk: int = 48):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner = heads * dim_head
        self.scale = dim_head ** -0.5
        self.qkv = nn.Linear(dim, inner * 3, bias=False)
        self.cope = CoPE(npos_max=num_patches + 1, head_dim=dim_head)  # +1 兼容 CLS
        self.attend = nn.Softmax(dim=-1)
        self.proj = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(p))
        self.topk = int(topk)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        # ---- 计算完整 logits（后续还会重算一次，用于最终 softmax）----
        attn_logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [B,H,N,N]
        if mask is not None:
            attn_logits = attn_logits + mask

        # ---- Top-k 阈值（就地 masked_fill_，避免额外大掩码张量）----
        with torch.no_grad():
            kth = attn_logits.topk(k=self.topk, dim=-1).values[..., -1:]  # [B,H,N,1]
        attn_logits.masked_fill_(attn_logits < kth, float('-inf'))

        # ---- 用“稀疏 logits”生成 CoPE 偏置 ----
        cope_bias = self.cope(q, attn_logits)  # [B,H,N,N]

        # ---- 重算完整 logits + 偏置 -> softmax ----
        attn_logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if mask is not None:
            attn_logits = attn_logits + mask
        attn_logits = attn_logits + cope_bias

        attn = self.attend(attn_logits)
        out  = torch.matmul(attn, v)                          # [B,H,N,D]
        out  = rearrange(out, 'b h n d -> b n (h d)')
        return self.proj(out)

# ---------------- Block（Pre-Norm） ----------------
class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, num_patches=196, p=0.1, topk: int = 48):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = Attention(dim, heads, dim//heads, num_patches, p=p, topk=topk)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = MLP(dim, int(dim * mlp_ratio), p=p)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# ---------------- ViT-Tiny + CLS ----------------
class ViTCoPE(nn.Module):
    def __init__(self,
                 image_size=224, patch_size=16, num_classes=1000,
                 dim=192, depth=12, heads=3, mlp_ratio=4.0,
                 dropout=0.1, emb_dropout=0.1, topk: int = 48):
        super().__init__()
        H, W = pair(image_size); pH, pW = pair(patch_size)
        assert H % pH == 0 and W % pW == 0
        self.h = H // pH
        self.w = W // pW
        num_patches = self.h * self.w
        patch_dim = pH * pW * 3

        self.to_patch = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=pH, p2=pW),
            nn.Linear(patch_dim, dim)
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.drop = nn.Dropout(emb_dropout)
        self.blocks = nn.Sequential(*[
            Block(dim, heads, mlp_ratio, num_patches, dropout, topk=topk)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, img):
        B = img.size(0)
        x = self.to_patch(img)                        # [B,N,C]  N=196
        cls = self.cls_token.expand(B, -1, -1)       # [B,1,C]
        x = torch.cat([cls, x], dim=1)               # [B,197,C]
        x = self.drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        logits = self.head(x[:, 0])                  # 用 CLS 读出
        return logits
