# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

# ---------------- helpers ----------------
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# ---------------- CoPE：在 attention logits 上加偏置 ----------------
class CoPE(nn.Module):
    """
    Adds bias to attention logits (paper-correct, pairwise).
    """
    def __init__(self, npos_max: int, head_dim: int):
        super().__init__()
        self.npos_max = int(npos_max)
        self.head_dim = int(head_dim)
        # pos_emb: [1, D, L]
        self.pos_emb = nn.Parameter(torch.zeros(1, head_dim, npos_max))
        nn.init.xavier_uniform_(self.pos_emb)

    @torch.no_grad()
    def _resize_pos_len_(self, cur_len: int):
        # 保证 L == 当前序列长度 N（含 CLS）
        if self.pos_emb.shape[-1] != cur_len:
            pe = F.interpolate(self.pos_emb, size=cur_len, mode='linear', align_corners=False)
            self.pos_emb.data.copy_(pe)

    def forward(self, query: torch.Tensor, attn_logits: torch.Tensor) -> torch.Tensor:
        """
        query:       [B, H, N, D]
        attn_logits: [B, H, N, N]  (before softmax)
        return:      [B, H, N, N]  bias to add to logits
        """
        B, H, N, D = query.shape
        self._resize_pos_len_(N)                           # ★ 长度自适配
        pos_emb_matrix = self.pos_emb[0]                   # [D, N]
        logits_int = torch.matmul(query, pos_emb_matrix)   # [B, H, N, N]

        # 逐行（query 维）处理，节约峰值内存
        bias = logits_int.new_empty(B, H, N, N)
        for i in range(N):
            attn_logits_i = attn_logits[:, :, i, :]              # [B, H, N]
            gates_i = torch.sigmoid(attn_logits_i)               # [B, H, N]
            pos_i = gates_i.flip(-1).cumsum(dim=-1).flip(-1)     # [B, H, N]
            pos_i = pos_i.clamp(max=N - 1)

            pos_floor = pos_i.floor().long()
            pos_ceil  = pos_i.ceil().long()
            w = (pos_i - pos_floor).to(query.dtype)              # ★ AMP 友好

            logits_i = logits_int[:, :, i, :]                    # [B, H, N]
            logits_floor = torch.gather(logits_i, dim=-1, index=pos_floor)
            logits_ceil  = torch.gather(logits_i, dim=-1, index=pos_ceil)

            bias_i = logits_ceil * w + logits_floor * (1. - w)   # [B, H, N]
            bias[:, :, i, :] = bias_i
        return bias

# ---------------- Self Attention with SCoPE（保表达力版） ----------------
class SelfAttn(nn.Module):
    """
    Self-Attention + CoPE (add-to-logits).
    SCoPE 融合：仅做“行级幅度缩放”，不改变列间相对关系，从而保留 pairwise 表达力。
    scale = clamp(1 + tau * (hk - cope), 0.25, 1.75);  tau = tanh(alpha) ∈ (0,1)
    """
    def __init__(self, dim, heads=3, dim_head=64, num_patches=196, dropout=0.):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.heads = heads
        self.head_dim = dim_head
        self.scale = dim_head ** -0.5

        inner = heads * dim_head
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)

        # ★ 这里仅传入 num_patches；内部再 +1 以包含 CLS，避免外部重复 +1
        self.cope = CoPE(npos_max=num_patches + 1, head_dim=dim_head)

        # 融合强度参数（标量），建议初值小一点
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.attend = nn.Softmax(dim=-1)
        self.drop = nn.Dropout(dropout)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))

    def forward(self, x):
        """
        x: [B, N, dim] (含 CLS)
        """
        B, N, C = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # 原始 logits
        attn_logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [B,H,N,N]

        # 1) CoPE 偏置（pairwise）
        cope_bias = self.cope(q, attn_logits)                            # [B,H,N,N]

        # 2) SCoPE（保表达力）：只调整“行的幅度”，不改列间相对关系
        #   用平滑的“尖锐度”替代 max：||p||_2^2 = sum(p^2)，范围 [1/N, 1]
        attn_probs = torch.softmax(attn_logits, dim=-1)                  # [B,H,N,N]
        hk_gate = (attn_probs * attn_probs).sum(dim=-1)                  # [B,H,N]
        cope_gate = torch.sigmoid(attn_logits.mean(dim=-1))              # [B,H,N]

        tau = torch.tanh(self.alpha)                                     # ∈ (0,1)
        scale = 1.0 + tau * (hk_gate - cope_gate)                        # [B,H,N]
        scale = torch.clamp(scale, 0.25, 1.75)                           # 稳定数值
        cope_bias = cope_bias * scale.unsqueeze(-1)                      # 仅行级放缩

        # 3) 加偏置 -> softmax
        attn_logits = attn_logits + cope_bias
        attn = self.attend(attn_logits)
        attn = self.drop(attn)

        out = torch.matmul(attn, v)                                      # [B,H,N,D]
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# ---------------- PreNorm & FFN ----------------
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, *args):
        return self.fn(self.norm(x), *args)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), nn.Dropout(dropout)
        )
    def forward(self, x): return self.net(x)

# ---------------- Transformer ----------------
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, num_patches, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PreNorm(dim, SelfAttn(dim, heads, dim_head, num_patches, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]) for _ in range(depth)
        ])
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

# ---------------- ViTScope（强制 CLS） ----------------
class ViTScope(nn.Module):
    """
    ViT with SCoPE (CoPE add-to-logits + row-scale fusion), with CLS.
    """
    def __init__(self, *, image_size, patch_size,
                 num_classes, dim, depth, heads, mlp_dim,
                 channels=3, dim_head=64,
                 dropout=0., emb_dropout=0.):
        super().__init__()
        H, W = pair(image_size); pH, pW = pair(patch_size)
        assert H % pH == 0 and W % pW == 0
        N = (H // pH) * (W // pW)
        patch_dim = channels * pH * pW

        self.to_patch = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=pH, p2=pW),
            nn.Linear(patch_dim, dim)
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, N, dropout)

        self.norm = nn.LayerNorm(dim)
        self.mlp_head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.mlp_head.weight, std=0.02)
        if self.mlp_head.bias is not None:
            nn.init.zeros_(self.mlp_head.bias)

    def forward(self, img):
        B = img.size(0)
        x = self.to_patch(img)                   # [B, N, dim]
        cls = self.cls_token.expand(B, 1, -1)    # [B, 1, dim]
        x = torch.cat([cls, x], dim=1)           # [B, 1+N, dim]
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.norm(x)
        return self.mlp_head(x[:, 0])
