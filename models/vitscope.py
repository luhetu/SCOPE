# -*- coding: utf-8 -*-
# ViT-SCoPE: Q-offset + HKGate + CLS (224x224, patch=16, ViT-Tiny friendly implementation)
import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange

def pair(t): return t if isinstance(t, tuple) else (t, t)

# ---------------- HKGate: align to patch grid + lightweight normalization ----------------
class HKGate(nn.Module):
    def __init__(self, out_hw):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.max4 = nn.MaxPool2d(4, 4)
        self.adapt = nn.AdaptiveAvgPool2d(out_hw)  # Force align to (h_p, w_p)
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):                       # x: [B,3,H,W]
        mu = self.global_avgpool(x)
        x  = x + (1 + self.alpha) * (x - mu)    # Contrast enhancement
        x  = self.max4(x)                       # 224->56
        x  = self.max4(x)                       # 56->14
        x  = self.adapt(x).mean(dim=1)          # [B, h_p, w_p]
        gate = x.flatten(1)                     # [B, N]
        # Per-sample normalize to (0,1)
        gate = (gate - gate.mean(dim=1, keepdim=True)) / (gate.std(dim=1, keepdim=True) + 1e-5)
        gate = torch.sigmoid(self.beta * gate)
        return gate                              # [B, N] ∈ (0,1)

# ---------------- CoPE: Q-offset (keep your implementation) ----------------
class CoPE(nn.Module):
    def __init__(self, npos_max, dim_head):
        super().__init__()
        self.npos_max = npos_max
        self.dim_head = dim_head
        self.pos_emb = nn.Parameter(torch.zeros(1, dim_head, npos_max))
        nn.init.xavier_uniform_(self.pos_emb)

    def forward(self, q, attn_logits):
        # q: [B,H,N,D] ; attn_logits: [B,H,N,N]
        import torch.nn.functional as F
        gates = torch.sigmoid(attn_logits.mean(dim=-1))                 # [B,H,N]
        pos = gates.flip(-1).cumsum(dim=-1).flip(-1)
        target_len = attn_logits.shape[-1]
        pos_emb = self.pos_emb
        if pos_emb.shape[-1] != target_len:
            pos_emb = F.interpolate(pos_emb, size=target_len, mode='linear', align_corners=False)
        pos = pos.clamp(0, target_len - 1)
        f = pos.floor().long(); c = pos.ceil().long(); w = (pos - f).unsqueeze(-1)
        table = pos_emb[0].transpose(0, 1)                               # [N, D]
        B,H,N = f.shape
        e_f = table.index_select(0, f.reshape(-1)).view(B,H,N,self.dim_head)
        e_c = table.index_select(0, c.reshape(-1)).view(B,H,N,self.dim_head)
        offset = e_f * (1 - w) + e_c * w                                # [B,H,N,D]
        return offset, gates

# ---------------- Base layers ----------------
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim); self.fn = fn
    def forward(self, x, *args, **kwargs): return self.fn(self.norm(x), *args, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), nn.Dropout(dropout)
        )
    def forward(self, x): return self.net(x)

# ---------------- Attention: Q-offset + λ convex combination fusion + CLS gate ----------------
class Attention(nn.Module):
    def __init__(self, dim, heads=6, dim_head=32, num_patches=196, dropout=0., use_cls_token=True):
        super().__init__()
        inner = dim_head * heads
        self.heads = heads
        self.use_cls_token = use_cls_token
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.cope  = CoPE(npos_max=num_patches + (1 if use_cls_token else 0), dim_head=dim_head)
        self.lam   = nn.Parameter(torch.tensor(0.5))                    # Learn λ for convex combination
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))

    def forward(self, x, hk_gate_1d):
        # x: [B, N(+CLS), dim], hk_gate_1d: [B,N] (aligned to patches)
        B, N_all, _ = x.shape
        expected_patches = N_all - 1 if self.use_cls_token else N_all
        assert hk_gate_1d.shape[1] == expected_patches, "HKGate length should equal patch count"
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q,k,v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1,-2)) * self.scale         # [B,H,N_all,N_all]
        offset, cope_gate = self.cope(q, dots)                           # offset:[B,H,N_all,D], cope_gate:[B,H,N_all]

        if self.use_cls_token:
            # CLS gate from CoPE column 0 (multi-head average), concatenated with HKGate
            cls_from_cope = cope_gate[:, :, 0].mean(dim=1, keepdim=True) # [B,1]
            hk_full = torch.cat([cls_from_cope, hk_gate_1d], dim=1)      # [B,N_all]
        else:
            hk_full = hk_gate_1d                                        # [B,N_all]
        hk_full = hk_full.unsqueeze(1).expand(-1, self.heads, -1)        # [B,H,N_all]

        # Convex combination fusion (avoid secondary sigmoid flattening dynamic range)
        lam = torch.sigmoid(self.lam)                                    # (0,1)
        fused_gate = lam * cope_gate + (1 - lam) * hk_full               # [B,H,N_all] ∈ (0,1)

        q_new = q + offset * fused_gate.unsqueeze(-1)                    # Q-offset with gate
        attn = torch.softmax(torch.matmul(q_new, k.transpose(-1,-2)) * self.scale, dim=-1)
        out  = torch.matmul(attn, v)                                     # [B,H,N_all,D]
        return self.to_out(rearrange(out, 'b h n d -> b n (h d)'))

# ---------------- Transformer ----------------
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, num_patches, dropout=0., use_cls_token=True):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, num_patches, dropout, use_cls_token)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout)),
            ]) for _ in range(depth)
        ])
    def forward(self, x, hk_gate_1d):
        for attn, ff in self.layers:
            x = attn(x, hk_gate_1d) + x
            x = ff(x) + x
        return x

# ---------------- ViT-SCoPE 主Body (含 CLS,Read CLS) ----------------
class ViTSCoPE(nn.Module):
    """
    Default:image_size=224, patch_size=16, dim=192, depth=12, heads=6, dim_head=32, mlp_dim=768

    """
    def __init__(self, *, image_size=224, patch_size=16,
                 num_classes=1000, dim=192, depth=12, heads=6, mlp_dim=768,
                 channels=3, dim_head=32, dropout=0.0, emb_dropout=0.0,
                 use_cls_token=True, pool="cls"):
        super().__init__()
        if pool == "cls" and not use_cls_token:
            raise ValueError("pool='cls' requires use_cls_token=True")
        assert pool in {"cls", "mean"}, "pool must be 'cls' or 'mean'"
        H, W = pair(image_size); pH, pW = pair(patch_size)
        assert H % pH == 0 and W % pW == 0
        h_p, w_p = H // pH, W // pW
        N = h_p * w_p
        patch_dim = channels * pH * pW

        self.to_patch = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=pH, p2=pW),
            nn.Linear(patch_dim, dim)
        )
        self.use_cls_token = use_cls_token
        self.pool = pool

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.drop = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, N, dropout, use_cls_token)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

        self.hk_gate = HKGate(out_hw=(h_p, w_p))   

    def forward(self, img: torch.Tensor):
        B = img.size(0)
        hk_gate_1d = self.hk_gate(img)                  # [B, N]

        x = self.to_patch(img)                          # [B, N, dim]
        if self.use_cls_token:
            cls = self.cls_token.expand(B, 1, -1)      # [B, 1, dim]
            x = torch.cat([cls, x], dim=1)             # [B, 1+N, dim]
        x = self.drop(x)

        x = self.transformer(x, hk_gate_1d)
        x = self.norm(x)
        if self.pool == "mean":
            x = x[:, 1:].mean(dim=1) if self.use_cls_token else x.mean(dim=1)
        else:
            x = x[:, 0]
        return self.head(x)

ViTScope = ViTSCoPE
