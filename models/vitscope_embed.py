import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange

# ————————————————
# Helpers
# ————————————————
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# ————————————————
# CNNGate + SCoPE 模块：动态位置偏移（Soft版本）
# ————————————————
class CNNGateV2(nn.Module):
    """CNN-based gating for position encoding"""
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
        
        # 混合池化
        x = self.max4(x)
        x = self.max4(x)
        x = self.max2(x)
        
        # 展平
        x = x.mean(dim=1)
        return x.flatten(1)

class SCoPE(nn.Module):
    """Soft CoPE: 动态位置偏移（Embedding层版本）"""
    def __init__(self, npos_max, dim_head):
        super().__init__()
        self.npos_max = npos_max
        self.dim_head = dim_head
        self.pos_emb = nn.Parameter(torch.zeros(1, dim_head, npos_max))
        nn.init.xavier_uniform_(self.pos_emb)

    def forward(self, q, attn_logits):
        # q:           [B, H, N, D_head]
        # attn_logits: [B, H, N, N]
        # Soft gating
        gate = torch.sigmoid(attn_logits.mean(dim=-1))  # [B, H, N]
        # Dynamic position
        pos = gate.flip(-1).cumsum(dim=-1).flip(-1)
        pos = pos.clamp(0, self.npos_max-1)
        # Floor/ceil + weights
        f = pos.floor().long()
        c = pos.ceil().long()
        w = (pos - f).unsqueeze(-1)  # [B, H, N, 1]
        # Lookup embeddings
        emb2d = self.pos_emb[0].transpose(0,1)  # [npos_max, dim_head]
        B, H, N = f.shape
        f_idx = f.reshape(-1)
        c_idx = c.reshape(-1)
        e_f = emb2d.index_select(0, f_idx).view(B, H, N, self.dim_head)
        e_c = emb2d.index_select(0, c_idx).view(B, H, N, self.dim_head)
        # Interpolate
        offset = e_f * (1 - w) + e_c * w  # [B, H, N, D_head]
        return offset

# ————————————————
# PreNorm & FeedForward
# ————————————————
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
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# ————————————————
# Attention with SCoPE
# ————————————————
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, num_patches=64, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.scope = SCoPE(npos_max=num_patches, dim_head=dim_head)
        self.attend = nn.Softmax(dim=-1)
        self.to_out = nn.Sequential(
            nn.Linear(inner, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: [B, N, dim]
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # Raw attention logits
        logits = torch.matmul(q, k.transpose(-1,-2)) * self.scale  # [B,H,N,N]
        # SCoPE offset
        offset = self.scope(q, logits)  # [B,H,N,D_head]

        # Apply to q
        q2 = q + offset
        # Recompute attention
        attn = self.attend(torch.matmul(q2, k.transpose(-1,-2)) * self.scale)
        out = torch.matmul(attn, v)  # [B,H,N,D_head]
        # Merge heads
        out2 = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out2)

# ————————————————
# Transformer & ViTScope（Embedding层版本，无固定位置编码）
# ————————————————
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, num_patches, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, num_patches, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]) for _ in range(depth)
        ])

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViTScope(nn.Module):
    def __init__(self, *, image_size, patch_size,
                 num_classes, dim, depth, heads, mlp_dim,
                 pool='mean', channels=3, dim_head=64,
                 dropout=0., emb_dropout=0.):
        super().__init__()
        H, W = pair(image_size)
        pH, pW = pair(patch_size)
        assert H % pH == 0 and W % pW == 0
        N = (H // pH) * (W // pW)
        patch_dim = channels * pH * pW

        # Patch embedding
        self.to_patch = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=pH, p2=pW),
            nn.Linear(patch_dim, dim)
        )
        self.dropout = nn.Dropout(emb_dropout)
        # Pure SCoPE transformer (no fixed position encoding)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, N, dropout)
        self.pool = pool
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, img):
        x = self.to_patch(img)  # [B, N, dim]
        x = self.dropout(x)
        x = self.transformer(x)  # [B, N, dim]
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        return self.mlp_head(x)

