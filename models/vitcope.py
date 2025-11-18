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
# CoPE module: dynamic position offset
# ————————————————
class CoPE(nn.Module):
    def __init__(self, npos_max, dim_head):
        super().__init__()
        self.npos_max = npos_max
        self.dim_head = dim_head
        # pos_emb: [1, dim_head, npos_max]
        self.pos_emb = nn.Parameter(torch.zeros(1, dim_head, npos_max))
        nn.init.xavier_uniform_(self.pos_emb)

    def forward(self, q, attn_logits):
        # q:           [B, H, N, D_head]
        # attn_logits: [B, H, N, N]
        # 1) gate = sigmoid(mean over last dim)
        gate = torch.sigmoid(attn_logits.mean(dim=-1))  # [B, H, N]
        # 2) dynamic pos = flip–cumsum–flip
        pos = gate.flip(-1).cumsum(dim=-1).flip(-1)
        pos = pos.clamp(0, self.npos_max-1)
        # 3) floor/ceil + weights
        f = pos.floor().long()
        c = pos.ceil().long()
        w = (pos - f).unsqueeze(-1)  # [B, H, N, 1]
        # 4) lookup embeddings
        emb2d = self.pos_emb[0].transpose(0,1)  # [npos_max, dim_head]
        B,H,N = f.shape
        f_idx = f.reshape(-1)
        c_idx = c.reshape(-1)
        e_f = emb2d.index_select(0, f_idx).view(B,H,N,self.dim_head)
        e_c = emb2d.index_select(0, c_idx).view(B,H,N,self.dim_head)
        # 5) interpolate
        offset = e_f * (1 - w) + e_c * w          # [B, H, N, D_head]
        return offset, gate  # Consistent with vitscope.py/vit_backbone.py

# ————————————————
# PreNorm & FeedForward
# ————————————————
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn   = fn
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
# Attention: only use CoPE, no CNNGate
# ————————————————
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, num_patches=64, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner   = dim_head * heads

        self.to_qkv  = nn.Linear(dim, inner * 3, bias=False)
        self.cope    = CoPE(npos_max=num_patches, dim_head=dim_head)
        self.attend  = nn.Softmax(dim=-1)
        self.to_out  = nn.Sequential(
            nn.Linear(inner, dim),
            nn.Dropout(dropout)
        )
        self.vis_attn = None
        self.vis_cope_gate = None

    def forward(self, x):
        # x: [B, N, dim]
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # raw attention logits
        logits = torch.matmul(q, k.transpose(-1,-2)) * self.scale  # [B,H,N,N]
        # CoPE offset (returns two values)
        offset, gate = self.cope(q, logits)                        # [B,H,N,D_head], [B,H,N]

        # apply to q
        q2 = q + offset
        # recompute attention
        new_logits = torch.matmul(q2, k.transpose(-1,-2)) * self.scale
        attn = self.attend(new_logits)
        out  = torch.matmul(attn, v)                               # [B,H,N,D_head]
        # merge heads
        out2 = rearrange(out, 'b h n d -> b n (h d)')
        # Cache for visualization
        self.vis_attn = attn.detach()
        self.vis_cope_gate = gate.detach()
        return self.to_out(out2)

# ————————————————
# Transformer & ViTcope (remove CLS, mean pooling)
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

class ViTcope(nn.Module):
    def __init__(self, *, image_size, patch_size,
                 num_classes, dim, depth, heads, mlp_dim,
                 pool='mean', channels=3, dim_head=64,
                 dropout=0., emb_dropout=0.):
        super().__init__()
        H,W = pair(image_size)
        pH,pW = pair(patch_size)
        assert H%pH==0 and W%pW==0
        N = (H//pH)*(W//pW)
        patch_dim = channels * pH * pW

        # patch embedding
        self.to_patch = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=pH, p2=pW),
            nn.Linear(patch_dim, dim)
        )
        self.dropout    = nn.Dropout(emb_dropout)
        # pure CoPE transformer
        self.transformer= Transformer(dim, depth, heads, dim_head, mlp_dim, N, dropout)
        self.pool       = pool
        self.mlp_head   = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, img):
        x = self.to_patch(img)         # [B, N, dim]
        x = self.dropout(x)
        x = self.transformer(x)        # [B, N, dim]
        x = x.mean(dim=1) if self.pool=='mean' else x[:,0]
        return self.mlp_head(x)

# Alias, maintain backward compatibility
ViTCoPE = ViTcope
