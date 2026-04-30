import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange

try:
    from timm.layers import DropPath
except ImportError:
    from timm.models.layers import DropPath


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

        self.pos_emb = nn.Parameter(torch.zeros(1, dim_head, npos_max))
        nn.init.xavier_uniform_(self.pos_emb)

    def forward(self, q, attn_logits):
        # q:           [B, H, N, D_head]
        # attn_logits: [B, H, N, N]
        import torch.nn.functional as F

        gate = torch.sigmoid(attn_logits.mean(dim=-1))  # [B, H, N]

        pos = gate.flip(-1).cumsum(dim=-1).flip(-1)

        target_len = attn_logits.shape[-1]
        pos_emb = self.pos_emb

        if pos_emb.shape[-1] != target_len:
            pos_emb = F.interpolate(
                pos_emb,
                size=target_len,
                mode="linear",
                align_corners=False,
            )

        pos = pos.clamp(0, target_len - 1)

        f = pos.floor().long()
        c = pos.ceil().long()
        w = (pos - f).unsqueeze(-1)  # [B, H, N, 1]

        table = pos_emb[0].transpose(0, 1)  # [npos, dim_head]

        B, H, N = f.shape

        e_f = table.index_select(0, f.reshape(-1)).view(B, H, N, self.dim_head)
        e_c = table.index_select(0, c.reshape(-1)).view(B, H, N, self.dim_head)

        offset = e_f * (1 - w) + e_c * w  # [B, H, N, D_head]

        return offset, gate


# ————————————————
# PreNorm & FeedForward
# ————————————————
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ————————————————
# Attention: CoPE Q-offset
# ————————————————
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, num_patches=64, dropout=0.):
        super().__init__()

        self.heads = heads
        self.scale = dim_head ** -0.5

        inner = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.cope = CoPE(npos_max=num_patches, dim_head=dim_head)
        self.attend = nn.Softmax(dim=-1)

        self.to_out = nn.Sequential(
            nn.Linear(inner, dim),
            nn.Dropout(dropout),
        )

        self.vis_attn = None
        self.vis_cope_gate = None

    def forward(self, x):
        # x: [B, N, dim]
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads),
            qkv,
        )

        logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [B,H,N,N]

        offset, gate = self.cope(q, logits)  # [B,H,N,D], [B,H,N]

        q = q + offset

        new_logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(new_logits)

        out = torch.matmul(attn, v)  # [B,H,N,D]
        out = rearrange(out, "b h n d -> b n (h d)")

        self.vis_attn = attn.detach()
        self.vis_cope_gate = gate.detach()

        return self.to_out(out)


# ————————————————
# Transformer with DropPath
# ————————————————
class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        num_tokens,
        dropout=0.,
        drop_path_rate=0.,
    ):
        super().__init__()

        dpr = torch.linspace(0, drop_path_rate, depth).tolist()

        self.layers = nn.ModuleList([
            nn.ModuleList([
                PreNorm(
                    dim,
                    Attention(
                        dim,
                        heads=heads,
                        dim_head=dim_head,
                        num_patches=num_tokens,
                        dropout=dropout,
                    ),
                ),
                PreNorm(
                    dim,
                    FeedForward(
                        dim,
                        mlp_dim,
                        dropout=dropout,
                    ),
                ),
                DropPath(dpr[i]) if dpr[i] > 0. else nn.Identity(),
                DropPath(dpr[i]) if dpr[i] > 0. else nn.Identity(),
            ])
            for i in range(depth)
        ])

    def forward(self, x):
        for attn, ff, drop_path1, drop_path2 in self.layers:
            x = x + drop_path1(attn(x))
            x = x + drop_path2(ff(x))

        return x


# ————————————————
# ViT-CoPE
# ————————————————
class ViTcope(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="mean",
        channels=3,
        dim_head=64,
        use_cls_token=False,
        dropout=0.,
        emb_dropout=0.,
        drop_path_rate=0.,
    ):
        super().__init__()

        H, W = pair(image_size)
        pH, pW = pair(patch_size)

        assert H % pH == 0 and W % pW == 0

        N = (H // pH) * (W // pW)
        patch_dim = channels * pH * pW
        num_tokens = N + (1 if use_cls_token else 0)

        self.to_patch = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=pH,
                p2=pW,
            ),
            nn.Linear(patch_dim, dim),
        )

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            num_tokens=num_tokens,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
        )

        self.pool = pool
        self.use_cls_token = use_cls_token

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
        )

    def forward(self, img):
        x = self.to_patch(img)  # [B, N, dim]

        if self.use_cls_token:
            cls = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat((cls, x), dim=1)

        x = self.dropout(x)
        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        return self.mlp_head(x)


ViTCoPE = ViTcope