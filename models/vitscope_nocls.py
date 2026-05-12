# -*- coding: utf-8 -*-
# ViT-SCoPE NoCLS-aware gate:
# Q-offset + HKGate + CLS token, but CLS gate uses neutral 1.0
import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange

try:
    from timm.layers import DropPath
except ImportError:
    from timm.models.layers import DropPath


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# ---------------- HKGate ----------------
class HKGate(nn.Module):
    def __init__(self, out_hw):
        super().__init__()

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.alpha = nn.Parameter(torch.tensor(0.1))

        self.max4 = nn.MaxPool2d(4, 4)
        self.adapt = nn.AdaptiveAvgPool2d(out_hw)

        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # x: [B, 3, H, W]
        mu = self.global_avgpool(x)

        x = x + (1 + self.alpha) * (x - mu)

        x = self.max4(x)
        x = self.max4(x)

        x = self.adapt(x).mean(dim=1)  # [B, h_p, w_p]

        gate = x.flatten(1)  # [B, N]

        gate = (gate - gate.mean(dim=1, keepdim=True)) / (
            gate.std(dim=1, keepdim=True) + 1e-5
        )

        gate = torch.sigmoid(self.beta * gate)

        return gate


# ---------------- CoPE ----------------
class CoPE(nn.Module):
    def __init__(self, npos_max, dim_head):
        super().__init__()

        self.npos_max = npos_max
        self.dim_head = dim_head

        self.pos_emb = nn.Parameter(torch.zeros(1, dim_head, npos_max))
        nn.init.xavier_uniform_(self.pos_emb)

    def forward(self, q, attn_logits):
        # q: [B, H, N, D]
        # attn_logits: [B, H, N, N]
        gates = torch.sigmoid(attn_logits.mean(dim=-1))  # [B,H,N]

        pos = gates.flip(-1).cumsum(dim=-1).flip(-1)
        pos = pos.clamp(0, self.npos_max - 1)

        f = pos.floor().long()
        c = pos.ceil().long()
        w = (pos - f).unsqueeze(-1)

        table = self.pos_emb[0].transpose(0, 1)  # [N, D]

        B, H, N = f.shape

        e_f = table.index_select(0, f.reshape(-1)).view(B, H, N, self.dim_head)
        e_c = table.index_select(0, c.reshape(-1)).view(B, H, N, self.dim_head)

        offset = e_f * (1 - w) + e_c * w

        return offset, gates


# ---------------- Base Layers ----------------
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


# ---------------- Attention ----------------
class Attention(nn.Module):
    def __init__(self, dim, heads=6, dim_head=32, num_patches=196, dropout=0., use_cls_token=True):
        super().__init__()

        inner = dim_head * heads

        self.heads = heads
        self.use_cls_token = use_cls_token
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)

        self.cope = CoPE(
            npos_max=num_patches + (1 if use_cls_token else 0),
            dim_head=dim_head,
        )

        self.lam = nn.Parameter(torch.tensor(0.5))

        self.to_out = nn.Sequential(
            nn.Linear(inner, dim),
            nn.Dropout(dropout),
        )

        self.vis_attn = None
        self.vis_cope_gate = None
        self.vis_fused_gate = None

    def forward(self, x, hk_gate_1d):
        # x: [B, N(+CLS), dim]
        # hk_gate_1d: [B, N]
        B, N_all, _ = x.shape

        expected_patches = N_all - 1 if self.use_cls_token else N_all
        assert hk_gate_1d.shape[1] == expected_patches, (
            f"HKGate length should equal patch count, "
            f"got hk_gate={hk_gate_1d.shape[1]}, patches={expected_patches}"
        )

        qkv = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads),
            qkv,
        )

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        offset, cope_gate = self.cope(q, dots)

        if self.use_cls_token:
            # No CLS-aware gating: CLS uses neutral gate = 1.0
            cls_gate = hk_gate_1d.new_ones((B, 1))  # [B,1]
            hk_full = torch.cat([cls_gate, hk_gate_1d], dim=1)  # [B,1+N]
        else:
            hk_full = hk_gate_1d  # [B,N]
        hk_full = hk_full.unsqueeze(1).expand(-1, self.heads, -1)  # [B,H,N_all]

        lam = torch.sigmoid(self.lam)

        fused_gate = lam * cope_gate + (1 - lam) * hk_full

        q = q + offset * fused_gate.unsqueeze(-1)

        attn = torch.softmax(
            torch.matmul(q, k.transpose(-1, -2)) * self.scale,
            dim=-1,
        )

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        self.vis_attn = attn.detach()
        self.vis_cope_gate = cope_gate.detach()
        self.vis_fused_gate = fused_gate.detach()

        return self.to_out(out)


# ---------------- Transformer with DropPath ----------------
class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        num_patches,
        dropout=0.,
        drop_path_rate=0.,
        use_cls_token=True,
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
                        num_patches=num_patches,
                        dropout=dropout,
                        use_cls_token=use_cls_token,
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

    def forward(self, x, hk_gate_1d):
        for attn, ff, drop_path1, drop_path2 in self.layers:
            x = x + drop_path1(attn(x, hk_gate_1d))
            x = x + drop_path2(ff(x))

        return x


# ---------------- ViT-SCoPE NoCLS-aware ----------------
class ViTSCoPE_NoCLS(nn.Module):
    def __init__(
        self,
        *,
        image_size=224,
        patch_size=16,
        num_classes=1000,
        dim=192,
        depth=12,
        heads=6,
        mlp_dim=768,
        channels=3,
        dim_head=32,
        dropout=0.0,
        emb_dropout=0.0,
        drop_path_rate=0.0,
        use_cls_token=True,
        pool="cls",
    ):
        super().__init__()
        if pool == "cls" and not use_cls_token:
            raise ValueError("pool='cls' requires use_cls_token=True")
        assert pool in {"cls", "mean"}, "pool must be 'cls' or 'mean'"

        H, W = pair(image_size)
        pH, pW = pair(patch_size)

        assert H % pH == 0 and W % pW == 0

        h_p, w_p = H // pH, W // pW
        N = h_p * w_p

        patch_dim = channels * pH * pW

        self.to_patch = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=pH,
                p2=pW,
            ),
            nn.Linear(patch_dim, dim),
        )

        self.use_cls_token = use_cls_token
        self.pool = pool

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.drop = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            num_patches=N,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
            use_cls_token=use_cls_token,
        )

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

        self.hk_gate = HKGate(out_hw=(h_p, w_p))

    def forward(self, img):
        B = img.size(0)

        hk_gate_1d = self.hk_gate(img)  # [B, N]

        x = self.to_patch(img)  # [B, N, dim]

        if self.use_cls_token:
            cls = self.cls_token.expand(B, 1, -1)
            x = torch.cat([cls, x], dim=1)

        x = self.drop(x)

        x = self.transformer(x, hk_gate_1d)

        x = self.norm(x)

        if self.pool == "mean":
            x = x[:, 1:].mean(dim=1) if self.use_cls_token else x.mean(dim=1)
        else:
            x = x[:, 0]

        return self.head(x)


ViTScope_NoCLS = ViTSCoPE_NoCLS