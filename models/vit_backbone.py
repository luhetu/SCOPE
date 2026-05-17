# -*- coding: utf-8 -*-
"""ViT / ViT-CoPE / ViT-SCoPE backbones for MMSegmentation and MMDetection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

try:
    from timm.layers import DropPath
except ImportError:
    from timm.models.layers import DropPath

try:
    from mmdet.models.builder import BACKBONES as MMDET_BACKBONES
    MMDET_AVAILABLE = True
except ImportError:
    MMDET_AVAILABLE = False

try:
    from mmseg.models.builder import BACKBONES as MMSEG_BACKBONES
    MMSEG_AVAILABLE = True
except ImportError:
    MMSEG_AVAILABLE = False


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
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


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        attn = self.attend(torch.matmul(q, k.transpose(-1, -2)) * self.scale)
        out = torch.matmul(attn, v)
        return self.to_out(rearrange(out, 'b h n d -> b n (h d)'))


class CoPE(nn.Module):
    def __init__(self, npos_max, dim_head):
        super().__init__()
        self.npos_max = npos_max
        self.dim_head = dim_head
        self.pos_emb = nn.Parameter(torch.zeros(1, dim_head, npos_max))
        nn.init.xavier_uniform_(self.pos_emb)

    def forward(self, q, attn_logits):
        gate = torch.sigmoid(attn_logits.mean(dim=-1))
        target_len = attn_logits.shape[-1]
        pos_emb = self.pos_emb
        if pos_emb.shape[-1] != target_len:
            pos_emb = F.interpolate(pos_emb.float(), size=target_len, mode='linear', align_corners=False).to(q.dtype)
        pos = gate.flip(-1).cumsum(dim=-1).flip(-1).clamp(0, target_len - 1)
        f = pos.floor().long()
        c = pos.ceil().long()
        w = (pos - f).unsqueeze(-1)
        table = pos_emb[0].transpose(0, 1)
        B, H, N = f.shape
        e_f = table.index_select(0, f.reshape(-1)).view(B, H, N, self.dim_head)
        e_c = table.index_select(0, c.reshape(-1)).view(B, H, N, self.dim_head)
        return e_f * (1 - w) + e_c * w, gate


class AttentionCoPE(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, num_tokens=196, dropout=0.0):
        super().__init__()
        inner = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.cope = CoPE(num_tokens, dim_head)
        self.attend = nn.Softmax(dim=-1)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))
        self.vis_attn = None
        self.vis_cope_gate = None

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        offset, gate = self.cope(q, logits)
        q = q + offset
        attn = self.attend(torch.matmul(q, k.transpose(-1, -2)) * self.scale)
        out = torch.matmul(attn, v)
        self.vis_attn = attn.detach()
        self.vis_cope_gate = gate.detach()
        return self.to_out(rearrange(out, 'b h n d -> b n (h d)'))


class HKGate(nn.Module):
    def __init__(self, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.max4 = nn.MaxPool2d(4, 4)
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        _, _, H, W = x.shape
        h_p, w_p = H // self.patch_size, W // self.patch_size
        mu = self.global_avgpool(x)
        x = x + (1 + self.alpha) * (x - mu)
        x = self.max4(self.max4(x))
        x = F.adaptive_avg_pool2d(x, (h_p, w_p)).mean(dim=1)
        gate = x.flatten(1)
        gate = (gate - gate.mean(dim=1, keepdim=True)) / (gate.std(dim=1, keepdim=True) + 1e-5)
        return torch.sigmoid(self.beta * gate)


class AttentionSCoPE(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, num_patches=196, dropout=0.0, use_cls_token=True):
        super().__init__()
        inner = heads * dim_head
        self.heads = heads
        self.use_cls_token = use_cls_token
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.cope = CoPE(num_patches + (1 if use_cls_token else 0), dim_head)
        self.lam = nn.Parameter(torch.tensor(0.5))
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))
        self.vis_attn = None
        self.vis_cope_gate = None
        self.vis_fused_gate = None

    def forward(self, x, hk_gate_1d):
        B, N_all, _ = x.shape
        expected = N_all - 1 if self.use_cls_token else N_all
        assert hk_gate_1d.shape[1] == expected, f'HKGate length {hk_gate_1d.shape[1]} != patch tokens {expected}'
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        offset, cope_gate = self.cope(q, dots)
        if self.use_cls_token:
            cls_gate = cope_gate[:, :, 0].mean(dim=1, keepdim=True)
            hk_full = torch.cat([cls_gate, hk_gate_1d], dim=1)
        else:
            hk_full = hk_gate_1d
        hk_full = hk_full.unsqueeze(1).expand(-1, self.heads, -1)
        fused_gate = torch.sigmoid(self.lam) * cope_gate + (1 - torch.sigmoid(self.lam)) * hk_full
        q = q + offset * fused_gate.unsqueeze(-1)
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * self.scale, dim=-1)
        out = torch.matmul(attn, v)
        self.vis_attn = attn.detach()
        self.vis_cope_gate = cope_gate.detach()
        self.vis_fused_gate = fused_gate.detach()
        return self.to_out(rearrange(out, 'b h n d -> b n (h d)'))


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, attention_type='vit', num_tokens=196, dropout=0.0, drop_path_rate=0.0):
        super().__init__()
        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.layers = nn.ModuleList([])
        for i in range(depth):
            if attention_type == 'vit':
                attn = Attention(dim, heads, dim_head, dropout)
            elif attention_type == 'cope':
                attn = AttentionCoPE(dim, heads, dim_head, num_tokens, dropout)
            else:
                raise ValueError(attention_type)
            self.layers.append(nn.ModuleList([
                PreNorm(dim, attn),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout)),
                DropPath(dpr[i]) if dpr[i] > 0 else nn.Identity(),
                DropPath(dpr[i]) if dpr[i] > 0 else nn.Identity(),
            ]))


class TransformerSCoPE(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, num_patches=196, dropout=0.0, drop_path_rate=0.0, use_cls_token=True):
        super().__init__()
        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, AttentionSCoPE(dim, heads, dim_head, num_patches, dropout, use_cls_token)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout)),
                DropPath(dpr[i]) if dpr[i] > 0 else nn.Identity(),
                DropPath(dpr[i]) if dpr[i] > 0 else nn.Identity(),
            ]))


class ResizeAdapter(nn.Module):
    def __init__(self, dim, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        self.lateral = nn.Conv2d(dim, dim, kernel_size=1)
        self.refine = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.lateral(x)
        if self.scale_factor != 1:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False, recompute_scale_factor=True)
        return self.refine(x)


class SimpleFPNAdapter(nn.Module):
    """Official XCiT-style adapter for p16 dense prediction: scales [4, 2, 1, 0.5]."""
    def __init__(self, dim, scale):
        super().__init__()
        if scale == 4:
            self.net = nn.Sequential(
                nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
                nn.SyncBatchNorm(dim),
                nn.GELU(),
                nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
            )
        elif scale == 2:
            self.net = nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2)
        elif scale == 1:
            self.net = nn.Identity()
        elif scale == 0.5:
            self.net = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            raise ValueError(f'Unsupported SimpleFPN scale: {scale}')

    def forward(self, x):
        return self.net(x)


class BackboneFeatureMixin:
    def _build_simple_fpn_adapters(self, dim, adapter_style='simple_fpn'):
        if adapter_style in ('identity', 'none', None):
            return nn.ModuleList([nn.Identity() for _ in self.out_indices])
        if len(self.out_indices) != 4:
            return nn.ModuleList([nn.Identity() for _ in self.out_indices])
        scales = [4, 2, 1, 0.5]
        adapter = ResizeAdapter if adapter_style == 'resize' else SimpleFPNAdapter
        return nn.ModuleList([adapter(dim, s) for s in scales])

    def _tokens_to_map(self, x, actual_h, actual_w, has_cls):
        if has_cls:
            x = x[:, 1:]
        return rearrange(x, 'b (h w) d -> b d h w', h=actual_h, w=actual_w)

    def _format_out(self, x, actual_h, actual_w, norm_idx, has_cls):
        out = self._tokens_to_map(x, actual_h, actual_w, has_cls)
        out = rearrange(out, 'b d h w -> b h w d')
        out = self.norms[norm_idx](out)
        out = rearrange(out, 'b h w d -> b d h w')
        return self.fpn_adapters[norm_idx](out)


def resize_abs_pos(pos_embedding, patch_size, H, W):
    cls_pos = pos_embedding[:, :1]
    patch_pos = pos_embedding[:, 1:]
    old = int(patch_pos.shape[1] ** 0.5)
    new_h, new_w = H // patch_size, W // patch_size
    patch_pos = patch_pos.reshape(1, old, old, -1).permute(0, 3, 1, 2)
    patch_pos = F.interpolate(patch_pos.float(), size=(new_h, new_w), mode='bicubic', align_corners=False).to(pos_embedding.dtype)
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, pos_embedding.shape[-1])
    return torch.cat([cls_pos, patch_pos], dim=1)


class ViTBackbone(nn.Module, BackboneFeatureMixin):
    def __init__(self, image_size=224, patch_size=16, dim=768, depth=12, heads=12, mlp_dim=3072, channels=3, dim_head=64, dropout=0.0, emb_dropout=0.0, drop_path_rate=0.0, out_indices=(3, 5, 7, 11), fpn_adapter_style='simple_fpn'):
        super().__init__()
        H, W = pair(image_size)
        pH, pW = pair(patch_size)
        num_patches = (H // pH) * (W // pW)
        self.patch_size = patch_size
        self.dim = dim
        self.out_indices = tuple(out_indices)
        self.to_patch_embedding = nn.Sequential(Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=pH, p2=pW), nn.Linear(channels * pH * pW, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, 'vit', num_patches + 1, dropout, drop_path_rate)
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in self.out_indices])
        self.fpn_adapters = self._build_simple_fpn_adapters(dim, fpn_adapter_style)

    def forward(self, img):
        B, _, H, W = img.shape
        actual_h, actual_w = H // self.patch_size, W // self.patch_size
        x = self.to_patch_embedding(img)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        pos = self.pos_embedding if x.shape[1] == self.pos_embedding.shape[1] else resize_abs_pos(self.pos_embedding, self.patch_size, H, W)
        x = self.dropout(x + pos)
        outs = []
        for i, (attn, ff, dp1, dp2) in enumerate(self.transformer.layers):
            x = x + dp1(attn(x))
            x = x + dp2(ff(x))
            if i in self.out_indices:
                idx = self.out_indices.index(i)
                outs.append(self._format_out(x, actual_h, actual_w, idx, True))
        return tuple(outs)

    def init_weights(self, pretrained=None):
        pass


class ViTCoPEBackbone(nn.Module, BackboneFeatureMixin):
    def __init__(self, image_size=224, patch_size=16, dim=768, depth=12, heads=12, mlp_dim=3072, channels=3, dim_head=64, use_cls_token=False, dropout=0.0, emb_dropout=0.0, drop_path_rate=0.0, out_indices=(3, 5, 7, 11), fpn_adapter_style='simple_fpn'):
        super().__init__()
        H, W = pair(image_size)
        pH, pW = pair(patch_size)
        num_patches = (H // pH) * (W // pW)
        num_tokens = num_patches + (1 if use_cls_token else 0)
        self.patch_size = patch_size
        self.dim = dim
        self.out_indices = tuple(out_indices)
        self.use_cls_token = use_cls_token
        self.to_patch = nn.Sequential(Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=pH, p2=pW), nn.Linear(channels * pH * pW, dim))
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, 'cope', num_tokens, dropout, drop_path_rate)
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in self.out_indices])
        self.fpn_adapters = self._build_simple_fpn_adapters(dim, fpn_adapter_style)

    def forward(self, img):
        B, _, H, W = img.shape
        actual_h, actual_w = H // self.patch_size, W // self.patch_size
        x = self.to_patch(img)
        if self.use_cls_token:
            x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = self.dropout(x)
        outs = []
        for i, (attn, ff, dp1, dp2) in enumerate(self.transformer.layers):
            x = x + dp1(attn(x))
            x = x + dp2(ff(x))
            if i in self.out_indices:
                idx = self.out_indices.index(i)
                outs.append(self._format_out(x, actual_h, actual_w, idx, self.use_cls_token))
        return tuple(outs)

    def init_weights(self, pretrained=None):
        pass


class ViTSCoPEBackbone(nn.Module, BackboneFeatureMixin):
    def __init__(self, image_size=224, patch_size=16, dim=768, depth=12, heads=12, mlp_dim=3072, channels=3, dim_head=64, dropout=0.0, emb_dropout=0.0, drop_path_rate=0.0, out_indices=(3, 5, 7, 11), fpn_adapter_style='simple_fpn', use_cls_token=True):
        super().__init__()
        H, W = pair(image_size)
        pH, pW = pair(patch_size)
        num_patches = (H // pH) * (W // pW)
        self.patch_size = patch_size
        self.dim = dim
        self.out_indices = tuple(out_indices)
        self.use_cls_token = use_cls_token
        self.to_patch = nn.Sequential(Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=pH, p2=pW), nn.Linear(channels * pH * pW, dim))
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.drop = nn.Dropout(emb_dropout)
        self.hk_gate = HKGate(patch_size)
        self.transformer = TransformerSCoPE(dim, depth, heads, dim_head, mlp_dim, num_patches, dropout, drop_path_rate, use_cls_token)
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in self.out_indices])
        self.fpn_adapters = self._build_simple_fpn_adapters(dim, fpn_adapter_style)

    def forward(self, img):
        B, _, H, W = img.shape
        actual_h, actual_w = H // self.patch_size, W // self.patch_size
        hk_gate = self.hk_gate(img)
        x = self.to_patch(img)
        if self.use_cls_token:
            x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = self.drop(x)
        outs = []
        for i, (attn, ff, dp1, dp2) in enumerate(self.transformer.layers):
            x = x + dp1(attn(x, hk_gate))
            x = x + dp2(ff(x))
            if i in self.out_indices:
                idx = self.out_indices.index(i)
                outs.append(self._format_out(x, actual_h, actual_w, idx, self.use_cls_token))
        return tuple(outs)

    def init_weights(self, pretrained=None):
        pass


if MMDET_AVAILABLE:
    MMDET_BACKBONES.register_module(name='ViTBackbone', module=ViTBackbone, force=True)
    MMDET_BACKBONES.register_module(name='ViTCoPEBackbone', module=ViTCoPEBackbone, force=True)
    MMDET_BACKBONES.register_module(name='ViTSCoPEBackbone', module=ViTSCoPEBackbone, force=True)

if MMSEG_AVAILABLE:
    MMSEG_BACKBONES.register_module(name='ViTBackbone', module=ViTBackbone, force=True)
    MMSEG_BACKBONES.register_module(name='ViTCoPEBackbone', module=ViTCoPEBackbone, force=True)
    MMSEG_BACKBONES.register_module(name='ViTSCoPEBackbone', module=ViTSCoPEBackbone, force=True)
