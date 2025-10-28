# -*- coding: utf-8 -*-
"""
ViT as Backbone for Object Detection
将 ViT 改造为可用于目标检测的多尺度特征提取器
"""
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

import sys
sys.path.append('..')
from mmdet.models.builder import BACKBONES


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


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


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


@BACKBONES.register_module()
class ViTBackbone(nn.Module):
    """
    ViT Backbone for Object Detection
    输出多尺度特征图用于 FPN
    """
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        channels=3,
        dim_head=64,
        dropout=0.,
        emb_dropout=0.,
        out_indices=(2, 5, 8, 11),  # 从哪些层输出特征
    ):
        super().__init__()
        
        image_height, image_width = (image_size, image_size) if isinstance(image_size, int) else image_size
        patch_height, patch_width = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size

        assert image_height % patch_height == 0 and image_width % patch_width == 0

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.patch_size = patch_size
        self.dim = dim
        self.out_indices = out_indices
        self.num_patches_h = image_height // patch_height
        self.num_patches_w = image_width // patch_width

        # Patch Embedding
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([])
        for _ in range(depth):
            self.transformer_blocks.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

        # 为每个输出添加 norm layer
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in out_indices])

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            tuple of feature maps at different scales
        """
        b = x.shape[0]
        
        # Patch embedding
        x = self.to_patch_embedding(x)
        
        # Add cls token
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)

        # 通过 Transformer blocks，在指定层输出特征
        outs = []
        for i, (attn, ff) in enumerate(self.transformer_blocks):
            x = attn(x) + x
            x = ff(x) + x
            
            if i in self.out_indices:
                # 移除 cls token，reshape 到 2D 特征图
                out = x[:, 1:]  # (B, N, D)
                out = rearrange(
                    out, 
                    'b (h w) d -> b d h w', 
                    h=self.num_patches_h, 
                    w=self.num_patches_w
                )
                # 应用 norm
                norm_idx = self.out_indices.index(i)
                out_normed = rearrange(out, 'b d h w -> b h w d')
                out_normed = self.norms[norm_idx](out_normed)
                out = rearrange(out_normed, 'b h w d -> b d h w')
                outs.append(out)

        return tuple(outs)

    def init_weights(self, pretrained=None):
        """Initialize weights"""
        if pretrained:
            # TODO: load pretrained weights
            pass
        else:
            # Initialize with default
            pass


# ======================== CoPE Module ========================
class CoPE(nn.Module):
    """Contextual Position Encoding - Dynamic position offset"""
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
        B, H, N = f.shape
        f_idx = f.reshape(-1)
        c_idx = c.reshape(-1)
        e_f = emb2d.index_select(0, f_idx).view(B, H, N, self.dim_head)
        e_c = emb2d.index_select(0, c_idx).view(B, H, N, self.dim_head)
        # 5) interpolate
        offset = e_f * (1 - w) + e_c * w  # [B, H, N, D_head]
        return offset


class CoPEAttention(nn.Module):
    """Attention with CoPE (Contextual Position Encoding)"""
    def __init__(self, dim, heads=8, dim_head=64, num_patches=64, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.cope = CoPE(npos_max=num_patches, dim_head=dim_head)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(
            nn.Linear(inner, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: [B, N, dim]
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # raw attention logits
        logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [B,H,N,N]
        # CoPE offset
        offset = self.cope(q, logits)  # [B,H,N,D_head]

        # apply to q
        q2 = q + offset
        # recompute attention
        attn = self.attend(torch.matmul(q2, k.transpose(-1, -2)) * self.scale)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # [B,H,N,D_head]
        # merge heads
        out2 = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out2)


@BACKBONES.register_module()
class ViTCoPEBackbone(nn.Module):
    """
    ViT with CoPE (Contextual Position Encoding) Backbone for Object Detection
    Based on vitcope_embed.py implementation
    """
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        channels=3,
        dim_head=64,
        dropout=0.,
        emb_dropout=0.,
        out_indices=(2, 5, 8, 11),  # 从哪些层输出特征
    ):
        super().__init__()
        
        image_height, image_width = (image_size, image_size) if isinstance(image_size, int) else image_size
        patch_height, patch_width = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size

        assert image_height % patch_height == 0 and image_width % patch_width == 0

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.patch_size = patch_size
        self.dim = dim
        self.out_indices = out_indices
        self.num_patches_h = image_height // patch_height
        self.num_patches_w = image_width // patch_width
        self.num_patches = num_patches

        # Patch Embedding
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # CoPE Transformer blocks
        self.transformer_blocks = nn.ModuleList([])
        for _ in range(depth):
            self.transformer_blocks.append(nn.ModuleList([
                PreNorm(dim, CoPEAttention(dim, heads=heads, dim_head=dim_head, 
                                           num_patches=num_patches+1, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

        # 为每个输出添加 norm layer
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in out_indices])

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            tuple of feature maps at different scales
        """
        b = x.shape[0]
        
        # Patch embedding
        x = self.to_patch_embedding(x)
        
        # Add cls token
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)

        # 通过 CoPE Transformer blocks，在指定层输出特征
        outs = []
        for i, (attn, ff) in enumerate(self.transformer_blocks):
            x = attn(x) + x
            x = ff(x) + x
            
            if i in self.out_indices:
                # 移除 cls token，reshape 到 2D 特征图
                out = x[:, 1:]  # (B, N, D)
                out = rearrange(
                    out, 
                    'b (h w) d -> b d h w', 
                    h=self.num_patches_h, 
                    w=self.num_patches_w
                )
                # 应用 norm
                norm_idx = self.out_indices.index(i)
                out_normed = rearrange(out, 'b d h w -> b h w d')
                out_normed = self.norms[norm_idx](out_normed)
                out = rearrange(out_normed, 'b h w d -> b d h w')
                outs.append(out)

        return tuple(outs)

    def init_weights(self, pretrained=None):
        """Initialize weights"""
        if pretrained:
            # TODO: load pretrained weights
            pass
        else:
            # Initialize with default
            pass


@BACKBONES.register_module()
class ViTSCoPEBackbone(ViTBackbone):
    """ViT with SCoPE for Object Detection"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # TODO: 添加 SCoPE 特定的实现
        pass

