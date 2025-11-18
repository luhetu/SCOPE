# -*- coding: utf-8 -*-
"""
ViT as Backbone for Object Detection
Transform ViT into a multi-scale feature extractor for object detection
"""
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

import sys
sys.path.append('..')

# Register to both mmdet and mmseg BACKBONES
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


class ViTBackbone(nn.Module):
    """
    ViT Backbone for Object Detection and Segmentation
    Output multi-scale feature maps for FPN/UPerNet
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
        out_indices=(2, 5, 8, 11),  # Which layers to output features from
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

        # Add norm layer for each output
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in out_indices])

    def _resize_pos_embed(self, pos_embed, img_h, img_w):
        """
        Interpolate position embeddings to adapt to different image sizes
        Args:
            pos_embed: (1, N+1, D) Original position embeddings (including cls token)
            img_h: Input image height
            img_w: Input image width
        Returns:
            (1, new_N+1, D) Interpolated position embeddings
        """
        import torch.nn.functional as F
        
        # Separate cls token and patch tokens
        cls_pos_embed = pos_embed[:, :1, :]  # (1, 1, D)
        patch_pos_embed = pos_embed[:, 1:, :]  # (1, N, D)
        
        # Calculate original patch grid size (assuming square)
        N = patch_pos_embed.shape[1]
        h = w = int(N ** 0.5)
        
        # Calculate new patch grid size (based on actual image size)
        new_h = img_h // self.patch_size
        new_w = img_w // self.patch_size
        
        # Reshape and interpolate
        patch_pos_embed = patch_pos_embed.reshape(1, h, w, -1).permute(0, 3, 1, 2)  # (1, D, h, w)
        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=(new_h, new_w),
            mode='bicubic',
            align_corners=False
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, pos_embed.shape[-1])  # (1, new_N, D)
        
        # Concatenate back to cls token
        return torch.cat([cls_pos_embed, patch_pos_embed], dim=1)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            tuple of feature maps at different scales
        """
        b, c, h, w = x.shape
        
        # Patch embedding
        patches = self.to_patch_embedding(x)
        
        # Add cls token
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, patches), dim=1)
        
        # Position embedding interpolation (handle different image sizes)
        if x.shape[1] != self.pos_embedding.shape[1]:
            pos_embed = self._resize_pos_embed(self.pos_embedding, h, w)
            x = x + pos_embed
        else:
            x = x + self.pos_embedding
        
        x = self.dropout(x)

        # Calculate actual patch grid size
        actual_h = h // self.patch_size
        actual_w = w // self.patch_size
        
        # Pass through Transformer blocks, output features at specified layers
        outs = []
        for i, (attn, ff) in enumerate(self.transformer_blocks):
            x = attn(x) + x
            x = ff(x) + x
            
            if i in self.out_indices:
                # Remove cls token, reshape to 2D feature map
                out = x[:, 1:]  # (B, N, D)
                out = rearrange(
                    out, 
                    'b (h w) d -> b d h w', 
                    h=actual_h, 
                    w=actual_w
                )
                # Apply norm
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


class CoPE(nn.Module):
    """Contextual Position Encoding"""
    def __init__(self, npos_max, dim_head):
        super().__init__()
        self.npos_max = npos_max
        self.dim_head = dim_head
        self.pos_emb = nn.Parameter(torch.zeros(1, dim_head, npos_max))
        nn.init.xavier_uniform_(self.pos_emb)

    def forward(self, q, attn_logits):
        gate = torch.sigmoid(attn_logits.mean(dim=-1))
        pos = gate.flip(-1).cumsum(dim=-1).flip(-1)
        pos = pos.clamp(0, self.npos_max-1)
        f = pos.floor().long()
        c = pos.ceil().long()
        w = (pos - f).unsqueeze(-1)
        emb2d = self.pos_emb[0].transpose(0,1)
        B,H,N = f.shape
        f_idx = f.reshape(-1)
        c_idx = c.reshape(-1)
        e_f = emb2d.index_select(0, f_idx).view(B,H,N,self.dim_head)
        e_c = emb2d.index_select(0, c_idx).view(B,H,N,self.dim_head)
        offset = e_f * (1 - w) + e_c * w
        return offset


class AttentionCoPE(nn.Module):
    """Attention with CoPE"""
    def __init__(self, dim, heads=8, dim_head=64, num_patches=196, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.cope = CoPE(npos_max=num_patches, dim_head=dim_head)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        logits = torch.matmul(q, k.transpose(-1,-2)) * self.scale
        offset = self.cope(q, logits)
        q2 = q + offset
        dots = torch.matmul(q2, k.transpose(-1,-2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class HKPool(nn.Module):
    """HK pool mechanism - dynamically align to patch grid"""
    def __init__(self, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.max4 = nn.MaxPool2d(4, 4)
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # x: [B,C,H,W]
        B, C, H, W = x.shape
        
        # Calculate target patch grid size (dynamic)
        h_p = H // self.patch_size
        w_p = W // self.patch_size
        
        mu = self.global_avgpool(x)
        x = x + (1 + self.alpha) * (x - mu)  # Contrast enhancement
        x = self.max4(x)  # Downsample
        x = self.max4(x)  # Downsample again
        
        # Dynamically align to (h_p, w_p)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (h_p, w_p))
        x = x.mean(dim=1)  # [B, h_p, w_p]
        gate = x.flatten(1)  # [B, N]
        
        # Per-sample normalize to (0,1)
        gate = (gate - gate.mean(dim=1, keepdim=True)) / (gate.std(dim=1, keepdim=True) + 1e-5)
        gate = torch.sigmoid(self.beta * gate)
        return gate  # [B, N] ∈ (0,1)


class AttentionSCoPE(nn.Module):
    """Attention with SCoPE - Q-offset + λ convex combination fusion + CLS gate (aligned with vitscope.py)"""
    def __init__(self, dim, heads=8, dim_head=64, num_patches=196, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.cope = CoPE(npos_max=num_patches + 1, dim_head=dim_head)  # +1 includes CLS
        self.lam = nn.Parameter(torch.tensor(0.5))  # Learn λ for convex combination
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, hk_gate_1d):
        # x: [B,1+N,dim], hk_gate_1d: [B,N] (aligned to patches)
        B, N_all, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1,-2)) * self.scale  # [B,H,N_all,N_all]
        offset = self.cope(q, dots)  # offset:[B,H,N_all,D]
        cope_gate = torch.sigmoid(dots.mean(dim=-1))  # [B,H,N_all]

        # CLS gate from CoPE column 0 (multi-head average), concatenated with HKGate
        cls_from_cope = cope_gate[:, :, 0].mean(dim=1, keepdim=True)  # [B,1]
        hk_full = torch.cat([cls_from_cope, hk_gate_1d], dim=1)  # [B,N_all]
        hk_full = hk_full.unsqueeze(1).expand(-1, self.heads, -1)  # [B,H,N_all]

        # Convex combination fusion (avoid secondary sigmoid flattening dynamic range)
        lam = torch.sigmoid(self.lam)  # (0,1)
        fused_gate = lam * cope_gate + (1 - lam) * hk_full  # [B,H,N_all] ∈ (0,1)

        q_new = q + offset * fused_gate.unsqueeze(-1)  # Q-offset with gate
        attn = torch.softmax(torch.matmul(q_new, k.transpose(-1,-2)) * self.scale, dim=-1)
        out = torch.matmul(attn, v)  # [B,H,N_all,D]
        return self.to_out(rearrange(out, 'b h n d -> b n (h d)'))


class ViTCoPEBackbone(nn.Module):
    """ViT with CoPE for Object Detection and Segmentation"""
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
        out_indices=(2, 5, 8, 11),
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
        
        # Transformer blocks with CoPE
        self.transformer_blocks = nn.ModuleList([])
        for _ in range(depth):
            self.transformer_blocks.append(nn.ModuleList([
                PreNorm(dim, AttentionCoPE(dim, heads=heads, dim_head=dim_head, num_patches=num_patches, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
        
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in out_indices])
    
    def _resize_pos_embed(self, pos_embed, img_h, img_w):
        """Interpolate position embeddings to adapt to different image sizes"""
        import torch.nn.functional as F
        cls_pos_embed = pos_embed[:, :1, :]
        patch_pos_embed = pos_embed[:, 1:, :]
        N = patch_pos_embed.shape[1]
        h = w = int(N ** 0.5)
        new_h = img_h // self.patch_size
        new_w = img_w // self.patch_size
        patch_pos_embed = patch_pos_embed.reshape(1, h, w, -1).permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(patch_pos_embed, size=(new_h, new_w), mode='bicubic', align_corners=False)
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, pos_embed.shape[-1])
        return torch.cat([cls_pos_embed, patch_pos_embed], dim=1)
    
    def forward(self, x):
        b, c, h, w = x.shape
        patches = self.to_patch_embedding(x)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, patches), dim=1)
        
        # Position embedding 插Value
        if x.shape[1] != self.pos_embedding.shape[1]:
            pos_embed = self._resize_pos_embed(self.pos_embedding, h, w)
            x = x + pos_embed
        else:
            x = x + self.pos_embedding
        
        x = self.dropout(x)
        
        # Calculate actual patch grid size
        actual_h = h // self.patch_size
        actual_w = w // self.patch_size
        
        outs = []
        for i, (attn, ff) in enumerate(self.transformer_blocks):
            x = attn(x) + x
            x = ff(x) + x
            
            if i in self.out_indices:
                out = x[:, 1:]
                out = rearrange(out, 'b (h w) d -> b d h w', h=actual_h, w=actual_w)
                norm_idx = self.out_indices.index(i)
                out_normed = rearrange(out, 'b d h w -> b h w d')
                out_normed = self.norms[norm_idx](out_normed)
                out = rearrange(out_normed, 'b h w d -> b d h w')
                outs.append(out)
        
        return tuple(outs)
    
    def init_weights(self, pretrained=None):
        pass


class ViTSCoPEBackbone(nn.Module):
    """ViT with SCoPE for Object Detection and Segmentation"""
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
        out_indices=(2, 5, 8, 11),
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
        
        # CNN Gate - dynamically align to patch grid
        self.hk_pool = HKPool(patch_size=patch_size)  # ✅ Pass patch_size for dynamic calculation
        
        # Transformer blocks with SCoPE
        self.transformer_blocks = nn.ModuleList([])
        for _ in range(depth):
            self.transformer_blocks.append(nn.ModuleList([
                PreNorm(dim, AttentionSCoPE(dim, heads=heads, dim_head=dim_head, num_patches=num_patches, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
        
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in out_indices])
    
    def _resize_pos_embed(self, pos_embed, img_h, img_w):
        """Interpolate position embeddings to adapt to different image sizes"""
        import torch.nn.functional as F
        cls_pos_embed = pos_embed[:, :1, :]
        patch_pos_embed = pos_embed[:, 1:, :]
        N = patch_pos_embed.shape[1]
        h = w = int(N ** 0.5)
        new_h = img_h // self.patch_size
        new_w = img_w // self.patch_size
        patch_pos_embed = patch_pos_embed.reshape(1, h, w, -1).permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(patch_pos_embed, size=(new_h, new_w), mode='bicubic', align_corners=False)
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, pos_embed.shape[-1])
        return torch.cat([cls_pos_embed, patch_pos_embed], dim=1)
    
    def forward(self, img):
        # img: Original image input [B, C, H, W]
        b, c, h, w = img.shape
        
        # 1. HK Gate from original image (aligned with vitscope.py)
        hk_gate_1d = self.hk_pool(img)  # [B, N] ✅ Use original image!
        
        # 2. Patch embedding
        patches = self.to_patch_embedding(img)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, patches), dim=1)  # [B, 1+N, dim]
        
        # 3. Position embedding interpolation
        if x.shape[1] != self.pos_embedding.shape[1]:
            pos_embed = self._resize_pos_embed(self.pos_embedding, h, w)
            x = x + pos_embed
        else:
            x = x + self.pos_embedding
        
        x = self.dropout(x)
        
        # Calculate actual patch grid size
        actual_h = h // self.patch_size
        actual_w = w // self.patch_size
        
        # 4. Transformer with hk_gate_1d
        outs = []
        for i, (attn, ff) in enumerate(self.transformer_blocks):
            x = attn(x, hk_gate_1d) + x  # ✅ Pass hk_gate_1d, not hk_feat!
            x = ff(x) + x
            
            if i in self.out_indices:
                out = x[:, 1:]
                out = rearrange(out, 'b (h w) d -> b d h w', h=actual_h, w=actual_w)
                norm_idx = self.out_indices.index(i)
                out_normed = rearrange(out, 'b d h w -> b h w d')
                out_normed = self.norms[norm_idx](out_normed)
                out = rearrange(out_normed, 'b h w d -> b d h w')
                outs.append(out)
        
        return tuple(outs)
    
    def init_weights(self, pretrained=None):
        pass



# ==================== Register to MMDetection and MMSegmentation ==================== #
if MMDET_AVAILABLE:
    MMDET_BACKBONES.register_module(name='ViTBackbone', module=ViTBackbone)
    MMDET_BACKBONES.register_module(name='ViTCoPEBackbone', module=ViTCoPEBackbone)
    MMDET_BACKBONES.register_module(name='ViTSCoPEBackbone', module=ViTSCoPEBackbone)

if MMSEG_AVAILABLE:
    MMSEG_BACKBONES.register_module(name='ViTBackbone', module=ViTBackbone)
    MMSEG_BACKBONES.register_module(name='ViTCoPEBackbone', module=ViTCoPEBackbone)
    MMSEG_BACKBONES.register_module(name='ViTSCoPEBackbone', module=ViTSCoPEBackbone)
