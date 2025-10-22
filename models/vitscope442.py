# -*- coding: utf-8 -*-
# === ViT-CoPE 版本：去掉cls token，直接 mean pool 所有 tokens ===
import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange

# 辅助函数
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class CNNGateV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max4 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        # 1) 全局对比增强
        mu    = self.global_avgpool(x)
        delta = x - mu
        x = x + (1 + self.alpha) * delta

        # 2) 混合池化：2×2 → 2×2 → 4×4 → 2×2
        x = self.max4(x)   # 256→128
        x = self.max4(x)   # 128→64
        
        x = self.max2(x)   # 16 → 8

        # 3) 把通道平均后展平，得到 [B, 8*8]
        x = x.mean(dim=1)  # [B, 8, 8]
        return x.flatten(1)  # [B, 64]
class CoPE(nn.Module):
    def __init__(self, npos_max, dim_head):
        super().__init__()
        self.npos_max = npos_max
        self.dim_head = dim_head
        # 初始化位置嵌入参数
        self.pos_emb = nn.Parameter(torch.zeros(1, dim_head, npos_max))
        nn.init.xavier_uniform_(self.pos_emb)

    def forward(self, q, attn_logits):
        # 计算 CoPE 门控值
        gates = torch.sigmoid(attn_logits.mean(dim=-1))  # [B, H, N]
        # 计算偏移位置
        pos = gates.flip(-1).cumsum(dim=-1).flip(-1)
        pos = pos.clamp(min=0, max=self.npos_max - 1)
        pos_floor = pos.floor().long()
        pos_ceil = pos.ceil().long()
        w = pos - pos_floor

        # 双线性插值获取偏移嵌入
        pos_emb_2d = self.pos_emb[0].transpose(0, 1)  # [N, D]
        b, h, n = pos_floor.shape
        gather_floor = pos_emb_2d.index_select(0, pos_floor.reshape(-1)).view(b, h, n, self.dim_head)
        gather_ceil = pos_emb_2d.index_select(0, pos_ceil.reshape(-1)).view(b, h, n, self.dim_head)
        w = w.unsqueeze(-1)
        offset = gather_floor * (1 - w) + gather_ceil * w
        return offset, gates  # 返回偏移和 CoPE gate

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        # 先 LayerNorm，再传递给 fn
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
    def __init__(self, dim, heads=8, dim_head=64, num_patches=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.cope = CoPE(npos_max=num_patches, dim_head=dim_head)
        self.alpha = nn.Parameter(torch.tensor(0.1))  # 融合时使用的可学习参数

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, cnn_feat):
        # x: [B, N, dim]; cnn_feat: [B, N]
        # 生成 Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        # 原始自注意力得分
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        offset, cope_gate = self.cope(q, dots)  # 计算偏移和 CoPE 门控 [B, H, N]

        # 融合 CoPE gate 与 CNN gate
        gate = cnn_feat.unsqueeze(1).expand(-1, self.heads, -1)  # [B, H, N]
        fused_gate = cope_gate + (1 + self.alpha) * (gate - cope_gate)
        fused_gate = torch.sigmoid(fused_gate)  # [B, H, N]

        # 根据融合后的门控加权偏移
        offset = offset * fused_gate.unsqueeze(-1)  # [B, H, N, D]
        q_new = q + offset

        # 使用加权后的 Q 计算新的注意力
        dots_new = torch.matmul(q_new, k.transpose(-1, -2)) * self.scale
        attn = torch.softmax(dots_new, dim=-1)
        out = torch.matmul(attn, v)  # [B, H, N, D]
        # 合并多头输出
        return self.to_out(rearrange(out, 'b h n d -> b n (h d)'))

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, num_patches, dropout=0.):
        super().__init__()
        # 按层堆叠注意力与前馈网络
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, num_patches, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ])
            for _ in range(depth)
        ])

    def forward(self, x, cnn_feat):
        for attn, ff in self.layers:
            x = attn(x, cnn_feat) + x
            x = ff(x) + x
        return x

class ViTcope(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim,
                 channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0
        # 计算 patch 数量
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        # Patch 嵌入
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim)
        )
        self.dropout = nn.Dropout(emb_dropout)
        # Transformer 编码器
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, num_patches, dropout)
        # 分类头
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        # CNNGate 分支
        self.cnn_gate = CNNGateV2()

    def forward(self, img):
        # 生成 patch tokens
        x = self.to_patch_embedding(img)
        x = self.dropout(x)
        # 计算 CNNGate 输出
        cnn_feat = self.cnn_gate(img)  # [B, N]
        # 通过 Transformer
        x = self.transformer(x, cnn_feat)
        # 对所有 token 做平均池化
        x = x.mean(dim=1)
        return self.mlp_head(x)