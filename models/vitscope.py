import torch
from torch import nn
import math
from einops import rearrange
from einops.layers.torch import Rearrange

# ————————————————
# Helpers
# ————————————————
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# ————————————————
# CoPE: 在 Attention Logits 上加偏置（与 vitcope.py 的 CoPE 核心逻辑一致）
# ————————————————
class CoPE(nn.Module):
    """
    CoPE Implementation: Adds bias to attention logits
    Core logic same as vitcope.py (flip-cumsum, interpolation)
    """
    def __init__(self, npos_max, head_dim):
        super().__init__()
        self.npos_max = npos_max
        self.head_dim = head_dim
        # pos_emb: [1, head_dim, npos_max]
        self.pos_emb = nn.Parameter(torch.zeros(1, head_dim, npos_max))
        nn.init.xavier_uniform_(self.pos_emb)

    def forward(self, query, attn_logits):
        """
        Args:
            query: [B, H, N, head_dim] - query embeddings
            attn_logits: [B, H, N, N] - attention logits (before softmax)
        Returns:
            bias: [B, H, N, N] - bias to add to attention logits
        """
        B, H, N, D = query.shape
        
        # Memory-efficient implementation: process per query position
        # Compute logits_int once (can be reused)
        pos_emb_matrix = self.pos_emb[0]  # [head_dim, npos_max]
        logits_int = torch.matmul(query, pos_emb_matrix)  # [B, H, N, npos_max]
        
        # Process each query position separately to save memory
        bias_output = []
        for i in range(N):
            # For query position i, process all key positions
            attn_logits_i = attn_logits[:, :, i, :]  # [B, H, N]
            gates_i = torch.sigmoid(attn_logits_i)  # [B, H, N]
            
            # Flip-cumsum-flip on the key dimension
            pos_i = gates_i.flip(-1).cumsum(dim=-1).flip(-1)  # [B, H, N]
            pos_i = pos_i.clamp(max=self.npos_max - 1)
            
            # Interpolate positions
            pos_ceil_i = pos_i.ceil().long().clamp(0, self.npos_max - 1)  # [B, H, N]
            pos_floor_i = pos_i.floor().long().clamp(0, self.npos_max - 1)  # [B, H, N]
            w_i = pos_i - pos_floor_i  # [B, H, N]
            
            # Gather from logits_int for this query position
            logits_i = logits_int[:, :, i, :]  # [B, H, npos_max]
            
            logits_ceil_i = torch.gather(logits_i, dim=2, index=pos_ceil_i)  # [B, H, N]
            logits_floor_i = torch.gather(logits_i, dim=2, index=pos_floor_i)  # [B, H, N]
            
            # Interpolate
            bias_i = logits_ceil_i * w_i + logits_floor_i * (1 - w_i)  # [B, H, N]
            bias_output.append(bias_i)
        
        # Stack: [B, H, N] * N -> [B, H, N, N]
        bias = torch.stack(bias_output, dim=2)  # [B, H, N, N]
        return bias

# ————————————————
# Self Attention with SCoPE (CoPE + HKGate Fusion)
# ————————————————
class SelfAttn(nn.Module):
    """Self Attention with SCoPE: FusedGate = CoPEGate + (1+α)·(HKGate - CoPEGate)"""
    def __init__(self, dim, heads=8, dim_head=64, num_patches=196, dropout=0.):
        super().__init__()
        self.heads = heads
        self.head_dim = dim_head
        self.scale = dim_head ** -0.5
        
        inner_dim = heads * dim_head
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.cope = CoPE(npos_max=num_patches + 1, head_dim=dim_head)  # +1 for CLS token
        self.alpha = nn.Parameter(torch.tensor(0.1))  # Fusion parameter
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Args:
            x: [B, N, dim] - input tokens (with CLS token)
        """
        B, N, C = x.shape
        
        # Compute QKV
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        # Compute attention logits
        attn_logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [B, H, N, N]
        
        # CoPE: Generate bias and gate
        cope_bias = self.cope(q, attn_logits)  # [B, H, N, N]
        cope_gate = torch.sigmoid(attn_logits.mean(dim=-1))  # [B, H, N] - CoPE gate
        
        # HKGate: From attention distribution
        hk_gate = torch.softmax(attn_logits, dim=-1).max(dim=-1).values  # [B, H, N] - HK gate
        
        # SCoPE Fusion: FusedGate = CoPEGate + (1+α)·(HKGate - CoPEGate)
        fused_gate = cope_gate + (1 + self.alpha) * (hk_gate - cope_gate)  # [B, H, N]
        fused_gate = torch.sigmoid(fused_gate)  # Ensure in [0, 1]
        
        # Apply fused gate to CoPE bias: bias * fused_gate
        fused_gate_expanded = fused_gate.unsqueeze(-1)  # [B, H, N, 1]
        cope_bias_gated = cope_bias * fused_gate_expanded  # [B, H, N, N]
        
        # Add gated CoPE bias to logits
        attn_logits = attn_logits + cope_bias_gated  # ✅ SCoPE: CoPE + HKGate fusion
        
        # Apply softmax
        attn = self.attend(attn_logits)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # [B, H, N, head_dim]
        out = rearrange(out, 'b h n d -> b n (h d)')  # [B, N, dim]
        return self.to_out(out)

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
# Transformer with SCoPE
# ————————————————
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, num_patches, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PreNorm(dim, SelfAttn(dim, heads, dim_head, num_patches + 1, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]) for _ in range(depth)
        ])

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

# ————————————————
# ViTScope with SCoPE (强制使用 CLS Token)
# ————————————————
class ViTScope(nn.Module):
    """
    ViT with SCoPE (Soft CoPE with HKGate Fusion)
    - CoPE core logic same as vitcope.py
    - SCoPE: FusedGate = CoPEGate + (1+α)·(HKGate - CoPEGate)
    - 强制使用 CLS Token (CLS Token Required)
    """
    def __init__(self, *, image_size, patch_size,
                 num_classes, dim, depth, heads, mlp_dim,
                 channels=3, dim_head=64,
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
        
        # CLS Token (强制使用)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        self.dropout = nn.Dropout(emb_dropout)
        
        # Transformer with SCoPE (num_patches + 1 for CLS token)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, N, dropout)
        
        # Classification head (使用 CLS token)
        self.norm = nn.LayerNorm(dim)
        self.mlp_head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.mlp_head.weight, std=0.02)
        if self.mlp_head.bias is not None:
            nn.init.zeros_(self.mlp_head.bias)

    def forward(self, img):
        B = img.size(0)
        
        # Patch embedding
        x = self.to_patch(img)  # [B, N, dim]
        
        # Add CLS token (强制使用)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, dim]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 1+N, dim]
        
        # Dropout
        x = self.dropout(x)
        
        # Transformer with SCoPE
        x = self.transformer(x)  # [B, 1+N, dim]
        
        # Use CLS token for classification (强制使用)
        x = self.norm(x)
        cls_feat = x[:, 0]  # [B, dim] - CLS token
        return self.mlp_head(cls_feat)
