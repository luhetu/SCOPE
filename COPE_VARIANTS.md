# 🔬 CoPE 变体模型对比

本项目实现了 **3 种 CoPE (Contextual Position Encoding)** 变体，用于消融实验和性能对比。

---

## 📊 三种变体对比

| 变体 | 文件 | CoPE 应用位置 | 固定位置编码 | 配置文件 |
|------|------|--------------|------------|----------|
| **ViTCoPE** | `vitcope.py` | Attention 层 | ❌ 删除 | `vitcope.yaml` |
| **ViTCoPE_Embed** | `vitcope_embed.py` | Embedding 层 | ❌ 删除 | `vitcope_embed.yaml` |
| **ViTCoPE_EmbedFull** | `vitcope_embedfull.py` | Embedding 层 | ✅ 保留 | `vitcope_embedfull.yaml` |

---

## 🔍 详细说明

### 1. ViTCoPE (原版，Attention层)

**文件**: `models/vitcope.py`

**特点**:
- 在 **Attention 机制**中应用 CoPE
- **删除**传统的固定位置编码
- 每个 Attention head 独立计算动态位置偏移

**实现**:
```python
class AttentionCoPE(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, num_patches=64, dropout=0.):
        # ...
        self.cope = CoPE(npos_max=num_patches, dim_head=dim_head)
    
    def forward(self, x):
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        logits = torch.matmul(q, k.transpose(-1,-2)) * self.scale
        offset = self.cope(q, logits)  # 动态位置偏移
        q2 = q + offset
        # ...
```

**优点**:
- 每层都能自适应位置编码
- 更灵活的位置表示

**缺点**:
- 计算量更大（每层都要计算 CoPE）

---

### 2. ViTCoPE_Embed (Embedding层，纯动态)

**文件**: `models/vitcope_embed.py`

**特点**:
- 在 **Embedding 阶段**应用 CoPE
- **删除**传统的固定位置编码
- 使用 **纯动态**位置编码
- Transformer 层使用标准 Attention（未修改）

**实现**:
```python
class CoPE(nn.Module):
    def __init__(self, npos_max, dim_head):
        self.pos_emb = nn.Parameter(torch.zeros(1, dim_head, npos_max))
    
    def forward(self, q, attn_logits):
        gate = torch.sigmoid(attn_logits.mean(dim=-1))
        pos = gate.flip(-1).cumsum(dim=-1).flip(-1)
        # 插值获取动态位置编码
        offset = interpolate(self.pos_emb, pos)
        return offset

class ViTcope(nn.Module):
    def forward(self, img):
        x = self.to_patch(img)  # [B, N, dim]
        # 注意：没有固定位置编码
        x = self.transformer(x)  # 标准 Transformer
        x = x.mean(dim=1)        # Mean pooling (无 CLS token)
        return self.mlp_head(x)
```

**优点**:
- 只在 embedding 阶段计算一次，更高效
- Transformer 层使用标准实现
- 对比实验：评估 CoPE 在不同位置的效果

**缺点**:
- 缺少固定位置编码的归纳偏置

---

### 3. ViTCoPE_EmbedFull (Embedding层，固定+动态)

**文件**: `models/vitcope_embedfull.py`

**特点**:
- 在 **Embedding 阶段**应用 CoPE
- **保留**传统的固定位置编码
- 使用 **固定 + 动态** 混合位置编码
- Transformer 层使用标准 Attention（未修改）

**实现**:
```python
class ViTCoPE_EmbedFull(nn.Module):
    def __init__(self, ...):
        # 固定位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim))
        # 动态位置偏移
        self.cope_emb = CoPEEmbedding(num_patches, dim)
    
    def forward(self, img):
        x = self.to_patch(img)           # [B, N, C]
        cope_offset = self.cope_emb(x)   # 动态偏移
        x = x + self.pos_embed + cope_offset  # 固定 + 动态
        if self.use_cls:
            x = torch.cat([cls_token, x], dim=1)
        x = self.blocks(x)               # 标准 Transformer
        return self.head(x[:, 0] if self.use_cls else x.mean(dim=1))
```

**优点**:
- 结合固定和动态位置编码的优势
- 保留传统 ViT 的归纳偏置
- 支持 CLS token

**缺点**:
- 参数量稍多（多了固定位置编码）

---

## 🧪 实验设计

这三个变体可以用于以下消融实验：

### 实验 1: CoPE 应用位置
- **ViTCoPE** vs **ViTCoPE_Embed**
- 对比：Attention 层 vs Embedding 层

### 实验 2: 固定位置编码的作用
- **ViTCoPE_Embed** vs **ViTCoPE_EmbedFull**
- 对比：纯动态 vs 固定+动态

### 实验 3: 与标准 ViT 对比
- **ViT** vs **ViTCoPE_Embed** vs **ViTCoPE_EmbedFull**
- 对比：固定位置编码 vs 动态位置编码

---

## 🚀 使用方法

### 测试所有模型

```bash
# 运行测试脚本
python test_new_models.py
```

### 训练

```bash
# 1. ViTCoPE (原版，Attention层)
python train.py --cfg configs/vitcope.yaml

# 2. ViTCoPE_Embed (Embedding层，纯动态)
python train.py --cfg configs/vitcope_embed.yaml

# 3. ViTCoPE_EmbedFull (Embedding层，固定+动态)
python train.py --cfg configs/vitcope_embedfull.yaml
```

### 在 NCC 集群上运行

```bash
# 提交单个任务
sbatch scripts/submit_slurm_cls.sh configs/vitcope_embed.yaml

# 批量提交所有 CoPE 变体
sbatch --job-name=cope scripts/submit_slurm_cls.sh configs/vitcope.yaml
sbatch --job-name=cope_embed scripts/submit_slurm_cls.sh configs/vitcope_embed.yaml
sbatch --job-name=cope_full scripts/submit_slurm_cls.sh configs/vitcope_embedfull.yaml
```

---

## 📊 配置参数

所有三个变体使用相同的配置参数：

```yaml
# 共同参数
task: cls
data_dir: /path/to/ImageNet
bs: 256
size: 224
n_epochs: 100
patch: 32
dim: 384
depth: 12
heads: 6
mlp_dim: 1536
lr: 3e-4
min_lr: 1e-5
warmup_epochs: 2
opt: adamw
amp: true
aug: true

# vitcope_embed.yaml 额外参数
dim_head: 64  # 每个 attention head 的维度
```

---

## 🔬 预期结果

基于理论分析，预期性能：

1. **ViTCoPE_EmbedFull** ≥ **ViTCoPE** ≥ **ViTCoPE_Embed**
   - 固定+动态可能效果最好
   - Attention 层 CoPE 更灵活但计算量大
   - 纯动态可能缺少归纳偏置

2. **计算效率**:
   - ViTCoPE_Embed = ViTCoPE_EmbedFull > ViTCoPE
   - Embedding 层只计算一次更快

3. **参数量**:
   - ViTCoPE_EmbedFull > ViTCoPE_Embed
   - 固定位置编码额外参数：`num_patches × dim`

---

## 📚 参考文献

- **CoPE**: [Contextual Position Encoding](https://arxiv.org/abs/2405.18719)
- **ViT**: [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)

---

**实验建议**:
1. 先用小模型（Tiny）快速验证
2. 在 CIFAR-10/100 上做消融实验
3. 最后在 ImageNet 上完整训练

