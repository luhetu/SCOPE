# 🎨 开始分割训练 - 完整指南

## 📋 任务概述

- **任务**: 语义分割（Semantic Segmentation）
- **数据集**: ADE20K（20,210 训练图像，150 类）
- **Decoder**: UPerNet（多尺度特征融合）
- **评估指标**: mIoU（Mean Intersection over Union）

---

## 🚀 快速开始

### 步骤 1: 切换环境

分割任务和检测任务共用同一个环境（都需要 mmcv-full）：

```bash
# 本地
source venv_swin_det/bin/activate

# NCC
conda activate scope_det
```

### 步骤 2: 验证环境

```bash
python -c "from mmseg.apis import train_segmentor; print('✅ MMSegmentation OK')"
```

### 步骤 3: 准备 ADE20K 数据集

#### 方法 A: 下载数据集

```bash
cd datasets/
wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
unzip ADEChallengeData2016.zip
```

#### 方法 B: 使用已有数据

```bash
# 如果已经在 datasets/ADE20K/ 下
ls datasets/ADE20K/ADEChallengeData2016/
```

#### 数据集结构

```
datasets/ADE20K/ADEChallengeData2016/
├── annotations/
│   ├── training/          # 20,210 张标注图像
│   │   ├── ADE_train_00000001.png
│   │   └── ...
│   └── validation/        # 2,000 张标注图像
│       ├── ADE_val_00000001.png
│       └── ...
├── images/
│   ├── training/          # 20,210 张训练图像
│   │   ├── ADE_train_00000001.jpg
│   │   └── ...
│   └── validation/        # 2,000 张验证图像
│       ├── ADE_val_00000001.jpg
│       └── ...
├── objectInfo150.txt      # 类别信息
└── sceneCategories.txt    # 场景类别
```

### 步骤 4: 修改配置文件

```bash
vim configs/seg_vit.yaml
```

修改数据路径：
```yaml
data_dir: /your/path/to/ADE20K/ADEChallengeData2016
```

### 步骤 5: 开始训练

```bash
# 本地训练
python train.py --cfg configs/seg_vit.yaml

# NCC 提交
sbatch scripts/submit_slurm_det.sh configs/seg_vit.yaml
```

---

## 📊 可用的分割配置

### 1. ViT-Tiny + UPerNet（首选，资源友好）

**配置**: `configs/seg_vit.yaml`

```yaml
model: vit
dim: 192
heads: 3
depth: 12
size: 512
bs: 2
n_epochs: 80
```

**预期性能**:
- mIoU: ~35-38
- 显存: ~8-10GB
- 训练时间: ~20小时（单 V100）

```bash
python train.py --cfg configs/seg_vit.yaml
```

---

### 2. CoPE + UPerNet（Attention层）

**配置**: `configs/seg_vitcope.yaml`

```yaml
model: vitcope
# CoPE 在 Attention 层
```

**对比 ViT**: 测试 Attention 层 CoPE 的效果

```bash
python train.py --cfg configs/seg_vitcope.yaml
```

---

### 3. CoPE_Embed + UPerNet（Embedding层，纯动态）

**配置**: `configs/seg_vitcope_embed.yaml`

```yaml
model: vitcope_embed
# CoPE 在 Embedding 层，无固定位置编码
```

**对比实验**: 纯动态位置编码 vs 固定位置编码

```bash
python train.py --cfg configs/seg_vitcope_embed.yaml
```

---

### 4. CoPE_EmbedFull + UPerNet（Embedding层，固定+动态）

**配置**: `configs/seg_vitcope_embedfull.yaml`

```yaml
model: vitcope_embedfull
# CoPE 在 Embedding 层，保留固定位置编码
```

**预期**: 可能表现最好（固定+动态结合）

```bash
python train.py --cfg configs/seg_vitcope_embedfull.yaml
```

---

### 5. Swin-Tiny + UPerNet（最高性能）

**配置**: `configs/seg_swin.yaml`

**预期性能**:
- mIoU: ~44-46
- 显存: ~10-12GB

```bash
python train.py --cfg configs/seg_swin.yaml
```

---

## 📈 训练监控

### 本地训练输出

```bash
Epoch [1/80] [100/2000] Loss: 1.234 | lr: 0.00006
Epoch [1/80] [200/2000] Loss: 1.156 | lr: 0.00006

Evaluating...
+--------+-------+-------+
| Class  | IoU   | Acc   |
+--------+-------+-------+
| wall   | 0.452 | 0.623 |
| floor  | 0.478 | 0.645 |
| ...    | ...   | ...   |
+--------+-------+-------+
| mIoU: 35.6 | Pixel Acc: 75.8 |
```

### NCC 查看日志

```bash
# 查看训练日志
tail -f logs/scope_det_*.out

# 查看任务状态
squeue -u $USER
```

### WandB 监控

训练会自动上传到 WandB（如果配置）：
- mIoU 曲线
- 训练 Loss
- 学习率变化
- 每类 IoU

---

## 🎯 完整实验设计

### 实验 1: Baseline 性能

```bash
# ViT-Tiny baseline
python train.py --cfg configs/seg_vit.yaml
```

**目的**: 建立基准性能

---

### 实验 2: CoPE 位置对比

```bash
# Attention 层 vs Embedding 层
python train.py --cfg configs/seg_vitcope.yaml       # Attention层
python train.py --cfg configs/seg_vitcope_embed.yaml # Embedding层
```

**对比指标**:
- mIoU 差异
- 训练速度
- 显存占用

---

### 实验 3: 固定 vs 动态位置编码

```bash
# 纯动态 vs 固定+动态
python train.py --cfg configs/seg_vitcope_embed.yaml     # 纯动态
python train.py --cfg configs/seg_vitcope_embedfull.yaml # 固定+动态
```

**对比指标**:
- mIoU 提升
- 训练稳定性

---

### 实验 4: 最佳配置

```bash
# 使用最高性能的 Swin
python train.py --cfg configs/seg_swin.yaml
```

**目的**: 获得最佳性能上限

---

## 🔄 批量实验（NCC）

### 提交所有分割实验

```bash
#!/bin/bash
# run_all_seg_experiments.sh

# ViT 系列
sbatch --job-name=seg_vit scripts/submit_slurm_det.sh configs/seg_vit.yaml
sbatch --job-name=seg_cope scripts/submit_slurm_det.sh configs/seg_vitcope.yaml
sbatch --job-name=seg_cope_emb scripts/submit_slurm_det.sh configs/seg_vitcope_embed.yaml
sbatch --job-name=seg_cope_full scripts/submit_slurm_det.sh configs/seg_vitcope_embedfull.yaml

# Swin (最高性能)
sbatch --job-name=seg_swin scripts/submit_slurm_det.sh configs/seg_swin.yaml

echo "✅ 所有分割实验已提交"
squeue -u $USER
```

---

## 📊 预期结果总结

| 模型 | mIoU (预期) | 训练时间 | 显存 |
|------|------------|---------|------|
| ViT-Tiny | 35-38 | ~20h | 8GB |
| CoPE (Attn) | 36-39 | ~22h | 8GB |
| CoPE_Embed | 36-38 | ~20h | 8GB |
| CoPE_EmbedFull | **37-40** | ~20h | 8GB |
| Swin-Tiny | **44-46** | ~24h | 12GB |

---

## ⚠️ 常见问题

### 问题 1: 环境错误

```bash
# 错误: ModuleNotFoundError: No module named 'mmseg'
# 解决: 确认在检测/分割环境
source venv_swin_det/bin/activate
python -c "import mmseg; print('OK')"
```

### 问题 2: 数据集路径

```bash
# 错误: FileNotFoundError: ADE20K not found
# 解决: 检查数据路径
ls /path/to/ADE20K/ADEChallengeData2016/images/training/
vim configs/seg_vit.yaml  # 修改 data_dir
```

### 问题 3: CUDA OOM

```bash
# 解决: 减小 batch size
vim configs/seg_vit.yaml
# bs: 2 -> bs: 1
```

---

## 📚 模型输出

### 检查点保存位置

```bash
work_dirs/upernet_vit_512x512_80k_ade20k/
├── epoch_10.pth
├── epoch_20.pth
├── ...
├── latest.pth
└── best_mIoU_epoch_65.pth  # 最佳模型
```

### 可视化结果

分割结果会保存在：
```bash
work_dirs/upernet_vit_*/vis_results/
├── ADE_val_00000001_pred.png
├── ADE_val_00000002_pred.png
└── ...
```

---

## 🎯 下一步

分割训练成功后：

1. **分析结果**: 对比不同模型的 mIoU
2. **可视化**: 查看分割结果图像
3. **消融实验**: 分析 CoPE 的贡献
4. **写论文**: 整理实验结果和图表

---

## 📞 需要帮助？

- 环境问题: 查看 `INSTALL_DETECTION_ON_NCC.md`
- 集群使用: 查看 `NCC_TUTORIAL.md`
- CoPE 对比: 查看 `COPE_VARIANTS.md`

---

**祝训练顺利！** 🎨🚀

从 ViT 开始是正确的选择，它会帮你：
- ✅ 验证分割流程
- ✅ 建立性能基准
- ✅ 为 CoPE 消融实验做准备

