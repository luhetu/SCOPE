# 🚀 SCOPE 项目快速开始指南

统一的 ViT / CoPE / SCoPE / Swin Transformer 训练框架，支持分类、检测、分割三大任务。

---

## 📂 项目结构

```
SCOPE/
├── configs/              # 配置文件
│   ├── vit.yaml         # 分类配置
│   ├── detection_*.yaml # 检测配置
│   └── seg_*.yaml       # 分割配置
├── models/              # 模型定义
│   ├── vit.py
│   ├── vitcope.py
│   ├── vitscope442.py
│   ├── swin_transformer.py
│   └── vit_backbone.py  # 检测/分割用
├── tasks/               # 任务实现
│   ├── classification.py
│   ├── detection.py
│   └── segmentation.py
├── datasets/            # 数据集
│   ├── ADE20K/
│   └── coco/
├── train.py             # 统一训练入口
├── requirements_classification.txt
└── requirements_detection.txt
```

---

## 🎯 支持的任务

### 1️⃣ 图像分类 (ImageNet-1K)
- ✅ ViT (Tiny: dim=192, heads=3)
- ✅ ViT + CoPE
- ✅ ViT + SCoPE
- ✅ Swin Transformer V2 (Tiny)

### 2️⃣ 目标检测 (COCO)
- ✅ Swin + Mask R-CNN
- ✅ ViT/CoPE/SCoPE + Mask R-CNN (Tiny, 512x512)

### 3️⃣ 语义分割 (ADE20K)
- ✅ Swin + UPerNet
- ✅ ViT/CoPE/SCoPE + UPerNet (Tiny)

---

## ⚙️ 环境设置

### 分类任务（基础环境）
```bash
pip install -r requirements_classification.txt
```

### 检测/分割任务（专用环境）
```bash
# 详见 INSTALL_DETECTION_ENV.md
python3.7 -m venv venv_detection
source venv_detection/bin/activate
pip install -r requirements_detection.txt
```

---

## 🏃 快速运行

### 分类训练
```bash
# ViT Tiny
python train.py --cfg configs/vit.yaml

# CoPE
python train.py --cfg configs/vitcope.yaml

# SCoPE
python train.py --cfg configs/vitscope.yaml

# Swin
python train.py --cfg configs/swin.yaml
```

### 检测训练
```bash
# 切换到检测环境
source venv_detection/bin/activate

# Swin 检测（推荐）
python train.py --cfg configs/detection_swin.yaml

# ViT 检测（Tiny + 512x512）
python train.py --cfg configs/detection_vit.yaml
```

### 分割训练
```bash
# 使用检测环境
source venv_detection/bin/activate

# Swin 分割
python train.py --cfg configs/seg_swin.yaml

# ViT 分割
python train.py --cfg configs/seg_vit.yaml
```

---

## 📊 配置说明

所有配置文件都在 `configs/` 目录下，使用 YAML 格式：

### 分类配置示例 (`configs/vit.yaml`)
```yaml
task: cls                # 任务类型: cls / det / seg
model: vit               # 模型: vit / vitcope / vitscope / swin
data_dir: /path/to/ImageNet
bs: 256                  # Batch size
size: 224                # 图像尺寸
patch: 16                # Patch 大小
dim: 192                 # 模型维度 (Tiny)
depth: 12                # Transformer 深度
heads: 3                 # 注意力头数 (Tiny)
mlp_dim: 768             # MLP 维度
n_epochs: 100            # 训练轮数
lr: 3e-4                 # 学习率
opt: adamw               # 优化器
amp: true                # 混合精度训练
aug: true                # 数据增强
nowandb: false           # WandB 日志
```

### 检测配置示例 (`configs/detection_vit.yaml`)
```yaml
task: det
model: vit
data_dir: /path/to/coco
bs: 1                    # ViT 显存占用大
img_scale: (512, 512)    # 降低分辨率
dim: 192                 # Tiny 模型
# ... 其他参数
```

---

## 💾 模型检查点

训练的模型会自动保存到 `checkpoint/` 目录：
- `{model}_best.pth` - 验证集最佳模型
- `{model}_last.pth` - 最后一个 epoch

检测/分割模型保存到 `work_dirs/` 目录。

---

## 📈 训练监控

### WandB 集成
所有任务都支持 WandB 日志记录：
- 分类: `imagenet-1k-scope` 项目
- 检测: `coco-detection` 项目
- 分割: `ade20k-segmentation` 项目

禁用 WandB: 在配置文件中设置 `nowandb: true`

### 输出格式
```
[========== 10010/10010 ========>] Step: 30ms | Tot: 36m12s | Loss:6.920 | Acc:0.10%
[Epoch 000] TrainAcc=0.10% | ValAcc=0.12% | LR=0.000400 | Time=45.01 min
💾 Saved: checkpoint/vit_best.pth (acc=0.12%)
```

---

## 🔧 显存优化

### GPU 显存要求
| 任务 | 模型 | 最小显存 | 推荐配置 |
|------|------|----------|----------|
| 分类 | ViT Tiny | 4GB | `bs: 256` |
| 分类 | Swin Tiny | 8GB | `bs: 128` |
| 检测 | Swin | 12GB | `bs: 2` |
| 检测 | ViT Tiny | 4GB | `bs: 1, 512x512` |
| 分割 | All | 8-12GB | `bs: 2` |

### 显存不足？
1. **降低 Batch Size**: `bs: 128 → 64`
2. **使用 Tiny 模型**: `dim: 768 → 192`
3. **降低分辨率**: `size: 800 → 512`
4. **关闭 AMP**: `amp: false`

---

## 🐛 常见问题

### 1. 环境错误
```
ModuleNotFoundError: No module named 'mmcv._ext'
```
**解决**: 检测/分割需要专用环境
```bash
source venv_detection/bin/activate
```

### 2. CUDA OOM
```
RuntimeError: CUDA out of memory
```
**解决**: 降低 batch size 或使用 Tiny 模型

### 3. 数据路径错误
**解决**: 修改配置文件中的 `data_dir`

---

## 📚 论文引用

```bibtex
@article{your_scope_paper,
  title={Soft Contextual Position Encoding for Vision Transformers},
  author={Your Name},
  journal={arXiv},
  year={2025}
}

@inproceedings{dosovitskiy2021vit,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and others},
  booktitle={ICLR},
  year={2021}
}

@inproceedings{liu2021swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and others},
  booktitle={ICCV},
  year={2021}
}
```

---

## 🤝 贡献

欢迎提交 Issue 和 PR！

---

**最后更新**: 2025-10-26

