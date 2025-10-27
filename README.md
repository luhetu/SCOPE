# 🎯 SCOPE: Unified Vision Transformer Framework

A unified training framework for **ViT**, **CoPE**, **SCoPE**, and **Swin Transformer** across three vision tasks: **Image Classification**, **Object Detection**, and **Semantic Segmentation**.

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Features

- ✅ **Unified Framework**: One codebase for classification, detection, and segmentation
- ✅ **Multiple Models**: ViT, CoPE, SCoPE, Swin Transformer V2
- ✅ **YAML Configuration**: Easy-to-modify config files
- ✅ **WandB Integration**: Real-time training monitoring
- ✅ **Memory Optimized**: Tiny models for resource-constrained scenarios
- ✅ **Multi-scale Features**: Backbones adapted for dense prediction tasks

---

## 📊 Supported Tasks & Models

| Task | Models | Datasets | Framework |
|------|--------|----------|-----------|
| **Classification** | ViT-Tiny, CoPE, SCoPE, Swin-Tiny | ImageNet-1K | PyTorch |
| **Object Detection** | Swin + Mask R-CNN<br>ViT/CoPE/SCoPE + Mask R-CNN | COCO | MMDetection |
| **Semantic Segmentation** | Swin + UPerNet<br>ViT/CoPE/SCoPE + UPerNet | ADE20K | MMSegmentation |

---

## 🏗️ Architecture

```
SCOPE/
├── configs/              # YAML configurations
│   ├── vit.yaml
│   ├── detection_*.yaml
│   └── seg_*.yaml
├── models/              # Model implementations
│   ├── vit.py
│   ├── vitcope.py
│   ├── vitscope442.py
│   ├── swin_transformer.py
│   └── vit_backbone.py
├── tasks/               # Task implementations
│   ├── classification.py
│   ├── detection.py
│   └── segmentation.py
├── mmdet/               # MMDetection framework
├── mmseg/               # MMSegmentation framework
└── train.py             # Unified training entry
```

---

## ⚙️ Installation

### Classification Environment (PyTorch 2.x)
```bash
pip install -r requirements_classification.txt
```

### Detection & Segmentation Environment (PyTorch 1.9 + MMCV)
```bash
# Python 3.7 + CUDA 11.1
python3.7 -m venv venv_detection
source venv_detection/bin/activate
pip install -r requirements_detection.txt
```

📖 **Detailed instructions**: See [INSTALL_DETECTION_ENV.md](INSTALL_DETECTION_ENV.md)

---

## 🏃 Quick Start

### Image Classification
```bash
# ViT Tiny (dim=192, heads=3)
python train.py --cfg configs/vit.yaml

# CoPE (Contextual Position Encoding)
python train.py --cfg configs/vitcope.yaml

# SCoPE (Soft Contextual Position Encoding)
python train.py --cfg configs/vitscope.yaml

# Swin Transformer V2
python train.py --cfg configs/swin.yaml
```

### Object Detection
```bash
# Switch to detection environment
source venv_detection/bin/activate

# Swin + Mask R-CNN (Recommended)
python train.py --cfg configs/detection_swin.yaml

# ViT Tiny + Mask R-CNN (Memory optimized: 512x512)
python train.py --cfg configs/detection_vit.yaml
```

### Semantic Segmentation
```bash
# Use detection environment
source venv_detection/bin/activate

# Swin + UPerNet
python train.py --cfg configs/seg_swin.yaml

# ViT Tiny + UPerNet
python train.py --cfg configs/seg_vit.yaml
```

📖 **More examples**: See [QUICK_START.md](QUICK_START.md)

---

## 📝 Configuration

All models use YAML configuration files in `configs/` directory:

```yaml
task: cls              # Task type: cls / det / seg
model: vit             # Model: vit / vitcope / vitscope / swin
data_dir: /path/to/ImageNet
bs: 256                # Batch size
size: 224              # Image size
patch: 16              # Patch size
dim: 192               # Model dimension (Tiny)
depth: 12              # Transformer depth
heads: 3               # Attention heads (Tiny)
mlp_dim: 768           # MLP dimension
n_epochs: 100          # Training epochs
lr: 3e-4               # Learning rate
opt: adamw             # Optimizer
amp: true              # Mixed precision training
aug: true              # Data augmentation
nowandb: false         # WandB logging
```

---

## 🎯 Key Models

### ViT (Vision Transformer)
- **Tiny Config**: 192 dim, 3 heads, 12 layers
- **Memory**: ~4GB for classification, optimized for detection/segmentation

### CoPE (Contextual Position Encoding)
- Replaces fixed position embeddings with contextual ones
- Better generalization to varying image sizes

### SCoPE (Soft Contextual Position Encoding)
- Soft gating mechanism for position encoding
- CNN features guide attention computation

### Swin Transformer V2
- Hierarchical architecture with shifted windows
- Optimal for dense prediction tasks (detection/segmentation)

---

## 📈 Training Monitoring

### WandB Integration
All tasks automatically log to WandB:
- **Classification**: `imagenet-1k-scope` project
- **Detection**: `coco-detection` project  
- **Segmentation**: `ade20k-segmentation` project

Disable WandB: Set `nowandb: true` in config

### Output Format
```
[========== 10010/10010 ========>] Step: 30ms | Tot: 36m12s | Loss:6.920 | Acc:75.2%
[Epoch 000] TrainAcc=75.1% | ValAcc=75.3% | LR=0.000400 | Time=45.01 min
💾 Saved: checkpoint/vit_best.pth (acc=75.3%)
```

---

## 💾 Model Checkpoints

### Classification
Models saved to `checkpoint/`:
- `{model}_best.pth` - Best validation accuracy
- `{model}_last.pth` - Last epoch

### Detection & Segmentation
Models saved to `work_dirs/`:
- Automatic checkpoint saving by MMDetection/MMSegmentation

---

## 🔧 Memory Optimization

| Task | Model | Min GPU | Recommended Config |
|------|-------|---------|-------------------|
| Classification | ViT Tiny | 4GB | `bs: 256` |
| Classification | Swin Tiny | 8GB | `bs: 128` |
| Detection | Swin | 12GB | `bs: 2, 1333x800` |
| Detection | ViT Tiny | 4GB | `bs: 1, 512x512` |
| Segmentation | All | 8-12GB | `bs: 2, 512x512` |

**Out of Memory?**
1. Reduce batch size: `bs: 128 → 64`
2. Use Tiny models: `dim: 768 → 192`
3. Lower resolution: `size: 800 → 512`
4. Disable AMP: `amp: false`

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📚 Citation

If you find this work useful, please consider citing:

```bibtex
@article{scope2025,
  title={SCOPE: Soft Contextual Position Encoding for Vision Transformers},
  author={Your Name},
  journal={arXiv},
  year={2025}
}
```

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- [Vision Transformer](https://github.com/google-research/vision_transformer)
- [Swin Transformer](https://github.com/microsoft/Swin-Transformer)
- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
- [timm](https://github.com/huggingface/pytorch-image-models)

---

**Last Updated**: 2025-10-26

