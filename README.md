# SCOPE: Scalable Contextual Position Encoding for Vision Transformers

A PyTorch implementation of Vision Transformer variants with enhanced position encoding mechanisms for image classification, object detection, and semantic segmentation tasks.

## 📋 Overview

This project implements three Vision Transformer architectures:
- **ViT**: Standard Vision Transformer with learnable position embeddings
- **CoPE**: Contextual Position Encoding - dynamic position encoding based on content
- **SCoPE**: Scalable Contextual Position Encoding with Q-offset, HKGate, and CLS token

## 🏆 Model Performance

### Segmentation (ADE20K)

| Rank | Model | mIoU | Epochs | Checkpoint |
|------|-------|------|--------|------------|
| 🥇 | **CoPE** | **27.98%** | 27 | `CoPE_seg_best_mIoU27.98.pth` |
| 🥈 | SCoPE | 26.45% | 31 | `SCoPE_seg_best_mIoU26.45.pth` |
| 🥉 | ViT | 25.73% | 31 | `ViT_seg_best_mIoU25.73.pth` |

### Object Detection (COCO)

| Rank | Model | bbox mAP | Epochs | Checkpoint |
|------|-------|----------|--------|------------|
| 🥇 | **ViT** | **12.90%** | 12 | `ViT_det_best_mAP12.90.pth` |
| 🥈 | CoPE | 11.00% | 12 | `CoPE_det_best_mAP11.00.pth` |
| 🥉 | SCoPE | 12.10% | 12 | Latest training |

## 🚀 Quick Start

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd SCOPE
```

2. **Install dependencies**

For classification tasks:
```bash
pip install torch torchvision timm einops pyyaml wandb
```

For detection/segmentation tasks:
```bash
bash scripts/install_detection_env.sh
```

### Training

#### Image Classification (ImageNet/CIFAR)

```bash
# Train ViT on ImageNet
python train.py --cfg configs/vit.yaml

# Train SCoPE on CIFAR-10
python train.py --cfg configs/vitscope_cifar10.yaml

# Train CoPE on ImageNet
python train.py --cfg configs/vitcope.yaml
```

#### Object Detection (COCO)

```bash
# Activate detection environment
source venv_swin_det/bin/activate

# Train ViT for detection
python train.py --cfg configs/detection_vit.yaml

# Train CoPE for detection
python train.py --cfg configs/detection_vitcope.yaml

# Train SCoPE for detection
python train.py --cfg configs/detection_vitscope.yaml
```

#### Semantic Segmentation (ADE20K)

```bash
# Activate detection environment
source venv_swin_det/bin/activate

# Train ViT for segmentation
python train.py --cfg configs/seg_vit.yaml

# Train CoPE for segmentation
python train.py --cfg configs/seg_vitcope.yaml

# Train SCoPE for segmentation
python train.py --cfg configs/seg_vitscope.yaml
```

### Evaluation

#### Segmentation Evaluation
```bash
# Evaluate CoPE (best model)
python eval_seg.py --cfg configs/seg_vitcope.yaml --checkpoint checkpoint/best_checkpoints/CoPE_seg_best_mIoU27.98.pth

# Show visualization results
python show_results.py --cfg configs/seg_vitcope.yaml --checkpoint checkpoint/best_checkpoints/CoPE_seg_best_mIoU27.98.pth
```

#### Detection Evaluation
```bash
# Evaluate ViT (best model)
python train.py --cfg configs/detection_vit.yaml --eval --checkpoint checkpoint/best_checkpoints/ViT_det_best_mAP12.90.pth
```

## 📁 Project Structure

```
SCOPE/
├── configs/              # Configuration files for all experiments
│   ├── vit.yaml         # ViT classification config
│   ├── vitcope.yaml     # CoPE classification config
│   ├── vitscope.yaml    # SCoPE classification config
│   ├── detection_*.yaml # Detection task configs
│   └── seg_*.yaml       # Segmentation task configs
├── models/              # Model architectures
│   ├── vit.py          # Standard ViT
│   ├── vitcope.py      # CoPE implementation
│   ├── vitscope.py     # SCoPE implementation
│   └── swin_transformer.py
├── tasks/               # Task-specific implementations
│   ├── classification.py
│   ├── detection.py
│   └── segmentation.py
├── datasets/            # Dataset loaders
│   ├── classification.py
│   ├── fast_imagefolder.py
│   └── timed_dataset.py
├── mmdet/              # MMDetection integration
├── mmseg/              # MMSegmentation integration
├── utils/              # Utility functions
│   ├── cfg.py         # Config loader
│   ├── optim.py       # Optimizer utilities
│   └── utils.py       # General utilities
├── checkpoint/         # Model checkpoints
│   ├── best_checkpoints/  # Best performing models
│   └── pretrained/        # Pretrained weights
├── train.py           # Main training script
├── eval_seg.py        # Segmentation evaluation
├── show_results.py    # Visualization tool
└── diagnose_results.py # Performance diagnosis
```

## 🔧 Configuration

All experiments are configured via YAML files in the `configs/` directory. Key parameters:

```yaml
task: cls              # Task type: cls/det/seg
model: vitscope        # Model: vit/vitcope/vitscope
data_dir: /path/to/data
bs: 256               # Batch size
size: 224             # Input image size
n_epochs: 100         # Training epochs
patch: 16             # Patch size
dim: 192              # Model dimension
depth: 12             # Number of transformer layers
heads: 3              # Number of attention heads
mlp_dim: 768          # MLP hidden dimension
lr: 3e-4              # Learning rate
min_lr: 1e-5          # Minimum learning rate
warmup_epochs: 2      # Warmup epochs
opt: adamw            # Optimizer
amp: true             # Mixed precision training
aug: false            # Data augmentation
```

## 🎯 Key Features

### 1. SCoPE Architecture
- **HKGate**: Hierarchical knowledge gate for token importance weighting
- **Q-offset**: Query-based position offset for dynamic spatial encoding
- **CLS Token**: Global classification token for aggregation

### 2. CoPE Architecture
- **Contextual Position Encoding**: Content-aware position embeddings
- Better performance on dense prediction tasks (segmentation)

### 3. Multi-Task Support
- Image Classification (ImageNet, CIFAR-10/100)
- Object Detection (COCO)
- Semantic Segmentation (ADE20K)

### 4. Training Optimization
- Mixed precision training (AMP)
- Distributed training support
- WandB integration for experiment tracking
- Automatic checkpoint management

### 5. Fast Data Loading
- OpenCV-based fast image loader
- Timed dataset for performance profiling
- Optimized for network storage

## 📊 Datasets

### Classification
- **ImageNet**: Place in `data_dir/train` and `data_dir/val`
- **CIFAR-10/100**: Automatically downloaded

### Object Detection
- **COCO**: Place in `datasets/coco/`
  - `train2017/`
  - `val2017/`
  - `annotations/`

### Semantic Segmentation
- **ADE20K**: Place in `datasets/ADE20K/`
  - `ADEChallengeData2016/images/`
  - `ADEChallengeData2016/annotations/`

## 🛠️ Utilities

### Performance Diagnosis
```bash
# Analyze model performance and bottlenecks
python diagnose_results.py --cfg configs/seg_vitcope.yaml
```

### Result Visualization
```bash
# Visualize segmentation results
python show_results.py --cfg configs/seg_vitcope.yaml --checkpoint <path>
```

### Pycocotools Patching
```bash
# Fix pycocotools for evaluation
python patch_pycocotools.py
```

## 📈 Training Tips

1. **For Classification**:
   - Use `amp: true` for faster training
   - Adjust `bs` based on GPU memory
   - Use `aug: true` for better generalization

2. **For Detection/Segmentation**:
   - Requires mmcv-full installation
   - Use separate virtual environment
   - Recommended GPU: ≥16GB memory

3. **Data Loading**:
   - Use `FastImageFolder` for faster loading on network storage
   - Set appropriate `num_workers` in DataLoader
   - Use `TimedImageFolder` to diagnose bottlenecks

## 🔬 Model Details

### ViT (Vision Transformer)
- Standard transformer architecture with patch embedding
- Learnable position embeddings
- Simple and effective baseline

### CoPE (Contextual Position Encoding)
- Dynamic position encoding based on content
- Better spatial awareness for dense prediction
- Best for segmentation tasks

### SCoPE (Scalable Contextual Position Encoding)
- Lightweight position encoding mechanism
- HKGate for token importance
- Q-offset for dynamic position adjustment
- Balanced performance across tasks

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{scope2025,
  title={SCOPE: Scalable Contextual Position Encoding for Vision Transformers},
  author={Your Name},
  year={2025}
}
```

## 🤝 Acknowledgments

This project builds upon:
- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
- [timm](https://github.com/rwightman/pytorch-image-models)

## 📄 License

This project is licensed under the Apache 2.0 License.

## 🐛 Troubleshooting

### Common Issues

1. **ImportError: No module named 'mmcv._ext'**
   - Solution: Activate detection environment `source venv_swin_det/bin/activate`

2. **CUDA out of memory**
   - Solution: Reduce batch size in config file

3. **Slow data loading**
   - Solution: Use `FastImageFolder` or increase `num_workers`

4. **WandB login required**
   - Solution: Run `wandb login` or set `nowandb: true` in config

## 📧 Contact

For questions and issues, please open an issue on GitHub or contact the maintainers.

---

**Happy Training! 🚀**

