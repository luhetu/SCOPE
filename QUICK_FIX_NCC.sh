#!/bin/bash
# 快速修复 NCC 上的安装问题

echo "================================================"
echo "🔧 快速修复：安装 Cython 和其他依赖"
echo "================================================"

# 步骤 1: 安装 Cython（pycocotools 的构建依赖）
echo ""
echo "📦 步骤 1/2: 安装 Cython 和 numpy"
pip install cython numpy

# 步骤 2: 安装其他依赖
echo ""
echo "📦 步骤 2/2: 安装其他依赖包"
pip install opencv-python Pillow matplotlib
pip install pycocotools terminaltables
pip install timm==0.6.12 einops
pip install wandb tensorboard
pip install scipy scikit-learn
pip install tqdm pyyaml

echo ""
echo "================================================"
echo "✅ 安装完成！正在验证..."
echo "================================================"

# 验证安装
python << 'PYEOF'
import sys
packages = [
    ('torch', 'PyTorch'),
    ('torchvision', 'TorchVision'),
    ('mmcv', 'MMCV'),
    ('cv2', 'OpenCV'),
    ('PIL', 'Pillow'),
    ('matplotlib', 'Matplotlib'),
    ('pycocotools', 'pycocotools'),
    ('timm', 'Timm'),
    ('einops', 'Einops'),
    ('wandb', 'WandB'),
]

print("\n检查核心包:")
all_ok = True
for module, name in packages:
    try:
        mod = __import__(module)
        version = getattr(mod, '__version__', 'unknown')
        print(f"  ✅ {name:15s} {version}")
    except ImportError:
        print(f"  ❌ {name:15s} NOT INSTALLED")
        all_ok = False

# 检查 CUDA
import torch
print(f"\n  🔥 CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  🔥 GPU 数量: {torch.cuda.device_count()}")
    print(f"  🔥 当前 GPU: {torch.cuda.get_device_name(0)}")

if all_ok:
    print("\n✅ 所有依赖安装成功！")
else:
    print("\n⚠️  有些包未安装，请检查上面的错误信息")
PYEOF

echo ""
echo "================================================"
echo "✅ 环境准备完成！"
echo "================================================"
echo ""
echo "下一步:"
echo "  1. 修改数据路径: vim configs/detection_swin.yaml"
echo "  2. 运行训练: python train.py --cfg configs/detection_swin.yaml"
echo ""
