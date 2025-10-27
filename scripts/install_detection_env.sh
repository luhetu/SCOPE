#!/bin/bash
# ====================================================================
# 检测/分割环境安装脚本
# 使用方法: bash scripts/install_detection_env.sh
# ====================================================================

set -e  # 遇到错误立即退出

echo "================================================"
echo "🔧 安装检测/分割环境依赖"
echo "================================================"

# 检查 Python 版本
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "当前 Python 版本: $python_version"

# ==================== 步骤 1: 安装 PyTorch ====================
echo ""
echo "📦 步骤 1/3: 安装 PyTorch 1.9.0 + CUDA 11.1"
echo "这可能需要几分钟..."

pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 \
    -f https://download.pytorch.org/whl/torch_stable.html

echo "✅ PyTorch 安装完成"

# 验证 PyTorch
python -c "import torch; print(f'PyTorch 版本: {torch.__version__}')"
python -c "import torch; print(f'CUDA 可用: {torch.cuda.is_available()}')"

# ==================== 步骤 2: 安装 MMCV ====================
echo ""
echo "📦 步骤 2/3: 安装 MMCV-Full 1.3.17"
echo "这可能需要较长时间，请耐心等待..."

pip install mmcv-full==1.3.17 \
    -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

echo "✅ MMCV-Full 安装完成"

# 验证 MMCV
python -c "from mmcv import _ext; print('✅ MMCV-Full 验证成功')" || echo "⚠️  MMCV 可能未正确安装"

# ==================== 步骤 3: 安装其他依赖 ====================
echo ""
echo "📦 步骤 3/3: 安装其他依赖包"

# 创建临时文件，排除 PyTorch 和 MMCV
cat requirements_detection.txt | \
    grep -v "^torch==" | \
    grep -v "^torchvision==" | \
    grep -v "^mmcv-full==" | \
    grep -v "^#" | \
    grep -v "^$" > /tmp/temp_requirements.txt

pip install -r /tmp/temp_requirements.txt

rm /tmp/temp_requirements.txt

echo "✅ 其他依赖安装完成"

# ==================== 验证安装 ====================
echo ""
echo "================================================"
echo "🎉 安装完成！正在验证..."
echo "================================================"

echo ""
echo "检查核心包:"
python << EOF
import sys
packages = [
    ('torch', 'PyTorch'),
    ('torchvision', 'TorchVision'),
    ('mmcv', 'MMCV'),
    ('cv2', 'OpenCV'),
    ('timm', 'Timm'),
    ('einops', 'Einops'),
    ('wandb', 'WandB'),
]

for module, name in packages:
    try:
        mod = __import__(module)
        version = getattr(mod, '__version__', 'unknown')
        print(f"  ✅ {name:15s} {version}")
    except ImportError:
        print(f"  ❌ {name:15s} NOT INSTALLED")

# 检查 CUDA
import torch
print(f"\n  🔥 CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  🔥 GPU 数量: {torch.cuda.device_count()}")
    print(f"  🔥 当前 GPU: {torch.cuda.get_device_name(0)}")
EOF

echo ""
echo "================================================"
echo "✅ 环境设置完成！"
echo "================================================"
echo ""
echo "下一步:"
echo "  1. 配置数据集路径: vim configs/detection_swin.yaml"
echo "  2. 运行检测训练: python train.py --cfg configs/detection_swin.yaml"
echo "  3. 或提交 SLURM 任务: sbatch scripts/submit_slurm_det.sh"
echo ""

