#!/bin/bash
# ================================================================
# 在集群上快速设置 SCOPE 环境
# 使用方法: bash scripts/setup_on_cluster.sh
# ================================================================

set -e  # 遇到错误立即退出

echo "================================================"
echo "🚀 SCOPE 集群环境设置"
echo "================================================"

# ==================== 检查环境 ==================== #
echo ""
echo "📋 步骤 1/5: 检查环境..."

# 检查 git
if ! command -v git &> /dev/null; then
    echo "❌ 未找到 git，请先安装 git"
    exit 1
fi

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "❌ 未找到 python3，请加载 Python 模块"
    echo "   尝试: module load python"
    exit 1
fi

echo "✅ 基础环境检查通过"
echo "   Git: $(git --version)"
echo "   Python: $(python3 --version)"

# ==================== 选择环境类型 ==================== #
echo ""
echo "📦 步骤 2/5: 选择要设置的环境"
echo "   1) 分类环境 (Python 3.12, PyTorch 2.x)"
echo "   2) 检测/分割环境 (Python 3.7, PyTorch 1.9)"
echo "   3) 两个都安装"
read -p "请选择 [1/2/3]: " env_choice

# ==================== 创建虚拟环境 ==================== #
echo ""
echo "🔧 步骤 3/5: 创建虚拟环境..."

case $env_choice in
    1|3)
        echo "创建分类环境..."
        python3 -m venv ~/venv/scope_cls
        source ~/venv/scope_cls/bin/activate
        pip install --upgrade pip setuptools wheel
        
        echo "安装 PyTorch..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        
        echo "安装其他依赖..."
        pip install -r requirements_classification.txt
        
        deactivate
        echo "✅ 分类环境创建完成: ~/venv/scope_cls"
        ;;
esac

case $env_choice in
    2|3)
        echo "创建检测环境..."
        # 检查 Python 3.7
        if ! command -v python3.7 &> /dev/null; then
            echo "⚠️  未找到 python3.7"
            echo "   请加载 Python 3.7 模块: module load python/3.7"
            echo "   或跳过检测环境安装"
            if [ "$env_choice" == "2" ]; then
                exit 1
            fi
        else
            python3.7 -m venv ~/venv/scope_det
            source ~/venv/scope_det/bin/activate
            pip install --upgrade pip setuptools wheel
            
            echo "安装 PyTorch 1.9..."
            pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 \
                -f https://download.pytorch.org/whl/torch_stable.html
            
            echo "安装 MMCV-Full..."
            pip install mmcv-full==1.3.17 \
                -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
            
            echo "安装其他依赖..."
            pip install -r requirements_detection.txt
            
            deactivate
            echo "✅ 检测环境创建完成: ~/venv/scope_det"
        fi
        ;;
esac

# ==================== 设置数据路径 ==================== #
echo ""
echo "📂 步骤 4/5: 配置数据路径"
read -p "ImageNet 数据路径 (留空跳过): " imagenet_path
read -p "COCO 数据路径 (留空跳过): " coco_path
read -p "ADE20K 数据路径 (留空跳过): " ade20k_path

# 更新配置文件
if [ ! -z "$imagenet_path" ]; then
    for config in configs/vit.yaml configs/vitcope.yaml configs/vitscope.yaml configs/swin.yaml; do
        sed -i "s|data_dir:.*|data_dir: $imagenet_path|" $config
    done
    echo "✅ 已更新分类配置的数据路径"
fi

if [ ! -z "$coco_path" ]; then
    for config in configs/detection_*.yaml; do
        sed -i "s|data_dir:.*|data_dir: $coco_path|" $config
    done
    echo "✅ 已更新检测配置的数据路径"
fi

if [ ! -z "$ade20k_path" ]; then
    for config in configs/seg_*.yaml; do
        sed -i "s|data_dir:.*|data_dir: $ade20k_path|" $config
    done
    echo "✅ 已更新分割配置的数据路径"
fi

# ==================== 创建必要目录 ==================== #
echo ""
echo "📁 步骤 5/5: 创建目录结构..."
mkdir -p logs
mkdir -p checkpoint
mkdir -p work_dirs
chmod +x scripts/*.sh
echo "✅ 目录结构创建完成"

# ==================== 完成 ==================== #
echo ""
echo "================================================"
echo "✅ 环境设置完成！"
echo "================================================"
echo ""
echo "下一步:"
echo ""

if [ "$env_choice" == "1" ] || [ "$env_choice" == "3" ]; then
    echo "🔵 分类任务:"
    echo "   source ~/venv/scope_cls/bin/activate"
    echo "   python train.py --cfg configs/vit.yaml"
    echo "   或使用 SLURM: sbatch scripts/submit_slurm_cls.sh"
    echo ""
fi

if [ "$env_choice" == "2" ] || [ "$env_choice" == "3" ]; then
    echo "🟢 检测/分割任务:"
    echo "   source ~/venv/scope_det/bin/activate"
    echo "   python train.py --cfg configs/detection_swin.yaml"
    echo "   或使用 SLURM: sbatch scripts/submit_slurm_det.sh"
    echo ""
fi

echo "📚 更多信息请查看:"
echo "   - DEPLOY_ON_CLUSTER.md (集群部署指南)"
echo "   - QUICK_START.md (快速开始)"
echo ""
echo "================================================"

