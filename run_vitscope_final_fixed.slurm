#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --job-name=vitscope_train      # 任务名
#SBATCH --output=logs/vitscope_%j.out  # 标准输出文件
#SBATCH --error=logs/vitscope_%j.err   # 错误输出文件
#SBATCH --time=144:00:00               # 最大运行时间 (6天)
#SBATCH --gres=gpu:ampere:1            # 指定使用的 GPU 类型
#SBATCH --qos=long-high-prio           # 长时间高优先级任务
#SBATCH --partition=res-gpu-small      # 指定分区
#SBATCH --mem=28G                      # 内存
#SBATCH --mail-type=ALL                # 邮件通知类型
#SBATCH --mail-user=dnrx52@durham.ac.uk # 邮件地址

echo "=========================================="
echo "🚀 ViTSCoPE 训练开始"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=========================================="

# 加载必要的模块
module load cuda/11.7

# 激活conda环境（正确方式）
eval "$(conda shell.bash hook)"
conda activate dnrx52

# 检查并安装opencv-python
echo "检查opencv-python..."
python -c "import cv2" 2>/dev/null || pip install opencv-python --quiet

# 切换到项目目录
cd /home2/dnrx52/SCOPE
mkdir -p logs

# 🚀 使用OpenCV加速（比PIL快50-100%）
export USE_FAST_LOADER=ultra

echo ""
echo "配置信息："
echo "  Model: vitscope"
echo "  Batch Size: 1024"
echo "  Learning Rate: 1.2e-3"
echo "  Loader: UltraFastImageFolder (OpenCV)"
echo "  Aug: false"
echo ""
echo "预期性能："
echo "  数据加载占比: ~60-65%"
echo "  速度: ~0.8-1.0 batches/sec"
echo "  100 epochs: ~3-4天"
echo ""
echo "性能报告每100个batch显示一次"
echo "=========================================="
echo ""

# 运行训练
python train.py --cfg configs/vitscope_balanced.yaml

echo ""
echo "=========================================="
echo "训练完成"
echo "End: $(date)"
echo "=========================================="

