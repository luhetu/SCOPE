#!/bin/bash
#SBATCH --job-name=scope_det
#SBATCH --partition=gpu               # 修改为你集群的分区名
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1                  # 1块GPU
#SBATCH --mem=64G                     # 检测需要更多内存
#SBATCH --time=72:00:00               # 检测训练更久
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your@email.com    # 修改为你的邮箱

# ==================== 作业信息 ==================== #
echo "================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "================================================"

# ==================== 环境设置 ==================== #
# 清空模块
module purge

# 加载必要模块（检测环境需要 Python 3.7）
module load python/3.7   # 必须是 Python 3.7
module load cuda/11.1    # 必须是 CUDA 11.1
module load cudnn/8.0

# 显示加载的模块
echo "Loaded modules:"
module list

# 激活检测环境
source ~/venv/scope_det/bin/activate

# 验证环境
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# 检查 MMCV
python -c "from mmcv import _ext; print('✅ MMCV-Full installed correctly')" || echo "❌ MMCV not found"

# ==================== 训练配置 ==================== #
# 读取命令行参数，默认为 detection_swin.yaml
CONFIG=${1:-configs/detection_swin.yaml}

echo "================================================"
echo "Detection configuration: $CONFIG"
echo "================================================"

# 创建日志目录
mkdir -p logs

# ==================== 开始训练 ==================== #
# 设置环境变量
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 运行检测训练
python train.py --cfg $CONFIG

# ==================== 完成 ==================== #
echo "================================================"
echo "End time: $(date)"
echo "Job completed!"
echo "================================================"

