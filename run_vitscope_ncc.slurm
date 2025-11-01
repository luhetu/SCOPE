#!/bin/bash
#SBATCH --job-name=vitscope_train       # 任务名称
#SBATCH --output=logs/vitscope_%j.out   # 标准输出日志
#SBATCH --error=logs/vitscope_%j.err    # 错误输出日志
#SBATCH --partition=res-gpu-small       # 分区名称
#SBATCH --cpus-per-task=4               # CPU核心数
#SBATCH --gres=gpu:ampere:1             # 1个Ampere GPU
#SBATCH --mem=28G                       # 内存
#SBATCH --time=144:00:00                # 最大运行时间 6天
#SBATCH --mail-type=ALL                 # 邮件通知
#SBATCH --mail-user=dnrx52@durham.ac.uk

# 打印任务信息
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# 设置工作目录
cd /home2/dnrx52/SCOPE

# 创建日志目录
mkdir -p logs

# 加载CUDA模块
module load cuda/11.7

# 初始化conda（使用eval方式，更可靠）
eval "$(conda shell.bash hook)"
conda activate dnrx52

# 打印环境信息
echo "Python: $(which python)"
echo "Python Version: $(python --version 2>&1)"
echo "PyTorch Version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'N/A')"
echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
echo "GPU Name: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\")' 2>/dev/null || echo 'N/A')"
echo "Data Dir: /home2/dnrx52/vitcope"
echo "=========================================="

# 运行训练脚本
echo "Starting training with vitscope.yaml..."
python train.py --cfg configs/vitscope.yaml

# 打印结束信息
echo "=========================================="
echo "End Time: $(date)"
echo "Exit Code: $?"
echo "=========================================="



