#!/bin/bash
#SBATCH --job-name=vitcope_embedfull    # 任务名称
#SBATCH --output=logs/vitcope_embedfull_%j.out  # 标准输出日志 (%j 会被替换为任务ID)
#SBATCH --error=logs/vitcope_embedfull_%j.err   # 错误输出日志
#SBATCH --partition=res-gpu-small        # 分区名称（根据您的权限）
#SBATCH --cpus-per-task=4                # 每个任务使用4个CPU核心
#SBATCH --gres=gpu:ampere:1              # 使用1个GPU (任意架构：ampere或turing)
#SBATCH --mem=28G                        # 内存28GB
#SBATCH --time=144:00:00                 # 最大运行时间 144小时(6天)
#SBATCH --mail-type=ALL                  # 邮件通知
#SBATCH --mail-user=dnrx52@durham.ac.uk  # 邮件地址



# 设置工作目录
cd /home2/dnrx52/SCOPE

# 创建日志目录（如果不存在）
mkdir -p logs

# 加载CUDA模块
module load cuda/11.7

# 激活conda环境
eval "$(conda shell.bash hook)"
conda activate dnrx52


# 打印环境信息
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Python: $(which python)"
echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
echo "=========================================="

# 运行训练脚本
echo "Starting training..."
python train.py --cfg configs/vitscope.yaml

# 打印结束信息
echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="


