#!/bin/bash
#SBATCH --job-name=vitcope_embedfull   # 任务名
#SBATCH --output=logs/vitcope_embedfull_%j.out # 标准输出文件
#SBATCH --error=logs/vitcope_embedfull_%j.err  # 错误输出文件
#SBATCH --time=144:00:00                # 最大运行时间 (6天)
#SBATCH --gres=gpu:turing:1             # 请求1个Turing架构GPU (RTX 2080 Ti)
#SBATCH --partition=gpu-private         # 指定分区
#SBATCH --nodelist=gpu5                 # 指定使用gpu5节点
#SBATCH --cpus-per-task=4               # 每个任务分配4个CPU核心
#SBATCH --mem=28G                       # 分配的内存28GB
#SBATCH --mail-type=ALL                 # 邮件通知类型
#SBATCH --mail-user=dnrx52@durham.ac.uk # 邮件地址

# 打印任务信息
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# 加载必要的模块
module load cuda/11.7             # 加载CUDA模块

# 激活conda环境
source /home2/dnrx52/bin/activate
conda activate dnrx52

# 切换到项目目录
cd /home2/dnrx52/SCOPE

# 创建日志目录（如果不存在）
mkdir -p logs

# 打印环境信息
echo "Python: $(which python)"
echo "Python Version: $(python --version)"
echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
echo "GPU Name: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\")' 2>/dev/null || echo 'N/A')"
echo "=========================================="

# 运行训练脚本
echo "Starting training with vitcope_embedfull.yaml..."
python train.py --cfg configs/vitcope_embedfull.yaml

# 打印结束信息
echo "=========================================="
echo "End Time: $(date)"
echo "Job Exit Code: $?"
echo "=========================================="



