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
#SBATCH -w gpu12                       # 指定gpu12节点
#SBATCH --mem=28G                      # 内存
#SBATCH --mail-type=ALL                # 邮件通知类型
#SBATCH --mail-user=dnrx52@durham.ac.uk # 邮件地址

# 打印任务信息
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# 加载必要的模块
module load cuda/11.7

# 切换到项目目录
cd /home2/dnrx52/SCOPE

# 创建日志目录
mkdir -p logs

# 激活conda环境（方法1：使用conda init生成的hook）
eval "$(conda shell.bash hook)"
conda activate dnrx52

# 如果上面的方法失败，尝试方法2（取消下面的注释）
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate dnrx52

# 如果上面都失败，尝试方法3（直接使用conda环境的python）
# export PATH="/home2/dnrx52/miniconda3/envs/dnrx52/bin:$PATH"

# 打印环境信息
echo "Python: $(which python)"
echo "Python Version: $(python --version 2>&1)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'N/A')"
echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\")' 2>/dev/null || echo 'N/A')"
echo "Config: configs/vitscope.yaml"
echo "Data Dir: /home2/dnrx52/vitcope"
echo "=========================================="

# 运行训练脚本
echo "Starting training..."
python train.py --cfg configs/vitscope.yaml

# 打印结束信息
echo "=========================================="
echo "End Time: $(date)"
echo "Exit Code: $?"
echo "=========================================="



