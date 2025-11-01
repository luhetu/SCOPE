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

# 加载必要的模块
module load cuda/11.7

# 激活conda环境
eval "$(conda shell.bash hook)"
conda activate dnrx52

# 安装opencv-python（如果没有）
pip install opencv-python --quiet

# 🚀 关键：使用OpenCV加速数据加载（比PIL快50-100%）
export USE_FAST_LOADER=ultra

# 切换到项目目录
cd /home2/dnrx52/SCOPE

python train.py --cfg configs/vitscope_balanced.yaml

