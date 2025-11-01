#!/bin/bash
#SBATCH --job-name=vitcope_embedfull   # 任务名
#SBATCH --output=logs/vitcope_embedfull_%j.out # 标准输出文件
#SBATCH --error=logs/vitcope_embedfull_%j.err  # 错误输出文件
#SBATCH --time=144:00:00                # 最大运行时间 (6天)
#SBATCH --gres=gpu:AA_A6000_0:1      # 指定使用的 GPU 类型 (例如: turing, ampere)
#SBATCH --partition=res-gpu-small # 指定分区
#SBATCH --cpus-per-task=4               # 每个任务分配8个CPU核心
#SBATCH --mem=28G                       
#SBATCH --mail-type=ALL                 # 邮件通知类型
#SBATCH --mail-user=dnrx52@durham.ac.uk # 邮件地址

# 加载必要的模块
module load cuda/11.7             # 加载CUDA模块

# 激活conda环境
source /home2/dnrx52/bin/activate
conda activate dnrx52

# 切换到项目目录
cd /home2/dnrx52/SCOPE

python train.py --cfg configs/vitscope.yaml

