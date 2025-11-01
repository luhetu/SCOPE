#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --job-name=vitscope_train      # 任务名
#SBATCH --output=logs/vitscope_%j.out  # 标准输出文件
#SBATCH --error=logs/vitscope_%j.err   # 错误输出文件
#SBATCH --time=144:00:00               # 最大运行时间 (6天)
#SBATCH --gres=gpu:ampere:1            # 指定使用的 GPU 类型
#SBATCH --qos=long-high-prio           # 长时间高优先级任务
#SBATCH --partition=res-gpu-small      # 指定分区 (修正)
#SBATCH -w gpu12                       # 指定gpu12节点
#SBATCH --mem=28G                      # 内存
#SBATCH --mail-type=ALL                # 邮件通知类型
#SBATCH --mail-user=dnrx52@durham.ac.uk # 邮件地址

echo "=========================================="
echo "🚀 开始训练 - 优化配置"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# 加载必要的模块
module load cuda/11.7

# 激活conda环境
eval "$(conda shell.bash hook)"
conda activate dnrx52

# 切换到项目目录
cd /home2/dnrx52/SCOPE

# 打印配置
echo ""
echo "使用配置: vitscope_gpu12.yaml"
echo "  Batch Size: 768 (优化后)"
echo "  Num Workers: 4"
echo "  Aug: false (减少CPU负担)"
echo "  Epochs: 100"
echo ""
echo "⏱️  每10个batch会显示性能分析报告"
echo "=========================================="
echo ""

# 运行训练 - 使用优化后的配置
python train.py --cfg configs/vitscope_gpu12.yaml

echo ""
echo "=========================================="
echo "训练完成"
echo "End Time: $(date)"
echo "=========================================="

