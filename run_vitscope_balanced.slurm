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
echo "🚀 训练开始 - 平衡配置"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# 加载环境
module load cuda/11.7
eval "$(conda shell.bash hook)"
conda activate dnrx52

cd /home2/dnrx52/SCOPE

# 🚀 使用OpenCV加速数据加载（比PIL快50-100%）
export USE_FAST_LOADER=ultra

echo ""
echo "配置: vitscope_balanced.yaml"
echo "  Batch Size: 512 (针对num_workers=4优化)"
echo "  Aug: false (减轻CPU负担)"
echo "  🚀 Loader: UltraFastImageFolder (OpenCV+numpy)"
echo ""
echo "预期性能提升："
echo "  PIL:   数据加载1943ms → 速度0.48 batches/sec"
echo "  Ultra: 数据加载~800ms → 速度~1.0 batches/sec"
echo ""

python train.py --cfg configs/vitscope_balanced.yaml

echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="

