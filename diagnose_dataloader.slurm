#!/bin/bash
#SBATCH --job-name=diagnose_data      # 数据加载诊断
#SBATCH --output=logs/diagnose_%j.out
#SBATCH --error=logs/diagnose_%j.err
#SBATCH --time=00:30:00               # 只需30分钟
#SBATCH --gres=gpu:ampere:1
#SBATCH --qos=short
#SBATCH --partition=res-gpu-small
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=dnrx52@durham.ac.uk

echo "=========================================="
echo "🔍 DataLoader详细诊断"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# 环境设置
module load cuda/11.7
eval "$(conda shell.bash hook)"
conda activate dnrx52

cd /home2/dnrx52/SCOPE
mkdir -p logs

# 🔍 启用详细计时
export USE_TIMED_DATASET=true

echo ""
echo "诊断配置："
echo "  - 启用DataLoader内部计时"
echo "  - 测试 bs=512"
echo "  - 只运行1个epoch进行诊断"
echo ""
echo "将输出："
echo "  1. 外层Timer: 整体数据加载时间"
echo "  2. 内层Timer: I/O读取、JPEG解码、Transform详细时间"
echo ""
echo "=========================================="
echo ""

# 运行诊断
python train.py --cfg configs/vitscope_balanced.yaml

echo ""
echo "=========================================="
echo "诊断完成！"
echo "End Time: $(date)"
echo "=========================================="
echo ""
echo "📊 查看结果："
echo "  1. 外层性能 - 每10个batch的总体统计"
echo "  2. 内层性能 - 每100个样本的详细分解"
echo ""
echo "关键指标："
echo "  - 如果I/O读取 > 50%: 网络存储瓶颈"
echo "  - 如果Transform > 50%: 数据增强太慢"
echo "  - 如果JPEG解码 > 30%: 解码性能问题"
echo ""

