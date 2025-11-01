#!/bin/bash
#SBATCH --job-name=perf_test          # 性能测试
#SBATCH --output=logs/perf_test_%j.out
#SBATCH --error=logs/perf_test_%j.err
#SBATCH --time=1:00:00              
#SBATCH --gres=gpu:ampere:1
#SBATCH --qos=short                   # 短时任务
#SBATCH --partition=gpu-private
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=dnrx52@durham.ac.uk

echo "=========================================="
echo "⏱️  性能测试开始"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# 加载环境
module load cuda/11.7
eval "$(conda shell.bash hook)"
conda activate dnrx52

# 切换目录
cd /home2/dnrx52/SCOPE
mkdir -p logs

# 打印配置信息
echo ""
echo "测试配置："
echo "  Batch Size: 768"
echo "  Num Workers: 4"
echo "  Aug: false"
echo "  Epochs: 1 (只测试性能)"
echo ""

# 运行性能测试（只跑1个epoch）
echo "开始训练..."
python train.py --cfg configs/vitscope_test.yaml

echo ""
echo "=========================================="
echo "测试完成！"
echo "End Time: $(date)"
echo "=========================================="
echo ""
echo "请查看上方的性能分析报告，关注："
echo "  1. 数据加载占比是否 > 50%"
echo "  2. 整体速度 (batches/sec)"
echo "  3. 每个batch的平均时间"
echo ""



