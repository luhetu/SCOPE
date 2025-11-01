#!/bin/bash
#SBATCH --job-name=test_opencv        # 测试OpenCV加速
#SBATCH --output=logs/test_opencv_%j.out
#SBATCH --error=logs/test_opencv_%j.err
#SBATCH --time=00:30:00               # 30分钟
#SBATCH --gres=gpu:ampere:1
#SBATCH --qos=short
#SBATCH --partition=res-gpu-small
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G

echo "=========================================="
echo "🧪 测试OpenCV vs PIL数据加载性能"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

module load cuda/11.7
eval "$(conda shell.bash hook)"
conda activate dnrx52

cd /home2/dnrx52/SCOPE
mkdir -p logs

echo ""
echo "测试1: PIL加载器 (默认torchvision)"
echo "--------------------------------------"
export USE_FAST_LOADER=pil
python train.py --cfg configs/vitscope_test.yaml

echo ""
echo ""
echo "测试2: OpenCV加载器 (读取用OpenCV)"
echo "--------------------------------------"
export USE_FAST_LOADER=opencv  
python train.py --cfg configs/vitscope_test.yaml

echo ""
echo ""
echo "测试3: Ultra加载器 (完全OpenCV+numpy)"
echo "--------------------------------------"
export USE_FAST_LOADER=ultra
python train.py --cfg configs/vitscope_test.yaml

echo ""
echo "=========================================="
echo "测试完成！对比3种加载器的性能差异"
echo "=========================================="

