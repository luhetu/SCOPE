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
#SBATCH --mem=28G                      # 内存
#SBATCH --mail-type=ALL                # 邮件通知类型
#SBATCH --mail-user=dnrx52@durham.ac.uk # 邮件地址

# 加载必要的模块
module load cuda/11.7

# ✅ 正确激活conda
eval "$(conda shell.bash hook)"
conda activate dnrx52

# 切换到项目目录
cd /home2/dnrx52/SCOPE

# ✅ 设置OpenCV加速
export USE_FAST_LOADER=ultra
export PREFETCH_FACTOR=4

# ✅ 限制CPU线程，减少与DataLoader的竞争
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENCV_OPENCL_RUNTIME=disabled

# ✅ 验证OpenCV可用
echo "检查OpenCV..."
python -c "import cv2; print('✅ OpenCV版本:', cv2.__version__)" || {
    echo "❌ OpenCV不可用，请先安装: pip install opencv-python"
    exit 1
}

echo "开始训练..."
echo "配置: vitscope_balanced.yaml (bs=256, 针对num_workers=4优化)"
python train.py --cfg configs/vitscope_balanced.yaml

