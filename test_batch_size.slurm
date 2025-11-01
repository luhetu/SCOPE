#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --job-name=bs_test            # 任务名
#SBATCH --output=logs/bs_test_%j.out  # 标准输出文件
#SBATCH --error=logs/bs_test_%j.err   # 错误输出文件
#SBATCH --time=01:00:00               # 只需1小时测试
#SBATCH --gres=gpu:ampere:1           # 指定使用的 GPU 类型
#SBATCH --qos=short                   # 短时任务
#SBATCH --partition=res-gpu-small     # 指定分区
#SBATCH -w gpu12                      # 指定gpu12节点
#SBATCH --mem=28G                     # 内存
#SBATCH --mail-type=FAIL              # 只在失败时发邮件
#SBATCH --mail-user=dnrx52@durham.ac.uk

echo "=========================================="
echo "Batch Size Test - Finding Optimal Configuration"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "=========================================="

# 加载CUDA和激活环境
module load cuda/11.7
eval "$(conda shell.bash hook)"
conda activate dnrx52

cd /home2/dnrx52/SCOPE
mkdir -p logs

# 测试函数
test_batch_size() {
    local config=$1
    local bs=$2
    echo ""
    echo "=========================================="
    echo "Testing Batch Size: $bs"
    echo "Config: $config"
    echo "=========================================="
    
    # 只运行50个steps测试
    timeout 5m python -c "
import torch
import sys
sys.path.insert(0, '/home2/dnrx52/SCOPE')
from tasks import build_task
from utils.cfg import load_cfg
import argparse

parser = argparse.ArgumentParser()
args = load_cfg(parser, cfg_path='$config')

try:
    task = build_task(args)
    print(f'✅ Batch Size $bs: Successfully loaded')
    
    # 运行几个step测试显存
    for i, (inputs, targets) in enumerate(task.trainloader):
        if i >= 3:
            break
        inputs, targets = inputs.cuda(), targets.cuda()
        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs = task.net(inputs)
            loss = task.criterion(outputs, targets)
        task.scaler.scale(loss).backward()
        task.scaler.step(task.optimizer)
        task.scaler.update()
        task.optimizer.zero_grad()
        
        # 打印显存使用
        mem_allocated = torch.cuda.memory_allocated() / 1024**3
        mem_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f'  Step {i+1}: Allocated={mem_allocated:.2f}GB, Reserved={mem_reserved:.2f}GB')
    
    print(f'✅ Batch Size $bs: Test PASSED')
    print(f'   Max Memory: {torch.cuda.max_memory_allocated()/1024**3:.2f}GB / 80GB')
    
except RuntimeError as e:
    if 'out of memory' in str(e):
        print(f'❌ Batch Size $bs: OUT OF MEMORY')
    else:
        print(f'❌ Batch Size $bs: Error - {e}')
    sys.exit(1)
"
    
    if [ $? -eq 0 ]; then
        echo "✅ Success"
        return 0
    else
        echo "❌ Failed"
        return 1
    fi
}

# 从小到大测试
echo ""
echo "Starting batch size tests..."
echo ""

# 测试 bs=512 (当前)
test_batch_size "configs/vitscope_gpu12.yaml" 512
result_512=$?

# 测试 bs=1024
test_batch_size "configs/vitscope_bs1024.yaml" 1024
result_1024=$?

# 测试 bs=1536
if [ $result_1024 -eq 0 ]; then
    test_batch_size "configs/vitscope_bs1536.yaml" 1536
    result_1536=$?
fi

# 测试 bs=2048
if [ $result_1536 -eq 0 ]; then
    test_batch_size "configs/vitscope_bs2048.yaml" 2048
    result_2048=$?
fi

echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo "Batch Size 512:  $([ $result_512 -eq 0 ] && echo '✅ PASS' || echo '❌ FAIL')"
echo "Batch Size 1024: $([ $result_1024 -eq 0 ] && echo '✅ PASS' || echo '❌ FAIL')"
[ -n "$result_1536" ] && echo "Batch Size 1536: $([ $result_1536 -eq 0 ] && echo '✅ PASS' || echo '❌ FAIL')"
[ -n "$result_2048" ] && echo "Batch Size 2048: $([ $result_2048 -eq 0 ] && echo '✅ PASS' || echo '❌ FAIL')"
echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="



