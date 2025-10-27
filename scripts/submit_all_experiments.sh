#!/bin/bash
# ================================================================
# 批量提交所有 SCOPE 实验
# 使用方法: bash scripts/submit_all_experiments.sh
# ================================================================

echo "================================================"
echo "🚀 批量提交 SCOPE 实验"
echo "================================================"

# 创建日志目录
mkdir -p logs

# ==================== 分类实验 ==================== #
echo ""
echo "📊 提交分类实验..."

# ViT Tiny
sbatch --job-name=vit_cls scripts/submit_slurm_cls.sh configs/vit.yaml
echo "✅ 提交 ViT 分类任务"
sleep 1

# CoPE
sbatch --job-name=cope_cls scripts/submit_slurm_cls.sh configs/vitcope.yaml
echo "✅ 提交 CoPE 分类任务"
sleep 1

# SCoPE
sbatch --job-name=scope_cls scripts/submit_slurm_cls.sh configs/vitscope.yaml
echo "✅ 提交 SCoPE 分类任务"
sleep 1

# Swin
sbatch --job-name=swin_cls scripts/submit_slurm_cls.sh configs/swin.yaml
echo "✅ 提交 Swin 分类任务"
sleep 1

# ==================== 检测实验 ==================== #
echo ""
echo "🎯 提交检测实验..."

# Swin Detection (推荐)
sbatch --job-name=swin_det scripts/submit_slurm_det.sh configs/detection_swin.yaml
echo "✅ 提交 Swin 检测任务"
sleep 1

# ViT Detection (可选，显存占用大)
# sbatch --job-name=vit_det scripts/submit_slurm_det.sh configs/detection_vit.yaml
# echo "✅ 提交 ViT 检测任务"
# sleep 1

# ==================== 分割实验 ==================== #
echo ""
echo "🖼️  提交分割实验..."

# Swin Segmentation
sbatch --job-name=swin_seg scripts/submit_slurm_det.sh configs/seg_swin.yaml
echo "✅ 提交 Swin 分割任务"
sleep 1

# ViT Segmentation (可选)
# sbatch --job-name=vit_seg scripts/submit_slurm_det.sh configs/seg_vit.yaml
# echo "✅ 提交 ViT 分割任务"
# sleep 1

# ==================== 查看作业状态 ==================== #
echo ""
echo "================================================"
echo "✅ 所有任务已提交！"
echo "================================================"
echo ""
echo "查看作业状态:"
echo "  squeue -u $USER"
echo ""
echo "查看特定作业:"
echo "  squeue -j JOB_ID"
echo ""
echo "查看日志:"
echo "  tail -f logs/JOBNAME_JOBID.out"
echo ""
echo "取消作业:"
echo "  scancel JOB_ID"
echo "  scancel -u $USER  # 取消所有作业"
echo "================================================"

# 显示当前作业
sleep 2
echo ""
echo "当前作业队列:"
squeue -u $USER

