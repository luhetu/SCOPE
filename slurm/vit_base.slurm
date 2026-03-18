#!/bin/bash
#SBATCH -N 1
#SBATCH -w gpu13
#SBATCH -c 8
#SBATCH --mem=56g
#SBATCH --gres=gpu:hopper:1
#SBATCH -p gpu-bigmem
#SBATCH --qos=long-high-prio
#SBATCH --time=6-23:59:59
#SBATCH -J vit_BASE
#SBATCH -o logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dnrx52@durham.ac.uk

set -euo pipefail

module load cuda/11.1

ROOT_DIR=/home3/dnrx52/SCOPE
PYTHON=/home3/dnrx52/.venvs/vitscope/bin/python
CONFIG=${1:-configs/vit_base.yaml}
CHECKPOINT=checkpoint/vit_cls_size224_patch16_dim768_depth12_last.pth

cd "$ROOT_DIR"
mkdir -p logs

echo "============================================================"
echo "Job    : $SLURM_JOB_ID  |  Node: $(hostname)"
echo "Config : $CONFIG"
echo "Resume : $CHECKPOINT"
echo "Start  : $(date)"
echo "============================================================"

$PYTHON -m pip install --quiet -r "$ROOT_DIR/requirements_minimal.txt"

$PYTHON train.py \
    --cfg "$CONFIG" \
    --resume "$CHECKPOINT" \
    --workers_per_gpu 8 \
    --time_profile \
    --time_profile_interval 500

echo "Done at $(date)"
