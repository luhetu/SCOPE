#!/bin/bash
#SBATCH -N 1
#SBATCH --mem=56G
#SBATCH --gres=gpu:ampere:1
#SBATCH -p gpu-bigmem
#SBATCH --qos=long-high-prio
#SBATCH --time=4-23:59:59
#SBATCH -J det_scope_tiny
#SBATCH -o logs/%x%j.out
#SBATCH --error=logs/%x%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dnrx52@durham.ac.uk






module load cuda/11.1

ROOT_DIR=/home3/dnrx52/SCOPE
PYTHON=$HOME/anaconda3/envs/vitseg/bin/python
CONFIG=${1:-configs/detection_vitscope_tiny.yaml}

cd "$ROOT_DIR"
mkdir -p logs

echo "Job: $SLURM_JOB_ID | Node: $(hostname) | Config: $CONFIG | Start: $(date)"
$PYTHON train.py --cfg "$CONFIG" --workers_per_gpu 8
echo "Done at $(date)"
