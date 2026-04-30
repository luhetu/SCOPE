#!/bin/bash
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --job-name=seg_vit_small
#SBATCH --output=logs/seg_vit_small_%j.out
#SBATCH --error=logs/seg_vit_small_%j.err
#SBATCH --time=144:00:00
#SBATCH --gres=gpu:ampere:1
#SBATCH --qos=long-high-prio
#SBATCH --partition=res-gpu-small
#SBATCH --mem=28G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dnrx52@durham.ac.uk

module load cuda/11.1

ROOT_DIR=/home3/dnrx52/SCOPE
PYTHON=/home3/dnrx52/anaconda3/envs/vitseg/bin/python
CONFIG=${1:-configs/seg_vit_small.yaml}

cd "$ROOT_DIR"
mkdir -p logs

echo "Job: $SLURM_JOB_ID | Node: $(hostname) | Start: $(date)"

$PYTHON train.py --cfg "$CONFIG" --workers_per_gpu 8

echo "Done at $(date)"
