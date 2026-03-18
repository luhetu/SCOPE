#!/bin/bash
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=56g
#SBATCH --gres=gpu:hopper:1
#SBATCH -p gpu-bigmem
#SBATCH --qos=long-high-prio
#SBATCH --time=6-23:59:59
#SBATCH -J scope_train
#SBATCH -o logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dnrx52@durham.ac.uk

set -euo pipefail

module load cuda/11.1

ROOT_DIR=/home3/dnrx52/SCOPE
PYTHON=/home3/dnrx52/.venvs/vitscope/bin/python
WORKERS=10

# ---- Config: set CFG, NFS_DATA_DIR, and optionally RESUME before submitting ----
CFG=${CFG:-configs/vitscope_nocls.yaml}
NFS_DATA_DIR=${NFS_DATA_DIR:-/home3/dnrx52/vitcope}
RESUME=${RESUME:-}

cd "$ROOT_DIR"
mkdir -p logs

echo "============================================================"
echo "Job    : $SLURM_JOB_ID on $(hostname)"
echo "Config : $CFG"
echo "Resume : ${RESUME:-none}"
echo "NFS src: $NFS_DATA_DIR"
echo "Start  : $(date)"
echo "============================================================"

# ---- Copy dataset to local scratch (/tmp is local NVMe on compute nodes) ----
LOCAL_DATA=${TMPDIR:-/tmp}/imagenet_${SLURM_JOB_ID}
echo ">>> Copying dataset to local scratch: $LOCAL_DATA"
echo ">>> Start rsync: $(date)"
mkdir -p "$LOCAL_DATA"
rsync -a --info=progress2 "$NFS_DATA_DIR"/ "$LOCAL_DATA"/
echo ">>> rsync done: $(date)"
echo ">>> Local disk usage: $(du -sh $LOCAL_DATA)"

# ---- Train ----
DATA_ARG="--data_dir $LOCAL_DATA"
RESUME_ARG=""
if [ -n "$RESUME" ]; then
    RESUME_ARG="--resume $RESUME"
fi

$PYTHON train.py \
    --cfg "$CFG" \
    $DATA_ARG \
    $RESUME_ARG \
    --workers_per_gpu "$WORKERS" \
    --time_profile \
    --time_profile_interval 1000

# ---- Cleanup ----
echo ">>> Cleaning up local scratch..."
rm -rf "$LOCAL_DATA"
echo ">>> Done at: $(date)"
