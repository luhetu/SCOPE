#!/bin/bash
#SBATCH -N 1
#SBATCH -w gpu13
#SBATCH -c 8
#SBATCH --mem=56g
#SBATCH --gres=gpu:hopper:1
#SBATCH -p gpu-bigmem
#SBATCH --qos=long-high-prio
#SBATCH --time=6-23:59:59
#SBATCH -J vitcope_im100_ab
#SBATCH -o logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dnrx52@durham.ac.uk

set -euo pipefail

module load cuda/11.1

ROOT_DIR=/home3/dnrx52/SCOPE
PYTHON=/home3/dnrx52/.venvs/vitscope/bin/python
WORKERS=8

CFG_CLS=configs/vitcope_imagenet100_cls.yaml
CFG_NOCLS=configs/vitcope_imagenet100_nocls.yaml

cd "$ROOT_DIR"
mkdir -p logs

echo "============================================================"
echo "Python: $PYTHON"
echo "Root  : $ROOT_DIR"
echo "Start : $(date)"
echo "============================================================"

if ! $PYTHON -m pip --version >/dev/null 2>&1; then
  $PYTHON -m ensurepip --upgrade
fi
$PYTHON -m pip install --quiet -r "$ROOT_DIR/requirements_minimal.txt"
$PYTHON -c "import sys, yaml; print('sys.executable:', sys.executable); print('pyyaml:', yaml.__version__)"

run_exp () {
  local cfg="$1"
  local tag="$2"
  local logfile="logs/${tag}_$(date +%Y%m%d_%H%M%S).log"
  echo ""
  echo ">>> Running $tag with $cfg"
  echo ">>> Log file: $logfile"
  echo ">>> Start   : $(date)"
  $PYTHON train.py --cfg "$cfg" --workers_per_gpu "$WORKERS" --time_profile --time_profile_interval 1000 2>&1 | tee "$logfile"
  echo ">>> End     : $(date)"
}

run_exp "$CFG_CLS" "vitcope_im100_cls"
run_exp "$CFG_NOCLS" "vitcope_im100_nocls"

echo ""
echo "All experiments finished at $(date)"

