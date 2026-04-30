#!/bin/bash
#SBATCH -N 1
#SBATCH -w gpu13
#SBATCH -c 16
#SBATCH --mem=56g
#SBATCH --gres=gpu:hopper:1
#SBATCH -p gpu-bigmem
#SBATCH --qos=hopper-test-qos
#SBATCH --time=4-23:59:59
#SBATCH -J vitcope_small
#SBATCH -o logs/%x%j.out
#SBATCH --error=logs/%x%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dnrx52@durham.ac.uk

module load cuda/11.1

mkdir -p logs
cd /home3/dnrx52/SCOPE

PYTHON=/home3/dnrx52/.venvs/vitscope/bin/python
CONFIG=${1:-configs/vitcope_small.yaml}

echo "Python: $PYTHON"
if ! $PYTHON -m pip --version >/dev/null 2>&1; then
  $PYTHON -m ensurepip --upgrade
fi
$PYTHON -m pip install --quiet -r /home3/dnrx52/SCOPE/requirements_minimal.txt
$PYTHON -c "import sys; print('sys.executable:', sys.executable); import yaml; print('pyyaml:', yaml.__version__)"
$PYTHON train.py --cfg "$CONFIG" --workers_per_gpu 16 --time_profile
