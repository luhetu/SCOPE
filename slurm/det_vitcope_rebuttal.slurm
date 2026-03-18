#!/bin/bash
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=56G
#SBATCH --gres=gpu:ampere:1
#SBATCH -p gpu-bigmem
#SBATCH --qos=long-high-prio
#SBATCH --time=144:00:00
#SBATCH -J det_vitcope_rebuttal
#SBATCH -o logs/%x%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dnrx52@durham.ac.uk

module load cuda/11.1

cd /home3/dnrx52/SCOPE

$HOME/anaconda3/envs/vitseg/bin/python train.py --cfg /home3/dnrx52/SCOPE/configs/detection_vitcope_rebuttal.yaml
