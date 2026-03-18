#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import sys
import warnings
import os

# Set environment variables to reduce output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['PYTHONWARNINGS'] = 'ignore'   # Suppress Python warnings

# Filter common warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*Torch was not compiled with flash attention.*')

from tasks import build_task
from utils.cfg import load_cfg


def check_environment(args):
    """Check if current environment is suitable for running the specified task"""
    import torch
    
    task = args.task
    pytorch_version = torch.__version__
    python_version = "{}.{}".format(sys.version_info.major, sys.version_info.minor)
    
    # Detection and segmentation tasks require mmcv-full
    if task in ['det', 'seg']:
        try:
            from mmcv import _ext
            task_name = "Detection" if task == 'det' else "Segmentation"
            print(f"✅ {task_name} environment correct (PyTorch {pytorch_version})")
        except ImportError:
            task_name = "Detection" if task == 'det' else "Segmentation"
            print("\n" + "="*60)
            print("⚠️  Environment Mismatch Warning")
            print("="*60)
            print(f"Current task: {task_name} ({task})")
            print(f"Current environment: Classification environment (PyTorch {pytorch_version})")
            print(f"\n{task_name} task requires a dedicated environment!")
            print("\nPlease run the following commands:")
            print("  source venv_swin_det/bin/activate")
            print(f"  python train.py --cfg {args.cfg}")
            print("\nIf you haven't created detection environment yet, run first:")
            print("  bash INSTALL_DETECTION_NOW.sh")
            print("="*60 + "\n")
            sys.exit(1)
    else:
        # Classification task
        print(f"✅ Classification environment (Python {python_version}, PyTorch {pytorch_version})")


def main():
    parser = argparse.ArgumentParser(description="Unified ViT/CoPE/SCoPE Trainer")
    parser.add_argument("--cfg", type=str, default="", help="YAML config path")
    parser.add_argument("--resume", type=str, default="", help="Resume from checkpoint")
    parser.add_argument("--workers_per_gpu", type=int, default=None, help="Override dataloader workers per GPU")
    parser.add_argument("--model", type=str, default=None, help="Override model name from config")
    parser.add_argument("--data_dir", type=str, default=None, help="Override data_dir (e.g. local scratch $TMPDIR)")
    parser.add_argument("--time_profile", action="store_true", help="Enable lightweight timing logs")
    parser.add_argument("--time_profile_interval", type=int, default=1000, help="Timing log interval (batches)")

    # ✅ Load config correctly
    args = load_cfg(parser)
    
    # ✅ Check environment
    check_environment(args)

    # ✅ Intelligently handle training configuration: classification/detection use epochs, segmentation uses iterations
    if args.task == 'seg':
        # Segmentation task: convert epochs to iterations
        # ADE20K: ~20k training images
        num_images = 20210
        
        iters_per_epoch = num_images // args.bs
        
        # Convert epochs to iterations
        args.max_iters = args.n_epochs * iters_per_epoch
        
        # Calculate warmup_iters (if warmup_epochs is configured)
        if hasattr(args, 'warmup_epochs') and args.warmup_epochs > 0:
            args.warmup_iters = int(args.warmup_epochs * iters_per_epoch)
        else:
            args.warmup_iters = 1500  # Default warmup
        
        # Save iters_per_epoch for later use
        args.iters_per_epoch = iters_per_epoch
        
        print(f"\n{'='*60}")
        print(f"🔧 Iteration-based configuration (segmentation)")
        print(f"{'='*60}")
        print(f"  Training images: {num_images:,}")
        print(f"  Batch Size: {args.bs}")
        print(f"  Iterations per epoch: {iters_per_epoch:,}")
        print(f"  Configured epochs: {args.n_epochs}")
        print(f"  ➜ max_iters: {args.max_iters:,}")
        if hasattr(args, 'warmup_epochs'):
            print(f"  Warmup epochs: {args.warmup_epochs}")
        print(f"  ➜ warmup_iters: {args.warmup_iters}")
        print(f"{'='*60}\n")
    elif args.task == 'det':
        # Detection task: keep epoch-based, only calculate warmup_iters
        # COCO: ~118k training images
        num_images = 118287
        iters_per_epoch = num_images // args.bs
        
        # Calculate warmup_iters
        if hasattr(args, 'warmup_epochs') and args.warmup_epochs > 0:
            args.warmup_iters = int(args.warmup_epochs * iters_per_epoch)
        else:
            args.warmup_iters = 500  # Default warmup
        
        print(f"\n{'='*60}")
        print(f"🔧 Epoch-based configuration (detection)")
        print(f"{'='*60}")
        print(f"  Training images: {num_images:,}")
        print(f"  Batch Size: {args.bs}")
        print(f"  Iterations per epoch: {iters_per_epoch:,}")
        print(f"  Total epochs: {args.n_epochs}")
        if hasattr(args, 'warmup_epochs'):
            print(f"  Warmup epochs: {args.warmup_epochs}")
        print(f"  ➜ warmup_iters: {args.warmup_iters}")
        print(f"{'='*60}\n")

    # Print key configuration information
    print(f"\n{'='*60}")
    print(f"🚀 Training Configuration")
    print(f"{'='*60}")
    print(f"  Task type: {args.task}")
    print(f"  Model: {args.model}")
    print(f"  Image size: {args.size}")
    print(f"  Patch size: {args.patch}")
    
    # Print different architecture parameters based on model type
    if args.model == 'swin':
        print(f"  Model dimension: {args.embed_dim}")
        print(f"  Depth: {args.depths}")
        print(f"  Attention heads: {args.num_heads}")
        print(f"  MLP ratio: 4.0")
        print(f"  Window size: {args.window_size}")
    else:
        # ViT/CoPE/SCoPE use dim/depth/heads
        if hasattr(args, 'dim'):
            print(f"  Model dimension: {args.dim}")
        if hasattr(args, 'depth'):
            print(f"  Depth: {args.depth}")
        if hasattr(args, 'heads'):
            print(f"  Attention heads: {args.heads}")
        if hasattr(args, 'mlp_dim'):
            print(f"  MLP dimension: {args.mlp_dim}")
    
    print(f"  Batch Size: {args.bs}")
    print(f"  Learning rate: {args.lr}")
    
    # Print different training length information based on task type
    if args.task == 'seg':
        print(f"  Training iterations: {args.max_iters:,} ({args.n_epochs} epochs)")
    elif args.task == 'det':
        print(f"  Training epochs: {args.n_epochs} (epoch-based)")
    else:
        print(f"  Training epochs: {args.n_epochs}")
    
    print(f"{'='*60}\n")

    # -------- Start task -------- #
    task = build_task(args)
    task.train()


if __name__ == "__main__":
    main()
