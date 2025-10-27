# -*- coding: utf-8 -*-
import argparse
import sys
import warnings
import os

# 设置环境变量以减少输出
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 抑制 TensorFlow 警告
os.environ['PYTHONWARNINGS'] = 'ignore'   # 抑制 Python 警告

# 过滤常见警告
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*Torch was not compiled with flash attention.*')

from tasks import build_task
from utils.cfg import load_cfg


def check_environment(args):
    """检查当前环境是否适合运行指定的任务"""
    import torch
    
    task = args.task
    pytorch_version = torch.__version__
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    
    # 检测和分割任务需要 mmcv-full
    if task in ['det', 'seg']:
        try:
            from mmcv import _ext
            task_name = "检测" if task == 'det' else "分割"
            print(f"✅ {task_name}环境正确 (PyTorch {pytorch_version})")
        except ImportError:
            task_name = "检测" if task == 'det' else "分割"
            print("\n" + "="*60)
            print("⚠️  环境不匹配警告")
            print("="*60)
            print(f"当前任务：{task_name} ({task})")
            print(f"当前环境：分类环境 (PyTorch {pytorch_version})")
            print(f"\n{task_name}任务需要使用专用环境！")
            print("\n请运行以下命令：")
            print("  source venv_swin_det/bin/activate")
            print(f"  python train.py --cfg {args.cfg}")
            print("\n如果还没有创建检测环境，请先运行：")
            print("  bash INSTALL_DETECTION_NOW.sh")
            print("="*60 + "\n")
            sys.exit(1)
    else:
        # 分类任务
        print(f"✅ 分类环境 (Python {python_version}, PyTorch {pytorch_version})")


def main():
    parser = argparse.ArgumentParser(description="Unified ViT/CoPE/SCoPE Trainer")
    parser.add_argument("--cfg", type=str, default="", help="YAML config path")

    # ✅ 正确加载配置
    args = load_cfg(parser)
    
    # ✅ 检查环境
    check_environment(args)

    # 打印关键配置信息
    print(f"\n{'='*60}")
    print(f"🚀 训练配置")
    print(f"{'='*60}")
    print(f"  任务类型: {args.task}")
    print(f"  模型: {args.model}")
    print(f"  图像尺寸: {args.size}")
    print(f"  Patch 尺寸: {args.patch}")
    
    # 根据模型类型打印不同的架构参数
    if args.model == 'swin':
        print(f"  模型维度: {args.embed_dim}")
        print(f"  深度: {args.depths}")
        print(f"  注意力头数: {args.num_heads}")
        print(f"  MLP比率: 4.0")
        print(f"  窗口大小: {args.window_size}")
    else:
        # ViT/CoPE/SCoPE 使用 dim/depth/heads
        if hasattr(args, 'dim'):
            print(f"  模型维度: {args.dim}")
        if hasattr(args, 'depth'):
            print(f"  深度: {args.depth}")
        if hasattr(args, 'heads'):
            print(f"  注意力头数: {args.heads}")
        if hasattr(args, 'mlp_dim'):
            print(f"  MLP维度: {args.mlp_dim}")
    
    print(f"  Batch Size: {args.bs}")
    print(f"  学习率: {args.lr}")
    print(f"  训练轮数: {args.n_epochs}")
    print(f"{'='*60}\n")

    # -------- 启动任务 -------- #
    task = build_task(args)
    task.train()


if __name__ == "__main__":
    main()
