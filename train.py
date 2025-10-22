# -*- coding: utf-8 -*-
import argparse
import os
import yaml
from tasks import build_task


def load_cfg(parser):
    """
    ⚙️ 加载 YAML 配置文件并合并到 args
    """
    # 先解析命令行参数
    args = parser.parse_args()
    
    # 如果指定了配置文件，加载并覆盖
    if args.cfg and os.path.isfile(args.cfg):
        print(f"✅ [cfg] 加载配置文件：{args.cfg}")
        with open(args.cfg, "r") as f:
            cfg = yaml.safe_load(f)
        
        # 将 YAML 配置直接设置到 args 对象
        for key, value in cfg.items():
            setattr(args, key, value)
        
        print(f"✅ [cfg] 成功加载 {len(cfg)} 个参数")
    elif args.cfg:
        print(f"⚠️  [cfg] 配置文件不存在：{args.cfg}")
    
    return args


def main():
    parser = argparse.ArgumentParser(description="Unified ViT/CoPE/SCoPE Trainer")
    parser.add_argument("--cfg", type=str, default="", help="YAML config path")

    # ✅ 正确加载配置
    args = load_cfg(parser)

    # -------- 默认兜底值（防止 YAML 缺少字段） -------- #
    if not hasattr(args, "task"):
        args.task = "cls"
    if not hasattr(args, "model"):
        args.model = "vit"
    if not hasattr(args, "data_dir"):
        args.data_dir = "./data/imagenet"
    if not hasattr(args, "size"):
        args.size = 224
    if not hasattr(args, "patch"):
        args.patch = 16
    if not hasattr(args, "dim"):
        args.dim = 192
    if not hasattr(args, "depth"):
        args.depth = 12
    if not hasattr(args, "heads"):
        args.heads = 3
    if not hasattr(args, "mlp_dim"):
        args.mlp_dim = 768
    if not hasattr(args, "bs"):
        args.bs = 256
    if not hasattr(args, "n_epochs"):
        args.n_epochs = 100
    if not hasattr(args, "lr"):
        args.lr = 3e-4
    if not hasattr(args, "min_lr"):
        args.min_lr = 1e-5
    if not hasattr(args, "warmup_epochs"):
        args.warmup_epochs = 2
    if not hasattr(args, "opt"):
        args.opt = "adamw"
    if not hasattr(args, "amp"):
        args.amp = True
    if not hasattr(args, "aug"):
        args.aug = True
    if not hasattr(args, "nowandb"):
        args.nowandb = False

    # 打印关键配置信息
    print(f"\n{'='*60}")
    print(f"🚀 训练配置")
    print(f"{'='*60}")
    print(f"  任务类型: {args.task}")
    print(f"  模型: {args.model}")
    print(f"  图像尺寸: {args.size}")
    print(f"  Patch 尺寸: {args.patch}")
    print(f"  模型维度: {args.dim}")
    print(f"  深度: {args.depth}")
    print(f"  注意力头数: {args.heads}")
    print(f"  Batch Size: {args.bs}")
    print(f"  学习率: {args.lr}")
    print(f"  训练轮数: {args.n_epochs}")
    print(f"{'='*60}\n")

    # -------- 启动任务 -------- #
    task = build_task(args)
    task.train()


if __name__ == "__main__":
    main()
