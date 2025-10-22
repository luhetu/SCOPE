# -*- coding: utf-8 -*-
import argparse
import os
import yaml
from tasks import build_task


def load_cfg(parser):
    """
    ⚙️ 正确加载 YAML 配置文件（修正版）：
    1️⃣ 先读取命令行中的 --cfg
    2️⃣ 再读取 YAML 内容并更新 parser 默认值
    3️⃣ 最后再解析完整参数（确保 YAML 字段都进入 args）
    """
    # 第一次解析，只为了拿到 --cfg
    temp_args, _ = parser.parse_known_args()
    cfg_path = getattr(temp_args, "cfg", "")

    # 如果传入了 YAML 文件路径
    if cfg_path and os.path.isfile(cfg_path):
        print(f"[cfg] 加载配置文件：{cfg_path}")
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)

        # ✅ 把 YAML 键值注册为 parser 参数
        for key, value in cfg.items():
            if not any(a.option_strings == [f"--{key}"] for a in parser._actions):
                parser.add_argument(f"--{key}", type=type(value), default=value)
            else:
                parser.set_defaults(**{key: value})

    # 再解析完整参数（合并 YAML + CLI）
    return parser.parse_args()


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

    print(f"\n🚀 Task={args.task} | Model={args.model} | ImageSize={args.size}\n")

    # -------- 启动任务 -------- #
    task = build_task(args)
    task.train()


if __name__ == "__main__":
    main()
