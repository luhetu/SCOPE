#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate ADE20K segmentation checkpoints with the same MMSeg config builder used
by training. This intentionally avoids the old hardcoded ViT eval path.
"""

import argparse
import os

import torch
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmseg.apis import single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset

from tasks.segmentation import SegmentationTask
from utils.cfg import load_cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ADE20K segmentation checkpoint")
    parser.add_argument("--cfg", "--config", dest="cfg", required=True, help="segmentation yaml config")
    parser.add_argument("--checkpoint", required=True, help="fine-tuned segmentation checkpoint")
    parser.add_argument("--workers_per_gpu", type=int, default=None, help="override dataloader workers")
    parser.add_argument("--data_dir", type=str, default=None, help="override ADE20K data root")
    parser.add_argument("--show", action="store_true", help="show predictions")
    parser.add_argument("--out-dir", default=None, help="directory to save visualized predictions")
    parser.add_argument("--efficient-test", action="store_true", help="save predictions as temporary numpy files")
    return load_cfg(parser)


def main():
    args = parse_args()

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # Evaluation should load the fine-tuned segmentation checkpoint directly.
    # Do not first load the ImageNet classification pretrain from the yaml.
    args.pretrained = ""
    args.nowandb = True

    print("\n🔧 Building segmentation task from training config")
    task = SegmentationTask(args)
    cfg = task.cfg
    model = task.model

    print(f"\n📊 Building ADE20K validation dataset from: {args.data_dir}")
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=int(getattr(args, "workers_per_gpu", None) or cfg.data.workers_per_gpu),
        dist=False,
        shuffle=False,
    )

    print(f"\n📥 Loading segmentation checkpoint: {args.checkpoint}")
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        model.CLASSES = dataset.CLASSES
    if hasattr(dataset, "PALETTE"):
        model.PALETTE = dataset.PALETTE

    if not torch.cuda.is_available():
        raise RuntimeError("Segmentation evaluation requires CUDA in the MMSeg environment.")

    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.eval()

    print("\n🚀 Running single-GPU evaluation")
    results = single_gpu_test(
        model,
        data_loader,
        show=args.show,
        out_dir=args.out_dir,
        efficient_test=args.efficient_test,
    )

    print("\n📈 Computing mIoU metrics")
    eval_results = dataset.evaluate(results, metric="mIoU", logger="silent")

    print("\n" + "=" * 60)
    print("SEGMENTATION EVALUATION RESULTS")
    print("=" * 60)
    for key, val in eval_results.items():
        if isinstance(val, float):
            print(f"{key:20s}: {val * 100:.2f}%")
        else:
            print(f"{key:20s}: {val}")
    print("=" * 60)

    return eval_results


if __name__ == "__main__":
    main()
