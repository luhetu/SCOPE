#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPU smoke checks for dense prediction task wiring.

This script validates the segmentation and detection model construction plus a
single synthetic training-loss pass. It does not require ADE20K/COCO data or
pretrained checkpoints, so it is useful for checking integration problems before
launching the full CUDA training entrypoint.
"""

import argparse
from argparse import Namespace
from pathlib import Path
import sys
import warnings

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def _base_args(task, model, size, dim, depth, heads):
    return dict(
        cfg=f"smoke_{task}_{model}",
        task=task,
        model=model,
        data_dir="/tmp/nonexistent",
        bs=2,
        pretrained="",
        size=size,
        patch=16,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=dim * 2,
        dim_head=max(8, dim // heads),
        out_indices=tuple(range(depth)),
        lr=1e-4,
        min_lr=0.0,
        warmup_iters=0,
        warmup_epochs=0,
        weight_decay=0.01,
        opt="adamw",
        amp=False,
        nowandb=True,
        use_cls_token=True,
        drop_path_rate=0.0,
        workers_per_gpu=0,
        log_interval=1,
        eval_interval=99,
        checkpoint_interval=99,
        seed=1,
        run_tag=None,
    )


def check_backbones(size, dim, depth, heads):
    from models.vit_backbone import (
        ViTBackbone,
        ViTCoPEBackbone,
        ViTSCoPEBackbone,
    )

    cases = [
        ("vit", ViTBackbone, {}),
        ("vitcope", ViTCoPEBackbone, {"use_cls_token": False}),
        ("vitscope", ViTSCoPEBackbone, {"use_cls_token": True}),
    ]
    for name, cls, extra in cases:
        model = cls(
            image_size=size,
            patch_size=16,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=dim * 2,
            dim_head=max(8, dim // heads),
            out_indices=tuple(range(depth)),
            **extra,
        ).eval()
        with torch.no_grad():
            outs = model(torch.randn(1, 3, size, size))
        shapes = [tuple(out.shape) for out in outs]
        assert len(outs) == depth, f"{name}: expected {depth} outputs, got {len(outs)}"
        assert all(out.shape[0] == 1 and out.shape[1] == dim for out in outs), shapes
        print(f"backbone {name}: {shapes}")


def check_segmentation(model_name, size, dim, depth, heads):
    from mmseg.models import build_segmentor
    from tasks.segmentation import SegmentationTask

    args = Namespace(
        **_base_args("seg", model_name, size, dim, depth, heads),
        backbone_size=size,
        test_img_scale=(size, size),
        img_scale=(size, size),
        crop_size=size,
        seg_neck_style="xcit_fpn",
        max_iters=1,
        n_epochs=1,
        layer_decay_rate=1.0,
        seg_head_dim=dim,
        seg_aux_dim=dim,
        seg_aux_in_index=min(2, depth - 1),
        seg_norm_type="BN",
        final_eval=False,
    )
    task = SegmentationTask.__new__(SegmentationTask)
    task.args = args
    task.run_name = task._build_run_name()
    cfg = task._build_mmseg_config()

    model = build_segmentor(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    model.train()
    img = torch.randn(2, 3, size, size)
    img_metas = [
        dict(
            img_shape=(size, size, 3),
            ori_shape=(size, size, 3),
            pad_shape=(size, size, 3),
            scale_factor=1.0,
            flip=False,
        )
        for _ in range(2)
    ]
    gt = torch.randint(0, 150, (2, 1, size, size), dtype=torch.long)
    losses = model.forward_train(img, img_metas, gt)
    _assert_finite_losses(losses)
    print(f"segmentation {model_name}: {sorted(losses.keys())}")


def check_detection(model_name, size, dim, depth, heads):
    from mmdet.core import BitmapMasks
    from mmdet.models import build_detector
    from tasks.detection import DetectionTask

    args = Namespace(
        **_base_args("det", model_name, size, dim, depth, heads),
        det_neck_type="fpn",
        img_scale=(size, size),
        n_epochs=1,
        weight_decay=0.05,
        betas=(0.9, 0.999),
    )
    task = DetectionTask.__new__(DetectionTask)
    task.args = args
    task.run_name = task._build_run_name()
    cfg = task._build_mmdet_config()

    model = build_detector(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    model.train()
    img = torch.randn(2, 3, size, size)
    img_metas = [
        dict(
            img_shape=(size, size, 3),
            ori_shape=(size, size, 3),
            pad_shape=(size, size, 3),
            scale_factor=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            flip=False,
        )
        for _ in range(2)
    ]
    boxes = [
        torch.tensor([[8.0, 8.0, size * 0.6, size * 0.65]], dtype=torch.float32),
        torch.tensor([[size * 0.2, size * 0.15, size * 0.75, size * 0.8]], dtype=torch.float32),
    ]
    labels = [torch.tensor([1], dtype=torch.long), torch.tensor([2], dtype=torch.long)]
    masks = []
    for bbox in boxes:
        x1, y1, x2, y2 = bbox[0].int().tolist()
        arr = np.zeros((1, size, size), dtype=np.uint8)
        arr[0, y1:y2, x1:x2] = 1
        masks.append(BitmapMasks(arr, size, size))

    losses = model.forward_train(img, img_metas, boxes, labels, gt_masks=masks)
    _assert_finite_losses(losses)
    print(f"detection {model_name}: {sorted(losses.keys())}")


def _assert_finite_losses(losses):
    for name, value in losses.items():
        values = value if isinstance(value, (list, tuple)) else [value]
        for item in values:
            if hasattr(item, "detach"):
                assert torch.isfinite(item).all(), f"{name} produced non-finite loss"


def main():
    parser = argparse.ArgumentParser(description="Smoke test segmentation/detection code paths")
    parser.add_argument("--model", default="vitscope", choices=("vit", "vitcope", "vitscope"))
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--skip-backbones", action="store_true")
    parser.add_argument("--skip-seg", action="store_true")
    parser.add_argument("--skip-det", action="store_true")
    args = parser.parse_args()

    if not args.skip_backbones:
        check_backbones(args.size, args.dim, args.depth, args.heads)
    if not args.skip_seg:
        check_segmentation(args.model, args.size, args.dim, args.depth, args.heads)
    if not args.skip_det:
        check_detection(args.model, args.size, args.dim, args.depth, args.heads)
    print("dense task smoke checks OK")


if __name__ == "__main__":
    main()
