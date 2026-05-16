#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke-test segmentation and detection model construction.

This check intentionally avoids dataset access and full train/eval loops. It
builds the local MMSeg/MMDet task configs with tiny ViT-family backbones and
runs feature extraction for segmentation and detection variants.

Use --stub-mmcv-ops only in lightweight environments that have pure-Python
mmcv but not mmcv-full. The stub is enough for import/model-construction checks
and feature extraction, but it does not validate compiled ops such as NMS or
RoIAlign.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import torch


def maybe_stub_mmcv_ops(enabled: bool) -> None:
    if not enabled:
        return

    try:
        import mmcv.utils.ext_loader
    except Exception as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("Unable to import mmcv.utils.ext_loader") from exc

    class DummyExt:
        def __getattr__(self, name):
            def _missing(*args, **kwargs):
                raise RuntimeError(
                    f"mmcv compiled op {name} is unavailable in this smoke-test environment"
                )

            return _missing

    mmcv.utils.ext_loader.load_ext = lambda *args, **kwargs: DummyExt()


def make_args(task: str, model: str, cli_args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        task=task,
        model=model,
        cfg="",
        run_tag="smoke",
        pretrained=None,
        data_dir="./datasets/nonexistent",
        bs=1,
        workers_per_gpu=0,
        size=cli_args.image_size,
        backbone_size=cli_args.image_size,
        crop_size=cli_args.image_size,
        img_scale=[cli_args.image_size, cli_args.image_size],
        test_img_scale=[cli_args.image_size, cli_args.image_size],
        patch=cli_args.patch_size,
        dim=cli_args.dim,
        depth=cli_args.depth,
        heads=cli_args.heads,
        mlp_dim=cli_args.mlp_dim,
        dim_head=cli_args.dim_head,
        out_indices=tuple(range(cli_args.depth)),
        lr=1e-4,
        min_lr=0.0,
        n_epochs=1,
        max_iters=cli_args.max_iters,
        warmup_iters=0,
        checkpoint_interval=1,
        eval_interval=2,
        log_interval=1,
        weight_decay=0.01,
        amp=False,
        nowandb=True,
        drop_path_rate=0.0,
        det_neck_type="fpn",
        use_cls_token=(model == "vitscope"),
        seg_neck_style="none",
        seg_head_dim=cli_args.dim,
        seg_aux_dim=cli_args.seg_aux_dim,
        seg_aux_in_index=min(2, cli_args.depth - 1),
        seg_norm_type="GN",
        final_eval=False,
        min_pretrained_match_rate=0.0,
    )


def smoke_segmentation(model_name: str, cli_args: argparse.Namespace) -> list[tuple[int, ...]]:
    from mmseg.models import build_segmentor
    from tasks.segmentation import SegmentationTask

    task = object.__new__(SegmentationTask)
    task.args = make_args("seg", model_name, cli_args)
    task.run_name = f"seg_{model_name}_smoke"
    cfg = task._build_mmseg_config()
    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get("train_cfg"),
        test_cfg=cfg.get("test_cfg"),
    ).eval()
    with torch.no_grad():
        features = model.extract_feat(torch.randn(1, 3, cli_args.image_size, cli_args.image_size))
    return [tuple(feature.shape) for feature in features]


def smoke_detection(model_name: str, cli_args: argparse.Namespace) -> list[tuple[int, ...]]:
    from mmdet.models import build_detector
    from tasks.detection import DetectionTask

    task = object.__new__(DetectionTask)
    task.args = make_args("det", model_name, cli_args)
    task.run_name = f"det_{model_name}_smoke"
    cfg = task._build_mmdet_config()
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get("train_cfg"),
        test_cfg=cfg.get("test_cfg"),
    ).eval()
    with torch.no_grad():
        features = model.extract_feat(torch.randn(1, 3, cli_args.image_size, cli_args.image_size))
    return [tuple(feature.shape) for feature in features]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stub-mmcv-ops", action="store_true")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--dim-head", type=int, default=32)
    parser.add_argument("--mlp-dim", type=int, default=128)
    parser.add_argument("--seg-aux-dim", type=int, default=32)
    parser.add_argument("--max-iters", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    maybe_stub_mmcv_ops(args.stub_mmcv_ops)

    if args.image_size % args.patch_size != 0:
        raise ValueError("--image-size must be divisible by --patch-size")
    if args.seg_aux_dim % 32 != 0:
        raise ValueError("--seg-aux-dim must be divisible by 32 for GroupNorm")
    if args.depth < 1:
        raise ValueError("--depth must be at least 1")

    for model_name in ("vit", "vitcope", "vitscope"):
        shapes = smoke_segmentation(model_name, args)
        print(f"seg:{model_name}: {shapes}")

    for model_name in ("vit", "vitcope", "vitscope"):
        shapes = smoke_detection(model_name, args)
        print(f"det:{model_name}: {shapes}")


if __name__ == "__main__":
    main()
