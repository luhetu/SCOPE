#!/usr/bin/env python3
"""Smoke checks for segmentation and detection wiring.

The full downstream training jobs require datasets plus a compiled mmcv-full
extension. This script verifies the Python/config/model wiring with tiny inputs
so it can run on a CPU-only development machine. When mmcv._ext is unavailable,
it installs a dynamic stub that is sufficient for import/build checks; it still
does not exercise custom CUDA/C++ ops such as NMS or RoIAlign.
"""

from __future__ import annotations

import argparse
import sys
import types
from argparse import Namespace
from pathlib import Path

import torch
import yaml


class _FakeMMCVExt(types.ModuleType):
    """Dynamic stand-in for mmcv._ext during CPU-only smoke checks."""

    def __init__(self) -> None:
        super().__init__("mmcv._ext")
        self.__file__ = "<fake mmcv._ext>"
        self.__package__ = "mmcv"

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)

        def _missing(*args, **kwargs):
            raise RuntimeError(
                f"mmcv._ext function {name} is unavailable in smoke mode. "
                "Install mmcv-full to run full detection/segmentation training."
            )

        setattr(self, name, _missing)
        return _missing


def _ensure_mmcv_ext(allow_fake: bool) -> None:
    try:
        __import__("mmcv._ext")
        return
    except Exception:
        if not allow_fake:
            raise

    sys.modules["mmcv._ext"] = _FakeMMCVExt()
    print("Using fake mmcv._ext for import/build-only smoke checks.")


def _check_backbone_feature_shapes() -> None:
    from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone

    cases = [
        ("vit_resize", ViTBackbone, {"fpn_adapter_style": "resize"}),
        ("vit_simple_fpn", ViTBackbone, {"fpn_adapter_style": "simple_fpn"}),
        ("vitcope_resize", ViTCoPEBackbone, {"fpn_adapter_style": "resize"}),
        ("vitcope_simple_fpn", ViTCoPEBackbone, {"fpn_adapter_style": "simple_fpn"}),
        ("vitscope_resize", ViTSCoPEBackbone, {"fpn_adapter_style": "resize"}),
        ("vitscope_simple_fpn", ViTSCoPEBackbone, {"fpn_adapter_style": "simple_fpn"}),
    ]
    expected = [
        (2, 32, 32, 32),
        (2, 32, 16, 16),
        (2, 32, 8, 8),
        (2, 32, 4, 4),
    ]

    for name, cls, extra in cases:
        torch.manual_seed(0)
        model = cls(
            image_size=64,
            patch_size=8,
            dim=32,
            depth=4,
            heads=4,
            mlp_dim=64,
            dim_head=8,
            out_indices=(0, 1, 2, 3),
            **extra,
        )
        model.eval()
        with torch.no_grad():
            shapes = [tuple(out.shape) for out in model(torch.randn(2, 3, 64, 64))]
        if shapes != expected:
            raise AssertionError(f"{name} shapes {shapes} != expected {expected}")

    print("Backbone feature-map smoke checks passed.")


def _base_args(task: str) -> dict:
    return {
        "cfg": "",
        "model": "vit",
        "data_dir": "/tmp/nonexistent",
        "bs": 2,
        "pretrained": None,
        "size": 64,
        "patch": 8,
        "dim": 32,
        "depth": 4,
        "heads": 4,
        "mlp_dim": 64,
        "dim_head": 8,
        "n_epochs": 1,
        "lr": 1e-4,
        "warmup_iters": 0,
        "warmup_epochs": 0,
        "weight_decay": 0.01 if task == "seg" else 0.05,
        "amp": False,
        "nowandb": True,
        "task": task,
        "drop_path_rate": 0.0,
        "out_indices": (0, 1, 2, 3),
        "workers_per_gpu": 0,
        "log_interval": 10,
        "eval_interval": 11,
        "checkpoint_interval": 50,
        "crop_size": 64,
        "img_scale": (64, 64),
        "test_img_scale": (64, 64),
        "seg_aux_dim": 32,
        "embed_dim": 32,
        "depths": [1, 1, 1, 1],
        "num_heads": [1, 2, 4, 8],
        "window_size": 4,
    }


def _check_tiny_task_models() -> None:
    from mmdet.models import build_detector
    from mmseg.models import build_segmentor
    from tasks.detection import DetectionTask
    from tasks.segmentation import SegmentationTask

    seg_args = Namespace(**_base_args("seg"))
    seg_task = SegmentationTask.__new__(SegmentationTask)
    seg_task.args = seg_args
    seg_task.run_name = "smoke_seg"
    seg_cfg = seg_task._build_mmseg_config()
    seg_cfg.model.decode_head.num_classes = 3
    seg_cfg.model.auxiliary_head.num_classes = 3
    seg_model = build_segmentor(
        seg_cfg.model,
        train_cfg=seg_cfg.get("train_cfg"),
        test_cfg=seg_cfg.get("test_cfg"),
    )

    seg_model.train()
    img = torch.randn(2, 3, 64, 64)
    img_metas = [
        {
            "img_shape": (64, 64, 3),
            "ori_shape": (64, 64, 3),
            "pad_shape": (64, 64, 3),
            "scale_factor": 1.0,
            "flip": False,
        }
        for _ in range(2)
    ]
    gt_seg = torch.randint(0, 3, (2, 1, 64, 64), dtype=torch.long)
    losses = seg_model.forward_train(img, img_metas, gt_seg)
    required_losses = {
        "decode.loss_seg",
        "decode.acc_seg",
        "aux.loss_seg",
        "aux.acc_seg",
    }
    if set(losses) != required_losses:
        raise AssertionError(f"Unexpected segmentation losses: {sorted(losses)}")

    det_args = Namespace(**_base_args("det"))
    det_task = DetectionTask.__new__(DetectionTask)
    det_task.args = det_args
    det_task.run_name = "smoke_det"
    det_cfg = det_task._build_mmdet_config()
    det_cfg.model.roi_head.bbox_head.num_classes = 3
    det_cfg.model.roi_head.mask_head.num_classes = 3
    det_model = build_detector(
        det_cfg.model,
        train_cfg=det_cfg.get("train_cfg"),
        test_cfg=det_cfg.get("test_cfg"),
    )

    det_model.eval()
    with torch.no_grad():
        feat_shapes = [
            tuple(feat.shape) for feat in det_model.extract_feat(torch.randn(2, 3, 64, 64))
        ]
    expected_shapes = [
        (2, 256, 32, 32),
        (2, 256, 16, 16),
        (2, 256, 8, 8),
        (2, 256, 4, 4),
        (2, 256, 2, 2),
    ]
    if feat_shapes != expected_shapes:
        raise AssertionError(
            f"Detection FPN shapes {feat_shapes} != expected {expected_shapes}"
        )

    print("Tiny segmentation/detection model smoke checks passed.")


def _normalize_cfg_values(cfg: dict) -> dict:
    float_keys = {"lr", "min_lr", "warmup_epochs", "drop_path_rate", "weight_decay"}
    int_keys = {
        "bs",
        "size",
        "n_epochs",
        "max_iters",
        "warmup_iters",
        "checkpoint_interval",
        "eval_interval",
        "log_interval",
        "patch",
        "dim",
        "depth",
        "heads",
        "mlp_dim",
        "dim_head",
        "seg_head_dim",
        "seg_aux_dim",
        "embed_dim",
        "window_size",
    }
    for key in float_keys:
        if key in cfg and cfg[key] is not None:
            cfg[key] = float(cfg[key])
    for key in int_keys:
        if key in cfg and cfg[key] is not None:
            cfg[key] = int(cfg[key])
    return cfg


def _config_defaults(task: str) -> dict:
    defaults = _base_args(task)
    defaults.update(
        {
            "size": 224,
            "patch": 16,
            "dim": 192,
            "depth": 12,
            "heads": 3,
            "mlp_dim": 768,
            "dim_head": 64,
            "bs": 1,
            "crop_size": 512,
            "img_scale": [2048, 512] if task == "seg" else [1333, 800],
            "test_img_scale": [2048, 512],
            "seg_aux_dim": 256,
            "embed_dim": 96,
            "depths": [2, 2, 6, 2],
            "num_heads": [3, 6, 12, 24],
            "window_size": 7,
        }
    )
    return defaults


def _check_downstream_configs(config_dir: Path) -> None:
    from tasks.detection import DetectionTask
    from tasks.segmentation import SegmentationTask

    checked = 0
    for path in sorted(config_dir.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text()) or {}
        task_name = raw.get("task")
        if task_name not in {"seg", "det"}:
            continue

        cfg = _config_defaults(task_name)
        cfg.update(_normalize_cfg_values(raw))
        cfg["cfg"] = str(path)
        args = Namespace(**cfg)

        if task_name == "seg":
            task = SegmentationTask.__new__(SegmentationTask)
            task.args = args
            task.run_name = path.stem
            task._build_mmseg_config()
        else:
            task = DetectionTask.__new__(DetectionTask)
            task.args = args
            task.run_name = path.stem
            task._build_mmdet_config()

        checked += 1

    if checked == 0:
        raise AssertionError(f"No downstream configs found in {config_dir}")

    print(f"Built {checked} downstream segmentation/detection configs.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs"),
        help="Directory containing YAML configs to validate.",
    )
    parser.add_argument(
        "--require-mmcv-ext",
        action="store_true",
        help="Fail instead of using a fake mmcv._ext when compiled ops are missing.",
    )
    args = parser.parse_args()

    _ensure_mmcv_ext(allow_fake=not args.require_mmcv_ext)
    _check_backbone_feature_shapes()
    _check_tiny_task_models()
    _check_downstream_configs(args.config_dir)
    print("Downstream segmentation/detection smoke checks passed.")


if __name__ == "__main__":
    main()
