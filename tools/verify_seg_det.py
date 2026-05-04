#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke-check segmentation and detection task wiring.

The full ADE20K/COCO training jobs require a dedicated OpenMMLab environment
and real datasets. This verifier keeps the check fast by validating config
files first, then (when torch/mmcv/mmseg/mmdet are installed) building tiny
segmentation and detection models and exercising their backbone/head contracts
with synthetic images.
"""

import argparse
import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIGS = (
    "configs/seg_vit_tiny.yaml",
    "configs/seg_vitcope_tiny.yaml",
    "configs/seg_vitscope_tiny.yaml",
    "configs/detection_vit_tiny.yaml",
    "configs/detection_vitcope_tiny.yaml",
    "configs/detection_vitscope_tiny.yaml",
)

COMMON_REQUIRED = (
    "task",
    "model",
    "data_dir",
    "bs",
    "size",
    "patch",
    "dim",
    "depth",
    "heads",
    "mlp_dim",
    "dim_head",
    "n_epochs",
    "lr",
)

TASK_REQUIRED = {
    "seg": ("crop_size", "img_scale", "test_img_scale", "max_iters"),
    "det": ("img_scale",),
}


def _load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a YAML mapping")
    return data


def _validate_config(path, data):
    missing = [key for key in COMMON_REQUIRED if key not in data]
    task = data.get("task")
    missing.extend(key for key in TASK_REQUIRED.get(task, ()) if key not in data)

    if task not in TASK_REQUIRED:
        raise ValueError(f"{path}: task must be one of {sorted(TASK_REQUIRED)}, got {task!r}")
    if missing:
        raise ValueError(f"{path}: missing required keys: {', '.join(sorted(set(missing)))}")

    if task == "seg" and data["model"] == "swin":
        raise ValueError(f"{path}: this verifier expects ViT-family segmentation configs")
    if task == "det" and data["model"] == "swin":
        raise ValueError(f"{path}: this verifier expects ViT-family detection configs")


def _check_dependency(module_name):
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "installed")
        return True, str(version)
    except Exception as exc:  # pragma: no cover - diagnostic path
        return False, f"{type(exc).__name__}: {exc}"


def _check_dependencies():
    checks = {
        "torch": _check_dependency("torch"),
        "timm": _check_dependency("timm"),
        "einops": _check_dependency("einops"),
        "mmcv": _check_dependency("mmcv"),
        "mmcv._ext": _check_dependency("mmcv._ext"),
        "mmseg": _check_dependency("mmseg"),
        "mmdet": _check_dependency("mmdet"),
    }
    return checks


def _as_namespace(data):
    args = dict(data)
    args.update(
        bs=1,
        workers_per_gpu=0,
        pretrained=None,
        nowandb=True,
        amp=False,
        size=64,
        patch=16,
        dim=64,
        depth=4,
        heads=4,
        mlp_dim=128,
        dim_head=16,
        out_indices=(0, 1, 2, 3),
        drop_path_rate=0.0,
        warmup_iters=0,
        run_tag="verify",
        cfg="",
    )
    if args["task"] == "seg":
        args.update(
            crop_size=64,
            img_scale=(64, 64),
            test_img_scale=(64, 64),
            max_iters=2,
            eval_interval=3,
            log_interval=2,
            checkpoint_interval=2,
            seg_head_dim=64,
            seg_aux_dim=64,
        )
    else:
        args.update(
            img_scale=(64, 64),
            n_epochs=1,
            log_interval=1,
        )
    return SimpleNamespace(**args)


def _run_segmentation_smoke(data):
    import torch
    from mmseg.models import build_segmentor
    from tasks.segmentation import SegmentationTask

    args = _as_namespace(data)
    task = SegmentationTask.__new__(SegmentationTask)
    task.args = args
    task.device = "cpu"
    task.run_name = task._build_run_name()
    cfg = task._build_mmseg_config()
    model = build_segmentor(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    model.eval()

    image = torch.randn(1, 3, args.crop_size, args.crop_size)
    with torch.no_grad():
        features = model.backbone(image)
        logits = model.decode_head(features)

    if len(features) != 4:
        raise AssertionError(f"expected 4 segmentation backbone features, got {len(features)}")
    if logits.shape[:2] != (1, 150):
        raise AssertionError(f"unexpected segmentation logits shape: {tuple(logits.shape)}")


def _run_detection_smoke(data):
    import torch
    from mmdet.models import build_detector
    from tasks.detection import DetectionTask

    args = _as_namespace(data)
    task = DetectionTask.__new__(DetectionTask)
    task.args = args
    task.device = "cpu"
    task.run_name = task._build_run_name()
    cfg = task._build_mmdet_config()
    model = build_detector(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    model.eval()

    image = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        features = model.backbone(image)
        pyramid = model.neck(features)
        rpn_scores, rpn_deltas = model.rpn_head(pyramid)

    if len(features) != 4:
        raise AssertionError(f"expected 4 detection backbone features, got {len(features)}")
    if len(pyramid) != 5:
        raise AssertionError(f"expected 5 FPN outputs, got {len(pyramid)}")
    if not rpn_scores or not rpn_deltas:
        raise AssertionError("RPN head did not produce classification and bbox outputs")


def main():
    parser = argparse.ArgumentParser(description="Verify segmentation/detection task wiring")
    parser.add_argument(
        "--config",
        action="append",
        dest="configs",
        help="Config path to verify. May be passed more than once.",
    )
    parser.add_argument(
        "--strict-deps",
        action="store_true",
        help="Return failure when torch/OpenMMLab dependencies are unavailable.",
    )
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="Only validate config files; skip dependency and model smoke checks.",
    )
    args = parser.parse_args()

    os.chdir(ROOT)
    configs = [Path(path) for path in (args.configs or DEFAULT_CONFIGS)]

    loaded = []
    print("== Config validation ==")
    for path in configs:
        data = _load_config(path)
        _validate_config(path, data)
        loaded.append((path, data))
        print(f"OK {path} ({data['task']}/{data['model']})")

    if args.config_only:
        return 0

    print("\n== Dependency check ==")
    checks = _check_dependencies()
    missing = []
    for name, (ok, detail) in checks.items():
        status = "OK" if ok else "MISSING"
        print(f"{status} {name}: {detail}")
        if not ok:
            missing.append(name)

    if missing:
        print("\nSkipping model smoke checks because dependencies are missing:")
        print("  " + ", ".join(missing))
        if args.strict_deps:
            return 1
        return 0

    print("\n== Model smoke checks ==")
    failed = []
    for path, data in loaded:
        try:
            if data["task"] == "seg":
                _run_segmentation_smoke(data)
            elif data["task"] == "det":
                _run_detection_smoke(data)
            else:
                raise ValueError(f"unsupported task: {data['task']}")
            print(f"OK {path}")
        except Exception as exc:
            failed.append((path, exc))
            print(f"FAIL {path}: {type(exc).__name__}: {exc}")

    if failed:
        return 1

    print("\nSegmentation and detection smoke checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
