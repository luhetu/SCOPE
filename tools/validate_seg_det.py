#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke-validate segmentation/detection configs and custom dense backbones.

The full training stack requires mmcv-full C++/CUDA extensions.  This script
stubs the small OpenMMLab API surface needed to exercise this repository's
config builders, then runs CPU forward checks for the custom ViT backbones.
"""

import argparse
import glob
import os
import sys
import types
from typing import Any, Dict, Iterable, List, Tuple

import torch
import yaml


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class Config(dict):
    """Minimal mmcv.Config replacement with attribute access."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class Registry:
    def __init__(self):
        self.module_dict = {}

    def register_module(self, name=None, module=None, force=False, **kwargs):
        def _register(cls):
            key = name or cls.__name__
            if not force and key in self.module_dict:
                return cls
            self.module_dict[key] = cls
            return cls

        if module is not None:
            key = name or module.__name__
            if force or key not in self.module_dict:
                self.module_dict[key] = module
            return module
        return _register


def _install_openmmlab_stubs() -> None:
    mmcv = types.ModuleType("mmcv")
    mmcv.Config = Config
    sys.modules["mmcv"] = mmcv

    mmcv_parallel = types.ModuleType("mmcv.parallel")
    mmcv_parallel.MMDataParallel = object
    sys.modules["mmcv.parallel"] = mmcv_parallel

    for root in ("mmdet", "mmseg"):
        sys.modules[root] = types.ModuleType(root)

        apis = types.ModuleType(f"{root}.apis")
        apis.set_random_seed = lambda *args, **kwargs: None
        if root == "mmdet":
            apis.train_detector = lambda *args, **kwargs: None
        else:
            apis.train_segmentor = lambda *args, **kwargs: None
            apis.single_gpu_test = lambda *args, **kwargs: []
        sys.modules[f"{root}.apis"] = apis

        datasets = types.ModuleType(f"{root}.datasets")
        datasets.build_dataset = lambda *args, **kwargs: None
        if root == "mmseg":
            datasets.build_dataloader = lambda *args, **kwargs: None
        sys.modules[f"{root}.datasets"] = datasets

        models = types.ModuleType(f"{root}.models")
        if root == "mmdet":
            models.build_detector = lambda *args, **kwargs: None
        else:
            models.build_segmentor = lambda *args, **kwargs: None
        sys.modules[f"{root}.models"] = models

        builder = types.ModuleType(f"{root}.models.builder")
        builder.BACKBONES = Registry()
        sys.modules[f"{root}.models.builder"] = builder


FLOAT_KEYS = {
    "lr",
    "min_lr",
    "warmup_epochs",
    "drop_path_rate",
    "weight_decay",
    "dropout",
    "emb_dropout",
    "layer_decay_rate",
}
INT_KEYS = {
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
    "seg_neck_dim",
    "seg_aux_in_index",
}
BOOL_KEYS = {"amp", "aug", "nowandb", "use_cls_token"}


def _coerce_value(key: str, value: Any) -> Any:
    if value is None:
        return None
    if key in FLOAT_KEYS:
        return float(value)
    if key in INT_KEYS:
        return int(value)
    if key in BOOL_KEYS:
        return bool(value)
    if key == "pretrained" and value == "null":
        return None
    return value


def _load_args(path: str) -> argparse.Namespace:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    # Include optional CLI/config fields as explicit None values to catch the
    # same default-handling path used by train.py after argparse parsing.
    args = argparse.Namespace(
        cfg=path,
        resume="",
        workers_per_gpu=None,
        model=None,
        data_dir=None,
        time_profile=False,
        time_profile_interval=1000,
        checkpoint_interval=None,
        eval_interval=None,
        log_interval=None,
        seg_head_dim=None,
        seg_aux_dim=None,
        seg_aux_in_index=None,
        seg_norm_type=None,
        seg_neck_style=None,
        seg_neck_dim=None,
        crop_size=None,
        img_scale=None,
        test_img_scale=None,
        backbone_size=None,
        det_neck_type=None,
        warmup_iters=None,
        warmup_epochs=None,
        betas=None,
        weight_decay=None,
        layer_decay_rate=None,
        seed=None,
        run_tag=None,
    )
    for key, value in raw.items():
        setattr(args, key, _coerce_value(key, value))
    return args


def _run_name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _validate_seg_configs() -> List[str]:
    from tasks.segmentation import SegmentationTask

    passed = []
    for path in sorted(glob.glob(os.path.join(REPO_ROOT, "configs", "seg_*.yaml"))):
        args = _load_args(path)
        task = SegmentationTask.__new__(SegmentationTask)
        task.args = args
        task.run_name = _run_name(path)
        cfg = task._build_mmseg_config()
        assert cfg.model["type"] == "EncoderDecoder"
        assert cfg.data["workers_per_gpu"] > 0
        assert cfg.log_config["interval"] > 0
        passed.append(os.path.relpath(path, REPO_ROOT))
    return passed


def _validate_det_configs() -> List[str]:
    from tasks.detection import DetectionTask

    passed = []
    for path in sorted(glob.glob(os.path.join(REPO_ROOT, "configs", "detection_*.yaml"))):
        args = _load_args(path)
        task = DetectionTask.__new__(DetectionTask)
        task.args = args
        task.run_name = _run_name(path)
        cfg = task._build_mmdet_config()
        assert cfg.model["type"] == "MaskRCNN"
        assert cfg.data["workers_per_gpu"] > 0
        assert cfg.log_config["interval"] > 0
        passed.append(os.path.relpath(path, REPO_ROOT))
    return passed


def _expected_shapes(batch: int, channels: int) -> Tuple[Tuple[int, int, int, int], ...]:
    return (
        (batch, channels, 8, 8),
        (batch, channels, 4, 4),
        (batch, channels, 2, 2),
        (batch, channels, 1, 1),
    )


def _backbone_cases() -> Iterable[Tuple[str, Any, Dict[str, Any]]]:
    from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone

    common = dict(
        image_size=32,
        patch_size=16,
        dim=32,
        depth=4,
        heads=4,
        mlp_dim=64,
        dim_head=8,
        out_indices=(0, 1, 2, 3),
    )
    for style in ("resize", "simple_fpn"):
        yield f"vit/{style}", ViTBackbone, dict(common, fpn_adapter_style=style)
        for use_cls_token in (False, True):
            yield (
                f"vitcope/{style}/cls={use_cls_token}",
                ViTCoPEBackbone,
                dict(common, fpn_adapter_style=style, use_cls_token=use_cls_token),
            )
            yield (
                f"vitscope/{style}/cls={use_cls_token}",
                ViTSCoPEBackbone,
                dict(common, fpn_adapter_style=style, use_cls_token=use_cls_token),
            )


def _validate_backbones() -> List[str]:
    batch, channels = 2, 32
    image = torch.randn(batch, 3, 32, 32)
    expected = _expected_shapes(batch, channels)
    passed = []
    with torch.no_grad():
        for name, cls, kwargs in _backbone_cases():
            model = cls(**kwargs).eval()
            outputs = model(image)
            shapes = tuple(tuple(out.shape) for out in outputs)
            assert shapes == expected, f"{name}: expected {expected}, got {shapes}"
            passed.append(name)
    return passed


def main() -> None:
    _install_openmmlab_stubs()

    seg = _validate_seg_configs()
    det = _validate_det_configs()
    backbones = _validate_backbones()

    print(f"Validated {len(seg)} segmentation configs")
    print(f"Validated {len(det)} detection configs")
    print(f"Validated {len(backbones)} custom backbone CPU forward cases")


if __name__ == "__main__":
    main()
