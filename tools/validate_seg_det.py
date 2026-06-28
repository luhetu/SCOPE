#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight validation for segmentation and detection task code.

The full training stack needs mmcv-full CUDA/C++ extensions plus ADE20K/COCO
data. This script validates the local config-builder logic and custom backbone
forward paths without those heavyweight runtime requirements.
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.update(dict(*args, **kwargs))

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = _wrap(value)

    def update(self, other=None, **kwargs):
        items = {}
        if other:
            items.update(other)
        items.update(kwargs)
        for key, value in items.items():
            self[key] = _wrap(value)


class Config(AttrDict):
    pass


def _wrap(value):
    if isinstance(value, AttrDict):
        return value
    if isinstance(value, dict):
        return AttrDict(value)
    if isinstance(value, list):
        return [_wrap(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_wrap(item) for item in value)
    return value


class Registry:
    def __init__(self, name):
        self.name = name
        self.module_dict = {}

    def register_module(self, name=None, module=None, force=False):
        def _register(cls):
            module_name = name or cls.__name__
            if not force and module_name in self.module_dict:
                return cls
            self.module_dict[module_name] = cls
            return cls

        if module is not None:
            return _register(module)
        return _register


def _new_module(name):
    module = types.ModuleType(name)
    sys.modules[name] = module
    return module


def _install_openmmlab_stubs():
    mmcv = _new_module("mmcv")
    mmcv.Config = Config

    mmcv_parallel = _new_module("mmcv.parallel")
    mmcv_parallel.MMDataParallel = object
    mmcv.parallel = mmcv_parallel

    mmseg = _new_module("mmseg")
    mmseg_apis = _new_module("mmseg.apis")
    mmseg_apis.set_random_seed = lambda *args, **kwargs: None
    mmseg_apis.single_gpu_test = lambda *args, **kwargs: []
    mmseg_apis.train_segmentor = lambda *args, **kwargs: None
    mmseg_datasets = _new_module("mmseg.datasets")
    mmseg_datasets.build_dataloader = lambda *args, **kwargs: None
    mmseg_datasets.build_dataset = lambda *args, **kwargs: None
    mmseg_models = _new_module("mmseg.models")
    mmseg_models.build_segmentor = lambda *args, **kwargs: None
    mmseg_builder = _new_module("mmseg.models.builder")
    mmseg_builder.BACKBONES = Registry("mmseg_backbone")
    mmseg.models = mmseg_models
    mmseg.apis = mmseg_apis
    mmseg.datasets = mmseg_datasets
    mmseg_models.builder = mmseg_builder

    mmdet = _new_module("mmdet")
    mmdet_apis = _new_module("mmdet.apis")
    mmdet_apis.set_random_seed = lambda *args, **kwargs: None
    mmdet_apis.train_detector = lambda *args, **kwargs: None
    mmdet_datasets = _new_module("mmdet.datasets")
    mmdet_datasets.build_dataset = lambda *args, **kwargs: None
    mmdet_models = _new_module("mmdet.models")
    mmdet_models.build_detector = lambda *args, **kwargs: None
    mmdet_builder = _new_module("mmdet.models.builder")
    mmdet_builder.BACKBONES = Registry("mmdet_backbone")
    mmdet.models = mmdet_models
    mmdet.apis = mmdet_apis
    mmdet.datasets = mmdet_datasets
    mmdet_models.builder = mmdet_builder


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
    "embed_dim",
    "window_size",
}

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

BOOL_KEYS = {"amp", "aug", "nowandb", "use_cls_token", "final_eval"}

DEFAULTS = {
    "cfg": "",
    "resume": "",
    "workers_per_gpu": None,
    "model": None,
    "data_dir": "./datasets",
    "time_profile": False,
    "time_profile_interval": 1000,
    "pretrained": "",
    "run_tag": None,
    "seed": None,
    "task": None,
    "bs": 2,
    "size": 224,
    "patch": 16,
    "dim": 192,
    "depth": 12,
    "heads": 3,
    "mlp_dim": 768,
    "dim_head": 64,
    "out_indices": (3, 5, 7, 11),
    "drop_path_rate": 0.0,
    "use_cls_token": False,
    "n_epochs": 1,
    "max_iters": None,
    "warmup_epochs": 0.0,
    "warmup_iters": None,
    "lr": 1e-4,
    "min_lr": 0.0,
    "weight_decay": 0.01,
    "betas": None,
    "amp": False,
    "nowandb": True,
    "final_eval": False,
    "checkpoint_interval": 5000,
    "eval_interval": 2001,
    "log_interval": None,
    "img_scale": None,
    "test_img_scale": None,
    "crop_size": None,
    "backbone_size": None,
    "seg_neck_style": "xcit_fpn",
    "seg_head_dim": None,
    "seg_aux_dim": None,
    "seg_neck_dim": None,
    "seg_aux_in_index": None,
    "seg_norm_type": "SyncBN",
    "layer_decay_rate": 1.0,
    "det_neck_type": "fpn",
    "embed_dim": 96,
    "depths": [2, 2, 6, 2],
    "num_heads": [3, 6, 12, 24],
    "window_size": 7,
}


def _coerce_value(key, value):
    if value == "null":
        return None
    if value is None:
        return None
    if key in INT_KEYS:
        return int(value)
    if key in FLOAT_KEYS:
        return float(value)
    if key in BOOL_KEYS:
        return bool(value)
    return value


def _load_args(config_path):
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    values = dict(DEFAULTS)
    values["cfg"] = str(config_path)
    for key, value in raw.items():
        values[key] = _coerce_value(key, value)
    return argparse.Namespace(**values)


def _build_task_config(task_cls, args, method_name):
    task = object.__new__(task_cls)
    task.args = args
    task.device = "cpu"
    task.run_name = task._build_run_name()
    return getattr(task, method_name)()


def _assert_workers(cfg, config_path):
    workers = cfg.data.workers_per_gpu
    if not isinstance(workers, int):
        raise AssertionError(f"{config_path}: workers_per_gpu is not int: {workers!r}")
    if workers < 0:
        raise AssertionError(f"{config_path}: workers_per_gpu must be non-negative")


def _validate_segmentation_configs(segmentation_cls):
    paths = sorted(CONFIG_DIR.glob("seg_*.yaml"))
    if not paths:
        raise AssertionError("No segmentation configs found")

    for path in paths:
        args = _load_args(path)
        if args.task != "seg":
            raise AssertionError(f"{path}: expected task=seg, got {args.task!r}")
        cfg = _build_task_config(segmentation_cls, args, "_build_mmseg_config")
        _assert_workers(cfg, path)
        if cfg.model["type"] != "EncoderDecoder":
            raise AssertionError(f"{path}: expected EncoderDecoder")
        backbone = cfg.model["backbone"]
        if args.model == "swin":
            if backbone["type"] != "SwinTransformer":
                raise AssertionError(f"{path}: Swin config built {backbone['type']}")
        elif backbone["type"] not in {"ViTBackbone", "ViTCoPEBackbone", "ViTSCoPEBackbone"}:
            raise AssertionError(f"{path}: unsupported segmentation backbone {backbone['type']}")

    print(f"Validated {len(paths)} segmentation configs")


def _validate_detection_configs(detection_cls):
    paths = sorted(CONFIG_DIR.glob("detection_*.yaml"))
    if not paths:
        raise AssertionError("No detection configs found")

    for path in paths:
        args = _load_args(path)
        if args.task != "det":
            raise AssertionError(f"{path}: expected task=det, got {args.task!r}")
        cfg = _build_task_config(detection_cls, args, "_build_mmdet_config")
        _assert_workers(cfg, path)
        if cfg.model["type"] != "MaskRCNN":
            raise AssertionError(f"{path}: expected MaskRCNN")
        backbone = cfg.model["backbone"]
        if args.model == "swin":
            if backbone["type"] != "SwinTransformer":
                raise AssertionError(f"{path}: Swin config built {backbone['type']}")
            if "image_size" in backbone or "dim" in backbone:
                raise AssertionError(f"{path}: Swin config leaked ViT-only keys")
        elif backbone["type"] not in {"ViTBackbone", "ViTCoPEBackbone", "ViTSCoPEBackbone"}:
            raise AssertionError(f"{path}: unsupported detection backbone {backbone['type']}")

    print(f"Validated {len(paths)} detection configs")


def _validate_backbone_forward():
    import torch

    from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone

    expected_shapes = [
        (1, 32, 8, 8),
        (1, 32, 4, 4),
        (1, 32, 2, 2),
        (1, 32, 1, 1),
    ]
    cases = [
        ("vit", ViTBackbone, {}),
        ("vitcope_no_cls", ViTCoPEBackbone, {"use_cls_token": False}),
        ("vitcope_cls", ViTCoPEBackbone, {"use_cls_token": True}),
        ("vitscope_no_cls", ViTSCoPEBackbone, {"use_cls_token": False}),
        ("vitscope_cls", ViTSCoPEBackbone, {"use_cls_token": True}),
    ]

    x = torch.randn(1, 3, 32, 32)
    common = dict(
        image_size=32,
        patch_size=16,
        dim=32,
        depth=4,
        heads=2,
        mlp_dim=64,
        dim_head=16,
        out_indices=(0, 1, 2, 3),
    )

    for adapter_style in ("resize", "simple_fpn"):
        for name, cls, extra in cases:
            model = cls(**common, fpn_adapter_style=adapter_style, **extra)
            model.eval()
            with torch.no_grad():
                outputs = model(x)
            shapes = [tuple(output.shape) for output in outputs]
            if shapes != expected_shapes:
                raise AssertionError(
                    f"{name}/{adapter_style}: expected {expected_shapes}, got {shapes}"
                )

    print("Validated custom backbone CPU forward smoke tests")


def main():
    sys.path.insert(0, str(ROOT))
    _install_openmmlab_stubs()

    from tasks.detection import DetectionTask
    from tasks.segmentation import SegmentationTask

    _validate_segmentation_configs(SegmentationTask)
    _validate_detection_configs(DetectionTask)
    _validate_backbone_forward()
    print("Segmentation and detection validation passed")


if __name__ == "__main__":
    main()
