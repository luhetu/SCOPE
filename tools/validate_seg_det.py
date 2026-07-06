#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight segmentation/detection validation for Cursor Cloud.

This script validates SCOPE config construction without requiring mmcv-full's
compiled extensions, then smoke-tests the custom ViT backbones on CPU.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import types
from types import SimpleNamespace

import yaml


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
BOOL_KEYS = {"amp", "aug", "nowandb", "use_cls_token"}


class Config(dict):
    """Minimal mmcv.Config-compatible object for task config builders."""

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

    def register_module(self, name=None, module=None, force=False):
        if module is None:
            def decorator(cls):
                self.register_module(name=name or cls.__name__, module=cls, force=force)
                return cls

            return decorator

        registry_name = name or module.__name__
        if registry_name in self.module_dict and not force:
            return module
        self.module_dict[registry_name] = module
        return module


def _noop(*args, **kwargs):
    return None


def install_openmmlab_stubs():
    mmcv = types.ModuleType("mmcv")
    mmcv.Config = Config
    mmcv_parallel = types.ModuleType("mmcv.parallel")
    mmcv_parallel.MMDataParallel = object
    mmcv.parallel = mmcv_parallel

    mmseg = types.ModuleType("mmseg")
    mmseg_apis = types.ModuleType("mmseg.apis")
    mmseg_apis.set_random_seed = _noop
    mmseg_apis.single_gpu_test = lambda *args, **kwargs: []
    mmseg_apis.train_segmentor = _noop
    mmseg_datasets = types.ModuleType("mmseg.datasets")
    mmseg_datasets.build_dataloader = _noop
    mmseg_datasets.build_dataset = _noop
    mmseg_models = types.ModuleType("mmseg.models")
    mmseg_models.build_segmentor = _noop
    mmseg_builder = types.ModuleType("mmseg.models.builder")
    mmseg_builder.BACKBONES = Registry()
    mmseg_models.builder = mmseg_builder
    mmseg.models = mmseg_models
    mmseg.apis = mmseg_apis
    mmseg.datasets = mmseg_datasets

    mmdet = types.ModuleType("mmdet")
    mmdet_apis = types.ModuleType("mmdet.apis")
    mmdet_apis.set_random_seed = _noop
    mmdet_apis.train_detector = _noop
    mmdet_datasets = types.ModuleType("mmdet.datasets")
    mmdet_datasets.build_dataset = _noop
    mmdet_models = types.ModuleType("mmdet.models")
    mmdet_models.build_detector = _noop
    mmdet_builder = types.ModuleType("mmdet.models.builder")
    mmdet_builder.BACKBONES = Registry()
    mmdet_models.builder = mmdet_builder
    mmdet.models = mmdet_models
    mmdet.apis = mmdet_apis
    mmdet.datasets = mmdet_datasets

    sys.modules.update(
        {
            "mmcv": mmcv,
            "mmcv.parallel": mmcv_parallel,
            "mmseg": mmseg,
            "mmseg.apis": mmseg_apis,
            "mmseg.datasets": mmseg_datasets,
            "mmseg.models": mmseg_models,
            "mmseg.models.builder": mmseg_builder,
            "mmdet": mmdet,
            "mmdet.apis": mmdet_apis,
            "mmdet.datasets": mmdet_datasets,
            "mmdet.models": mmdet_models,
            "mmdet.models.builder": mmdet_builder,
        }
    )


def convert_value(key, value):
    if value == "null":
        return None
    if value is None:
        return None
    if key in INT_KEYS:
        return int(value)
    if key in FLOAT_KEYS:
        return float(value)
    if key in BOOL_KEYS:
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "on"}
        return bool(value)
    return value


def load_args(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    args = SimpleNamespace(
        cfg=cfg_path,
        resume="",
        workers_per_gpu=None,
        model=None,
        data_dir=None,
        time_profile=False,
        time_profile_interval=1000,
    )
    for key, value in raw.items():
        setattr(args, key, convert_value(key, value))
    return args


def assert_train_summary_fields(args):
    required = ["task", "model", "bs", "lr", "size", "patch"]
    if args.model == "swin":
        required.extend(["embed_dim", "depths", "num_heads", "window_size"])
    else:
        required.extend(["dim", "depth", "heads", "mlp_dim"])
    missing = [name for name in required if not hasattr(args, name)]
    if missing:
        raise AssertionError(f"{args.cfg} missing fields used by train.py: {missing}")


def validate_configs():
    from tasks.detection import DetectionTask
    from tasks.segmentation import SegmentationTask

    cfg_paths = sorted(glob.glob("configs/seg_*.yaml")) + sorted(glob.glob("configs/detection_*.yaml"))
    if not cfg_paths:
        raise AssertionError("No segmentation or detection config files found")

    for cfg_path in cfg_paths:
        args = load_args(cfg_path)
        assert_train_summary_fields(args)
        if args.task == "seg":
            task = object.__new__(SegmentationTask)
            task.args = args
            task.run_name = task._build_run_name()
            cfg = task._build_mmseg_config()
            assert cfg.model["type"] == "EncoderDecoder"
            assert cfg.data["workers_per_gpu"] == 4
            assert len(cfg.model["decode_head"]["in_channels"]) == 4
        elif args.task == "det":
            task = object.__new__(DetectionTask)
            task.args = args
            task.run_name = task._build_run_name()
            cfg = task._build_mmdet_config()
            assert cfg.model["type"] == "MaskRCNN"
            assert cfg.data["workers_per_gpu"] == 4
            assert cfg.model["neck"]["num_outs"] == 5
        else:
            raise AssertionError(f"{cfg_path} has unsupported task {args.task!r}")
        print(f"OK config: {cfg_path}")


def validate_backbones():
    import torch

    from models.vit_backbone import ViTBackbone, ViTCoPEBackbone, ViTSCoPEBackbone

    expected_hw = [(8, 8), (4, 4), (2, 2), (1, 1)]
    cases = [
        ("vit", ViTBackbone, {}),
        ("vitcope_no_cls", ViTCoPEBackbone, {"use_cls_token": False}),
        ("vitcope_cls", ViTCoPEBackbone, {"use_cls_token": True}),
        ("vitscope_no_cls", ViTSCoPEBackbone, {"use_cls_token": False}),
        ("vitscope_cls", ViTSCoPEBackbone, {"use_cls_token": True}),
    ]
    for style in ("resize", "simple_fpn"):
        for name, cls, extra in cases:
            model = cls(
                image_size=32,
                patch_size=16,
                dim=48,
                depth=4,
                heads=3,
                mlp_dim=96,
                dim_head=16,
                out_indices=(0, 1, 2, 3),
                fpn_adapter_style=style,
                **extra,
            )
            model.eval()
            with torch.no_grad():
                outputs = model(torch.randn(1, 3, 32, 32))
            assert len(outputs) == 4, f"{name}/{style} returned {len(outputs)} outputs"
            for output, (height, width) in zip(outputs, expected_hw):
                assert tuple(output.shape) == (1, 48, height, width), (
                    f"{name}/{style} output shape {tuple(output.shape)} != "
                    f"(1, 48, {height}, {width})"
                )
            print(f"OK backbone: {name} ({style})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs-only", action="store_true", help="Skip CPU backbone smoke tests")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    install_openmmlab_stubs()
    validate_configs()
    if not args.configs_only:
        validate_backbones()
    print("Segmentation/detection validation passed.")


if __name__ == "__main__":
    main()
